"""One fixed classifier in the separate A+B expansion experiment."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/.cache"))

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from sepsis_warning.data import config,environment
from sepsis_warning.ab_expansion import PROCESSED,OUTPUT,EVALUATION_SPLITS,content_hash,write_json,result_directory
from sepsis_warning.history_study import digest
from sepsis_warning.regularized import patient_bags
from sepsis_warning.evaluation import ranking_metrics,select_alert_threshold,alert_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",required=True)
    parser.add_argument("--model",choices=["lr","ebm","lightgbm"],required=True)
    args = parser.parse_args()
    directory = PROCESSED/"cases"/args.case
    metadata = json.loads((directory/"case.json").read_text())
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    parameters = protocol["model_settings"]["models"][args.model]
    stopping = protocol["model_settings"]["lightgbm_stopping"]
    signature = digest({"case":metadata["signature"],"parameters":parameters,"stopping":stopping,"worker":content_hash(__file__)})
    out = result_directory(args.case)
    out.mkdir(parents=True,exist_ok=True)
    result_path = out/f"{args.model}_results.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("signature") == signature:
            assert all((out/f"{args.model}_{split}_probability.npy").exists() for split in EVALUATION_SPLITS)
            print("Reusing completed expansion model",flush=True)
            return
    rows = pd.read_csv(directory/"train_rows.csv.gz")
    y = np.load(directory/"train_y.npy")
    np.testing.assert_array_equal(rows.label,y)
    processor = joblib.load(directory/"preprocessor.joblib")
    assert len(processor.names_) == metadata["features"]
    reused = metadata["cohort"] == "A_only"
    details = {}
    cfg = config()
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reused:
            reference = OUTPUT/"references"/f"{metadata['profile']}_{args.model}_model.joblib"
            bundle = joblib.load(reference)
            original = json.loads(reference.with_name(f"{metadata['profile']}_{args.model}_results.json").read_text())
            assert bundle["parameters"] == parameters and bundle["preprocessor"].names_ == processor.names_
            model = bundle["model"]
            fit_seconds = original["fit_seconds"]
            training_metrics = original["training"]
            details = {**original["details"],"reference_snapshot":str(reference.relative_to(ROOT))}
        else:
            x = np.load(directory/"train_X.npy",mmap_mode="r")
            assert x.shape == (len(rows),metadata["features"]) and not np.isinf(x).any()
            if args.model == "lr":
                assert np.isfinite(x).all()
                model = LogisticRegression(**parameters)
                with threadpool_limits(limits=cfg["lr"]["threads"]):model.fit(x,y)
                details["iterations"] = int(model.n_iter_[0])
                assert details["iterations"] < parameters["max_iter"]
            elif args.model == "ebm":
                from interpret.glassbox import ExplainableBoostingClassifier
                bags,assignments = patient_bags(rows,n_bags=parameters["outer_bags"],seed=2026)
                assignments.to_csv(out/"ebm_internal_patients.csv",index=False)
                model = ExplainableBoostingClassifier(feature_names=processor.names_,feature_types=["continuous"]*len(processor.names_),**parameters)
                with parallel_backend("threading",n_jobs=parameters["n_jobs"]),threadpool_limits(limits=1):model.fit(x,y,bags=bags)
                details["executed_cycles_including_patience"] = (model.best_iteration_[0]/len(model.term_features_)).tolist()
            else:
                import lightgbm as lgb
                bags,assignments = patient_bags(rows,n_bags=1,seed=2026)
                assignments.to_csv(out/"lightgbm_internal_patients.csv",index=False)
                inner,held = bags[:,0] > 0,bags[:,0] < 0
                internal = lgb.LGBMClassifier(**parameters)
                internal.fit(x[inner],y[inner],eval_set=[(x[held],y[held])],eval_metric=stopping["metric"],feature_name=processor.names_,
                             callbacks=[lgb.early_stopping(stopping["rounds"],first_metric_only=True,verbose=False,min_delta=stopping["min_delta"]),lgb.log_evaluation(0)])
                trees = max(1,int(internal.best_iteration_))
                details.update({"inner_selected_trees":trees,"early_stopping_patient_grouped":True,"refit_on_all_training_patients":True})
                del internal
                model = lgb.LGBMClassifier(**{**parameters,"n_estimators":trees})
                model.fit(x,y,feature_name=processor.names_)
            fit_seconds = time.monotonic()-started
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                training_probability = model.booster_.predict(x) if args.model == "lightgbm" else model.predict_proba(x)[:,1]
            training_metrics = ranking_metrics(y,training_probability)
            del x,training_probability
        def predict(split):
            matrix = np.load(directory/f"{split}_X.npy",mmap_mode="r")
            assert matrix.shape[1] == metadata["features"] and not np.isinf(matrix).any()
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                probability = model.booster_.predict(matrix) if args.model == "lightgbm" else model.predict_proba(matrix)[:,1]
            assert np.isfinite(probability).all() and ((probability >= 0)&(probability <= 1)).all()
            np.save(out/f"{args.model}_{split}_probability.npy",probability)
            return probability
        probabilities = {split:predict(split) for split in ["A_validation","B_validation"]}
        evaluation_rows = {split:pd.read_csv(PROCESSED/f"{split}_rows.csv.gz") for split in ["A_validation","B_validation"]}
        pooled_rows = pd.concat([evaluation_rows["A_validation"],evaluation_rows["B_validation"]],ignore_index=True)
        pooled_probability = np.r_[probabilities["A_validation"],probabilities["B_validation"]]
        threshold,search = select_alert_threshold(pooled_rows,pooled_probability,cfg)
        search.to_csv(out/f"{args.model}_threshold_search.csv",index=False)
        # B test is scored only after the fit and pooled-validation threshold are fixed.
        probabilities["B_test"] = predict("B_test")
        evaluation_rows["B_test"] = pd.read_csv(PROCESSED/"B_test_rows.csv.gz")
    evaluation,alerts = {},{}
    for split in EVALUATION_SPLITS:
        table = evaluation_rows[split]
        assert not set(rows.patient_id)&set(table.patient_id)
        evaluation[split] = ranking_metrics(table.label.to_numpy(),probabilities[split])
        alerts[split],episodes,_ = alert_replay(table,probabilities[split],threshold,cfg["alerts"]["suppression_hours"])
        episodes.to_csv(out/f"{args.model}_{split}_alert_episodes.csv",index=False)
    evaluation["pooled_validation"] = ranking_metrics(pooled_rows.label.to_numpy(),pooled_probability)
    alerts["pooled_validation"],_,_ = alert_replay(pooled_rows,pooled_probability,threshold,cfg["alerts"]["suppression_hours"])
    if reused:
        key = "validation" if metadata["profile"] == "common" else "evaluation"
        for name,value in evaluation["A_validation"].items():
            np.testing.assert_allclose(value,original[key][name],rtol=0,atol=1e-12)
    temporary = out/f"{args.model}_model.tmp"
    joblib.dump({"model":model,"preprocessor":processor,"case":metadata,"parameters":parameters,"threshold":threshold},temporary,compress=3)
    os.replace(temporary,out/f"{args.model}_model.joblib")
    runtime = environment()
    runtime["packages"].update({"interpret-core":version("interpret-core"),"lightgbm":version("lightgbm")})
    write_json(result_path,{"signature":signature,"case":metadata,"model":args.model,"parameters":parameters,"training":training_metrics,
               "evaluation":evaluation,"alerts":alerts,"threshold":threshold,"threshold_fit_on":["A_validation","B_validation"],
               "fit_seconds":round(fit_seconds,2),"reused_A_only_model":reused,"details":details,"warnings":[str(w.message) for w in caught],
               "A_test_evaluated":False,"B_test_is_unseen_hospital":False,"environment":runtime})
    print(f"Finished {metadata['cohort']}/{metadata['profile']}/{args.model}: A-val AP={evaluation['A_validation']['auprc_average_precision']:.4f}, B-test AP={evaluation['B_test']['auprc_average_precision']:.4f}",flush=True)


if __name__ == "__main__":main()
