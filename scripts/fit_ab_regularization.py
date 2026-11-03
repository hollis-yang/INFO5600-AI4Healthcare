"""Fit one frozen regularization candidate on an internal fold or full training pool."""
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
import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import PROCESSED,OUTPUT,AB_CASE,AB_MODELS,VALIDATION_SPLITS,model_metrics,output_directory
from sepsis_warning.ab_expansion import content_hash,write_json
from sepsis_warning.history_study import digest
from sepsis_warning.data import config,environment
from sepsis_warning.regularized import patient_bags
from sepsis_warning.evaluation import select_alert_threshold,alert_replay


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["cv","validation"],required=True)
    parser.add_argument("--fold",type=int,default=-1)
    parser.add_argument("--model",choices=["lr","ebm","lightgbm"],required=True)
    parser.add_argument("--candidate",required=True)
    args=parser.parse_args()
    protocol=json.loads((OUTPUT/"protocol.json").read_text())
    candidate=next(c for c in protocol["candidates"] if c["model"]==args.model and c["candidate"]==args.candidate)
    parameters=candidate["parameters"]
    directory=PROCESSED/(f"fold{args.fold}" if args.stage=="cv" else "full")
    meta=json.loads((directory/"case.json").read_text())
    if args.stage=="validation":
        selection=json.loads((OUTPUT/"selection.json").read_text())
        assert selection["selected_candidates"][args.model]==args.candidate
        assert not selection["outer_validation_used_for_selection"]
    out=output_directory(args.stage,args.fold,args.model,args.candidate);out.mkdir(parents=True,exist_ok=True)
    signature=digest({"case":meta["signature"],"parameters":parameters,"worker":content_hash(__file__),"protocol":digest(protocol)})
    result_path=out/"results.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("signature")==signature:
        assert (out/"model.joblib").exists()
        print("Using matching completed candidate",flush=True);return
    rows=pd.read_csv(directory/"train_rows.csv.gz")
    y=np.load(directory/"train_y.npy")
    np.testing.assert_array_equal(rows.label,y)
    matrix_directory=directory if args.stage=="cv" else AB_CASE
    x=np.load(matrix_directory/"train_X.npy",mmap_mode="r")
    assert x.shape==(len(rows),312) and np.isfinite(x).all()
    processor=joblib.load(directory/"preprocessor.joblib")
    assert processor.names_==protocol["feature_columns"]
    reused=args.stage=="validation" and args.candidate=="baseline"
    details={}
    started=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reused:
            reference=joblib.load(AB_MODELS/f"{args.model}_model.joblib")
            assert reference["parameters"]==parameters
            model=reference["model"]
            details["reused_source"]=str((AB_MODELS/f"{args.model}_model.joblib").relative_to(ROOT))
            fit_seconds=0.0
        elif args.model=="lr":
            model=LogisticRegression(**parameters)
            with threadpool_limits(limits=protocol["settings"]["study"]["lr_threads"]):model.fit(x,y)
            details["iterations"]=int(model.n_iter_[0]);assert details["iterations"]<parameters["max_iter"]
            fit_seconds=time.monotonic()-started
        elif args.model=="ebm":
            from interpret.glassbox import ExplainableBoostingClassifier
            bags,assignments=patient_bags(rows,n_bags=parameters["outer_bags"],seed=2026)
            assignments.to_csv(out/"internal_patients.csv",index=False)
            model=ExplainableBoostingClassifier(feature_names=processor.names_,feature_types=["continuous"]*312,**parameters)
            with parallel_backend("threading",n_jobs=parameters["n_jobs"]),threadpool_limits(limits=1):model.fit(x,y,bags=bags)
            details["executed_cycles_including_patience"]=(model.best_iteration_[0]/len(model.term_features_)).tolist()
            fit_seconds=time.monotonic()-started
        else:
            import lightgbm as lgb
            bags,assignments=patient_bags(rows,n_bags=1,seed=2026)
            assignments.to_csv(out/"internal_patients.csv",index=False)
            inner,held=bags[:,0]>0,bags[:,0]<0
            stop=protocol["lightgbm_stopping"]
            internal=lgb.LGBMClassifier(**parameters)
            internal.fit(x[inner],y[inner],eval_set=[(x[held],y[held])],eval_metric=stop["metric"],feature_name=processor.names_,
                callbacks=[lgb.early_stopping(stop["rounds"],first_metric_only=True,verbose=False,min_delta=stop["min_delta"]),lgb.log_evaluation(0)])
            trees=max(1,int(internal.best_iteration_));del internal
            details.update({"inner_selected_trees":trees,"early_stopping_patient_grouped":True,"refit_on_all_training_patients":True})
            model=lgb.LGBMClassifier(**{**parameters,"n_estimators":trees});model.fit(x,y,feature_name=processor.names_)
            fit_seconds=time.monotonic()-started
        def predict(matrix):
            with threadpool_limits(limits=protocol["settings"]["study"]["lr_threads"]):
                p=model.booster_.predict(matrix) if args.model=="lightgbm" else model.predict_proba(matrix)[:,1]
            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            return p
        training=model_metrics(rows,predict(x))
        if args.stage=="cv":
            evaluation_rows=pd.read_csv(directory/"evaluation_rows.csv.gz")
            assert not set(rows.patient_id)&set(evaluation_rows.patient_id)
            p=predict(np.load(directory/"evaluation_X.npy",mmap_mode="r"))
            np.save(out/"probability.npy",p)
            evaluation=model_metrics(evaluation_rows,p)
            alarms=None;threshold=None
        else:
            evaluation,probabilities,evaluation_rows={},{},{}
            for split in VALIDATION_SPLITS:
                table=pd.read_csv(PROCESSED/f"{split}_rows.csv.gz")
                assert not set(rows.patient_id)&set(table.patient_id)
                p=predict(np.load(AB_CASE/f"{split}_X.npy",mmap_mode="r"))
                evaluation_rows[split],probabilities[split]=table,p
                np.save(out/f"{split}_probability.npy",p)
                from sepsis_warning.evaluation import ranking_metrics
                evaluation[split]=ranking_metrics(table.label.to_numpy(),p)
            pooled_rows=pd.concat([evaluation_rows[s] for s in VALIDATION_SPLITS],ignore_index=True)
            pooled_p=np.concatenate([probabilities[s] for s in VALIDATION_SPLITS])
            evaluation["pooled_validation"]=ranking_metrics(pooled_rows.label.to_numpy(),pooled_p)
            evaluation["macro_hospital_auprc"]=float(np.mean([evaluation[s]["auprc_average_precision"] for s in VALIDATION_SPLITS]))
            threshold,search=select_alert_threshold(pooled_rows,pooled_p,config())
            search.to_csv(out/"threshold_search.csv",index=False)
            alarms={}
            for split,table,p in [(s,evaluation_rows[s],probabilities[s]) for s in VALIDATION_SPLITS]+[("pooled_validation",pooled_rows,pooled_p)]:
                alarms[split],episodes,_=alert_replay(table,p,threshold,config()["alerts"]["suppression_hours"])
                episodes.to_csv(out/f"{split}_alert_episodes.csv",index=False)
            if reused:
                original=json.loads((AB_MODELS/f"{args.model}_results.json").read_text())
                for split in VALIDATION_SPLITS:
                    for metric,value in evaluation[split].items():np.testing.assert_allclose(value,original["evaluation"][split][metric],rtol=0,atol=1e-12)
    temporary=out/"model.tmp"
    joblib.dump({"model":model,"preprocessor":processor,"parameters":parameters,"case":meta,"threshold":threshold},temporary,compress=3)
    os.replace(temporary,out/"model.joblib")
    runtime=environment();runtime["packages"].update({"interpret-core":version("interpret-core"),"lightgbm":version("lightgbm")})
    write_json(result_path,{"signature":signature,"case":meta,"model":args.model,"candidate":args.candidate,"parameters":parameters,
        "evaluation":evaluation,"training":training,"alerts":alarms,"threshold":threshold,"fit_seconds":round(fit_seconds,2),"reused_full_baseline":reused,
        "details":details,"warnings":[str(w.message) for w in caught],"test_sets_evaluated":[],"environment":runtime})
    print(f"Finished {args.stage}/{args.fold}/{args.model}/{args.candidate}: macro AP={evaluation['macro_hospital_auprc']:.5f}; fit={fit_seconds:.1f}s",flush=True)


if __name__=="__main__":main()
