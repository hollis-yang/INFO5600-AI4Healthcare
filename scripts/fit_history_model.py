"""One isolated model worker for a prespecified history/sample trial."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/"outputs/.cache"))

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from sepsis_warning.data import config, save_json, environment
from sepsis_warning.history_study import settings, digest
from sepsis_warning.regularized import patient_bags
from sepsis_warning.evaluation import ranking_metrics, select_alert_threshold, alert_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True)
    parser.add_argument("--model", choices=["lr", "ebm", "lightgbm"], required=True)
    args = parser.parse_args()
    cfg, study = config(), settings()
    trial = ROOT/"processed_data/history_study"/args.trial
    output = ROOT/"outputs/history_study"/args.trial
    metadata = json.loads((trial/"trial.json").read_text())
    prior_study = json.loads((ROOT/"reports/ebm_regularized/experiment_results.json").read_text())
    ebm_name = prior_study["selection"]["selected_case"]
    ebm_settings = next(case["settings"] for case in prior_study["validation_cases"] if case["name"] == ebm_name)
    if args.model == "lr":
        parameters = {"C": cfg["lr"]["C"], "solver": "lbfgs", "penalty": "l2", "class_weight": None,
                      "max_iter": cfg["lr"]["max_iter"], "tol": cfg["lr"]["tolerance"], "random_state": 42}
    elif args.model == "ebm":
        parameters = ebm_settings
    else:
        parameters = study["lightgbm"]
    signature = digest({"trial": metadata["signature"], "model": args.model, "parameters": parameters,
                        "stopping": study["lightgbm_stopping"] if args.model == "lightgbm" else None,
                        "implementation": "three_model_history_v1"})
    results_path = output/f"{args.model}_results.json"
    if results_path.exists() and json.loads(results_path.read_text()).get("signature") == signature:
        print("Reusing matching completed model", flush=True)
        return
    x = np.load(ROOT/metadata["train_X"], mmap_mode="r")
    y = np.load(trial/"train_y.npy")
    val_x = np.load(ROOT/metadata["validation_X"], mmap_mode="r")
    val_y = np.load(ROOT/"processed_data/validation_y.npy")
    train_rows = pd.read_csv(trial/"train_rows.csv.gz")
    val_rows = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    assert len(x) == len(y) == len(train_rows)
    assert np.array_equal(train_rows.label.to_numpy(), y)
    assert x.shape[1] == val_x.shape[1] == metadata["features"]
    assert np.isfinite(x).all() and np.isfinite(val_x).all()
    processor = joblib.load(trial/"preprocessor.joblib")
    reused = metadata["retention_hours"] == 6 and metadata["fraction"] == 1 and args.model in ["lr", "ebm"]
    started = time.monotonic()
    details = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reused:
            source = ROOT/"outputs/lr_baseline.joblib" if args.model == "lr" else ROOT/f"outputs/ebm_regularized/{ebm_name}/model.joblib"
            bundle = joblib.load(source)
            model = bundle["model"]
            assert processor.names_ == bundle["preprocessor"].names_
            reference_results = json.loads((ROOT/"outputs/lr_results.json" if args.model == "lr" else ROOT/f"outputs/ebm_regularized/{ebm_name}/validation_results.json").read_text())
            fit_seconds = reference_results["fit_seconds"]
            details["reused_from"] = str(source.relative_to(ROOT))
        elif args.model == "lr":
            model = LogisticRegression(**parameters)
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                model.fit(x, y)
            fit_seconds = time.monotonic()-started
            details["iterations"] = int(model.n_iter_[0])
            if details["iterations"] >= cfg["lr"]["max_iter"]:
                raise RuntimeError("LR reached its iteration cap")
        elif args.model == "ebm":
            from interpret.glassbox import ExplainableBoostingClassifier
            bags, assignments = patient_bags(train_rows, n_bags=parameters["outer_bags"], holdout_fraction=.2, seed=2026)
            assignments.to_csv(output/"ebm_patient_bags.csv", index=False)
            model = ExplainableBoostingClassifier(feature_names=processor.names_,
                       feature_types=["continuous"]*len(processor.names_), **parameters)
            with parallel_backend("threading", n_jobs=parameters["n_jobs"]), threadpool_limits(limits=1):
                model.fit(x, y, bags=bags)
            fit_seconds = time.monotonic()-started
            details["executed_cycles_including_patience"] = (model.best_iteration_[0]/len(model.term_features_)).tolist()
        else:
            import lightgbm as lgb
            bags, assignments = patient_bags(train_rows, n_bags=1, holdout_fraction=.2, seed=2026)
            assignments.to_csv(output/"lightgbm_inner_patients.csv", index=False)
            inner, stopping = bags[:, 0] > 0, bags[:, 0] < 0
            stopping_settings = study["lightgbm_stopping"]
            internal = lgb.LGBMClassifier(**parameters)
            internal.fit(x[inner], y[inner], eval_set=[(x[stopping], y[stopping])],
                         eval_metric=stopping_settings["metric"], feature_name=processor.names_,
                         callbacks=[lgb.early_stopping(stopping_settings["rounds"], first_metric_only=True,
                                       verbose=False, min_delta=stopping_settings["min_delta"]), lgb.log_evaluation(0)])
            best_iteration = max(1, int(internal.best_iteration_))
            details["inner_selected_trees"] = best_iteration
            details["early_stopping_patient_grouped"] = True
            details["refit_on_all_selected_training_patients"] = True
            del internal
            final_parameters = {**parameters, "n_estimators": best_iteration}
            model = lgb.LGBMClassifier(**final_parameters)
            model.fit(x, y, feature_name=processor.names_)
            fit_seconds = time.monotonic()-started
        if args.model == "lightgbm":
            scores = model.booster_.predict(val_x)
            train_scores = model.booster_.predict(x)
            pd.DataFrame({"feature": processor.names_, "gain": model.booster_.feature_importance("gain")}).sort_values("gain", ascending=False).to_csv(output/"lightgbm_feature_importance.csv", index=False)
        else:
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                scores = model.predict_proba(val_x)[:, 1]
                train_scores = model.predict_proba(x)[:, 1]
    metrics = ranking_metrics(val_y, scores)
    if reused:
        for key in metrics:
            np.testing.assert_allclose(metrics[key], reference_results["validation"][key], atol=1e-12)
    np.save(output/f"{args.model}_validation_probability.npy", scores)
    threshold, thresholds = select_alert_threshold(val_rows, scores, cfg)
    alarms, episodes, _ = alert_replay(val_rows, scores, threshold, cfg["alerts"]["suppression_hours"])
    thresholds.to_csv(output/f"{args.model}_threshold_search.csv", index=False)
    episodes.to_csv(output/f"{args.model}_alert_episodes.csv", index=False)
    joblib.dump({"model": model, "preprocessor": processor, "threshold": threshold,
                 "trial": metadata, "parameters": parameters}, output/f"{args.model}_model.joblib", compress=3)
    run_environment = environment()
    run_environment["packages"].update({"interpret-core": version("interpret-core"), "lightgbm": version("lightgbm")})
    result = {"signature": signature, "trial": metadata, "model": args.model, "parameters": parameters,
              "validation": metrics, "training": ranking_metrics(y, train_scores), "alerts": alarms,
              "fit_seconds": round(fit_seconds, 2), "cached_reference_model": reused,
              "details": details, "warnings": [str(w.message) for w in caught],
              "test_sets_evaluated": [], "environment": run_environment}
    save_json(results_path, result)
    print(f"Finished {args.model}: validation {metrics}, fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
