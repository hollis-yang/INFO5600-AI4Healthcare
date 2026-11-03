"""Fit one frozen model for an internal-CV or selected validation data case."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
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
from sepsis_warning.evaluation import ranking_metrics, select_alert_threshold, alert_replay
from sepsis_warning.history_study import digest, sha_file
from sepsis_warning.optimization import PROCESSED, OUTPUT
from sepsis_warning.regularized import patient_bags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", choices=["lr", "ebm", "lightgbm"], required=True)
    args = parser.parse_args()
    cfg = config()
    directory = PROCESSED/"cases"/args.case
    meta = json.loads((directory/"case.json").read_text())
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    parameters = protocol["model_settings"]["models"][args.model]
    stopping = protocol["model_settings"]["lightgbm_stopping"]
    signature = digest({"case": meta["signature"], "parameters": parameters,
                        "stopping": stopping, "worker": sha_file(__file__)})
    output = OUTPUT/meta["stage"]/args.case
    output.mkdir(parents=True, exist_ok=True)
    result_path = output/f"{args.model}_results.json"
    if result_path.exists() and json.loads(result_path.read_text()).get("signature") == signature:
        assert (output/f"{args.model}_probability.npy").exists()
        print("Using matching cached result", flush=True)
        return
    x = np.load(directory/"train_X.npy", mmap_mode="r")
    evaluation_x = np.load(directory/"evaluation_X.npy", mmap_mode="r")
    y = np.load(directory/"train_y.npy")
    evaluation_y = np.load(directory/"evaluation_y.npy")
    rows = pd.read_csv(directory/"train_rows.csv.gz")
    evaluation_rows = pd.read_csv(directory/"evaluation_rows.csv.gz")
    assert len(x) == len(y) == len(rows) and len(evaluation_x) == len(evaluation_y) == len(evaluation_rows)
    np.testing.assert_array_equal(y, rows.label)
    np.testing.assert_array_equal(evaluation_y, evaluation_rows.label)
    assert not set(rows.patient_id) & set(evaluation_rows.patient_id)
    assert not np.isinf(x).any() and not np.isinf(evaluation_x).any()
    if args.model == "lr":
        assert np.isfinite(x).all() and np.isfinite(evaluation_x).all()
    processor = joblib.load(directory/"preprocessor.joblib")
    reused = meta["stage"] == "validation" and meta["representation"] == {
        "source": "baseline", "native": False, "clip": True, "drop_context": False, "splines": False}
    details = {}
    started = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if reused:
            original = ROOT/f"outputs/history_study/history6h_full/{args.model}_model.joblib"
            reference = json.loads((original.parent/f"{args.model}_results.json").read_text())
            bundle = joblib.load(original)
            assert bundle["parameters"] == parameters and bundle["preprocessor"].names_ == processor.names_
            model = bundle["model"]
            fit_seconds = reference["fit_seconds"]
            details = {**reference["details"], "reused_from": str(original.relative_to(ROOT))}
        elif args.model == "lr":
            model = LogisticRegression(**parameters)
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                model.fit(x, y)
            fit_seconds = time.monotonic()-started
            details["iterations"] = int(model.n_iter_[0])
            if details["iterations"] >= parameters["max_iter"]:
                raise RuntimeError("LR reached its frozen iteration cap")
        elif args.model == "ebm":
            from interpret.glassbox import ExplainableBoostingClassifier
            bags, assignments = patient_bags(rows, n_bags=parameters["outer_bags"], seed=2026)
            assignments.to_csv(output/"ebm_internal_patients.csv", index=False)
            model = ExplainableBoostingClassifier(feature_names=processor.names_,
                       feature_types=["continuous"]*len(processor.names_), **parameters)
            with parallel_backend("threading", n_jobs=parameters["n_jobs"]), threadpool_limits(limits=1):
                model.fit(x, y, bags=bags)
            fit_seconds = time.monotonic()-started
            details["executed_cycles_including_patience"] = (model.best_iteration_[0]/len(model.term_features_)).tolist()
        else:
            import lightgbm as lgb
            bags, assignments = patient_bags(rows, n_bags=1, seed=2026)
            assignments.to_csv(output/"lightgbm_internal_patients.csv", index=False)
            inner, held = bags[:, 0] > 0, bags[:, 0] < 0
            internal = lgb.LGBMClassifier(**parameters)
            internal.fit(x[inner], y[inner], eval_set=[(x[held], y[held])], eval_metric=stopping["metric"],
                         feature_name=processor.names_, callbacks=[lgb.early_stopping(stopping["rounds"],
                           first_metric_only=True, verbose=False, min_delta=stopping["min_delta"]), lgb.log_evaluation(0)])
            count = max(1, int(internal.best_iteration_))
            details.update({"inner_selected_trees": count, "early_stopping_patient_grouped": True,
                            "refit_on_all_selected_training_patients": True})
            del internal
            model = lgb.LGBMClassifier(**{**parameters, "n_estimators": count})
            model.fit(x, y, feature_name=processor.names_)
            fit_seconds = time.monotonic()-started
        if args.model == "lightgbm":
            probability, training_probability = model.booster_.predict(evaluation_x), model.booster_.predict(x)
            pd.DataFrame({"feature": processor.names_, "gain": model.booster_.feature_importance("gain")}).sort_values(
                "gain", ascending=False).to_csv(output/f"{args.model}_feature_importance.csv", index=False)
        else:
            with threadpool_limits(limits=cfg["lr"]["threads"]):
                probability, training_probability = model.predict_proba(evaluation_x)[:, 1], model.predict_proba(x)[:, 1]
    metrics = ranking_metrics(evaluation_y, probability)
    if reused:
        for name, value in metrics.items():
            np.testing.assert_allclose(value, reference["validation"][name], rtol=0, atol=1e-12)
    np.save(output/f"{args.model}_probability.npy", probability)
    alarms, threshold = None, None
    if meta["stage"] == "validation":
        threshold, search = select_alert_threshold(evaluation_rows, probability, cfg)
        alarms, episodes, _ = alert_replay(evaluation_rows, probability, threshold, cfg["alerts"]["suppression_hours"])
        search.to_csv(output/f"{args.model}_threshold_search.csv", index=False)
        episodes.to_csv(output/f"{args.model}_alert_episodes.csv", index=False)
    joblib.dump({"model": model, "preprocessor": processor, "parameters": parameters, "case": meta,
                 "threshold": threshold}, output/f"{args.model}_model.joblib", compress=3)
    save_json(result_path, {"signature": signature, "case": meta, "model": args.model, "parameters": parameters,
              "evaluation": metrics, "training": ranking_metrics(y, training_probability), "alerts": alarms,
              "fit_seconds": round(fit_seconds, 2), "reused_baseline": reused,
              "details": details, "warnings": [str(w.message) for w in caught],
              "test_sets_evaluated": [], "environment": environment()})
    print(f"Finished {args.model}: AP={metrics['auprc_average_precision']:.5f}, AUROC={metrics['auroc']:.5f}, fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
