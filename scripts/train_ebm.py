"""Fit one additive EBM on the exact cached LR inputs; reserve both test sets."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from threadpoolctl import threadpool_limits
from sepsis_warning.data import config, environment, save_json
from sepsis_warning.evaluation import (alert_replay, ranking_metrics,
                                      select_alert_threshold, threshold_metrics)


def fingerprint(paths):
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def main():
    started = time.monotonic()
    cfg = config()
    with (ROOT / "configs/ebm.toml").open("rb") as stream:
        settings = tomllib.load(stream)["model"]
    processed = ROOT / cfg["data"]["processed_directory"]
    output = ROOT / cfg["data"]["output_directory"]
    paths = [processed / f"{split}_{part}.npy" for split in ["train", "validation"] for part in ["X", "y"]]
    paths += [ROOT / cfg["data"]["split_file"], output / "preprocessor.joblib",
              processed / "validation_rows.csv.gz", output / "lr_validation_predictions.csv.gz"]
    before = fingerprint(paths)
    save_json(output / "ebm_input_fingerprint.json", before)
    train_x = np.load(processed / "train_X.npy", mmap_mode="r")
    train_y = np.load(processed / "train_y.npy")
    val_x = np.load(processed / "validation_X.npy", mmap_mode="r")
    val_y = np.load(processed / "validation_y.npy")
    rows = pd.read_csv(processed / "validation_rows.csv.gz")
    lr_predictions = pd.read_csv(output / "lr_validation_predictions.csv.gz")
    pd.testing.assert_frame_equal(rows, lr_predictions[rows.columns])
    assert np.array_equal(rows.label.to_numpy(), val_y)
    lr_bundle = joblib.load(output / "lr_baseline.joblib")
    processor = joblib.load(output / "preprocessor.joblib")
    assert processor.names_ == lr_bundle["feature_names"]
    assert train_x.shape[1] == val_x.shape[1] == len(processor.names_)
    with threadpool_limits(limits=cfg["lr"]["threads"]):
        np.testing.assert_allclose(lr_bundle["model"].predict_proba(val_x)[:, 1],
                                   lr_predictions.probability.to_numpy(), atol=1e-12, rtol=1e-10)
    n_terms = len(processor.names_)
    model = ExplainableBoostingClassifier(feature_names=processor.names_,
                                          feature_types=["continuous"] * n_terms,
                                          **settings)
    print(f"Fitting additive EBM on {len(train_y):,} identical LR training rows, {n_terms} features", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=cfg["lr"]["threads"]):
            fit_started = time.monotonic()
            model.fit(train_x, train_y)
            fit_seconds = time.monotonic()-fit_started
            scores = model.predict_proba(val_x)[:, 1]
            train_scores = model.predict_proba(train_x)[:, 1]
    assert all(len(term) == 1 for term in model.term_features_)
    assert np.isfinite(scores).all() and np.all((scores >= 0) & (scores <= 1))
    np.testing.assert_allclose(model.decision_function(val_x[:128]),
                               model.eval_terms(val_x[:128]).sum(axis=1)+float(model.intercept_[0]), atol=1e-10)
    metrics = ranking_metrics(val_y, scores)
    print(f"EBM validation: {metrics}; fit {fit_seconds:.1f}s", flush=True)
    predictions = rows.copy()
    predictions["probability"] = scores
    predictions.to_csv(output / "ebm_validation_predictions.csv.gz", index=False, compression="gzip")
    threshold, threshold_table = select_alert_threshold(rows, scores, cfg)
    alert_metrics, episodes, per_patient = alert_replay(rows, scores, threshold, cfg["alerts"]["suppression_hours"])
    threshold_table.to_csv(output / "ebm_validation_threshold_search.csv", index=False)
    episodes.to_csv(output / "ebm_validation_alert_episodes.csv", index=False)
    per_patient.to_csv(output / "ebm_validation_patient_alerts.csv", index=False)
    joblib.dump({"model": model, "preprocessor": processor, "threshold": threshold,
                 "configuration": cfg, "model_configuration": settings,
                 "feature_names": processor.names_}, output / "ebm_baseline.joblib", compress=3)
    importances = pd.DataFrame({"feature": model.term_names_, "mean_absolute_log_odds_contribution": model.term_importances()})
    importances.sort_values("mean_absolute_log_odds_contribution", ascending=False).to_csv(output / "ebm_feature_importance.csv", index=False)
    after = fingerprint(paths)
    assert before == after, "Cached data or LR reference was changed during EBM training"
    run_environment = environment()
    run_environment["packages"]["interpret-core"] = version("interpret-core")
    results = {"evaluation_split": "hospital_A_validation", "test_sets_evaluated": [],
               "training_hours": len(train_y), "validation_hours": len(rows),
               "validation_patients": int(rows.patient_id.nunique()), "features": n_terms,
               "validation": metrics, "training": ranking_metrics(train_y, train_scores),
               "validation_selected_threshold_alert_metrics": alert_metrics,
               "validation_selected_threshold_hour_metrics": threshold_metrics(val_y, scores, threshold),
               "default_threshold_0_5": threshold_metrics(val_y, scores, 0.5),
               "model_configuration": settings, "additive_terms_only": True,
               "no_internal_random_hour_validation": True, "tuning_configurations": 1,
               "fit_seconds": round(fit_seconds, 2), "total_seconds": round(time.monotonic()-started, 2),
               "training_warnings": [str(w.message) for w in caught],
               "input_fingerprints_unchanged": True, "lr_predictions_reproduced": True,
               "additive_decomposition_verified": True,
               "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "environment": run_environment,
               "limitations": ["One fixed additive EBM configuration; no hyperparameter search or CV.",
                               "No bag-based uncertainty estimates; metric uncertainty uses patient bootstrap.",
                               "Alert threshold chosen and reported on the same validation set."]}
    save_json(output / "ebm_results.json", results)
    print(f"EBM alert operating point: {alert_metrics}", flush=True)
    print("EBM complete. Run scripts/compare_models.py for paired comparisons and plots.", flush=True)


if __name__ == "__main__":
    main()
