"""Fit one frozen A-only model with patient CV or a final full-A refit.

Outer-CV thresholds are calibrated on patients excluded from the inner fit.
The final models instead receive a frozen threshold derived from A OOF scores.
This worker never fits on or evaluates hospital B.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

import sepsis_warning.a_cv_refit as study
from sepsis_warning.a_cv_refit import DATA, OUT, REPORT, predict_full_patients
from sepsis_warning.ab_expansion import FixedColumnsPreprocessor, content_hash, write_json
from sepsis_warning.data import FEATURE_NAMES, config, environment
from sepsis_warning.evaluation import ranking_metrics, select_alert_threshold
from sepsis_warning.history_study import digest
from sepsis_warning.regularized import patient_bags


def predict(model, model_name, features):
    """Bound prediction threads and reject invalid probabilities."""
    with threadpool_limits(limits=2 if model_name == "lr" else 3):
        probability = (model.booster_.predict(features, num_threads=3) if model_name == "lightgbm"
                       else model.predict_proba(features)[:, 1])
    probability = np.asarray(probability)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    return probability


def fit_model(model_name, features, labels, rows, processor, parameters,
              directory, prefix, stopping=None, calibration=None):
    """Fit with the frozen settings; optional calibration is only for GBM stopping."""
    assert set(rows.hospital) == {"A"}, "Only hospital A may enter fitting"
    assert len(features) == len(labels) == len(rows)
    assert features.shape[1] == len(processor.names_) == 312
    assert np.isfinite(features).all()
    details = {}
    started = time.monotonic()
    if model_name == "lr":
        model = LogisticRegression(**parameters)
        with threadpool_limits(limits=2):
            model.fit(features, labels)
        details["iterations"] = int(model.n_iter_[0])
        assert details["iterations"] < parameters["max_iter"], "LR reached its iteration cap"
    elif model_name == "ebm":
        from interpret.glassbox import ExplainableBoostingClassifier

        bags, assignments = patient_bags(rows, n_bags=parameters["outer_bags"],
                                         holdout_fraction=0.2, seed=2026)
        assignments.to_csv(directory / f"{prefix}_ebm_patient_bags.csv", index=False)
        model = ExplainableBoostingClassifier(
            feature_names=processor.names_, feature_types=["continuous"] * 312,
            **parameters)
        # Three bag workers, with one native thread per worker, match the old fit.
        with parallel_backend("threading", n_jobs=3), threadpool_limits(limits=1):
            model.fit(features, labels, bags=bags)
        details["executed_cycles_including_patience"] = (
            model.best_iteration_[0] / len(model.term_features_)).tolist()
        details["patient_bag_seed"] = 2026
        details["patient_bag_holdout_fraction"] = 0.2
        details["bag_preprocessing"] = "Shared within this fit's A training patients"
    else:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**parameters)
        options = {"feature_name": processor.names_}
        if calibration is not None:
            calibration_features, calibration_labels = calibration
            assert stopping is not None
            assert np.isfinite(calibration_features).all()
            options.update({
                "eval_set": [(calibration_features, calibration_labels)],
                "eval_metric": stopping["metric"],
                "callbacks": [lgb.early_stopping(
                    stopping["rounds"], first_metric_only=True, verbose=False,
                    min_delta=stopping["min_delta"]), lgb.log_evaluation(0)],
            })
        with threadpool_limits(limits=3):
            model.fit(features, labels, **options)
        if calibration is not None:
            details["inner_selected_trees"] = max(1, int(model.best_iteration_))
        else:
            details["fitted_trees"] = int(model.booster_.num_trees())
    return model, details, time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lr", "ebm", "lightgbm"], required=True)
    parser.add_argument("--fold", type=int, choices=[-1, 0, 1, 2], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads((OUT / "protocol.json").read_text())
    parameters = dict(protocol["parameters"][args.model])
    allowed = protocol["feature_columns"]
    assert len(allowed) == len(set(allowed)) == 312
    assert set(allowed) <= set(FEATURE_NAMES)
    cfg = config()
    final = args.fold == -1
    out = OUT / "final" / args.model if final else OUT / "cv" / f"fold{args.fold}" / args.model
    out.mkdir(parents=True, exist_ok=True)

    threshold_inputs = None
    if final:
        threshold_inputs = json.loads((OUT / "thresholds.json").read_text())
        threshold = float(threshold_inputs[args.model])
        assert np.isfinite(threshold) and 0 <= threshold <= 1
    signature = digest({
        "protocol": digest(protocol), "worker": content_hash(__file__),
        "shared_module": content_hash(study.__file__), "model": args.model,
        "fold": args.fold, "configuration": cfg,
        "thresholds": threshold_inputs,
    })
    result_path = out / "results.json"
    if result_path.exists():
        cached = json.loads(result_path.read_text())
        if cached.get("signature") == signature:
            required = [out / "model.joblib"]
            if not final:
                required += [out / name for name in ["probability.npy", "labels.npy",
                            "task_eligible.npy", "patients.csv", "patient_utilities.csv",
                            "verification.json", "threshold_search.csv", "inner_patients.csv"]]
            assert all(path.exists() and path.stat().st_size > 0 for path in required)
            assert content_hash(out / "model.joblib") == cached["model_sha256"]
            print(f"Reusing completed A-only {args.model}, fold={args.fold}", flush=True)
            return

    rows = pd.read_csv(DATA / "A_rows.csv.gz", dtype={"patient_id": str})
    patients = pd.read_csv(DATA / "A_patients.csv", dtype={"patient_id": str})
    raw = np.load(DATA / "A_features.npy", mmap_mode="r")
    assert set(rows.hospital) == {"A"}, "This worker must not read B training data"
    assert raw.shape == (len(rows), len(FEATURE_NAMES))
    assert patients.patient_id.is_unique
    assert set(patients.fold) == {0, 1, 2}
    assert set(rows.patient_id) == set(patients.patient_id)
    observed_events = rows.groupby("patient_id").label.max()
    np.testing.assert_array_equal(
        patients.set_index("patient_id").has_event.sort_index().to_numpy(dtype=int),
        observed_events.sort_index().to_numpy(dtype=int))
    outer_train = patients if final else patients.loc[patients.fold != args.fold]
    outer_train = outer_train.sort_values("patient_id").copy()
    held_ids = [] if final else patients.loc[patients.fold == args.fold, "patient_id"].tolist()
    training_ids = outer_train.patient_id.tolist()
    assert not set(training_ids) & set(held_ids)
    training_indices = np.flatnonzero(rows.patient_id.isin(training_ids).to_numpy())
    training_rows = rows.iloc[training_indices].reset_index(drop=True)
    training_labels = training_rows.label.to_numpy()
    details = {"training_patients": len(training_ids),
               "training_event_patients": int(outer_train.has_event.sum()),
               "training_hours": len(training_rows), "features": 312,
               "fit_hospitals": ["A"], "inner_split_seed": 2026,
               "outer_fold": args.fold, "refit_on_all_selected_A_training_patients": True}
    fit_seconds = 0.0
    need_inner = not final or args.model == "lightgbm"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if need_inner:
            inner_ids, calibration_ids = train_test_split(
                outer_train.patient_id.to_numpy(), test_size=0.2, random_state=2026,
                stratify=outer_train.has_event.to_numpy())
            assert not set(inner_ids) & set(calibration_ids)
            assert set(inner_ids) | set(calibration_ids) == set(training_ids)
            assert not (set(inner_ids) | set(calibration_ids)) & set(held_ids)
            assignment = outer_train.copy()
            assignment["role"] = np.where(assignment.patient_id.isin(inner_ids),
                                           "fit", "calibration")
            assignment["assignment"] = np.where(assignment.role == "fit", 1, -1)
            assignment.to_csv(out / "inner_patients.csv", index=False)
            inner_indices = np.flatnonzero(rows.patient_id.isin(inner_ids).to_numpy())
            calibration_indices = np.flatnonzero(rows.patient_id.isin(calibration_ids).to_numpy())
            inner_rows = rows.iloc[inner_indices].reset_index(drop=True)
            calibration_rows = rows.iloc[calibration_indices].reset_index(drop=True)
            inner_processor = FixedColumnsPreprocessor(FEATURE_NAMES, allowed).fit(raw[inner_indices])
            assert inner_processor.names_ == allowed
            inner_x = inner_processor.transform(raw[inner_indices])
            calibration_x = inner_processor.transform(raw[calibration_indices])
            inner_y = inner_rows.label.to_numpy()
            calibration_y = calibration_rows.label.to_numpy()
            model, inner_details, elapsed = fit_model(
                args.model, inner_x, inner_y, inner_rows, inner_processor, parameters,
                out, "inner_fit", stopping=protocol["lightgbm_stopping"],
                calibration=(calibration_x, calibration_y) if args.model == "lightgbm" else None)
            fit_seconds += elapsed
            details["inner_model"] = inner_details
            details["inner_fit_patients"] = len(inner_ids)
            details["inner_calibration_patients"] = len(calibration_ids)
            details["inner_preprocessor_fit_on"] = "Inner fitting A patients only"
            if args.model == "lightgbm":
                selected_trees = inner_details["inner_selected_trees"]
                details["inner_selected_trees"] = selected_trees
            if not final:
                calibration_probability = predict(model, args.model, calibration_x)
                np.save(out / "inner_calibration_probability.npy", calibration_probability)
                threshold, search = select_alert_threshold(calibration_rows, calibration_probability, cfg)
                search.to_csv(out / "threshold_search.csv", index=False)
                details["inner_calibration_ranking"] = ranking_metrics(calibration_y, calibration_probability)
                details["threshold_fit_on"] = "Disjoint inner calibration A patients"
                del calibration_probability
            del model, inner_processor, inner_x, calibration_x, inner_y, calibration_y
            del inner_rows, calibration_rows, inner_indices, calibration_indices
            gc.collect()
        if final:
            details["threshold_fit_on"] = "All-A OOF predictions; frozen before final fitting"
        processor = FixedColumnsPreprocessor(FEATURE_NAMES, allowed).fit(raw[training_indices])
        assert processor.names_ == allowed
        features = processor.transform(raw[training_indices])
        final_parameters = dict(parameters)
        if args.model == "lightgbm":
            final_parameters["n_estimators"] = selected_trees
        model, fitted_details, elapsed = fit_model(
            args.model, features, training_labels, training_rows, processor,
            final_parameters, out, "refit")
        fit_seconds += elapsed
        details["refit_model"] = fitted_details
        training_probability = predict(model, args.model, features)
        training = ranking_metrics(training_labels, training_probability)
        del features, training_probability, raw
        gc.collect()
        bundle = {"model": model, "preprocessor": processor, "parameters": parameters,
                  "fitted_parameters": final_parameters, "threshold": float(threshold),
                  "model_name": args.model, "fold": args.fold, "signature": signature,
                  "training_patients": len(training_ids), "fit_hospitals": ["A"]}
        temporary = out / "model.tmp"
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, out / "model.joblib")
        evaluation = None if final else predict_full_patients(bundle, held_ids, out)

    runtime = environment()
    runtime["packages"].update({"interpret-core": version("interpret-core"),
                                "lightgbm": version("lightgbm")})
    record = {"signature": signature, "model": args.model, "fold": args.fold,
              "stage": "final" if final else "cv", "parameters": parameters,
              "fitted_parameters": final_parameters, "threshold": float(threshold),
              "training_patients": len(training_ids),
              "training_event_patients": int(outer_train.has_event.sum()),
              "training_hours": len(training_rows), "features": 312,
              "training": training, "evaluation": evaluation, "details": details,
              "fit_seconds": round(fit_seconds, 2),
              "execution_seconds": round(time.monotonic() - started, 2),
              "model_sha256": content_hash(out / "model.joblib"),
              "warnings": [str(w.message) for w in caught],
              "B_evaluated": False, "environment": runtime,
              "limitations": [
                  "Fixed historical hyperparameters; no new feature or regularization search.",
                  "EBM internal bags share preprocessing fitted within the selected A training patients.",
                  "A calibration thresholds transfer from the inner fit to an outer refit with more training patients."
                  if not final else
                  "The final threshold transfers from cross-fitted A models to the full-A refit.",
              ]}
    write_json(result_path, record)
    print(f"Completed A-only {args.model}, fold={args.fold}, fit={fit_seconds:.1f}s; "
          f"threshold={threshold:.12g}", flush=True)


if __name__ == "__main__":
    main()
