"""Control EBM overfitting, then compare class weights without changing outer splits."""
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
from joblib import parallel_backend
from scipy.special import expit
from threadpoolctl import threadpool_limits
from sepsis_warning.data import config, save_json, environment
from sepsis_warning.evaluation import ranking_metrics, select_alert_threshold, alert_replay
from sepsis_warning.regularized import patient_bags, oob_probabilities, prior_correct, frozen_cut_types
from train_ebm import fingerprint


def main():
    started = time.monotonic()
    cfg = config()
    with (ROOT / "configs/ebm_regularized.toml").open("rb") as stream:
        study = tomllib.load(stream)
    output = ROOT / "outputs/ebm_regularized"
    output.mkdir(parents=True, exist_ok=True)
    processed = ROOT / cfg["data"]["processed_directory"]
    paths = [processed/f"{split}_{part}.npy" for split in ["train", "validation"] for part in ["X", "y"]]
    paths += [ROOT/cfg["data"]["split_file"], ROOT/"outputs/preprocessor.joblib",
              ROOT/"outputs/lr_validation_predictions.csv.gz", ROOT/"outputs/ebm_validation_predictions.csv.gz"]
    initial = fingerprint(paths)
    save_json(output/"input_fingerprints.json", initial)
    x = np.load(processed/"train_X.npy", mmap_mode="r")
    y = np.load(processed/"train_y.npy")
    train_rows = pd.read_csv(processed/"train_rows.csv.gz")
    assert np.array_equal(y, train_rows.label.to_numpy())
    split_list = pd.read_csv(ROOT/cfg["data"]["split_file"])
    assert set(train_rows.patient_id) == set(split_list.loc[split_list.split == "train", "patient_id"])
    bags, assignments = patient_bags(train_rows, study["common"]["outer_bags"],
                                      study["experiment"]["holdout_patient_fraction"], study["experiment"]["random_seed"])
    assignments.to_csv(processed/"splits/ebm_inner_patient_bags.csv", index=False)
    np.save(processed/"ebm_inner_bags.npy", bags)
    bag_summary = []
    for bag in range(bags.shape[1]):
        assignment = assignments[f"bag_{bag}"]
        bag_summary.append({"bag": bag, "training_patients": int((assignment > 0).sum()),
                            "holdout_patients": int((assignment < 0).sum()),
                            "training_hours": int((bags[:, bag] > 0).sum()),
                            "holdout_hours": int((bags[:, bag] < 0).sum()),
                            "holdout_event_patients": int(assignments.loc[assignment < 0, "has_event"].sum())})
    save_json(output/"patient_bags.json", {"bags": bag_summary, "outer_split_unchanged": True,
        "description": "All hours of a patient have the same sign within each bag; held-out labels only guide stopping/selection.",
        "preprocessing": "Frozen LR preprocessing and unweighted quantile bins are fitted within A training only; inner OOB is a development estimate, not independent testing."})
    processor = joblib.load(ROOT/"outputs/preprocessor.joblib")
    records = []

    def fit_case(name, params, weight, feature_types=None):
        destination = output/name
        destination.mkdir(exist_ok=True)
        settings = {**study["common"], **params}
        signature_data = {"settings": settings, "weight": weight, "inputs": initial,
                          "bags_sha256": hashlib.sha256(bags.tobytes()).hexdigest(),
                          "feature_types": feature_types}
        signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        if (destination/"metrics.json").exists():
            saved = json.loads((destination/"metrics.json").read_text())
            if saved["signature"] == signature:
                print(f"Reusing completed case {name}", flush=True)
                records.append(saved)
                return saved
        print(f"Fitting {name}: positive weight={weight}, bins={settings['max_bins']}, three patient-grouped bags", flush=True)
        model = ExplainableBoostingClassifier(feature_names=processor.names_,
            feature_types=feature_types or ["continuous"]*x.shape[1], **settings)
        weights = None if weight == 1 else np.where(y == 1, weight, 1.0)
        case_start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with parallel_backend("threading", n_jobs=settings["n_jobs"]), threadpool_limits(limits=1):
                model.fit(x, y, sample_weight=weights, bags=bags)
        fit_seconds = time.monotonic()-case_start
        if feature_types is not None:
            for expected, actual in zip(feature_types, model.bins_):
                if isinstance(expected, list):
                    np.testing.assert_array_equal(expected, actual[0])
        oob_raw, covered = oob_probabilities(model, x, bags)
        oob = prior_correct(oob_raw[covered], weight)
        metrics = ranking_metrics(y[covered], oob)
        np.save(destination/"oob_probability_raw.npy", oob_raw)
        joblib.dump({"model": model, "preprocessor": processor, "positive_weight": weight,
                     "prior_correction": "subtract log(weight) from the raw log-odds",
                     "settings": settings}, destination/"model.joblib", compress=3)
        record = {"name": name, "signature": signature, "positive_weight": weight,
                  "settings": settings, "oob": metrics, "oob_raw": ranking_metrics(y[covered], oob_raw[covered]),
                  "oob_hours": int(covered.sum()), "oob_patients": int(train_rows.loc[covered, "patient_id"].nunique()),
                  "fit_seconds": round(fit_seconds, 2), "executed_boosting_steps": model.best_iteration_.tolist(),
                  "executed_cycles_including_patience": (model.best_iteration_[0]/len(model.term_features_)).tolist(),
                  "saved_model": "Native best-loss checkpoint, not necessarily the final executed step.",
                  "warnings": [str(w.message) for w in caught], "binning_source": "unweighted frozen cuts" if feature_types is not None else "unweighted A training quantiles"}
        save_json(destination/"metrics.json", record)
        records.append(record)
        print(f"Finished {name}: OOB AP={metrics['auprc_average_precision']:.4f}, AUROC={metrics['auroc']:.4f}, cycles={record['executed_cycles_including_patience']}, fit={fit_seconds:.1f}s", flush=True)
        return record

    # Stage 1: choose complexity using training-internal OOB predictions only.
    unweighted = []
    for candidate in study["regularization"]:
        candidate = dict(candidate)
        name = candidate.pop("name")
        unweighted.append(fit_case(name, candidate, 1.0))
    best_reg = sorted(unweighted, key=lambda r: (-r["oob"]["auprc_average_precision"], r["oob"]["brier_score"]))[0]
    source_model = joblib.load(output/best_reg["name"]/"model.joblib")["model"]
    fixed_types = frozen_cut_types(source_model)
    parameters = {key: best_reg["settings"][key] for key in ["max_bins", "min_samples_leaf", "reg_lambda"]}
    weight_cases = [best_reg]
    print(f"Selected complexity internally: {best_reg['name']}", flush=True)
    # Stage 2: fix the complexity, patient bags and cuts; change only class weights.
    for weight in study["experiment"]["positive_weights"]:
        if weight != 1:
            weight_cases.append(fit_case(f"{best_reg['name']}_weight{weight:g}", parameters, weight, fixed_types))
    chosen = sorted(weight_cases, key=lambda r: (-r["oob"]["auprc_average_precision"], r["positive_weight"]))[0]
    selection = {"criterion": "Training-internal patient OOB average precision; lower weight breaks ties.",
                 "selected_regularization": best_reg["name"], "selected_case": chosen["name"],
                 "selected_positive_weight": chosen["positive_weight"], "cases": records,
                 "outer_validation_used_for_selection": False,
                 "class_weight_correction": "Analytic prior shift, not fitted calibration; raw and corrected Brier are both reported."}
    save_json(output/"internal_selection.json", selection)
    # Evaluate outer validation only after recording training-internal choices.
    val_x = np.load(processed/"validation_X.npy", mmap_mode="r")
    val_y = np.load(processed/"validation_y.npy")
    val_rows = pd.read_csv(processed/"validation_rows.csv.gz")
    assert np.array_equal(val_y, val_rows.label.to_numpy())
    validation_records = []
    for record in records:
        destination = output/record["name"]
        model = joblib.load(destination/"model.joblib")["model"]
        raw_scores = model.predict_proba(val_x)[:, 1]
        scores = prior_correct(raw_scores, record["positive_weight"])
        training_scores = prior_correct(model.predict_proba(x)[:, 1], record["positive_weight"])
        predictions = val_rows.copy()
        predictions["probability"] = scores
        predictions["raw_weighted_probability"] = raw_scores
        predictions.to_csv(destination/"validation_predictions.csv.gz", index=False, compression="gzip")
        threshold, threshold_table = select_alert_threshold(val_rows, scores, cfg)
        alarms, episodes, per_patient = alert_replay(val_rows, scores, threshold, cfg["alerts"]["suppression_hours"])
        threshold_table.to_csv(destination/"threshold_search.csv", index=False)
        episodes.to_csv(destination/"alert_episodes.csv", index=False)
        per_patient.to_csv(destination/"patient_alerts.csv", index=False)
        validation_record = {**record, "validation": ranking_metrics(val_y, scores),
                             "validation_raw": ranking_metrics(val_y, raw_scores),
                             "training": ranking_metrics(y, training_scores), "alerts": alarms,
                             "selected_internally": record["name"] == chosen["name"],
                             "test_sets_evaluated": []}
        save_json(destination/"validation_results.json", validation_record)
        validation_records.append(validation_record)
        print(f"Outer validation {record['name']}: AP={validation_record['validation']['auprc_average_precision']:.4f}, event sensitivity={alarms['event_sensitivity']:.3f}, PPV={alarms['alert_ppv']:.3f}", flush=True)
    assert initial == fingerprint(paths)
    chosen_result = next(record for record in validation_records if record["selected_internally"])
    chosen_bundle = joblib.load(output/chosen["name"]/"model.joblib")
    chosen_bundle["threshold"] = chosen_result["alerts"]["threshold"]
    chosen_bundle["selection"] = selection
    joblib.dump(chosen_bundle, output/"selected_model.joblib", compress=3)
    run_environment = environment()
    run_environment["packages"]["interpret-core"] = version("interpret-core")
    result = {"selection": selection, "validation_cases": validation_records,
              "input_fingerprints_unchanged": True, "test_sets_evaluated": [],
              "base_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "execution_elapsed_seconds": round(time.monotonic()-started, 2),
              "total_recorded_case_fit_seconds": round(sum(record["fit_seconds"] for record in records), 2),
              "environment": run_environment,
              "caveats": ["OOB predictions are used for early stopping and selection, not independent performance estimates.",
                          "Outer A validation has been viewed in prior development; final test sets remain reserved.",
                          "Unsupervised preprocessing and bin definitions are fitted within all A training, shared across internal bags.",
                          "Analytic class-weight prior correction is approximate; it is not learned probability calibration."]}
    save_json(output/"experiment_results.json", result)
    print("Regularization/weight experiments complete; original outer splits and test sets preserved.", flush=True)


if __name__ == "__main__":
    main()
