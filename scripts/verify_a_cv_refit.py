"""Independently verify saved full-A CV/refit and full-B evaluation artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from sepsis_warning.data import FEATURE_NAMES, causal_features, config

DATA = ROOT / "processed_data/a_only_cv_refit"
OUT = ROOT / "outputs/a_only_cv_refit"
REPORT = ROOT / "reports/a_only_cv_refit"
MODELS = ["lr", "ebm", "lightgbm"]
COMPONENTS = ["observed", "best", "inaction", "worst"]


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    path = Path(path)
    assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), path
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    assert size == path.stat().st_size, path
    return digest.hexdigest()


def close(actual, expected, atol=1e-12):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol)


def ranking(labels, probability):
    return {"auroc": float(roc_auc_score(labels, probability)),
            "auprc": float(average_precision_score(labels, probability))}


def utility(frame):
    totals = {name: float(frame[name].sum()) for name in COMPONENTS}
    denominator = totals["best"] - totals["inaction"]
    assert denominator > 0
    return {**totals, "normalization_denominator": denominator,
            "normalized_utility": (totals["observed"] - totals["inaction"]) / denominator}


def load_official_scorer():
    directory = ROOT / "third_party/physionet2019"
    upstream = read_json(directory / "UPSTREAM.json")
    assert upstream["commit"] == "467c49b514542be7a4a0bafe40fa2c3b064dda2e"
    for name, metadata in upstream["files"].items():
        assert sha256(directory / name) == metadata["sha256"], name
    spec = importlib.util.spec_from_file_location(
        "independent_physionet2019_score", directory / "evaluate_sepsis_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_components(scorer, labels, predictions):
    best = np.zeros(len(labels), dtype=np.int8)
    if labels.any():
        onset = int(np.argmax(labels)) + 6
        best[max(0, onset - 12):min(len(labels), onset + 4)] = 1
    inactive = np.zeros(len(labels), dtype=np.int8)
    return {name: float(scorer.compute_prediction_utility(labels, prediction))
            for name, prediction in [("observed", predictions), ("best", best),
                                     ("inaction", inactive), ("worst", 1 - best)]}


def verify_bundle(path, result, protocol, training_patients, training_hours, fold):
    assert sha256(path) == result["model_sha256"]
    bundle = joblib.load(path)
    assert bundle["training_patients"] == result["training_patients"] == training_patients
    assert result["training_hours"] == training_hours
    assert bundle["preprocessor"].fit_rows_ == training_hours
    assert bundle["preprocessor"].names_ == protocol["feature_columns"]
    assert bundle["fit_hospitals"] == result["details"]["fit_hospitals"] == ["A"]
    assert bundle["fold"] == result["fold"] == fold
    assert bundle["parameters"] == result["parameters"] == protocol["parameters"][result["model"]]
    assert bundle["model_name"] == result["model"]
    assert bundle["fitted_parameters"] == result["fitted_parameters"]
    if result["model"] == "lightgbm":
        assert bundle["model"].get_params()["n_estimators"] == result["fitted_parameters"]["n_estimators"]
    close(bundle["threshold"], result["threshold"])
    assert result["features"] == 312 and not result["B_evaluated"]
    return bundle


def replay_probabilities(bundle, patients, probability, eligible, raw, audit, cfg):
    positions = np.flatnonzero(eligible)
    positions = positions[np.unique(np.linspace(0, len(positions) - 1, 32, dtype=int))]
    owners = np.searchsorted(patients.stop.to_numpy(), positions, side="right")
    replayed = 0
    for owner in np.unique(owners):
        patient = patients.iloc[int(owner)]
        record = audit.loc[patient.patient_id]
        values = raw[int(record.raw_start):int(record.raw_stop)]
        selected = positions[owners == owner]
        offsets = selected - int(patient.start)
        features = causal_features(values, cfg)[offsets]
        transformed = bundle["preprocessor"].transform(features)
        with threadpool_limits(limits=2):
            model = bundle["model"]
            expected = (model.booster_.predict(transformed, num_threads=2)
                        if hasattr(model, "booster_") else model.predict_proba(transformed)[:, 1])
        close(probability[selected], expected)
        replayed += len(selected)
    assert replayed == 32
    return replayed


def verify_predictions(directory, bundle, identifiers, raw, audit, cfg, scorer):
    patients = pd.read_csv(directory / "patients.csv")
    assert patients.patient_id.is_unique and set(patients.patient_id) == set(identifiers)
    probability = np.load(directory / "probability.npy", mmap_mode="r")
    labels = np.load(directory / "labels.npy", mmap_mode="r")
    mask = np.load(directory / "task_eligible.npy", mmap_mode="r")
    assert len(probability) == len(labels) == len(mask) == int(patients.hours.sum())
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    assert np.isin(labels, [0, 1]).all() and mask.dtype == np.dtype(bool)
    np.testing.assert_array_equal(patients.start, np.r_[0, patients.stop.to_numpy()[:-1]])
    np.testing.assert_array_equal(patients.stop - patients.start, patients.hours)
    assert int(patients.stop.iloc[-1]) == len(labels)
    # Independently reconstruct eligibility from original timestamps and unshifted labels.
    for patient in patients.itertuples(index=False):
        record = audit.loc[patient.patient_id]
        assert bool(record.included) and patient.hospital == record.hospital
        values = raw[int(record.raw_start):int(record.raw_stop)]
        begin, end = int(patient.start), int(patient.stop)
        np.testing.assert_array_equal(labels[begin:end], values[:, 40])
        assert patient.hours == len(values) and patient.first_hour == int(values[0, 39])
        assert bool(patient.has_event) == bool(values[:, 40].any())
        assert values[0, 40] == 0 and values[0, 34] >= 18
        times, original_labels = values[:, 39], values[:, 40]
        if original_labels.any():
            onset = times[np.flatnonzero(original_labels)[0]] + 6
            assert onset <= times[-1]
            expected = times < onset
        else:
            expected = times + 6 <= times[-1]
        np.testing.assert_array_equal(mask[begin:end], expected)
    contributions = pd.read_csv(directory / "patient_utilities.csv")
    assert contributions.patient_id.is_unique
    assert set(contributions.patient_id) == set(identifiers)
    assert np.isfinite(contributions[COMPONENTS].to_numpy()).all()
    by_id = contributions.set_index("patient_id")
    event_positions = np.flatnonzero(patients.has_event.to_numpy(dtype=bool))
    control_positions = np.flatnonzero(~patients.has_event.to_numpy(dtype=bool))
    assert len(event_positions) and len(control_positions)
    probes = np.unique([0, len(patients) // 2, len(patients) - 1,
                        int(event_positions[0]), int(control_positions[0])])
    for position in probes:
        patient = patients.iloc[int(position)]
        begin, end = int(patient.start), int(patient.stop)
        expected = official_components(scorer, labels[begin:end],
                                       probability[begin:end] >= bundle["threshold"])
        for name, value in expected.items():
            close(by_id.loc[patient.patient_id, name], value)
    rank = ranking(labels[mask], probability[mask])
    score = utility(contributions)
    saved = read_json(directory / "evaluation.json")
    for name, value in rank.items():
        close(value, saved["ranking"]["auprc_average_precision" if name == "auprc" else name])
    for name, value in score.items():
        close(value, saved["utility"][name],
              atol=1e-12 if name == "normalized_utility" else 1e-8)
    close(saved["threshold"], bundle["threshold"])
    assert saved["patients"] == len(patients)
    assert saved["event_patients"] == int(patients.has_event.sum())
    assert saved["eligible_hours"] == int(mask.sum()) and saved["full_hours"] == len(labels)
    probes_replayed = replay_probabilities(bundle, patients, probability, mask, raw, audit, cfg)
    return {"patients": patients, "probability": probability, "labels": labels, "mask": mask,
            "contributions": contributions, "ranking": rank, "utility": score,
            "official_patient_probes": len(probes), "prediction_probes": probes_replayed}


def main():
    assert (OUT / "completed.json").exists(), "Run after the CV/refit pipeline completes"
    protocol, case = read_json(OUT / "protocol.json"), read_json(DATA / "case.json")
    assert protocol["feature_columns"] and len(set(protocol["feature_columns"])) == 312
    assert set(protocol["feature_columns"]) <= set(FEATURE_NAMES)
    assert protocol["B_selection_used"] is False
    for name, expected in protocol["input_fingerprints"].items():
        assert sha256(ROOT / name) == expected, name
    # The historical protocol also fingerprints both original processed split files.
    prior = read_json(ROOT / "reports/ab_regularization/protocol.json")["input_fingerprints"]
    for name in ["processed_data/splits/patient_splits.csv",
                 "processed_data/ab_expansion/B_patient_splits.csv",
                 "processed_data/ab_expansion/patient_splits.csv"]:
        assert sha256(ROOT / name) == prior[name], name
    for name, key in [("A_features.npy", "features_sha256"), ("A_rows.csv.gz", "rows_sha256"),
                      ("A_patients.csv", "patients_sha256")]:
        assert sha256(DATA / name) == case[key], name
    rows = pd.read_csv(DATA / "A_rows.csv.gz")
    patients = pd.read_csv(DATA / "A_patients.csv")
    original = pd.read_csv(ROOT / "processed_data/splits/patient_splits.csv")
    a_ids = set(original.loc[original.hospital.eq("A"), "patient_id"])
    b_ids = set(original.loc[original.hospital.eq("B"), "patient_id"])
    assert len(a_ids) == len(patients) == 20131 and len(b_ids) == 19743 and not a_ids & b_ids
    assert patients.patient_id.is_unique and set(patients.patient_id) == set(rows.patient_id) == a_ids
    assert len(rows) == 670920 and int(rows.label.sum()) == 9510 and rows.hospital.eq("A").all()
    assert int(patients.has_event.sum()) == 1585 and not rows.duplicated(["patient_id", "hour"]).any()
    event = rows.groupby("patient_id").label.max()
    np.testing.assert_array_equal(patients.set_index("patient_id").has_event.sort_index(), event.sort_index())
    features = np.load(DATA / "A_features.npy", mmap_mode="r")
    assert features.shape == (670920, 318) and features.dtype == np.dtype(np.float32)
    assert set(patients.fold) == {0, 1, 2}
    for fold, (_, held) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
                                    .split(patients.patient_id, patients.has_event)):
        assert set(patients.iloc[held].patient_id) == set(patients.loc[patients.fold.eq(fold), "patient_id"])
    raw = np.load(ROOT / "processed_data/raw_records.npy", mmap_mode="r")
    audit = pd.read_csv(ROOT / "outputs/patient_summary.csv").set_index("patient_id")
    cfg, scorer = config(), load_official_scorer()
    metrics = pd.read_csv(REPORT / "metrics.csv").set_index(["evaluation", "model"])
    assert metrics.index.is_unique and len(metrics) == 6
    thresholds, freeze = read_json(OUT / "thresholds.json"), read_json(OUT / "frozen_before_B.json")
    selection = read_json(OUT / "selection.json")
    assert set(thresholds) == set(MODELS) and not freeze["B_used_for_selection"]
    assert selection["thresholds"] == thresholds and selection["B_used"] is False
    assert selection["model_settings_changed"] is False
    assert sha256(OUT / "thresholds.json") == freeze["thresholds_sha256"]
    assert sha256(OUT / "selection.json") == freeze["selection_sha256"]
    row_folds = rows.patient_id.map(patients.set_index("patient_id").fold).to_numpy()
    checks = []
    for model in MODELS:
        oof, fold_components, all_held = np.full(len(rows), np.nan), [], []
        for fold in range(3):
            directory = OUT / "cv" / f"fold{fold}" / model
            result = read_json(directory / "results.json")
            assert result["model"] == model
            held_ids = set(patients.loc[patients.fold.eq(fold), "patient_id"])
            training_ids = a_ids - held_ids
            training_hours = int(rows.patient_id.isin(training_ids).sum())
            inner = pd.read_csv(directory / "inner_patients.csv")
            assert inner.patient_id.is_unique and set(inner.patient_id) == training_ids
            assert not set(inner.patient_id) & held_ids and set(inner.role) == {"fit", "calibration"}
            bundle = verify_bundle(directory / "model.joblib", result, protocol,
                                   len(training_ids), training_hours, fold)
            checked = verify_predictions(directory, bundle, held_ids, raw, audit, cfg, scorer)
            held_rows = rows.loc[row_folds == fold]
            mapping = checked["patients"].set_index("patient_id")
            locations = (held_rows.patient_id.map(mapping.start).to_numpy() + held_rows.hour.to_numpy()
                         - held_rows.patient_id.map(mapping.first_hour).to_numpy()).astype(int)
            assert checked["mask"][locations].all() and len(locations) == int(checked["mask"].sum())
            np.testing.assert_array_equal(checked["labels"][locations], held_rows.label)
            oof[row_folds == fold] = checked["probability"][locations]
            fold_components.append(checked["contributions"].assign(fold=fold))
            all_held.extend(checked["patients"].patient_id)
            for name, value in checked["ranking"].items():
                close(value, result["evaluation"]["ranking"]["auprc_average_precision" if name == "auprc" else name])
            close(checked["utility"]["normalized_utility"], result["evaluation"]["utility"]["normalized_utility"])
            checks.append({"stage": "cv", "model": model, "fold": fold,
                           "patients": len(held_ids), "official_patient_probes": checked["official_patient_probes"],
                           "prediction_probes": checked["prediction_probes"]})
        assert len(all_held) == len(set(all_held)) == 20131 and set(all_held) == a_ids
        assert np.isfinite(oof).all()
        np.testing.assert_array_equal(oof, np.load(OUT / "oof" / model / "eligible_probability.npy"))
        combined = pd.concat(fold_components, ignore_index=True)
        saved_components = pd.read_csv(OUT / "oof" / model / "honest_cv_patient_utilities.csv")
        pd.testing.assert_frame_equal(combined.sort_values("patient_id").reset_index(drop=True),
                                      saved_components.sort_values("patient_id").reset_index(drop=True))
        rank, score = ranking(rows.label.to_numpy(), oof), utility(combined)
        summary = metrics.loc[("A_patient_3CV_OOF", model)]
        assert int(summary.patients) == 20131 and int(summary.event_patients) == 1585
        assert int(summary.eligible_hours) == 670920 and int(summary.full_hours) == 788042
        close([summary.auroc, summary.auprc, summary.utility_score],
              [rank["auroc"], rank["auprc"], score["normalized_utility"]])
        close(summary.final_OOF_threshold, thresholds[model])
        final_path = OUT / "final" / model / "model.joblib"
        assert sha256(final_path) == freeze["model_sha256"][model]
        result = read_json(final_path.with_name("results.json"))
        assert result["model"] == model
        assert result["training_event_patients"] == 1585
        bundle = verify_bundle(final_path, result, protocol, 20131, 670920, -1)
        close(bundle["threshold"], thresholds[model])
        checked = verify_predictions(OUT / "B" / model, bundle, b_ids, raw, audit, cfg, scorer)
        assert len(checked["patients"]) == 19743 and int(checked["patients"].has_event.sum()) == 905
        assert len(checked["labels"]) == 757924 and int(checked["mask"].sum()) == 641470
        assert int(checked["labels"][checked["mask"]].sum()) == 5430
        summary = metrics.loc[("B_after_full_A_refit", model)]
        assert [int(summary.patients), int(summary.event_patients), int(summary.eligible_hours), int(summary.full_hours)] == [19743, 905, 641470, 757924]
        close([summary.auroc, summary.auprc, summary.utility_score, summary.final_OOF_threshold],
              [checked["ranking"]["auroc"], checked["ranking"]["auprc"],
               checked["utility"]["normalized_utility"], thresholds[model]])
        checks.append({"stage": "B", "model": model, "patients": 19743,
                       "official_patient_probes": checked["official_patient_probes"],
                       "prediction_probes": checked["prediction_probes"]})
    for model in MODELS:
        assert sha256(OUT / "final" / model / "model.joblib") == freeze["model_sha256"][model]
    assert sha256(OUT / "thresholds.json") == freeze["thresholds_sha256"]
    result = {"verified_at_utc": datetime.now(timezone.utc).isoformat(), "passed": True,
              "original_inputs_and_splits_unchanged": True, "patient_disjoint_cv_complete": True,
              "original_raw_labels_and_eligibility_verified": True,
              "AUROC_AP_and_pooled_official_utility_recomputed": True,
              "models_and_thresholds_match_pre_B_freeze": True,
              "official_scorer_commit": "467c49b514542be7a4a0bafe40fa2c3b064dda2e",
              "prediction_directories": checks, "verifier_sha256": sha256(Path(__file__))}
    (REPORT / "independent_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": True, "directories": len(checks),
                      "probability_replay_probes": sum(c["prediction_probes"] for c in checks),
                      "official_patient_probes": sum(c["official_patient_probes"] for c in checks)}), flush=True)


if __name__ == "__main__":
    main()
