"""Independently verify saved matched full-record 563-feature model results."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "processed_data/friend_563_comparison"
OUT = ROOT / "outputs/friend_563_comparison"
REPORT = ROOT / "reports/friend_563_comparison"
MAIN = "32ee355b9373e81a9d3bdb1f1b4a22883e04678e"
MODELS = ["xgboost", "lightgbm"]
EXPECTED = {"train": (27911, 1742, 1081518, 16952),
            "validation": (5981, 374, 233738, 3630),
            "test": (5982, 374, 230710, 3630)}


def read_json(path):
    return json.loads(Path(path).read_text())


def available(path):
    path = Path(path)
    assert path.exists() and path.stat().st_size > 0, path
    assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), f"Cloud placeholder: {path}"
    return path


def sha256(path):
    path = available(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    assert size == path.stat().st_size
    return digest.hexdigest()


def close(actual, expected, *, raw_utility=False):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8 if raw_utility else 1e-12)


def rank(labels, probability):
    assert len(labels) == len(probability)
    assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
    return {"auroc": float(roc_auc_score(labels, probability)),
            "auprc_average_precision": float(average_precision_score(labels, probability))}


def independent_delta180(labels, patients):
    """Exact integer marginal utilities for the immutable official default constants."""
    result = np.full(len(labels), -9, dtype=np.int64)
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        y = labels[a:b]
        if not y.any():
            continue
        relative = np.arange(len(y)) - (int(np.flatnonzero(y)[0]) + 6)
        early = (relative >= -12) & (relative <= -6)
        late = (relative > -6) & (relative <= 3)
        target = result[a:b]
        target[early] = 30 * (relative[early] + 12)
        target[late] = 20 * (relative[late] + 15)
        target[relative > 3] = 0
    return result


def independent_best_threshold(probability, delta):
    """Aggregate ties in a dictionary and search exact integer utility gains."""
    groups = {}
    for score, increment in zip(probability, delta):
        score = float(score)
        groups[score] = groups.get(score, 0) + int(increment)
    best_threshold = float(np.nextafter(float(np.max(probability)), np.inf))
    best_gain = running = 0
    for score in sorted(groups, reverse=True):
        running += groups[score]
        # Strict improvement preserves the highest threshold on a tied maximum.
        if running > best_gain:
            best_gain, best_threshold = running, score
    denominator = int(delta[delta > 0].sum())
    assert denominator > 0
    return best_threshold, best_gain / denominator, len(groups) + 1


def official_scorer():
    directory = ROOT / "third_party/physionet2019"
    upstream = read_json(directory / "UPSTREAM.json")
    assert upstream["commit"] == "467c49b514542be7a4a0bafe40fa2c3b064dda2e"
    for name, metadata in upstream["files"].items():
        assert sha256(directory / name) == metadata["sha256"]
    specification = importlib.util.spec_from_file_location("independent_friend_563_official", directory / "evaluate_sepsis_score.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def official_components(scorer, labels, predictions):
    best = np.zeros(len(labels), dtype=np.int8)
    if labels.any():
        onset = int(np.flatnonzero(labels)[0]) + 6
        best[max(0, onset - 12):min(len(labels), onset + 4)] = 1
    inactive = np.zeros(len(labels), dtype=np.int8)
    return {name: float(scorer.compute_prediction_utility(labels, action))
            for name, action in [("observed", predictions), ("best", best),
                                 ("inaction", inactive), ("worst", 1 - best)]}


def load_model(name, directory):
    if name == "xgboost":
        import xgboost as xgb
        model = xgb.Booster(model_file=directory / "model.ubj")
        model.set_param({"nthread": 2})
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(directory / "model.txt"))
    return model


def model_path(name, directory):
    return directory / ("model.ubj" if name == "xgboost" else "model.txt")


def replay_model(name, model, features, expected):
    prediction = (model.inplace_predict(features) if name == "xgboost"
                  else model.predict(features, num_threads=2))
    close(prediction, expected)


def verify_original_features(protocol, raw, people):
    source = DATA / "source"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", MAIN, "sepsis_pipeline"], cwd=ROOT, text=True).splitlines()
    for name in paths:
        if not name.endswith(".py"):
            continue
        blob = subprocess.check_output(["git", "show", f"{MAIN}:{name}"], cwd=ROOT)
        assert available(source / name).read_bytes() == blob
        assert hashlib.sha256(blob).hexdigest() == protocol["source_hashes"][name]
    sys.path.insert(0, str(source))
    from sepsis_pipeline.features import FEATURE_NAMES, transform
    assert FEATURE_NAMES["augmented"] == protocol["feature_names"]
    assert len(set(protocol["feature_names"])) == 563
    forbidden = {"patient_id", "subject_id", "hospital", "split", "cv_fold", "label", "SepsisLabel", "Unit1", "Unit2"}
    assert not forbidden & set(protocol["feature_names"])
    checks = []
    for split, (patients, matrix, _) in people.items():
        for _, group in patients.groupby(["hospital", "has_event"]):
            sample = group.iloc[0]
            values = raw[int(sample.raw_start):int(sample.raw_stop)]
            actual = transform(values[:, :40], "augmented")
            np.testing.assert_array_equal(actual, matrix[int(sample.start):int(sample.stop)])
            for stop in sorted(set([1, min(6, len(values)), min(12, len(values)), max(1, len(values)//2), len(values)])):
                np.testing.assert_array_equal(transform(values[:stop, :40], "augmented"), actual[:stop])
            try:
                transform(values, "augmented")
            except ValueError:
                pass
            else:
                raise AssertionError("Feature interface accepted the outcome column")
            # Verify the cached float64 input against the original committed PSV bytes.
            name = f"data/training_set{sample.hospital}/{sample.patient_id[2:]}.psv"
            blob = subprocess.check_output(["git", "show", f"{MAIN}:{name}"], cwd=ROOT)
            original = np.loadtxt(io.BytesIO(blob), delimiter="|", skiprows=1, ndmin=2)
            np.testing.assert_array_equal(original, values)
            checks.append({"split": split, "patient_id": sample.patient_id, "hours": len(values)})
    return transform, checks


def verify_data():
    raw_case = read_json(DATA / "raw_case.json")
    for name, key in [("raw_float64.npy", "raw_sha256"), ("raw_audit.csv", "audit_sha256"),
                      ("patient_splits.csv", "patient_splits_sha256")]:
        assert sha256(DATA / name) == raw_case[key]
    assert raw_case["all_original_PSV_sha256_verified"]
    raw = np.load(available(DATA / "raw_float64.npy"), mmap_mode="r")
    assert raw.shape == (1552210, 41) and raw.dtype == np.dtype(np.float64)
    audit = pd.read_csv(available(DATA / "raw_audit.csv"))
    assert len(audit) == 40336 and audit.patient_id.is_unique
    all_ids, people = [], {}
    for split, expected in EXPECTED.items():
        case = read_json(DATA / f"{split}_case.json")
        for suffix, key in [("features.npy", "feature_sha256"), ("labels.npy", "labels_sha256"),
                            ("patients.csv", "patients_sha256")]:
            assert sha256(DATA / f"{split}_{suffix}") == case[key]
        matrix = np.load(available(DATA / f"{split}_features.npy"), mmap_mode="r")
        labels = np.load(available(DATA / f"{split}_labels.npy"), mmap_mode="r")
        patients = pd.read_csv(available(DATA / f"{split}_patients.csv"))
        assert matrix.shape == (len(labels), 563) and matrix.dtype == np.dtype(np.float32)
        actual = len(patients), int(patients.has_event.sum()), len(labels), int(labels.sum())
        assert actual == expected
        assert patients.patient_id.is_unique and np.isin(labels, [0, 1]).all()
        np.testing.assert_array_equal(patients.start, np.r_[0, patients.stop.to_numpy()[:-1]])
        np.testing.assert_array_equal(patients.stop - patients.start, patients.hours)
        assert int(patients.stop.iloc[-1]) == len(labels)
        if split == "train":
            assert set(patients.cv_fold) == {0, 1, 2}
        else:
            assert patients.cv_fold.eq(-1).all()
        for patient in patients.itertuples(index=False):
            values = raw[int(patient.raw_start):int(patient.raw_stop)]
            a, b = int(patient.start), int(patient.stop)
            assert len(values) == patient.hours and int(values[0, 39]) == patient.first_hour
            np.testing.assert_array_equal(labels[a:b], values[:, 40])
            assert int(values[:, 40].max()) == patient.has_event
        all_ids.extend(patients.patient_id)
        people[split] = patients, matrix, labels
    assert len(all_ids) == len(set(all_ids)) == 39874
    assert set(all_ids) == set(audit.loc[audit.included, "patient_id"])
    # Cross-check historical patient assignments independently of the new manifest.
    old_a = pd.read_csv(ROOT / "processed_data/splits/patient_splits.csv")
    old_a = old_a.loc[old_a.hospital.eq("A"), ["patient_id", "split"]]
    old_b = pd.read_csv(ROOT / "reports/ab_expansion/B_patient_splits.csv")
    old_b = old_b.rename(columns={"expansion_split": "split"})
    old_b["split"] = old_b.split.str.removeprefix("B_")
    old = pd.concat([old_a, old_b[["patient_id", "split"]]]).set_index("patient_id")
    for split, (patients, _, _) in people.items():
        assert old.loc[patients.patient_id, "split"].eq(split).all()
    old_folds = pd.read_csv(ROOT / "reports/ab_regularization/patient_cv_folds.csv").set_index("patient_id")
    train = people["train"][0].set_index("patient_id")
    np.testing.assert_array_equal(train.cv_fold, old_folds.loc[train.index, "fold"])
    return raw, people


def prediction_probes(name, model, patients, probability, raw, transform):
    positions = np.unique(np.linspace(0, len(probability)-1, 32, dtype=int))
    owners = np.searchsorted(patients.stop.to_numpy(), positions, side="right")
    for owner in np.unique(owners):
        patient = patients.iloc[int(owner)]
        selected = positions[owners == owner]
        values = raw[int(patient.raw_start):int(patient.raw_stop), :40]
        features = transform(values, "augmented")[selected - int(patient.start)]
        replay_model(name, model, features, probability[selected])
    return len(positions)


def verify_cv(protocol, people, raw, transform):
    patients, _, labels = people["train"]
    summaries = []
    for name in MODELS:
        held_ids, selected_rounds = [], []
        for fold in range(3):
            directory = OUT / "cv" / f"fold{fold}" / name
            result = read_json(directory / "results.json")
            held = pd.read_csv(directory / "held_patients.csv")
            expected = patients.loc[patients.cv_fold.eq(fold)].copy()
            assert list(held.patient_id) == list(expected.patient_id)
            expected_labels = np.concatenate([labels[int(p.start):int(p.stop)] for p in expected.itertuples(index=False)])
            y = np.load(available(directory / "labels.npy"))
            np.testing.assert_array_equal(y, expected_labels)
            np.testing.assert_array_equal(held.start, np.r_[0, held.stop.to_numpy()[:-1]])
            np.testing.assert_array_equal(held.stop - held.start, held.hours)
            assert result["training_patients"] == len(patients)-len(held)
            assert result["training_hours"] == len(labels)-len(y)
            assert result["evaluation_patients"] == len(held) and result["evaluation_hours"] == len(y)
            assert not result["test_read"] and not result["validation_used_for_fitting"]
            assert result["parameters"] == protocol["parameters"][name]
            assert result["feature_count"] == 563
            assert sha256(model_path(name, directory)) == result["model_sha256"]
            probability = np.load(available(directory / "probability.npy"))
            for metric, value in rank(y, probability).items():
                close(value, result["evaluation"][metric])
            model = load_model(name, directory)
            rounds = int(result["selected_rounds"])
            assert 1 <= rounds <= protocol["maximum_rounds"]
            assert (model.num_boosted_rounds() if name == "xgboost" else model.current_iteration()) == rounds
            assert model.num_features() == 563 if name == "xgboost" else model.num_feature() == 563
            probes = prediction_probes(name, model, held, probability, raw, transform)
            selected_rounds.append(rounds); held_ids.extend(held.patient_id)
            summaries.append({"model": name, "fold": fold, "prediction_probes": probes})
        assert len(held_ids) == len(set(held_ids)) == len(patients)
        assert set(held_ids) == set(patients.patient_id)
        saved_rounds = int(read_json(OUT / "rounds.json")[name])
        assert saved_rounds == int(np.median(selected_rounds))
    return summaries


def verify_evaluation(name, split, threshold, people, raw, transform, scorer):
    patients, _, labels = people[split]
    directory = OUT / "evaluation" / split / name
    probability = np.load(available(directory / "probability.npy"), mmap_mode="r")
    saved = read_json(directory / "metrics.json")
    close(saved["threshold"], threshold)
    for metric, value in rank(labels, probability).items():
        close(value, saved["ranking"][metric])
    contributions = pd.read_csv(available(directory / "patient_utilities.csv"))
    assert list(contributions.patient_id) == list(patients.patient_id)
    observed = {key: 0.0 for key in ["observed", "best", "inaction", "worst"]}
    for row, patient in zip(contributions.itertuples(index=False), patients.itertuples(index=False)):
        a, b = int(patient.start), int(patient.stop)
        expected = official_components(scorer, labels[a:b], probability[a:b] >= threshold)
        for component, value in expected.items():
            close(getattr(row, component), value, raw_utility=True)
            observed[component] += value
    denominator = observed["best"]-observed["inaction"]
    utility = (observed["observed"]-observed["inaction"])/denominator
    close(utility, saved["utility"]["normalized_utility"])
    close(denominator, saved["utility"]["normalization_denominator"], raw_utility=True)
    for component, value in observed.items():
        close(value, saved["utility"][component], raw_utility=True)
    delta = independent_delta180(labels, patients)
    exact = int(delta[probability >= threshold].sum()) / int(delta[delta > 0].sum())
    close(exact, utility)
    model = load_model(name, OUT / "final" / name)
    probes = prediction_probes(name, model, patients, probability, raw, transform)
    if split == "validation":
        best_threshold, best_utility, groups = independent_best_threshold(probability, delta)
        close(best_threshold, threshold)
        info = read_json(OUT / "final" / name / "threshold.json")
        close(best_utility, info["max_validation_utility"])
        close(best_utility, utility)
        assert int(info["candidate_groups"]) == groups
        assert int(info["positive_hours_predicted"]) == int((probability >= threshold).sum())
        worker_probability = np.load(available(OUT / "final" / name / "validation_probability.npy"))
        np.testing.assert_array_equal(probability, worker_probability)
    else:
        groups = None
    return {"model": name, "split": split, "patients": len(patients),
            "prediction_probes": probes, "utility": utility, "threshold_candidate_groups": groups,
            **rank(labels, probability)}


def main():
    assert (OUT / "completed.json").exists(), "Run the verifier after the experiment finishes"
    protocol = read_json(OUT / "protocol.json")
    assert protocol["main_commit"] == MAIN and protocol["standard_id"] == "SEPSIS_FULL_RECORD_V2"
    raw, people = verify_data()
    transform, feature_checks = verify_original_features(protocol, raw, people)
    cv_checks = verify_cv(protocol, people, raw, transform)
    scorer = official_scorer()
    freeze = read_json(OUT / "frozen_before_test.json")
    assert freeze["test_used_for_selection"] is False
    assert sha256(OUT / "rounds.json") == freeze["rounds_sha256"]
    metric_table = pd.read_csv(REPORT / "metrics.csv").set_index(["split", "model"])
    assert metric_table.index.is_unique and set(metric_table.index) == {(s, m) for s in ["validation", "test"] for m in MODELS}
    evaluations = []
    for name in MODELS:
        directory = OUT / "final" / name
        final = read_json(directory / "results.json")
        assert final["training_patients"] == 27911 and final["training_hours"] == 1081518
        assert not final["test_read"] and not final["validation_used_for_fitting"]
        assert final["parameters"] == protocol["parameters"][name]
        assert int(final["selected_rounds"]) == int(read_json(OUT / "rounds.json")[name])
        assert sha256(model_path(name, directory)) == freeze["model_sha256"][name] == final["model_sha256"]
        threshold_path = directory / "threshold.json"
        assert sha256(threshold_path) == freeze["threshold_sha256"][name]
        threshold = float(read_json(threshold_path)["threshold"])
        for split in ["validation", "test"]:
            result = verify_evaluation(name, split, threshold, people, raw, transform, scorer)
            for column in ["auroc", "auprc"]:
                metric = "auprc_average_precision" if column == "auprc" else column
                close(result[metric], metric_table.loc[(split, name), column])
            close(result["utility"], metric_table.loc[(split, name), "utility_score"])
            close(threshold, metric_table.loc[(split, name), "threshold"])
            evaluations.append(result)
    report = {"passed": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "all_39874_patient_labels_checked_against_original_float64": True,
              "original_patient_split_and_CV_folds_match": True,
              "feature_source_commit": MAIN, "feature_patient_checks": feature_checks,
              "CV_model_checks": cv_checks, "evaluation_checks": evaluations,
              "all_validation_and_test_patient_utilities_recomputed_with_official_scorer": True,
              "validation_thresholds_exact_integer_global_optima": True,
              "frozen_final_models_and_threshold_hashes_match": True,
              "probability_replay_probes": sum(c["prediction_probes"] for c in cv_checks+evaluations)}
    path = REPORT / "independent_verification.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"passed": True, "patients": 39874,
                      "probability_replay_probes": report["probability_replay_probes"],
                      "validation_thresholds_globally_optimal": True}), flush=True)


if __name__ == "__main__":
    main()
