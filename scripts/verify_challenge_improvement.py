"""Independent read-only checks for the competition-inspired experiments.

--development-only checks available completed CV fits without reading validation
or test data. --final additionally verifies all final evaluation artifacts and
writes reports/challenge_improvement/independent_verification.json. Neither
mode changes the experiment's outputs or cached matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE = ROOT / "processed_data/friend_563_comparison"
DATA = ROOT / "processed_data/challenge_improvement"
OUT = ROOT / "outputs/challenge_improvement"
REPORT = ROOT / "reports/challenge_improvement"
MAIN = "32ee355b9373e81a9d3bdb1f1b4a22883e04678e"
BLOCKS = {"base": (), "lag": ("lag",), "lag_observed": ("lag", "observed"),
          "all": ("lag", "observed", "clinical")}


def available(path):
    path = Path(path)
    assert path.is_file() and path.stat().st_size > 0, path
    assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), path
    return path


def read_json(path):
    return json.loads(available(path).read_text())


def sha256(path):
    result = hashlib.sha256()
    with available(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(left, right, atol=1e-12):
    np.testing.assert_allclose(left, right, rtol=0, atol=atol)


def scorer():
    directory = ROOT / "third_party/physionet2019"
    metadata = read_json(directory / "UPSTREAM.json")
    assert metadata["commit"] == "467c49b514542be7a4a0bafe40fa2c3b064dda2e"
    for name, record in metadata["files"].items():
        assert sha256(directory / name) == record["sha256"]
    spec = importlib.util.spec_from_file_location("independent_challenge_score", directory / "evaluate_sepsis_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_prediction_utility


def official_components(function, labels, alarms):
    inactive = np.zeros(len(labels), np.int8)
    best = inactive.copy()
    if np.any(labels):
        onset = int(np.flatnonzero(labels)[0]) + 6
        best[max(0, onset-12):min(len(labels), onset+4)] = 1
    return {name: float(function(labels, action)) for name, action in
            [("observed", alarms), ("best", best), ("inaction", inactive), ("worst", 1-best)]}


def independent_delta(labels, patients):
    # Difference of the official TP/FN curves, expressed in exact integer units.
    delta = np.empty(len(labels), dtype=np.int64)
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        target = delta[a:b]
        target[:] = -9
        positive = np.flatnonzero(labels[a:b])
        if not len(positive):
            continue
        onset = int(positive[0]) + 6
        for t in range(b-a):
            r = t-onset
            if -12 <= r <= -6:
                target[t] = 30*(r+12)
            elif -6 < r <= 3:
                target[t] = 20*(r+15)
            elif r > 3:
                target[t] = 0
    return delta


def independent_threshold(scores, delta):
    groups = {}
    for score, increment in zip(scores, delta):
        value = float(score)
        groups[value] = groups.get(value, 0) + int(increment)
    threshold = float(np.nextafter(float(np.max(scores)), np.inf))
    running = best = 0
    for value in sorted(groups, reverse=True):
        running += groups[value]
        if running > best:
            best, threshold = running, value
    denominator = int(delta[delta > 0].sum())
    assert denominator > 0
    return threshold, best/denominator


def local_patients(patients):
    result = patients.copy().reset_index(drop=True)
    result["stop"] = result.hours.cumsum()
    result["start"] = result.stop-result.hours
    return result


def gather(labels, patients):
    return np.concatenate([labels[int(p.start):int(p.stop)] for p in patients.itertuples(index=False)])


def check_map(patients, labels, raw):
    assert patients.patient_id.is_unique
    np.testing.assert_array_equal(patients.start, np.r_[0, patients.stop.to_numpy()[:-1]])
    np.testing.assert_array_equal(patients.stop-patients.start, patients.hours)
    assert int(patients.stop.iloc[-1]) == len(labels)
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        record = raw[int(patient.raw_start):int(patient.raw_stop)]
        assert len(record) == b-a and int(record[0, 39]) == patient.first_hour
        np.testing.assert_array_equal(labels[a:b], record[:, 40])
        assert int(np.max(labels[a:b])) == patient.has_event


def ranking(labels, scores):
    assert len(labels) == len(scores) and np.isfinite(scores).all()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def normalized_official(function, labels, scores, patients, threshold, saved=None):
    totals = {name: 0. for name in ("observed", "best", "inaction", "worst")}
    if saved is not None:
        assert list(saved.patient_id) == list(patients.patient_id)
    for i, patient in enumerate(patients.itertuples(index=False)):
        a, b = int(patient.start), int(patient.stop)
        values = official_components(function, labels[a:b], (scores[a:b] >= threshold).astype(np.int8))
        for name, value in values.items():
            totals[name] += value
            if saved is not None:
                close(value, saved.iloc[i][name], atol=1e-9)
    denominator = totals["best"]-totals["inaction"]
    assert denominator > 0
    return (totals["observed"]-totals["inaction"])/denominator


def feature_functions(protocol):
    from sepsis_warning.challenge_features import engineer_patient, feature_names
    source_protocol = read_json(ROOT / "outputs/friend_563_comparison/protocol.json")
    for name, expected in source_protocol["source_hashes"].items():
        source = BASE / "source" / name
        assert sha256(source) == expected
    sys.path.insert(0, str(BASE / "source"))
    from sepsis_pipeline.features import transform, FEATURE_NAMES
    forbidden = {"patient_id", "subject_id", "hospital", "split", "cv_fold", "label", "SepsisLabel", "has_event"}
    for name, blocks in BLOCKS.items():
        names = FEATURE_NAMES["augmented"]+feature_names(blocks)
        assert protocol["feature_names"][name] == names and len(names) == len(set(names))
        assert not forbidden.intersection(names)
    return transform, engineer_patient


def verify_extra(split, patients, labels, raw, transform, engineer):
    case = read_json(DATA / f"{split}_extra_case.json")
    assert sha256(ROOT / "src/sepsis_warning/challenge_features.py") == case["source_sha256"]
    assert sha256(DATA / f"{split}_extra.npy") == case["feature_sha256"]
    extra = np.load(available(DATA / f"{split}_extra.npy"), mmap_mode="r")
    base = np.load(available(BASE / f"{split}_features.npy"), mmap_mode="r")
    assert extra.shape == (len(labels), 228) and extra.dtype == np.dtype("float32")
    assert base.shape == (len(labels), 563) and base.dtype == np.dtype("float32")
    checked = []
    for _, group in patients.groupby(["hospital", "has_event"]):
        for _, patient in group.iloc[[0, len(group)//2]].iterrows():
            values = raw[int(patient.raw_start):int(patient.raw_stop)]
            # The entire float64 input is anchored below to its original-PSV
            # provenance SHA. Avoid reopening iCloud-offloaded Git objects.
            a, b = int(patient.start), int(patient.stop)
            actual = engineer(values[:, :40])
            np.testing.assert_array_equal(actual, extra[a:b])
            np.testing.assert_array_equal(transform(values[:, :40], "augmented"), base[a:b])
            for cutoff in sorted({1, min(6, len(values)), min(12, len(values)), max(1, len(values)//2), len(values)}):
                np.testing.assert_array_equal(engineer(values[:cutoff, :40]), actual[:cutoff])
            try:
                engineer(values)
            except ValueError:
                pass
            else:
                raise AssertionError("Feature interface accepted SepsisLabel")
            checked.append(patient.patient_id)
    return checked


def verify_roles(protocol, patients):
    path = DATA / "cv_patient_roles.csv"
    assert sha256(path) == protocol["roles_sha256"]
    assert sha256(BASE / "patient_splits.csv") == protocol["patient_splits_sha256"]
    roles = pd.read_csv(path)
    assert len(roles) == 3*len(patients)
    result = {}
    for fold in range(3):
        subset = roles.loc[roles.outer_fold.eq(fold)].set_index("patient_id")
        assert subset.index.is_unique and set(subset.index) == set(patients.patient_id)
        other = patients.loc[patients.cv_fold.ne(fold)]
        fit, calibration = train_test_split(other.patient_id, test_size=.15, random_state=62026+fold,
                                           stratify=other.hospital+"_"+other.has_event.astype(str))
        expected = {"fit": set(fit), "calibration": set(calibration),
                    "held": set(patients.loc[patients.cv_fold.eq(fold), "patient_id"])}
        parts = {}
        for name, ids in expected.items():
            assert set(subset.loc[subset.role.eq(name)].index) == ids
            parts[name] = patients.loc[patients.patient_id.isin(ids)].copy()
        assert not (expected["fit"] & expected["calibration"] or expected["fit"] & expected["held"]
                    or expected["calibration"] & expected["held"])
        result[fold] = parts
    return result


def load_model(candidate, directory, rounds):
    if candidate["model"] == "xgboost":
        import xgboost as xgb
        model = xgb.Booster(model_file=directory / "model.ubj")
        model.set_param({"nthread": 2})
        assert model.num_boosted_rounds() == rounds
    else:
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(directory / "model.txt"))
        assert model.current_iteration() == rounds
    return model


def prediction_replay(candidate, directory, rounds, patients, scores, raw, transform, engineer):
    model = load_model(candidate, directory, rounds)
    feature_set = candidate["feature_set"]
    positions = np.unique(np.linspace(0, len(scores)-1, 24, dtype=int))
    owners = np.searchsorted(patients.stop.to_numpy(), positions, side="right")
    for owner in np.unique(owners):
        patient = patients.iloc[int(owner)]
        raw40 = raw[int(patient.raw_start):int(patient.raw_stop), :40]
        selected = positions[owners == owner]
        indices = selected-int(patient.start)
        features = transform(raw40, "augmented")[indices]
        if BLOCKS[feature_set]:
            features = np.column_stack([features, engineer(raw40, BLOCKS[feature_set])[indices]])
        actual = (model.inplace_predict(features) if candidate["model"] == "xgboost"
                  else model.predict(features, num_iteration=rounds, num_threads=2))
        close(actual, scores[selected])
    return len(positions)


def verify_cv(protocol, candidates, labels, delta, roles, raw, transform, engineer, function):
    checks = []
    for result_path in sorted((OUT / "cv").glob("fold*/*/results.json")):
        directory = result_path.parent
        saved = read_json(result_path)
        candidate = candidates[saved["candidate"]]
        fold = int(saved["fold"])
        parts = roles[fold]
        for name, key in [("fit", "fit"), ("calibration", "inner_calibration")]:
            part = parts[name]
            expected = {"patients": len(part), "event_patients": int(part.has_event.sum()),
                        "hours": int(part.hours.sum()), "patient_ids_sha256": digest(part.patient_id.tolist())}
            assert saved[key] == expected
        assert not saved["test_read"] and not saved["validation_used_for_fitting"]
        assert not saved["outer_holdout_used_for_selection"]
        assert saved["feature_count"] == len(protocol["feature_names"][candidate["feature_set"]])
        assert saved["model"] == candidate["model"] and saved["target"] == candidate["target"]
        rounds = int(saved["selected_rounds"])
        assert 1 <= rounds <= candidate["max_rounds"]
        model_file = directory / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt")
        assert sha256(model_file) == saved["model_sha256"]
        assert sha256(directory / "held_score.npy") == saved["score_sha256"]
        for name in ("calibration", "held"):
            csv_name = "cal_patients.csv" if name == "calibration" else "held_patients.csv"
            pd.testing.assert_frame_equal(pd.read_csv(directory / csv_name), local_patients(parts[name]), check_dtype=False)
        cal_y, held_y = [gather(labels, parts[name]) for name in ("calibration", "held")]
        cal_d, held_d = [gather(delta, parts[name]) for name in ("calibration", "held")]
        cal_score = np.load(available(directory / "cal_score.npy"))
        held_score = np.load(available(directory / "held_score.npy"))
        threshold, maximum = independent_threshold(cal_score, cal_d)
        close(threshold, saved["calibration"]["threshold"])
        close(maximum, saved["calibration"]["max_validation_utility"])
        close(roc_auc_score(cal_y, cal_score), saved["calibration"]["auroc"])
        for key, value in ranking(held_y, held_score).items():
            close(value, saved["evaluation"][key])
        official = normalized_official(function, held_y, held_score, local_patients(parts["held"]), threshold)
        close(official, saved["evaluation"]["utility_score"])
        close(official, held_d[held_score >= threshold].sum()/held_d[held_d > 0].sum())
        if candidate["target"] == "utility_classification":
            fit_weight_mean = np.abs(gather(delta, parts["fit"])/180.).mean()
            close(fit_weight_mean, saved["weight_mean_before_normalization"])
            close(1./fit_weight_mean, saved["weight_scale"])
            assert float(saved["parameters"].get("scale_pos_weight", 1.)) == 1.
        else:
            assert saved["weight_mean_before_normalization"] is None and saved["weight_scale"] is None
        if candidate["target"] != "utility_regression":
            assert ((held_score >= 0) & (held_score <= 1)).all()
        if candidate["selection_metric"] == "utility":
            history = saved["utility_iteration_history"]
            assert history and any(row["rounds"] == rounds for row in history)
            winner = next(row for row in history if row["rounds"] == rounds)
            close(winner["inner_utility"], maximum)
        signature = digest({"candidate": candidate, "feature_names": protocol["feature_names"][candidate["feature_set"]],
                            "protocol_sha256": sha256(OUT / "protocol.json"),
                            "worker_sha256": sha256(ROOT / "scripts/fit_challenge_improvement.py"),
                            "shared_module_sha256": sha256(ROOT / "src/sepsis_warning/challenge_improvement.py"),
                            "fold": fold, "rounds": None})
        assert saved["signature"] == signature
        probes = prediction_replay(candidate, directory, rounds, local_patients(parts["held"]), held_score,
                                   raw, transform, engineer)
        checks.append({"candidate": candidate["id"], "fold": fold, "prediction_replays": probes,
                       "official_utility_patients": len(parts["held"]), "selected_rounds": rounds})
    return checks


def verify_cv_summaries(labels, delta, roles):
    path = REPORT / "cv_metrics.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    checked = []
    for row in frame.itertuples(index=False):
        scores = np.full(len(labels), np.nan)
        alarms = np.zeros(len(labels), dtype=bool)
        rounds, aucs, aps = [], [], []
        for fold in range(3):
            part = roles[fold]["held"]
            index = np.concatenate([np.arange(p.start, p.stop) for p in part.itertuples(index=False)])
            directory = OUT / "cv" / f"fold{fold}" / row.candidate
            saved = read_json(directory / "results.json")
            scores[index] = np.load(available(directory / "held_score.npy"))
            alarms[index] = scores[index] >= saved["calibration"]["threshold"]
            rounds.append(saved["selected_rounds"])
            aucs.append(saved["evaluation"]["auroc"])
            aps.append(saved["evaluation"]["auprc"])
        assert np.isfinite(scores).all()
        np.testing.assert_array_equal(scores, np.load(available(OUT / "cv_predictions" / f"{row.candidate}_score.npy")))
        np.testing.assert_array_equal(alarms, np.load(available(OUT / "cv_predictions" / f"{row.candidate}_alarm.npy")))
        for key, value in ranking(labels, scores).items():
            close(value, getattr(row, key))
        close(np.mean(aucs), row.mean_fold_auroc)
        close(np.mean(aps), row.mean_fold_auprc)
        close(delta[alarms].sum()/delta[delta > 0].sum(), row.utility_score)
        assert int(np.median(rounds)) == row.final_rounds
        checked.append(row.candidate)
    return checked


def independent_blend(definition, scores, references):
    left, right = definition["components"]
    arrays = []
    for identifier in (left, right):
        values = scores[identifier]
        if definition["method"] == "percentile_rank":
            reference = np.sort(references[identifier])
            values = np.searchsorted(reference, values, side="right") / len(reference)
        else:
            assert definition["method"] == "probability_mean"
        arrays.append(values)
    weight = definition["weight_first"]
    return weight*arrays[0]+(1-weight)*arrays[1]


def verify_ensemble_cv(labels, delta, roles, function):
    definition_path = OUT / "ensemble_definitions.json"
    if not definition_path.exists():
        return []
    definitions = read_json(definition_path)
    summary_path = REPORT / "cv_ensemble_metrics.csv"
    summaries = pd.read_csv(summary_path).set_index("candidate") if summary_path.exists() else None
    checks = []
    for definition in definitions:
        pooled_score = np.full(len(labels), np.nan)
        pooled_alarm = np.zeros(len(labels), bool)
        aucs, aps = [], []
        for fold in range(3):
            directory = OUT / "ensembles/cv" / f"fold{fold}" / definition["id"]
            if not (directory / "results.json").exists():
                continue
            result = read_json(directory / "results.json")
            assert result["definition"] == definition
            assert not result["outer_holdout_used_for_threshold"]
            component_cal, component_held = {}, {}
            for identifier in definition["components"]:
                component = OUT / "cv" / f"fold{fold}" / identifier
                component_cal[identifier] = np.load(available(component / "cal_score.npy"))
                component_held[identifier] = np.load(available(component / "held_score.npy"))
            cal_scores = independent_blend(definition, component_cal, component_cal)
            held_scores = independent_blend(definition, component_held, component_cal)
            np.testing.assert_array_equal(cal_scores, np.load(available(directory / "cal_score.npy")))
            np.testing.assert_array_equal(held_scores, np.load(available(directory / "held_score.npy")))
            cal_y, held_y = [gather(labels, roles[fold][name]) for name in ("calibration", "held")]
            cal_d = gather(delta, roles[fold]["calibration"])
            threshold, maximum = independent_threshold(cal_scores, cal_d)
            close(threshold, result["calibration"]["threshold"])
            close(maximum, result["calibration"]["max_validation_utility"])
            ranking_values = ranking(held_y, held_scores)
            for key, value in ranking_values.items():
                close(value, result["evaluation"][key])
            official = normalized_official(function, held_y, held_scores,
                                          local_patients(roles[fold]["held"]), threshold)
            close(official, result["evaluation"]["utility_score"])
            held = roles[fold]["held"]
            index = np.concatenate([np.arange(p.start, p.stop) for p in held.itertuples(index=False)])
            pooled_score[index], pooled_alarm[index] = held_scores, held_scores >= threshold
            aucs.append(ranking_values["auroc"])
            aps.append(ranking_values["auprc"])
            checks.append({"ensemble": definition["id"], "fold": fold, "official_utility_patients": len(held)})
        if np.isfinite(pooled_score).all():
            identifier = definition["id"]
            np.testing.assert_array_equal(pooled_score, np.load(available(OUT / "cv_predictions" / f"{identifier}_score.npy")))
            np.testing.assert_array_equal(pooled_alarm, np.load(available(OUT / "cv_predictions" / f"{identifier}_alarm.npy")))
            assert summaries is not None
            row = summaries.loc[identifier]
            for key, value in ranking(labels, pooled_score).items():
                close(value, row[key])
            close(np.mean(aucs), row.mean_fold_auroc)
            close(np.mean(aps), row.mean_fold_auprc)
            close(delta[pooled_alarm].sum()/delta[delta > 0].sum(), row.utility_score)
    return checks


def verify_input_manifest(splits, declared):
    expected = {}
    for split in splits:
        case = read_json(BASE / f"{split}_case.json")
        for suffix, key in (("features.npy", "feature_sha256"), ("labels.npy", "labels_sha256"),
                            ("patients.csv", "patients_sha256")):
            path = BASE / f"{split}_{suffix}"
            actual = sha256(path)
            assert actual == case[key]
            expected[str(path.relative_to(ROOT))] = actual
        extra_case = read_json(DATA / f"{split}_extra_case.json")
        path = DATA / f"{split}_extra.npy"
        actual = sha256(path)
        assert actual == extra_case["feature_sha256"]
        assert extra_case["source_sha256"] == sha256(ROOT / "src/sepsis_warning/challenge_features.py")
        expected[str(path.relative_to(ROOT))] = actual
    path = DATA / "train_delta.npy"
    expected[str(path.relative_to(ROOT))] = sha256(path)
    assert declared == expected
    return expected


def verify_final(protocol, candidates, people, raw, transform, engineer, function):
    # The coordinator freezes choices before constructing/scoring final test data.
    frozen = read_json(OUT / "frozen_before_test.json")
    assert not frozen["test_used_for_selection"] and frozen["historical_test_exposure"]
    verify_input_manifest(("train", "validation"), frozen["input_sha256"])
    verify_input_manifest(("test",), read_json(REPORT / "test_input_sha256.json"))
    expected_sources = {
        "src/sepsis_warning/challenge_improvement.py", "src/sepsis_warning/challenge_features.py",
        "src/sepsis_warning/challenge_utility.py", "src/sepsis_warning/challenge_evaluation.py",
        "scripts/fit_challenge_improvement.py", "scripts/run_challenge_improvement.py"}
    assert set(frozen["source_sha256"]) == expected_sources
    for name, expected_hash in frozen["source_sha256"].items():
        assert sha256(ROOT / name) == expected_hash
    for name, key in [("rounds.json", "rounds_sha256"), ("final_selection.json", "selection_sha256"),
                      ("model_selection.json", "model_selection_sha256"),
                      ("ensemble_definitions.json", "ensemble_definitions_sha256")]:
        assert sha256(OUT / name) == frozen[key]
    selection = read_json(OUT / "final_selection.json")
    assert selection["models"] == frozen["final_candidate_ids"]
    assert selection["ensembles"] == frozen["ensembles"]
    rounds = read_json(OUT / "rounds.json")
    definitions = {entry["id"]: entry for entry in frozen["ensembles"]}
    references = {}
    for identifier, expected_hash in frozen["rank_reference_sha256"].items():
        path = OUT / "cv_predictions" / f"{identifier}_score.npy"
        assert sha256(path) == expected_hash
        references[identifier] = np.load(available(path))
        assert len(references[identifier]) == int(people["train"].hours.sum())
    for identifier in frozen["final_candidate_ids"]:
        directory = OUT / "final" / identifier
        candidate = candidates[identifier]
        result = read_json(directory / "results.json")
        model_file = directory / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt")
        assert sha256(model_file) == frozen["model_sha256"][identifier] == result["model_sha256"]
        assert sha256(directory / "validation_score.npy") == result["score_sha256"]
        fold_rounds = [read_json(OUT / "cv" / f"fold{fold}" / identifier / "results.json")["selected_rounds"]
                       for fold in range(3)]
        assert result["selected_rounds"] == rounds[identifier] == int(np.median(fold_rounds))
        assert result["fold"] == -1 and result["fit"]["patients"] == len(people["train"])
        assert result["inner_calibration"] is None and result["calibration"] is None
        assert not result["test_read"] and not result["validation_used_for_fitting"]
        assert result["feature_count"] == len(protocol["feature_names"][candidate["feature_set"]])
        if candidate["target"] == "utility_classification":
            train_y = np.load(available(BASE / "train_labels.npy"), mmap_mode="r")
            train_delta = independent_delta(train_y, people["train"])
            mean = np.abs(train_delta/180.).mean()
            close(mean, result["weight_mean_before_normalization"])
            close(1./mean, result["weight_scale"])
    for identifier, expected_hash in frozen["threshold_sha256"].items():
        assert sha256(OUT / "final" / identifier / "threshold.json") == expected_hash
    checks = []
    for split in ("validation", "test"):
        patients = pd.read_csv(available(BASE / f"{split}_patients.csv"))
        labels = np.load(available(BASE / f"{split}_labels.npy"), mmap_mode="r")
        check_map(patients, labels, raw)
        people[split] = patients
        verify_extra(split, patients, labels, raw, transform, engineer)
        directories = sorted((OUT / "evaluation" / split).glob("*/metrics.json"))
        expected_ids = set(frozen["final_candidate_ids"]) | set(definitions)
        assert {path.parent.name for path in directories} == expected_ids
        component_scores = {identifier: np.load(available(OUT / "evaluation" / split / identifier / "score.npy"), mmap_mode="r")
                            for identifier in frozen["final_candidate_ids"]}
        for metric_path in directories:
            directory = metric_path.parent
            saved = read_json(metric_path)
            scores = np.load(available(directory / "score.npy"), mmap_mode="r")
            threshold = float(saved["threshold"])
            close(threshold, read_json(OUT / "final" / directory.name / "threshold.json")["threshold"])
            if directory.name in candidates:
                final = OUT / "final" / directory.name
                candidate = candidates[directory.name]
                result = read_json(final / "results.json")
                prediction_replay(candidate, final, result["selected_rounds"], patients, scores, raw, transform, engineer)
                if split == "validation":
                    np.testing.assert_array_equal(scores, np.load(available(final / "validation_score.npy")))
            else:
                np.testing.assert_array_equal(scores, independent_blend(definitions[directory.name], component_scores, references))
            for key, value in ranking(labels, scores).items():
                close(value, saved[key])
            contributions = pd.read_csv(available(directory / "patient_utilities.csv"))
            actual = normalized_official(function, labels, scores, patients, threshold, contributions)
            close(actual, saved["utility_score"])
            if split == "validation":
                delta = independent_delta(labels, patients)
                optimum, maximum = independent_threshold(scores, delta)
                close(threshold, optimum)
                close(actual, maximum)
            else:
                validation_saved = read_json(OUT / "evaluation/validation" / directory.name / "metrics.json")
                close(threshold, validation_saved["threshold"])
            checks.append({"split": split, "id": directory.name, "official_utility_patients": len(patients)})
    all_ids = np.concatenate([frame.patient_id.to_numpy() for frame in people.values()])
    assert len(all_ids) == len(set(all_ids)) == 39874
    assert read_json(OUT / "frozen_before_test.json") == frozen
    return checks


def main():
    # Native-model replay needs the same pinned XGBoost/OpenMP runtime as fits.
    # Bootstrap only at the executable entry point, never during imports.
    if os.environ.get("INFO5600_VERIFY_CHALLENGE_RUNTIME_READY") != "1":
        from sepsis_warning.friend_563 import runtime_environment
        environment = runtime_environment()
        environment["INFO5600_VERIFY_CHALLENGE_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", "-u", __file__, *sys.argv[1:]], environment)
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development-only", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = parser.parse_args()
    protocol = read_json(OUT / "protocol.json")
    assert protocol == read_json(REPORT / "protocol.json")
    raw_case = read_json(BASE / "raw_case.json")
    assert raw_case["all_original_PSV_sha256_verified"]
    assert sha256(BASE / "raw_float64.npy") == raw_case["raw_sha256"]
    raw = np.load(available(BASE / "raw_float64.npy"), mmap_mode="r")
    labels = np.load(available(BASE / "train_labels.npy"), mmap_mode="r")
    patients = pd.read_csv(available(BASE / "train_patients.csv"))
    check_map(patients, labels, raw)
    delta = independent_delta(labels, patients)
    close(np.load(available(DATA / "train_delta.npy")), delta/180.)
    function = scorer()
    # Independently anchor marginal targets to direct official flips on real patients.
    flip_checks = 0
    for patient in patients.sample(120, random_state=2026).itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        inactive = np.zeros(b-a, np.int8)
        reference = function(labels[a:b], inactive)
        for t in np.unique(np.linspace(0, b-a-1, min(5, b-a), dtype=int)):
            action = inactive.copy()
            action[t] = 1
            close(function(labels[a:b], action)-reference, delta[a+t]/180., atol=1e-10)
            flip_checks += 1
    transform, engineer = feature_functions(protocol)
    prefix_checks = verify_extra("train", patients, labels, raw, transform, engineer)
    roles = verify_roles(protocol, patients)
    candidates_path = OUT / "candidates.json"
    candidates = {entry["id"]: entry for entry in read_json(candidates_path)} if candidates_path.exists() else {}
    cv_checks = verify_cv(protocol, candidates, labels, delta, roles, raw, transform, engineer, function)
    cv_summaries = verify_cv_summaries(labels, delta, roles)
    ensemble_checks = verify_ensemble_cv(labels, delta, roles, function)
    final_checks = verify_final(protocol, candidates, {"train": patients}, raw, transform, engineer, function) if args.final else []
    result = {"passed": True, "mode": "final" if args.final else "development_only",
              "verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "train_patients": len(patients), "train_hours": len(labels),
              "official_real_patient_marginal_checks": flip_checks,
              "training_feature_prefix_patient_ids": prefix_checks,
              "roles_sha256": sha256(DATA / "cv_patient_roles.csv"),
              "cv_checks": cv_checks, "final_checks": final_checks,
              "cv_summary_candidates": cv_summaries, "ensemble_cv_checks": ensemble_checks,
              "test_data_read": bool(args.final),
              "interpretation": "Independent implementation and artifact consistency checks; existing test remains an historically exposed exploratory benchmark."}
    if args.final:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / "independent_verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
