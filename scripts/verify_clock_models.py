"""Independently verify clock ablations and matched LR/EBM/tree comparisons.

The verifier never fits a predictive model or changes caches/results.  Final
verification writes one small report after all checks succeed.  Native replay
uses features rebuilt from the original verified float64 records, while the
official scorer and threshold search are independent of the new fitting code.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = ROOT / "processed_data/friend_563_comparison"
PREVIOUS_DATA = ROOT / "processed_data/challenge_improvement"
PREVIOUS_OUT = ROOT / "outputs/challenge_improvement"
OUT = ROOT / "outputs/clock_models"
REPORT = ROOT / "reports/clock_models"
HISTORICAL = {"lgb_789": "lgb_bce_lag_observed_leaf63", "xgb_789": "xgb_lag_observed_depth5"}


def available(path):
    path = Path(path)
    assert path.is_file() and path.stat().st_size, path
    assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), f"Cloud-evicted input: {path}"
    return path


def read_json(path):
    return json.loads(available(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with available(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual, expected, atol=1e-12, rtol=0):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def load_reference_verifier():
    path = available(ROOT / "scripts/verify_challenge_improvement.py")
    cache = Path(importlib.util.cache_from_source(str(path)))
    if cache.exists():
        available(cache)
    spec = importlib.util.spec_from_file_location("independent_clock_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def indices(patients):
    return np.concatenate([np.arange(int(p.start), int(p.stop)) for p in patients.itertuples(index=False)])


def local(patients):
    result = patients.copy().reset_index(drop=True)
    result["stop"] = result.hours.cumsum()
    result["start"] = result.stop-result.hours
    return result


def role_record(patients):
    return {"patients": len(patients), "event_patients": int(patients.has_event.sum()),
            "hours": int(patients.hours.sum()), "patient_ids_sha256": digest(patients.patient_id.tolist())}


def rankings(labels, scores):
    assert np.isfinite(scores).all() and len(labels) == len(scores)
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def verify_feature_views(protocol):
    old = read_json(PREVIOUS_OUT / "protocol.json")
    names = old["feature_names"]["lag_observed"]
    assert len(names) == len(set(names)) == 789
    clocks = [name for name in names if "ICULOS" in name or "HospAdmTime" in name or name == "sep_prefix_length"]
    assert len(clocks) == 55
    expected = {"baseline_789": [],
                "raw_clocks_only_736": [name for name in clocks if name not in {"ICULOS", "HospAdmTime"}],
                "explicit_clock_free_734": clocks}
    widths = {"baseline_789": 789, "raw_clocks_only_736": 736, "explicit_clock_free_734": 734}
    assert protocol["primary_feature_view"] == "baseline_789"
    assert protocol["primary_models"] == ["lr", "ebm", "lightgbm"]
    assert protocol["supplementary_models"] == ["xgboost"] and not protocol["ensemble_enabled"]
    for view, dropped in expected.items():
        record = protocol["feature_views"][view]
        kept = [i for i,name in enumerate(names) if name not in set(dropped)]
        assert record["kept_column_indices"] == kept
        assert record["dropped_features"] == dropped
        assert record["feature_count"] == widths[view] == len(kept)
        assert protocol["feature_names"][view] == [names[i] for i in kept]
        assert all(i in kept for i in range(563,789)), "Measurement age/count/history additions were removed"
    previous_candidates = {entry["id"]: entry for entry in read_json(PREVIOUS_OUT / "candidates.json")}
    lgb_parameters = previous_candidates[HISTORICAL["lgb_789"]]["parameters"]
    for identifier in ("lgb_789", "lgb_736", "lgb_734"):
        candidate = protocol["candidates"][identifier]
        assert candidate["parameters"] == lgb_parameters
        assert candidate["target"] == "binary" and candidate["selection_metric"] == "auroc"
    assert protocol["candidates"]["xgb_789"]["parameters"] == previous_candidates[HISTORICAL["xgb_789"]]["parameters"]
    for candidate in protocol["candidates"].values():
        if candidate["model"] in {"lr", "ebm", "xgboost"}:
            assert candidate["feature_view"] == "baseline_789"
        if candidate["model"] == "ebm":
            settings = candidate["parameters"]
            assert settings["interactions"] == settings["validation_size"] == settings["inner_bags"] == 0
            assert settings["outer_bags"] == 1 and settings["early_stopping_rounds"] == 0
    return old, names


def verify_historical_sources():
    frozen = read_json(PREVIOUS_OUT / "frozen_before_test.json")
    for name, expected in frozen["source_sha256"].items():
        assert sha256(ROOT / name) == expected
    assert read_json(ROOT / "reports/challenge_improvement/independent_verification.json")["passed"]
    return frozen


def verify_reuse(protocol, study):
    manifest = read_json(OUT / "reuse_manifest.json")
    assert set(manifest["models"]) == set(HISTORICAL)
    checked = []
    for alias, historical in HISTORICAL.items():
        entry = manifest["models"][alias]
        assert entry["historical_candidate"] == historical
        for fold in (0,1,2,-1):
            record = entry["final"] if fold == -1 else entry["cv"][str(fold)]
            expected = PREVIOUS_OUT / "final" / historical if fold == -1 else PREVIOUS_OUT / "cv" / f"fold{fold}" / historical
            directory = ROOT / record["directory"]
            assert directory == expected == study.resolve_directory(alias, fold)
            for name, expected_hash in record["files_sha256"].items():
                assert sha256(directory/name) == expected_hash
            target = OUT / "final" / alias if fold == -1 else OUT / "cv" / f"fold{fold}" / alias
            assert read_json(target / "reuse.json") == {"historical_candidate": historical, **record}
            saved = read_json(directory / "results.json")
            assert saved["candidate"] == historical and saved["fold"] == fold
            assert saved["feature_count"] == 789
            checked.append({"alias": alias, "fold": fold, "historical_candidate": historical})
    return checked


def independent_features(raw40, view, protocol, transform, engineer):
    full = np.column_stack([transform(raw40, "augmented"), engineer(raw40, ("lag", "observed"))])
    assert full.shape[1] == 789
    return full[:, protocol["feature_views"][view]["kept_column_indices"]]


def check_feature_replays(protocol, study, patients, raw, transform, engineer):
    checked = []
    for _, group in patients.groupby(["hospital", "has_event"]):
        for _, person in group.iloc[[0, len(group)//2]].iterrows():
            original = raw[int(person.raw_start):int(person.raw_stop), :40]
            global_index = np.arange(int(person.start), int(person.stop))
            for view in protocol["feature_views"]:
                expected = independent_features(original, view, protocol, transform, engineer)
                actual = study.load_features("train", view, global_index)
                np.testing.assert_array_equal(actual, expected)
                for cutoff in sorted({1, min(6,len(original)), min(12,len(original)), len(original)}):
                    np.testing.assert_array_equal(independent_features(original[:cutoff], view, protocol, transform, engineer), expected[:cutoff])
            checked.append(person.patient_id)
    return checked


def load_predictor(candidate, directory, rounds):
    model_name = candidate["model"]
    if model_name == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(available(directory / "model.txt")))
        assert model.current_iteration() == rounds
        return model, None
    if model_name == "xgboost":
        import xgboost as xgb
        model = xgb.Booster(model_file=available(directory / "model.ubj"))
        model.set_param({"nthread": 2})
        assert model.num_boosted_rounds() == rounds
        return model, None
    import joblib
    bundle = joblib.load(available(directory / "model.joblib"))
    assert bundle["model_name"] == model_name
    assert bundle["parameters"] == candidate["parameters"]
    model = bundle["model"]
    for name, value in candidate["parameters"].items():
        assert model.get_params()[name] == value, (name, model.get_params()[name], value)
    assert model.n_features_in_ == len(bundle["feature_names"])
    if model_name == "ebm":
        assert bundle["imputer"] is None and bundle["scaler"] is None
        assert model.interactions == model.validation_size == model.inner_bags == 0
        assert model.outer_bags == 1
        assert all(len(term) == 1 for term in model.term_features_)
        assert len(model.term_features_) == len(bundle["feature_names"])
    return model, bundle


def predict_independent(candidate, model, bundle, features, rounds):
    if candidate["model"] == "lightgbm":
        return model.predict(features, num_iteration=rounds, num_threads=2)
    if candidate["model"] == "xgboost":
        return model.inplace_predict(features, iteration_range=(0, rounds))
    if candidate["model"] == "lr":
        features = features.copy()
        for column, value in enumerate(bundle["imputer"]["statistics_"]):
            missing = np.isnan(features[:,column])
            features[missing,column] = value
        bundle["scaler"].transform(features, copy=False)
    return model.predict_proba(features)[:,1]


def prediction_replay(candidate, directory, rounds, patients, scores, raw, protocol, transform, engineer):
    model, bundle = load_predictor(candidate, directory, rounds)
    view = candidate["feature_view"]
    if bundle is not None:
        assert bundle["feature_names"] == protocol["feature_names"][view]
    positions = np.unique(np.linspace(0, len(scores)-1, 24, dtype=int))
    owners = np.searchsorted(patients.stop.to_numpy(), positions, side="right")
    for owner in np.unique(owners):
        person = patients.iloc[int(owner)]
        raw40 = raw[int(person.raw_start):int(person.raw_stop), :40]
        selected = positions[owners == owner]
        features = independent_features(raw40, view, protocol, transform, engineer)[selected-int(person.start)]
        actual = predict_independent(candidate, model, bundle, features, rounds)
        close(actual, scores[selected], atol=1e-10 if candidate["model"] == "lr" else 1e-12)
    del model, bundle
    gc.collect()
    return len(positions)


def verify_lr_statistics(candidate, directory, fit_patients, protocol, study):
    import joblib
    bundle = joblib.load(available(directory / "model.joblib"))
    assert bundle["model_name"] == "lr"
    assert bundle["fit_patient_ids_sha256"] == digest(fit_patients.patient_id.tolist())
    imputer, scaler = bundle["imputer"], bundle["scaler"]
    assert imputer["strategy"] == "median" and imputer["keep_empty_features"] is True
    matrix = study.load_features("train", candidate["feature_view"], indices(fit_patients))
    assert len(matrix) == int(fit_patients.hours.sum())
    empty = np.zeros(matrix.shape[1], bool)
    medians = np.empty(matrix.shape[1], np.float64)
    for column in range(matrix.shape[1]):
        observed = matrix[:,column][~np.isnan(matrix[:,column])]
        empty[column] = not len(observed)
        medians[column] = float(np.median(observed)) if len(observed) else 0.
    close(medians, imputer["statistics_"], atol=1e-7, rtol=1e-7)
    np.testing.assert_array_equal(empty, imputer["all_missing_features"])
    # Independent batch moments after median filling (Chan's merge formula).
    count = 0
    mean = np.zeros(matrix.shape[1], np.float64)
    m2 = np.zeros_like(mean)
    for start in range(0, len(matrix), 10000):
        block = matrix[start:start+10000]
        for column,value in enumerate(medians):
            missing = np.isnan(block[:,column])
            block[missing,column] = value
        batch_mean = block.mean(axis=0, dtype=np.float64)
        batch_m2 = block.var(axis=0, dtype=np.float64)*len(block)
        total = count+len(block)
        difference = batch_mean-mean
        m2 += batch_m2 + difference*difference*count*len(block)/total
        mean += difference*len(block)/total
        count = total
    variance = m2/count
    assert int(scaler.n_samples_seen_) == count
    close(scaler.mean_, mean, atol=1e-7, rtol=1e-10)
    close(scaler.var_, variance, atol=1e-7, rtol=1e-7)
    assert np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all()
    epsilon = np.finfo(np.float64).eps
    constant = variance <= count*epsilon*variance + (count*mean*epsilon)**2
    expected_scale = np.sqrt(variance)
    expected_scale[constant] = 1.
    close(scaler.scale_, expected_scale, atol=1e-7, rtol=1e-7)
    metadata = read_json(directory/"preprocessor.json")
    assert metadata["fit_rows"] == metadata["scaler_samples_seen"] == count
    arrays = {"imputer_statistics": imputer["statistics_"],
              "all_missing_features": imputer["all_missing_features"],
              "scaler_mean": scaler.mean_, "scaler_variance": scaler.var_,
              "scaler_scale": scaler.scale_}
    for name, array in arrays.items():
        np.testing.assert_array_equal(np.asarray(metadata[name]), array)
        assert metadata["array_sha256"][name] == hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    del matrix, bundle
    gc.collect()
    return {"fit_rows": count, "columns": len(medians), "all_missing_columns": int(empty.sum())}


def assert_role_metadata(saved, fit_patients, calibration_patients):
    assert saved["fit"] == role_record(fit_patients)
    assert saved["inner_calibration"] == (None if calibration_patients is None else role_record(calibration_patients))
    assert not saved["test_read"] and not saved["validation_used_for_fitting"]
    if calibration_patients is not None:
        assert not saved["outer_holdout_used_for_selection"]


def verify_signature(identifier, candidate, directory, fold, protocol):
    if identifier in HISTORICAL:
        return
    saved = read_json(directory/"results.json")
    record = {"candidate": dict(candidate),
              "feature_names": protocol["feature_names"][candidate["feature_view"]],
              "protocol_sha256": sha256(OUT/"protocol.json"),
              "shared_module_sha256": sha256(ROOT/"src/sepsis_warning/clock_models.py"),
              "fold": fold}
    if candidate["model"] in {"lr", "ebm"}:
        record.update(worker_sha256=sha256(ROOT/"scripts/fit_interpretable_clock.py"),
                      implementation_sha256=sha256(ROOT/"src/sepsis_warning/clock_interpretable.py"))
    else:
        record["candidate"].setdefault("auc_patience_rounds", protocol["auc_early_stopping_rounds"])
        record["candidate"].setdefault("min_delta", protocol["min_delta"])
        record.update(worker_sha256=sha256(ROOT/"scripts/fit_clock_models.py"),
                      native_helper_sha256=sha256(ROOT/"scripts/fit_challenge_improvement.py"),
                      rounds=read_json(OUT/"rounds.json")[identifier] if fold == -1 else None)
    assert digest(record) == saved["signature"], identifier
    if candidate["model"] in {"lr", "ebm"}:
        assert sha256(directory/"preprocessor.json") == saved["preprocessor_sha256"]
        import joblib
        bundle = joblib.load(available(directory/"model.joblib"))
        assert bundle["fit_patient_ids_sha256"] == saved["fit"]["patient_ids_sha256"]
        if candidate["model"] == "ebm":
            meta = read_json(directory/"preprocessor.json")
            assert meta["type"] == "native_NaN" and meta["fit_rows"] == saved["fit"]["hours"]
            assert meta["imputation"] is meta["standardization"] is None
            assert not meta["native_random_hour_validation"]


def verify_cv(protocol, study, reference, roles, labels, delta, raw, transform, engineer, official):
    checks = []
    for identifier,candidate in protocol["candidates"].items():
        for fold in range(3):
            directory = study.resolve_directory(identifier, fold)
            path = directory / "results.json"
            if not path.exists():
                continue
            saved = read_json(path)
            expected_id = HISTORICAL.get(identifier, identifier)
            assert saved["candidate"] == expected_id and saved["model"] == candidate["model"] and saved["fold"] == fold
            assert saved["feature_count"] == len(protocol["feature_names"][candidate["feature_view"]])
            assert saved["parameters"] == candidate["parameters"]
            verify_signature(identifier, candidate, directory, fold, protocol)
            parts = roles[fold]
            assert_role_metadata(saved, parts["fit"], parts["calibration"])
            for name, key in (("calibration", "cal_patients.csv"), ("held", "held_patients.csv")):
                pd.testing.assert_frame_equal(pd.read_csv(available(directory/key)), local(parts[name]), check_dtype=False)
            cal_index, held_index = indices(parts["calibration"]), indices(parts["held"])
            cal_scores = np.load(available(directory / "cal_score.npy"))
            held_scores = np.load(available(directory / "held_score.npy"))
            threshold, maximum = reference.independent_threshold(cal_scores, delta[cal_index])
            assert float(threshold) == float(saved["calibration"]["threshold"]), "Exact score group / highest-threshold tie rule differs"
            np.testing.assert_array_equal(cal_scores >= threshold, cal_scores >= saved["calibration"]["threshold"])
            close(maximum, saved["calibration"]["max_validation_utility"])
            for key,value in rankings(labels[cal_index], cal_scores).items():
                if key in saved["calibration"]:
                    close(value, saved["calibration"][key])
                else:
                    assert identifier in HISTORICAL
            for key,value in rankings(labels[held_index], held_scores).items():
                close(value, saved["evaluation"][key])
            utility = reference.normalized_official(official, labels[held_index], held_scores, local(parts["held"]), threshold)
            close(utility, saved["evaluation"]["utility_score"])
            assert sha256(directory / "held_score.npy") == saved["score_sha256"]
            for name in ("cal_score.npy", "cal_patients.csv", "held_patients.csv"):
                key = name.removesuffix(".npy").removesuffix(".csv") + "_sha256"
                if key in saved:
                    assert sha256(directory/name) == saved[key]
                else:
                    assert identifier in HISTORICAL, "Only old, manifest-hashed artifacts may lack per-result hashes"
            assert saved["evaluation_patients"] == len(parts["held"])
            assert saved["evaluation_hours"] == len(held_index)
            suffix = "model.txt" if candidate["model"] == "lightgbm" else "model.ubj" if candidate["model"] == "xgboost" else "model.joblib"
            assert sha256(directory/suffix) == saved["model_sha256"]
            rounds = int(saved["selected_rounds"])
            if candidate["model"] == "ebm":
                assert rounds == candidate["parameters"]["max_rounds"]
            stats = verify_lr_statistics(candidate, directory, parts["fit"], protocol, study) if candidate["model"] == "lr" else None
            replay = prediction_replay(candidate, directory, rounds, local(parts["held"]), held_scores, raw, protocol, transform, engineer)
            checks.append({"candidate": identifier, "fold": fold, "source_reused": identifier in HISTORICAL,
                           "official_patients": len(parts["held"]), "native_prediction_replays": replay,
                           "lr_fit_only_statistics": stats})
            print(f"Verified CV {identifier}/fold{fold}", flush=True)
    return checks


def verify_summaries(protocol, study, reference, roles, labels, delta):
    path = REPORT / "cv_metrics.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(available(path))
    checked = []
    for row in frame.itertuples(index=False):
        scores = np.full(len(labels), np.nan)
        alarms = np.zeros(len(labels), bool)
        rounds, aucs, aps = [], [], []
        for fold in range(3):
            directory = study.resolve_directory(row.candidate, fold)
            saved = read_json(directory / "results.json")
            selected = indices(roles[fold]["held"])
            scores[selected] = np.load(available(directory / "held_score.npy"))
            alarms[selected] = scores[selected] >= saved["calibration"]["threshold"]
            rounds.append(saved["selected_rounds"])
            aucs.append(saved["evaluation"]["auroc"])
            aps.append(saved["evaluation"]["auprc"])
        assert np.isfinite(scores).all()
        np.testing.assert_array_equal(scores, np.load(available(OUT/"cv_predictions"/f"{row.candidate}_score.npy")))
        np.testing.assert_array_equal(alarms, np.load(available(OUT/"cv_predictions"/f"{row.candidate}_alarm.npy")))
        for key,value in rankings(labels, scores).items():
            close(value, getattr(row,key))
        close(np.mean(aucs), row.mean_fold_auroc)
        close(np.mean(aps), row.mean_fold_auprc)
        close(delta[alarms].sum()/np.maximum(delta,0).sum(), row.utility_score)
        if hasattr(row, "median_cv_iterations"):
            assert int(np.median(rounds)) == row.median_cv_iterations
        if protocol["candidates"][row.candidate]["model"] in {"lightgbm", "xgboost"}:
            assert int(np.median(rounds)) == row.final_rounds
        elif protocol["candidates"][row.candidate]["model"] == "lr":
            assert row.final_rounds == protocol["candidates"][row.candidate]["parameters"]["max_iter"]
        else:
            assert row.final_rounds == protocol["candidates"][row.candidate]["parameters"]["max_rounds"]
        checked.append(row.candidate)
    return checked


def verify_model_selection(protocol):
    path = OUT/"model_selection.json"
    if not path.exists():
        return None
    frame = pd.read_csv(available(REPORT/"cv_metrics.csv"))
    assert set(frame.candidate) == set(protocol["candidates"])
    ebms = frame.loc[frame.model.eq("ebm")].sort_values(
        ["mean_fold_auroc", "final_rounds"], ascending=[False, True])
    assert len(ebms) == 2
    selected = ebms.iloc[0].candidate
    choice = read_json(path)
    assert choice["selected_ebm"] == selected
    assert choice["primary_feature_view"] == "baseline_789"
    assert choice["primary_candidates"] == ["lr_789", selected, "lgb_789"]
    assert choice["supplementary_candidates"] == ["xgb_789"]
    assert choice["ablation_candidates"] == ["lgb_789", "lgb_736", "lgb_734"]
    assert not choice["ensemble_enabled"]
    return choice


def verify_final(protocol, study, reference, roles, train_labels, people, raw, transform, engineer, official):
    frozen = read_json(OUT / "frozen_before_test.json")
    assert not frozen["test_used_for_selection"]
    selection = read_json(OUT/"model_selection.json")
    expected_identifiers = set(selection["primary_candidates"] + selection["supplementary_candidates"] + selection["ablation_candidates"])
    assert len(expected_identifiers) == 6 and set(frozen["model_sha256"]) == expected_identifiers
    assert set(frozen["final_candidate_ids"]) == expected_identifiers
    assert not frozen["ensemble_enabled"] and frozen["historical_test_exposure"]
    for name,key in (("rounds.json","rounds_sha256"), ("final_selection.json","selection_sha256"),
                     ("model_selection.json","model_selection_sha256"), ("protocol.json","protocol_sha256"),
                     ("reuse_manifest.json","reuse_manifest_sha256")):
        assert sha256(OUT/name) == frozen[key]
        assert read_json(OUT/name) == read_json(REPORT/name)
    if "lr_execution_environment_sha256" in frozen:
        runtime_path = OUT/"final/lr_789/execution_environment.json"
        assert sha256(runtime_path) == frozen["lr_execution_environment_sha256"]
        runtime = read_json(runtime_path)
        assert runtime == read_json(REPORT/"lr_execution_environment.json")
        assert runtime["JOBLIB_MULTIPROCESSING"] == "0" and runtime["VECLIB_MAXIMUM_THREADS"] == "4"
        assert not runtime["estimator_parameters_changed"]
        assert not runtime["historical_cv_retrained_for_runtime_change"]
    if "scheduling_events_sha256" in frozen:
        assert sha256(OUT/"scheduling_events.json") == frozen["scheduling_events_sha256"]
        events = read_json(OUT/"scheduling_events.json")
        assert events == read_json(REPORT/"scheduling_events.json")
        for event in events:
            assert not event["test_read"]
            assert not event.get("model_parameters_changed", False)
            assert event.get("selection_rule_unchanged", True)
    for name,expected in read_json(REPORT/"test_input_sha256.json").items():
        assert sha256(ROOT/name) == expected
    for name,expected in frozen.get("source_sha256", {}).items():
        assert sha256(ROOT/name) == expected
    for name,expected in frozen.get("input_sha256", {}).items():
        assert sha256(ROOT/name) == expected
    rounds = read_json(OUT / "rounds.json")
    for identifier,expected in frozen["model_sha256"].items():
        candidate = protocol["candidates"][identifier]
        directory = study.resolve_directory(identifier,-1)
        saved = read_json(directory / "results.json")
        suffix = "model.txt" if candidate["model"] == "lightgbm" else "model.ubj" if candidate["model"] == "xgboost" else "model.joblib"
        assert directory/suffix == ROOT/frozen["model_paths"][identifier]
        assert sha256(directory/suffix) == expected == saved["model_sha256"]
        assert_role_metadata(saved, people["train"], None)
        verify_signature(identifier, candidate, directory, -1, protocol)
        assert saved["fold"] == -1 and saved["feature_count"] == len(protocol["feature_names"][candidate["feature_view"]])
        if candidate["model"] in {"lightgbm", "xgboost"}:
            selected = [read_json(study.resolve_directory(identifier,fold)/"results.json")["selected_rounds"] for fold in range(3)]
            assert saved["selected_rounds"] == rounds[identifier] == int(np.median(selected))
        if candidate["model"] == "lr":
            verify_lr_statistics(candidate, directory, people["train"], protocol, study)
        elif candidate["model"] == "ebm":
            assert saved["selected_rounds"] == candidate["parameters"]["max_rounds"]
    checks = []
    metric_paths = list((OUT / "evaluation/validation").glob("*/metrics.json"))
    identifiers = {path.parent.name for path in metric_paths}
    assert identifiers == set(frozen["model_sha256"])
    assert not any("blend" in name or "ensemble" in name for name in identifiers)
    for split in ("validation", "test"):
        assert {path.parent.name for path in (OUT/"evaluation"/split).glob("*/metrics.json")} == expected_identifiers
        patients = pd.read_csv(available(BASE/f"{split}_patients.csv"))
        labels = np.load(available(BASE/f"{split}_labels.npy"), mmap_mode="r")
        reference.check_map(patients, labels, raw)
        people[split] = patients
        delta = reference.independent_delta(labels,patients)
        for identifier in sorted(identifiers):
            directory = OUT / "evaluation" / split / identifier
            saved = read_json(directory / "metrics.json")
            scores = np.load(available(directory / "score.npy"), mmap_mode="r")
            if "score_sha256" in saved:
                assert sha256(directory/"score.npy") == saved["score_sha256"]
            candidate = protocol["candidates"][identifier]
            model_dir = study.resolve_directory(identifier,-1)
            final = read_json(model_dir / "results.json")
            threshold_path = OUT / "final" / identifier / "threshold.json"
            assert sha256(threshold_path) == frozen["threshold_sha256"][identifier]
            threshold = read_json(threshold_path)["threshold"]
            assert float(saved["threshold"]) == float(threshold)
            for key,value in rankings(labels,scores).items():
                close(value,saved[key])
            contributions = pd.read_csv(available(directory / "patient_utilities.csv"))
            utility = reference.normalized_official(official, labels, scores, patients, threshold, contributions)
            close(utility,saved["utility_score"])
            if split == "validation":
                optimum,maximum = reference.independent_threshold(scores,delta)
                assert float(optimum) == float(threshold)
                close(maximum,utility)
                np.testing.assert_array_equal(scores,np.load(available(model_dir/"validation_score.npy")))
            else:
                close(threshold,read_json(OUT/"evaluation/validation"/identifier/"metrics.json")["threshold"])
            replay = prediction_replay(candidate,model_dir,int(final["selected_rounds"]),patients,scores,raw,protocol,transform,engineer)
            checks.append({"split":split,"candidate":identifier,"official_patients":len(patients),"native_prediction_replays":replay})
            print(f"Verified {split} {identifier}",flush=True)
    identifiers = np.concatenate([part.patient_id.to_numpy() for part in people.values()])
    assert len(identifiers) == len(set(identifiers)) == 39874
    assert read_json(OUT/"frozen_before_test.json") == frozen
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--development-only", action="store_true")
    modes.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if os.environ.get("INFO5600_VERIFY_CLOCK_RUNTIME_READY") != "1":
        from sepsis_warning.friend_563 import runtime_environment
        environment = runtime_environment()
        environment["INFO5600_VERIFY_CLOCK_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable,[sys.executable,"-B","-u",__file__,*sys.argv[1:]],environment)
    import sepsis_warning.clock_models as study
    reference = load_reference_verifier()
    protocol = read_json(OUT/"protocol.json")
    assert protocol == read_json(REPORT/"protocol.json")
    old_protocol,names = verify_feature_views(protocol)
    verify_historical_sources()
    reused = verify_reuse(protocol,study)
    raw_case = read_json(BASE/"raw_case.json")
    assert raw_case["all_original_PSV_sha256_verified"]
    assert sha256(BASE/"raw_float64.npy") == raw_case["raw_sha256"]
    raw = np.load(available(BASE/"raw_float64.npy"),mmap_mode="r")
    labels = np.load(available(BASE/"train_labels.npy"),mmap_mode="r")
    patients = pd.read_csv(available(BASE/"train_patients.csv"))
    reference.check_map(patients,labels,raw)
    delta = reference.independent_delta(labels,patients)
    close(delta/180.,np.load(available(PREVIOUS_DATA/"train_delta.npy")))
    transform,engineer = reference.feature_functions(old_protocol)
    prefix_checks = check_feature_replays(protocol,study,patients,raw,transform,engineer)
    roles = reference.verify_roles(old_protocol,patients)
    for fold in range(3):
        for actual,key in zip(study.cv_roles(fold),("fit","calibration","held")):
            pd.testing.assert_frame_equal(actual,roles[fold][key])
    official = reference.scorer()
    cv_checks = verify_cv(protocol,study,reference,roles,labels,delta,raw,transform,engineer,official)
    summary_checks = verify_summaries(protocol,study,reference,roles,labels,delta)
    selection_check = verify_model_selection(protocol)
    if args.final:
        assert selection_check is not None
        completion = read_json(OUT/"completed.json")
        assert completion["new_CV_fits"] == 15 and completion["reused_CV_fits"] == 6
        assert completion["new_final_fits"] == 4 and completion["reused_final_fits"] == 2
        assert completion["evaluations"] == 12 and not completion["ensemble_enabled"]
        assert not completion["test_used_for_selection"]
        if "completed_new_final_candidates" in completion:
            expected_new = set(selection_check["primary_candidates"] + selection_check["ablation_candidates"]) - set(HISTORICAL)
            assert len(expected_new) == 4
            completed_new = set(completion["completed_new_final_candidates"])
            assert expected_new <= completed_new
            unused_ebms = {key for key,value in protocol["candidates"].items()
                           if value["model"] == "ebm" and key != selection_check["selected_ebm"]}
            assert completed_new-expected_new <= unused_ebms
        assert len(cv_checks) == 3*len(protocol["candidates"]), "Every declared training CV fit must be complete"
    final_checks = verify_final(protocol,study,reference,roles,labels,{"train":patients},raw,transform,engineer,official) if args.final else []
    result = {"passed":True,"mode":"final" if args.final else "development_only",
              "verified_at_utc":datetime.now(timezone.utc).isoformat(),"ensemble_enabled":False,
              "patient_roles_reconstructed":True,"feature_view_counts":{key:len(value) for key,value in protocol["feature_names"].items()},
              "raw_feature_prefix_patient_ids":prefix_checks,"reused_artifact_checks":reused,
              "cv_checks":cv_checks,"cv_summary_candidates":summary_checks,"model_selection":selection_check,"final_checks":final_checks,
              "test_data_read":bool(args.final),"verifier_sha256":sha256(__file__),
              "reference_verifier_sha256":sha256(ROOT/"scripts/verify_challenge_improvement.py")}
    if args.final:
        REPORT.mkdir(parents=True,exist_ok=True)
        (REPORT/"independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
