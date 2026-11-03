"""Matched full-record XGBoost/LightGBM experiment on the published main features."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.data import ROOT
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.ab_regularization import runtime_environment as old_runtime
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.utility import patient_utility, normalized_utility, official_scorer

DATA = ROOT / "processed_data/friend_563_comparison"
OUT = ROOT / "outputs/friend_563_comparison"
REPORT = ROOT / "reports/friend_563_comparison"
MAIN = "32ee355b9373e81a9d3bdb1f1b4a22883e04678e"
MODELS = ["xgboost", "lightgbm"]
EXPECTED = {"train": (27911, 1742, 1081518, 16952),
            "validation": (5981, 374, 233738, 3630), "test": (5982, 374, 230710, 3630)}


def _feature_process_init():
    global _process_raw, _process_transform
    sys.path.insert(0, str(DATA / "source"))
    _process_transform = importlib.import_module("sepsis_pipeline.features").transform
    _process_raw = np.load(DATA / "raw_float64.npy", mmap_mode="r")


def _feature_process_one(bounds):
    a, b = bounds
    return _process_transform(_process_raw[a:b, :40], "augmented")


def runtime_environment():
    env = old_runtime()
    env["PYTHONPATH"] = "/private/tmp/info5600_xgb_runtime" + os.pathsep + env.get("PYTHONPATH", "")
    return env


def friend_modules():
    directory = DATA / "source"
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", MAIN, "sepsis_pipeline"], cwd=ROOT, text=True).splitlines()
    hashes = {}
    for name in files:
        if not name.endswith(".py"):
            continue
        blob = subprocess.check_output(["git", "show", f"{MAIN}:{name}"], cwd=ROOT)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != blob:
            target.write_bytes(blob)
        hashes[name] = hashlib.sha256(blob).hexdigest()
    sys.path.insert(0, str(directory))
    return (importlib.import_module("sepsis_pipeline.preprocess"),
            importlib.import_module("sepsis_pipeline.features"), hashes)


def prepare():
    for directory in [DATA, OUT, REPORT]:
        directory.mkdir(parents=True, exist_ok=True)
    preprocessing, features, source_hashes = friend_modules()
    parameters = {
        "xgboost": {"objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
                    "device": "cpu", "max_depth": 4, "max_leaves": 16, "eta": .05,
                    "min_child_weight": 5, "lambda": 10., "alpha": 0., "max_bin": 255,
                    "subsample": 1., "colsample_bytree": 1., "scale_pos_weight": 1., "seed": 42, "nthread": 4},
        "lightgbm": {"objective": "binary", "metric": "auc", "learning_rate": .05,
                     "max_depth": 4, "num_leaves": 16, "min_data_in_leaf": 200,
                     "lambda_l2": 10., "lambda_l1": 0., "max_bin": 255,
                     "feature_fraction": 1., "bagging_fraction": 1., "bagging_freq": 0,
                     "seed": 42, "num_threads": 4, "deterministic": True,
                     "force_col_wise": True, "verbosity": -1}}
    design = {"main_commit": MAIN, "feature_profile": "augmented", "feature_names": features.FEATURE_NAMES["augmented"],
              "source_hashes": source_hashes, "parameters": parameters, "maximum_rounds": 1000,
              "early_stopping_rounds": 50, "min_delta": 1e-5,
              "standard_id": "SEPSIS_FULL_RECORD_V2", "split_seed": 42, "cv_seed": 2026,
              "scope": "All original hours of the established 39,874 included patients; original SepsisLabel",
              "class_weights": "None; no resampling for either algorithm",
              "preprocessing": "Exact published main 563 features, original float64 PSV input, float32 output, native NaN; no learned imputation/scaling/clipping",
              "selection": "One prespecified configuration each; training-only 3-fold AUC early stopping; final rounds = median of fold best rounds",
              "CV_interpretation": "Development scores: each CV holdout also selects its early-stopping iteration; not independent nested-CV estimates",
              "threshold_policy": "One threshold per model maximizes full-record utility on validation only; ties choose highest threshold",
              "test_policy": "Freeze final models and validation thresholds before building/scoring test features; no test selection",
              "limitations": "Friend training parameters are unpublished; this is a controlled baseline, not an exact private-model reproduction. Original source patients have been examined in earlier experiments.",
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    assert len(design["feature_names"]) == len(set(design["feature_names"])) == 563
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text())
        a, b = dict(old), dict(design)
        a.pop("created_at_utc"); b.pop("created_at_utc")
        assert a == b, "Protocol changed: use a new experiment directory"
        design = old
    write_json(protocol_path, design)
    if not (DATA / "raw_case.json").exists():
        blob = subprocess.check_output(["git", "show", f"{MAIN}:data/manifest.csv"], cwd=ROOT)
        manifest = pd.read_csv(io.BytesIO(blob)).sort_values("relative_path").reset_index(drop=True)
        assert len(manifest) == 40336 and int(manifest.hours.sum()) == 1552210
        raw = np.lib.format.open_memmap(DATA / "raw_float64.npy", mode="w+", dtype=np.float64, shape=(1552210, 41))
        git = subprocess.Popen(["git", "cat-file", "--batch"], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        records, people, seen, position = [], [], set(), 0
        from sepsis_pipeline.schema import HEADER
        for i, row in enumerate(manifest.itertuples(index=False), 1):
            git.stdin.write(f"{MAIN}:{row.relative_path}\n".encode()); git.stdin.flush()
            header = git.stdout.readline().decode().strip().split()
            assert len(header) == 3 and header[1] == "blob", header
            payload = git.stdout.read(int(header[2])); assert git.stdout.read(1) == b"\n"
            sha = hashlib.sha256(payload).hexdigest(); assert sha == row.sha256
            assert payload.splitlines()[0].decode().strip().split("|") == HEADER
            values = np.loadtxt(io.BytesIO(payload), delimiter="|", skiprows=1, ndmin=2)
            assert values.shape == (row.hours, 41)
            hospital = "A" if "/training_setA/" in row.relative_path else "B"
            identifier = hospital + "_" + Path(row.relative_path).stem
            reason, onset = preprocessing.classify_record(values)
            if sha in seen:
                reason = "duplicate_complete_record"
            seen.add(sha)
            stop = position + len(values); raw[position:stop] = values
            record = {"patient_id": identifier, "hospital": hospital, "raw_path": row.relative_path,
                      "raw_sha256": sha, "raw_start": position, "raw_stop": stop, "hours": len(values),
                      "first_hour": int(values[0, 39]), "has_event": int(values[:, 40].max()),
                      "positive_hours": int(values[:, 40].sum()), "included": reason == "included", "reason": reason}
            records.append(record)
            if reason == "included":
                people.append({**record, "subject_id": identifier, "event": record["has_event"], "record_hours": len(values)})
            position = stop
            if i % 5000 == 0:
                print(f"Verified original float64 PSV: {i:,}/40,336", flush=True)
        git.stdin.close(); assert git.wait() == 0 and position == len(raw)
        raw.flush(); del raw
        audit = pd.DataFrame(records); audit.to_csv(DATA / "raw_audit.csv", index=False)
        people = preprocessing.assign_splits(pd.DataFrame(people))
        assert len(people) == 39874 and people.subject_id.is_unique
        # Compare exact original patient and fold assignments, not just aggregate counts.
        old = pd.read_csv(ROOT / "processed_data/splits/patient_splits.csv")
        b = pd.read_csv(ROOT / "reports/ab_expansion/B_patient_splits.csv")
        b = b.rename(columns={"expansion_split": "split"}); b["split"] = b.split.str.removeprefix("B_")
        original = pd.concat([old.loc[old.hospital.eq("A"), ["patient_id", "split"]], b[["patient_id", "split"]]]).set_index("patient_id")
        new = people.set_index("subject_id")
        assert set(original.index) == set(new.index)
        np.testing.assert_array_equal(new.split, original.loc[new.index, "split"])
        folds = pd.read_csv(ROOT / "reports/ab_regularization/patient_cv_folds.csv").set_index("patient_id")
        np.testing.assert_array_equal(new.loc[folds.index, "cv_fold"], folds["fold"])
        people.to_csv(DATA / "patient_splits.csv", index=False)
        write_json(DATA / "raw_case.json", {"raw_patients": 40336, "included": 39874,
            "raw_dtype": "float64", "manifest_sha256": hashlib.sha256(blob).hexdigest(),
            "all_original_PSV_sha256_verified": True, "patient_splits_match": True, "CV_folds_match": True,
            "raw_sha256": content_hash(DATA / "raw_float64.npy"),
            "audit_sha256": content_hash(DATA / "raw_audit.csv"), "patient_splits_sha256": content_hash(DATA / "patient_splits.csv")})
    for split in ["train", "validation"]:
        build_split(split)
    return design


def build_split(split):
    _, features, _ = friend_modules()
    case_path = DATA / f"{split}_case.json"
    if case_path.exists():
        for suffix in ["features.npy", "labels.npy", "patients.csv"]:
            path = DATA / f"{split}_{suffix}"
            assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), path
        return
    people = pd.read_csv(DATA / "patient_splits.csv")
    people = people.loc[people.split.eq(split)].sort_values(["hospital", "patient_id"])
    n = int(people.hours.sum())
    x = np.lib.format.open_memmap(DATA / f"{split}_features.npy", mode="w+", dtype=np.float32, shape=(n, 563))
    y = np.lib.format.open_memmap(DATA / f"{split}_labels.npy", mode="w+", dtype=np.int8, shape=(n,))
    raw = np.load(DATA / "raw_float64.npy", mmap_mode="r")
    records, position, checked = [], 0, set()
    pool = ProcessPoolExecutor(max_workers=4, initializer=_feature_process_init)
    generated_rows = pool.map(_feature_process_one,
        [(int(p.raw_start), int(p.raw_stop)) for p in people.itertuples(index=False)], chunksize=8)
    for i, (person, generated) in enumerate(zip(people.itertuples(index=False), generated_rows), 1):
        values = raw[int(person.raw_start):int(person.raw_stop)]
        assert generated.shape == (len(values), 563) and not np.isinf(generated).any()
        key = person.hospital, person.has_event
        if key not in checked:
            for cutoff in sorted(set([1, min(6, len(values)), max(1, len(values)//2), len(values)])):
                np.testing.assert_allclose(features.transform(values[:cutoff, :40], "augmented"), generated[:cutoff], rtol=0, atol=0, equal_nan=True)
            checked.add(key)
        stop = position + len(values)
        x[position:stop] = generated; y[position:stop] = values[:, 40]
        records.append({"patient_id": person.patient_id, "hospital": person.hospital, "has_event": person.has_event,
                        "cv_fold": person.cv_fold, "start": position, "stop": stop, "hours": len(values),
                        "raw_start": person.raw_start, "raw_stop": person.raw_stop, "first_hour": person.first_hour})
        position = stop
        if i % 1000 == 0:
            print(f"Built exact 563 features: {split} {i:,}/{len(people):,}", flush=True)
    pool.shutdown()
    x.flush(); y.flush(); del x
    patients = pd.DataFrame(records); patients.to_csv(DATA / f"{split}_patients.csv", index=False)
    actual = len(patients), int(patients.has_event.sum()), n, int(y.sum())
    assert actual == EXPECTED[split], actual
    write_json(case_path, {"patients": actual[0], "event_patients": actual[1], "hours": n, "positive_hours": actual[3],
                          "feature_sha256": content_hash(DATA / f"{split}_features.npy"),
                          "labels_sha256": content_hash(DATA / f"{split}_labels.npy"),
                          "patients_sha256": content_hash(DATA / f"{split}_patients.csv"), "prefix_groups_checked": len(checked)})


def utility_delta(labels, patients):
    """Per-hour utility of predicting one minus zero, using pinned default reward curves."""
    official_scorer()  # verifies the upstream source checksum
    delta = np.full(len(labels), -.05, dtype=np.float64)
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        y = labels[a:b]
        if not y.any():
            continue
        r = np.arange(len(y)) - (int(np.argmax(y)) + 6)
        u1 = np.where(r < -12, -.05, np.where(r <= -6, (r+12)/6, np.where(r <= 3, (3-r)/9, 0)))
        u0 = np.where(r <= -6, 0, np.where(r <= 3, -2*(r+6)/9, 0))
        delta[a:b] = u1-u0
    return delta


def select_utility_threshold(labels, probability, patients):
    delta = utility_delta(labels, patients)
    order = np.argsort(-probability, kind="stable")
    scores = probability[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores)-1]
    scaled = np.rint(delta*180).astype(np.int64)
    np.testing.assert_allclose(delta*180, scaled, rtol=0, atol=1e-9)
    gains = np.r_[0, np.cumsum(scaled[order])[ends]]
    thresholds = np.r_[np.nextafter(float(scores[0]), np.inf), scores[ends]]
    # The pinned default rewards are rational multiples of 1/180; integer sums
    # preserve exact utility ties without changing any prediction score.
    winner = int(np.argmax(gains))
    denominator = int(np.maximum(scaled, 0).sum())
    return {"threshold": float(thresholds[winner]), "max_validation_utility": float(gains[winner]/denominator),
            "tie_rule": "Highest threshold / fewest alarms among equal-utility score groups",
            "candidate_groups": len(thresholds), "positive_hours_predicted": int((probability >= thresholds[winner]).sum())}


def load_model(model, directory):
    if model == "xgboost":
        import xgboost as xgb
        booster = xgb.Booster(); booster.load_model(directory / "model.ubj"); booster.set_param({"nthread": 4})
    else:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(directory / "model.txt"))
    return booster


def predict(model, booster, x):
    result = np.empty(len(x), dtype=np.float64)
    for a in range(0, len(x), 10000):
        block = x[a:a+10000]
        p = booster.inplace_predict(block) if model == "xgboost" else booster.predict(block, num_threads=4)
        result[a:a+len(block)] = p
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result


def score(split, model, probability, threshold):
    labels = np.load(DATA / f"{split}_labels.npy")
    patients = pd.read_csv(DATA / f"{split}_patients.csv")
    contributions = []
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        contributions.append({"patient_id": patient.patient_id, **patient_utility(labels[a:b], probability[a:b] >= threshold)})
    frame = pd.DataFrame(contributions)
    out = OUT / "evaluation" / split / model; out.mkdir(parents=True, exist_ok=True)
    np.save(out / "probability.npy", probability)
    frame.to_csv(out / "patient_utilities.csv", index=False)
    result = {"ranking": ranking_metrics(labels, probability), "utility": normalized_utility(frame),
              "threshold": float(threshold), "patients": len(patients), "hours": len(labels),
              "positive_hours": int(labels.sum())}
    write_json(out / "metrics.json", result)
    return result
