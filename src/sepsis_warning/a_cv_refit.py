"""Frozen full-A patient CV and refit, with B excluded from development."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.data import ROOT, assess_patient, causal_features, config
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.history_study import digest
from sepsis_warning.utility import normalized_utility, patient_utility

DATA = ROOT / "processed_data/a_only_cv_refit"
OUT = ROOT / "outputs/a_only_cv_refit"
REPORT = ROOT / "reports/a_only_cv_refit"
MODELS = ["lr", "ebm", "lightgbm"]


def prepare():
    for directory in [DATA, OUT, REPORT]:
        directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / f"processed_data/{s}_{suffix}" for s in ["train", "validation", "test"]
               for suffix in ["features.npy", "rows.csv.gz"]]
    sources += [ROOT / "processed_data/raw_records.npy", ROOT / "outputs/patient_summary.csv",
                ROOT / "processed_data/splits/patient_splits.csv",
                ROOT / "reports/ab_expansion/B_patient_splits.csv",
                ROOT / "reports/history_study/model_settings.json",
                ROOT / "reports/ab_regularization/protocol.json", ROOT / "configs/baseline.toml"]
    for path in sources:
        assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), path
    fingerprints = {str(p.relative_to(ROOT)): content_hash(p) for p in sources}
    settings = json.loads((ROOT / "reports/history_study/model_settings.json").read_text())
    columns = json.loads((ROOT / "reports/ab_regularization/protocol.json").read_text())["feature_columns"]
    assert len(columns) == 312
    case_signature = digest({"input_fingerprints": fingerprints, "fold_seed": 2026, "folds": 3})
    case_path = DATA / "case.json"
    if not case_path.exists() or json.loads(case_path.read_text())["signature"] != case_signature:
        tables, offsets, start = [], [], 0
        matrix = np.lib.format.open_memmap(DATA / "A_features.npy", mode="w+", dtype=np.float32, shape=(670920, 318))
        for name in ["train", "validation", "test"]:
            table = pd.read_csv(ROOT / f"processed_data/{name}_rows.csv.gz")
            source = np.load(ROOT / f"processed_data/{name}_features.npy", mmap_mode="r")
            assert source.shape == (len(table), 318) and table.hospital.eq("A").all()
            for i in range(0, len(table), 10000):
                matrix[start+i:start+min(i+10000, len(table))] = source[i:i+10000]
            offsets.append({"source": name, "start": start, "stop": start+len(table)})
            start += len(table)
            tables.append(table)
        assert start == 670920
        matrix.flush()
        del matrix
        rows = pd.concat(tables, ignore_index=True)
        assert not rows.duplicated(["patient_id", "hour"]).any()
        rows.to_csv(DATA / "A_rows.csv.gz", index=False, compression="gzip")
        patients = rows.groupby("patient_id", sort=True).agg(has_event=("label", "max"), hospital=("hospital", "first")).reset_index()
        assert len(patients) == 20131 and int(patients.has_event.sum()) == 1585
        assert int(rows.label.sum()) == 9510
        patients["fold"] = -1
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
        for fold, (_, held) in enumerate(splitter.split(patients.patient_id, patients.has_event)):
            patients.loc[held, "fold"] = fold
        assert patients.fold.isin([0, 1, 2]).all()
        patients.to_csv(DATA / "A_patients.csv", index=False)
        write_json(case_path, {"signature": case_signature, "patients": len(patients), "event_patients": 1585,
                              "eligible_hours": len(rows), "positive_hours": 9510, "source_offsets": offsets,
                              "features_sha256": content_hash(DATA / "A_features.npy"),
                              "rows_sha256": content_hash(DATA / "A_rows.csv.gz"),
                              "patients_sha256": content_hash(DATA / "A_patients.csv")})
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "A_case_signature": case_signature,
        "input_fingerprints": fingerprints,
        "parameters": settings["models"], "lightgbm_stopping": settings["lightgbm_stopping"],
        "feature_columns": columns, "folds": 3, "fold_seed": 2026,
        "inner_patient_holdout_fraction": .2, "inner_seed": 2026,
        "hyperparameter_search": "None: previously frozen 312-feature baseline settings; only fitted stopping iterations vary",
        "A_primary_metrics": "Patient-disjoint pooled OOF AUROC/AP; utility uses each outer fold's internally calibrated threshold",
        "inner_preprocessing": "Fit on inner fitting patients only; refit on all outer training patients afterward",
        "EBM_internal_binning": "Binning sees current fitting partition, including internal early-stopping bags; outer evaluation excluded",
        "final_threshold": "Pooled eligible A OOF predictions; fixed existing clinical alarm-budget criterion",
        "threshold_criterion": "Maximize event sensitivity under <=1 false alert episode per 100 eligible hours, 6-hour suppression",
        "utility": "Pinned official scorer, full trajectories, original labels, hourly predictions without suppression",
        "final_refit": "All 20,131 included A patients; original A test now enters development and fitting",
        "B_selection_used": False,
        "interpretation": "Exploratory transfer to 19,743 included B patients; B was examined in prior experiments",
    }
    path = OUT / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        a, b = dict(previous), dict(protocol)
        a.pop("created_at_utc"); b.pop("created_at_utc")
        assert a == b, "Preserve earlier protocol/results instead of silently changing them"
        protocol = previous
    write_json(path, protocol)
    return protocol


def predict_full_patients(bundle, identifiers, directory):
    """Score raw full trajectories using frozen models; never fit on these patients."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    identifiers = list(identifiers)
    assert len(identifiers) == len(set(identifiers)) > 0
    audit = pd.read_csv(ROOT / "outputs/patient_summary.csv").set_index("patient_id")
    selected = audit.loc[identifiers]
    assert selected.included.all()
    n = int(selected.n_hours.sum())
    raw = np.load(ROOT / "processed_data/raw_records.npy", mmap_mode="r")
    probability = np.lib.format.open_memmap(directory / "probability.npy", mode="w+", dtype=np.float64, shape=(n,))
    labels = np.empty(n, dtype=np.int8)
    eligible = np.empty(n, dtype=bool)
    cfg, blocks, records, position, batch_start = config(), [], [], 0, 0
    prefix_checks = 0

    def flush():
        nonlocal batch_start
        features = np.concatenate(blocks)
        x = bundle["preprocessor"].transform(features)
        classifier = bundle["model"]
        with threadpool_limits(limits=2):
            p = (classifier.booster_.predict(x, num_threads=2) if hasattr(classifier, "booster_")
                 else classifier.predict_proba(x)[:, 1])
        assert len(p) == position-batch_start and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        probability[batch_start:position] = p
        blocks.clear()
        batch_start = position

    for i, (identifier, patient) in enumerate(selected.iterrows()):
        values = raw[int(patient.raw_start):int(patient.raw_stop)]
        reason, mask, _ = assess_patient(values, cfg)
        assert reason == "included" and len(values) == patient.n_hours
        features = causal_features(values, cfg)
        if i in [0, len(selected)//2, len(selected)-1]:
            cutoff = max(1, len(values)//2)
            np.testing.assert_allclose(causal_features(values[:cutoff], cfg), features[:cutoff], rtol=0, atol=0, equal_nan=True)
            prefix_checks += 1
        stop = position+len(values)
        labels[position:stop] = values[:, 40]
        eligible[position:stop] = mask
        records.append({"patient_id": identifier, "hospital": patient.hospital, "start": position, "stop": stop,
                        "hours": len(values), "first_hour": int(values[0, 39]), "has_event": bool(values[:, 40].any())})
        blocks.append(features)
        position = stop
        if position-batch_start >= 10000:
            flush()
    if blocks:
        flush()
    assert position == n
    probability.flush()
    patients = pd.DataFrame(records)
    patients.to_csv(directory / "patients.csv", index=False)
    np.save(directory / "labels.npy", labels)
    np.save(directory / "task_eligible.npy", eligible)
    threshold = float(bundle["threshold"])
    contributions = []
    for patient in patients.itertuples(index=False):
        begin, end = int(patient.start), int(patient.stop)
        contributions.append({"patient_id": patient.patient_id,
                              **patient_utility(labels[begin:end], probability[begin:end] >= threshold)})
    frame = pd.DataFrame(contributions)
    frame.to_csv(directory / "patient_utilities.csv", index=False)
    result = {"ranking": ranking_metrics(labels[eligible], probability[eligible]),
              "utility": normalized_utility(frame), "patients": len(patients),
              "event_patients": int(patients.has_event.sum()), "eligible_hours": int(eligible.sum()),
              "full_hours": n, "threshold": threshold}
    assert int(labels[eligible].sum()) == 6*int(patients.has_event.sum())
    write_json(directory / "verification.json", {"prefix_checks": prefix_checks, "finite_probabilities": True,
                                                 "original_labels_retained": True, "no_fit_during_prediction": True})
    write_json(directory / "evaluation.json", result)
    return result
