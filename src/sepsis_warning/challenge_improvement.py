"""Patient-isolated development of utility objectives and causal feature additions."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sepsis_warning.data import ROOT
from sepsis_warning import friend_563 as baseline
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.challenge_features import engineer_patient, feature_names
from sepsis_warning.challenge_utility import utility_targets

DATA = ROOT / "processed_data/challenge_improvement"
OUT = ROOT / "outputs/challenge_improvement"
REPORT = ROOT / "reports/challenge_improvement"
FEATURE_BLOCKS = {"base": (), "lag": ("lag",),
                  "lag_observed": ("lag", "observed"),
                  "all": ("lag", "observed", "clinical")}


def available(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Cached input is a cloud placeholder: {path}")
    return path


def load_protocol():
    return json.loads(available(OUT / "protocol.json").read_text())


def candidate_by_id(identifier):
    values = json.loads(available(OUT / "candidates.json").read_text())
    matches = [value for value in values if value["id"] == identifier]
    assert len(matches) == 1, identifier
    return matches[0]


def local_patients(patients):
    result = patients.copy().reset_index(drop=True)
    result["stop"] = result.hours.cumsum()
    result["start"] = result.stop - result.hours
    return result


def row_indices(patients):
    return np.concatenate([np.arange(int(p.start), int(p.stop))
                           for p in patients.itertuples(index=False)])


def cv_roles(fold):
    patients = pd.read_csv(available(baseline.DATA / "train_patients.csv"))
    roles = pd.read_csv(available(DATA / "cv_patient_roles.csv"))
    mapping = roles.loc[roles.outer_fold.eq(fold)].set_index("patient_id").role
    assert set(mapping.index) == set(patients.patient_id)
    assigned = patients.patient_id.map(mapping)
    result = tuple(patients.loc[assigned.eq(role)].copy() for role in ("fit", "calibration", "held"))
    assert sum(map(len, result)) == len(patients)
    assert result[2].cv_fold.eq(fold).all()
    return result


def load_features(split, feature_set, indices=None):
    """Read only the requested split; append cached additions in bounded batches."""
    blocks = FEATURE_BLOCKS[feature_set]
    base = np.load(available(baseline.DATA / f"{split}_features.npy"), mmap_mode="r")
    if not blocks:
        return base if indices is None else base[indices]
    extra = np.load(available(DATA / f"{split}_extra.npy"), mmap_mode="r")
    width = len(feature_names(blocks))
    assert base.shape[1] == 563 and extra.shape == (len(base), 228)
    n = len(base) if indices is None else len(indices)
    result = np.empty((n, 563 + width), dtype=np.float32)
    for start in range(0, n, 10000):
        stop = min(n, start + 10000)
        selected = slice(start, stop) if indices is None else indices[start:stop]
        result[start:stop, :563] = base[selected]
        result[start:stop, 563:] = extra[selected, :width]
    return result


def ranking_scores(labels, scores):
    assert np.isfinite(scores).all()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def utility_for_scores(labels, scores, patients, threshold):
    scaled = np.rint(utility_targets(labels, patients)["delta"] * 180).astype(np.int64)
    denominator = np.maximum(scaled, 0).sum()
    assert denominator > 0
    return float(scaled[np.asarray(scores) >= threshold].sum() / denominator)


def _extra_init():
    global _raw
    _raw = np.load(available(baseline.DATA / "raw_float64.npy"), mmap_mode="r")


def _extra_patient(bounds):
    start, stop = bounds
    return engineer_patient(_raw[start:stop, :40])


def build_extra(split):
    fingerprint = content_hash(Path(__file__).with_name("challenge_features.py"))
    marker = DATA / f"{split}_extra_case.json"
    if marker.exists():
        assert json.loads(marker.read_text())["source_sha256"] == fingerprint
        available(DATA / f"{split}_extra.npy")
        return
    patients = pd.read_csv(available(baseline.DATA / f"{split}_patients.csv"))
    n = int(patients.hours.sum())
    target = np.lib.format.open_memmap(DATA / f"{split}_extra.npy", mode="w+", dtype=np.float32, shape=(n, 228))
    raw = np.load(available(baseline.DATA / "raw_float64.npy"), mmap_mode="r")
    checks, position = set(), 0
    bounds = [(int(p.raw_start), int(p.raw_stop)) for p in patients.itertuples(index=False)]
    with ProcessPoolExecutor(max_workers=4, initializer=_extra_init) as pool:
        for i, (patient, values) in enumerate(zip(patients.itertuples(index=False), pool.map(_extra_patient, bounds, chunksize=16)), 1):
            assert values.shape == (patient.hours, 228) and not np.isinf(values).any()
            assert int(patient.start) == position
            stop = int(patient.stop)
            target[position:stop] = values
            key = patient.hospital, patient.has_event
            if key not in checks:
                original = raw[int(patient.raw_start):int(patient.raw_stop), :40]
                for cutoff in sorted({1, min(6, len(original)), max(1, len(original)//2), len(original)}):
                    np.testing.assert_allclose(engineer_patient(original[:cutoff]), values[:cutoff], rtol=0, atol=0, equal_nan=True)
                checks.add(key)
            position = stop
            if i % 3000 == 0:
                print(f"Causal additions: {split} {i:,}/{len(patients):,} patients", flush=True)
    target.flush()
    assert position == n
    write_json(marker, {"source_sha256": fingerprint, "feature_sha256": content_hash(DATA / f"{split}_extra.npy"),
                        "patients": len(patients), "hours": n, "features": 228, "prefix_groups": len(checks)})


def prepare():
    for directory in (DATA, OUT, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    source = json.loads(available(baseline.OUT / "protocol.json").read_text())
    patients = pd.read_csv(available(baseline.DATA / "train_patients.csv"))
    roles_path = DATA / "cv_patient_roles.csv"
    if not roles_path.exists():
        records = []
        for fold in range(3):
            other = patients.loc[patients.cv_fold.ne(fold)]
            fit, cal = train_test_split(other.patient_id, test_size=.15, random_state=62026+fold,
                                        stratify=other.hospital+"_"+other.has_event.astype(str))
            fit, cal = set(fit), set(cal)
            for p in patients.itertuples(index=False):
                role = "fit" if p.patient_id in fit else "calibration" if p.patient_id in cal else "held"
                assert (role == "held") == (p.cv_fold == fold)
                records.append({"outer_fold": fold, "patient_id": p.patient_id, "hospital": p.hospital,
                                "has_event": p.has_event, "role": role})
        pd.DataFrame(records).to_csv(roles_path, index=False)
    protocol = {
        "baseline_commit": "02a2863bdb3fdbc963591aeeaea6cfce863d2783",
        "standard_id": "SEPSIS_FULL_RECORD_V2", "maximum_rounds": 1000,
        "auc_early_stopping_rounds": 50, "utility_check_interval": 10,
        "utility_early_stopping_checks": 8, "min_delta": 1e-5,
        "feature_names": {key: source["feature_names"]+feature_names(blocks) for key, blocks in FEATURE_BLOCKS.items()},
        "base_parameters": source["parameters"],
        "roles_sha256": content_hash(roles_path),
        "patient_splits_sha256": content_hash(baseline.DATA / "patient_splits.csv"),
        "selection": "Existing three hospital/event-stratified patient folds. Within each outer training pool, reserve 15% hospital/event-stratified patients for early stopping and utility threshold. Outer held patients select neither. Candidate selection uses outer development metrics; post-selection CV is not a nested unbiased estimate.",
        "final_rounds": "Median of three inner-calibration-selected iteration counts; refit all original training patients.",
        "final_threshold": "Exact maximum full-record US on original validation patients, after candidates and rounds are frozen.",
        "test_policy": "All model/feature/ensemble choices frozen using training patients. Existing test has historical exposure; exploratory fixed benchmark only, no external-hospital or pristine-test claim.",
        "candidate_stages": [
            "Unchanged563: LGB binary, XGB binary, LGB delta regression, LGB utility-label weighted classification.",
            "LGB binary and training-CV-best LGB utility target: add lag, then real-observation history, then two clinical combinations.",
            "Two capacity/randomization alternatives for LGB AUC and US leaders; two XGB alternatives on training-selected AUC feature set.",
            "Predeclared equal binary-probability blend and percentile-rank blend of AUC/US leaders at weights .25/.5/.75. Select using training CV only."],
        "target_weighting": "Utility classification labels=1[delta>0], weights=abs(delta), divided by mean weight on fit patients only. Zero delta gets zero weight. Regression predicts unnormalized marginal utility; not a sepsis probability.",
        "sources": ["https://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf",
                    "https://www.cinc.org/archives/2019/pdf/CinC2019-459.pdf",
                    "https://www.cinc.org/archives/2019/pdf/CinC2019-423.pdf",
                    "https://www.cinc.org/archives/2019/pdf/CinC2019-020.pdf"],
        "future_extension": "Signature transforms deferred until low-cost objective/causal-feature/capacity/ensemble ablations are complete; not required for this round."}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Protocol changed; use a new experiment directory"
    else:
        write_json(path, protocol)
    labels = np.load(available(baseline.DATA / "train_labels.npy"))
    delta_path = DATA / "train_delta.npy"
    targets = utility_targets(labels, patients)
    if delta_path.exists():
        np.testing.assert_array_equal(np.load(available(delta_path)), targets["delta"])
    else:
        np.save(delta_path, targets["delta"])
    build_extra("train")
    write_json(REPORT / "protocol.json", protocol)
    pd.read_csv(roles_path).to_csv(REPORT / "cv_patient_roles.csv", index=False)
    return protocol


def candidate(identifier, model, target="binary", feature_set="base", overrides=None):
    protocol = load_protocol()
    params = dict(protocol["base_parameters"][model])
    params.update(overrides or {})
    metric = "auroc" if target in ("binary", "class_weighted") else "utility"
    if model == "lightgbm":
        params["objective"] = "regression" if target == "utility_regression" else "binary"
        params["metric"] = "auc" if metric == "auroc" else "None"
    else:
        params["objective"] = "reg:squarederror" if target == "utility_regression" else "binary:logistic"
        if metric == "utility":
            params.pop("eval_metric", None)
            params["disable_default_eval_metric"] = 1
    return {"id": identifier, "model": model, "target": target, "feature_set": feature_set,
            "parameters": params, "selection_metric": metric, "max_rounds": protocol["maximum_rounds"]}


def save_candidates(additions):
    path = OUT / "candidates.json"
    current = json.loads(path.read_text()) if path.exists() else []
    known = {entry["id"]: entry for entry in current}
    for entry in additions:
        if entry["id"] in known:
            assert known[entry["id"]] == entry
        else:
            current.append(entry)
            known[entry["id"]] = entry
    write_json(path, current)
    return current
