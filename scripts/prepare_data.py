"""Audit every raw file, construct the cohort, fix splits, and cache causal features."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sepsis_warning.data import (COLUMNS, CONTEXT, FEATURE_NAMES, PHYSIOLOGY,
                                TrainPreprocessor, assess_patient, causal_features,
                                config, create_splits, environment, save_json)


def main():
    started = time.monotonic()
    cfg = config()
    output = ROOT / cfg["data"]["output_directory"]
    processed = ROOT / cfg["data"]["processed_directory"]
    output.mkdir(exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "data/manifest.csv"
    manifest = pd.read_csv(manifest_path)
    reference = json.loads((ROOT / "data/download_report.json").read_text())
    expected = set(manifest.relative_path)
    actual = {str(p.relative_to(ROOT)) for p in (ROOT / "data").glob("training_set*/*.psv")}
    assert actual == expected and len(expected) == len(manifest) == reference["patient_files"]
    raw = np.lib.format.open_memmap(processed / "raw_records.npy", mode="w+", dtype=np.float32,
                                    shape=(int(manifest.hours.sum()), 41))
    row_codes = np.lib.format.open_memmap(processed / "hour_exclusion_codes.npy", mode="w+",
                                          dtype=np.uint8, shape=(int(manifest.hours.sum()),))
    patient_rows, offset = [], 0
    for i, record in enumerate(manifest.itertuples(index=False)):
        path = ROOT / record.relative_path
        body = path.read_bytes()
        if len(body) != record.bytes or hashlib.sha256(body).hexdigest() != record.sha256 or hashlib.md5(body).hexdigest() != record.md5:
            raise ValueError(f"Checksum mismatch: {record.relative_path}")
        if body.splitlines()[0].decode() != "|".join(COLUMNS):
            raise ValueError(f"Header mismatch: {record.relative_path}")
        values = np.loadtxt(io.BytesIO(body), delimiter="|", skiprows=1, dtype=np.float32, ndmin=2)
        assert values.shape == (record.hours, 41)
        assert np.isin(values[:, 40], [0, 1]).all()
        assert bool(values[:, 40].any()) == bool(record.has_positive_label)
        raw[offset:offset + len(values)] = values
        reason, eligible, onset = assess_patient(values, cfg)
        # 0=included, 1=excluded patient, 2=post-onset, 3=incomplete negative follow-up.
        row_codes[offset:offset+len(values)] = np.where(eligible, 0, 2 if np.isfinite(onset) else 3) if reason == "included" else 1
        positive = np.flatnonzero(values[:, 40] == 1)
        hospital = "A" if "training_setA" in record.relative_path else "B"
        patient_rows.append({
            "patient_id": f"{hospital}_{path.stem}", "hospital": hospital,
            "relative_path": record.relative_path, "raw_start": offset, "raw_stop": offset + len(values),
            "n_hours": len(values), "first_hour": float(values[0, 39]), "last_hour": float(values[-1, 39]),
            "age": float(values[0, 34]), "recorded_gender": float(values[0, 35]),
            "has_positive_label": bool(len(positive)),
            "first_positive_hour": float(values[positive[0], 39]) if len(positive) else np.nan,
            "estimated_onset_hour": onset, "included": reason == "included", "exclusion_reason": reason,
            "n_eligible_hours": int(eligible.sum()),
            "n_positive_eligible_hours": int(values[eligible, 40].sum()),
            "n_removed_post_onset_hours": int(np.sum(values[:, 39] >= onset)) if reason == "included" and np.isfinite(onset) else 0,
            "n_removed_incomplete_negative_hours": int((~eligible).sum()) if reason == "included" and not len(positive) else 0,
            "n_infinite_inputs": int(np.isinf(values[:, :40]).sum()),
            "n_negative_physiology_values": int(np.sum(values[:, [0, 1, 2, 3, 4, 5, 6]] < 0)),
        })
        offset += len(values)
        if (i+1) % 5000 == 0:
            print(f"Audited {i+1:,}/{len(manifest):,} patients", flush=True)
    assert offset == reference["hourly_records"]
    assert sum(p["has_positive_label"] for p in patient_rows) == reference["patients_with_positive_label"]
    raw.flush()
    row_codes.flush()
    patients = pd.DataFrame(patient_rows)
    patients.to_csv(output / "patient_summary.csv", index=False)
    exclusions = patients.groupby(["hospital", "exclusion_reason"], dropna=False).agg(
        patients=("patient_id", "size"), raw_hours=("n_hours", "sum"),
        eligible_hours=("n_eligible_hours", "sum"), positive_patients=("has_positive_label", "sum")).reset_index()
    exclusions.to_csv(output / "cohort_flow.csv", index=False)
    splits = create_splits(patients, cfg).sort_values(["split", "patient_id"]).reset_index(drop=True)
    split_path = ROOT / cfg["data"]["split_file"]
    split_path.parent.mkdir(parents=True, exist_ok=True)
    splits[["patient_id", "hospital", "split", "has_positive_label"]].to_csv(split_path, index=False)
    split_summary = splits.groupby(["hospital", "split"]).agg(
        patients=("patient_id", "size"), event_patients=("has_positive_label", "sum"),
        eligible_hours=("n_eligible_hours", "sum"), positive_hours=("n_positive_eligible_hours", "sum")).reset_index()
    split_summary["event_patient_prevalence"] = split_summary.event_patients / split_summary.patients
    split_summary["positive_hour_prevalence"] = split_summary.positive_hours / split_summary.eligible_hours
    split_summary.to_csv(output / "split_summary.csv", index=False)
    print(split_summary.to_string(index=False), flush=True)

    train_patients = splits.loc[splits.split == "train"]
    missing = np.zeros(40, np.int64)
    never = np.zeros(40, np.int64)
    max_gap = np.zeros(40, np.int64)
    measurements, group_hours, group_missing = [], Counter(), {False: np.zeros(40), True: np.zeros(40)}
    row_totals = Counter()
    for split, group in splits.groupby("split", sort=True):
        n_rows = int(group.n_eligible_hours.sum())
        features = np.lib.format.open_memmap(processed / f"{split}_features.npy", mode="w+",
                                             dtype=np.float32, shape=(n_rows, len(FEATURE_NAMES)))
        metadata, position = [], 0
        for patient in group.itertuples(index=False):
            values = raw[patient.raw_start:patient.raw_stop]
            reason, eligible, onset = assess_patient(values, cfg)
            assert reason == "included"
            patient_features = causal_features(values, cfg)[eligible]
            n = len(patient_features)
            features[position:position+n] = patient_features
            for hour, label in zip(values[eligible, 39], values[eligible, 40]):
                metadata.append((patient.patient_id, patient.hospital, int(hour), int(label), onset))
            position += n
            if split == "train":
                original = values[eligible, :40]
                absent = ~np.isfinite(original)
                missing += absent.sum(axis=0)
                never += absent.all(axis=0)
                gap = np.zeros(40, np.int64)
                for row in absent:
                    gap = np.where(row, gap + 1, 0)
                    max_gap = np.maximum(max_gap, gap)
                group_missing[patient.has_positive_label] += absent.sum(axis=0)
                group_hours[patient.has_positive_label] += n
                # Measurement intervals are described for physiological variables only.
                for j, name in enumerate(PHYSIOLOGY):
                    observed_hours = values[eligible, 39][~absent[:, j]]
                    if len(observed_hours) > 1:
                        measurements.extend((name, float(gap)) for gap in np.diff(observed_hours))
        assert position == n_rows
        features.flush()
        pd.DataFrame(metadata, columns=["patient_id", "hospital", "hour", "label", "onset_hour"]).to_csv(
            processed / f"{split}_rows.csv.gz", index=False, compression="gzip")
        row_totals[split] = n_rows
        print(f"Saved {split}: {n_rows:,} hours x {len(FEATURE_NAMES)} causal features", flush=True)
        del features

    n_train = row_totals["train"]
    missing_table = pd.DataFrame({"feature": COLUMNS[:40], "missing_hours": missing,
                                  "hour_missing_rate": missing / n_train,
                                  "never_measured_patients": never,
                                  "never_measured_patient_rate": never / len(train_patients),
                                  "max_consecutive_missing_hours": max_gap,
                                  "event_patient_hour_missing_rate": group_missing[True] / group_hours[True],
                                  "nonevent_patient_hour_missing_rate": group_missing[False] / group_hours[False]})
    missing_table.to_csv(output / "training_missingness.csv", index=False)
    if measurements:
        intervals = pd.DataFrame(measurements, columns=["feature", "gap_hours"])
        intervals.groupby("feature").gap_hours.agg(["count", "median", "mean", "max"]).to_csv(output / "training_measurement_intervals.csv")
        del measurements, intervals

    categorical = {f"{name}__current" for name in ["Gender", "Unit1", "Unit2"]}
    feature_types = ["binary" if name.endswith(("__missing_now", "__never_measured")) else
                     "categorical" if name in categorical else "numeric" for name in FEATURE_NAMES]
    pd.DataFrame({"feature": FEATURE_NAMES, "type": feature_types}).to_csv(output / "feature_dictionary.csv", index=False)
    import joblib
    train = np.load(processed / "train_features.npy", mmap_mode="r")
    processor = TrainPreprocessor(FEATURE_NAMES, cfg["features"]["clip_lower_quantile"], cfg["features"]["clip_upper_quantile"]).fit(train)
    joblib.dump(processor, output / "preprocessor.joblib", compress=3)
    pd.DataFrame({"feature": processor.names_, "imputation_value": processor.fill_,
                  "clip_lower": processor.lower_, "clip_upper": processor.upper_,
                  "scaling_mean": processor.scaler_.mean_, "scaling_scale": processor.scaler_.scale_}).to_csv(output / "preprocessing_parameters.csv", index=False)
    save_json(output / "preprocessing_summary.json", {"input_features": len(FEATURE_NAMES),
              "retained_features": len(processor.names_), "dropped_all_missing_in_train": processor.dropped_,
              "fit_split": "train", "training_rows": len(train),
              "numeric_imputation": "training median", "categorical_imputation": "training mode",
              "numeric_clipping_quantiles": [processor.lower_quantile, processor.upper_quantile],
              "scaling": "training mean and standard deviation", "feature_statistics": "observed measurements only; no future/backward fill"})
    # Apply the frozen training transform to every split, without evaluating test outcomes.
    for split in row_totals:
        original = np.load(processed / f"{split}_features.npy", mmap_mode="r")
        transformed = np.lib.format.open_memmap(processed / f"{split}_X.npy", mode="w+",
                                                dtype=np.float32, shape=(len(original), len(processor.names_)))
        for start in range(0, len(original), 20000):
            transformed[start:start+20000] = processor.transform(original[start:start+20000])
        transformed.flush()
        labels = pd.read_csv(processed / f"{split}_rows.csv.gz", usecols=["label"]).label.to_numpy(np.int8)
        np.save(processed / f"{split}_y.npy", labels)
        assert np.isfinite(transformed).all() and len(labels) == len(transformed)
        print(f"Applied training-fitted preprocessing to {split}", flush=True)
        del original, transformed
    checks = {"all_raw_files_match_manifest": True, "raw_patient_files": len(manifest),
              "raw_hourly_records": offset, "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "patient_id_unique": bool(patients.patient_id.is_unique),
              "no_patient_overlap_between_splits": bool(splits.patient_id.is_unique),
              "configuration": cfg, "environment": environment(),
              "eligible_patients": int(patients.included.sum()), "eligible_hours": int(splits.n_eligible_hours.sum()),
              "excluded_patients": int((~patients.included).sum()),
              "infinite_input_values": int(patients.n_infinite_inputs.sum()),
              "negative_vital_values_flagged_not_deleted": int(patients.n_negative_physiology_values.sum()),
              "elapsed_seconds": round(time.monotonic()-started, 2)}
    save_json(output / "audit_report.json", checks)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sorted_missing = missing_table.sort_values("hour_missing_rate")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(sorted_missing.feature, sorted_missing.hour_missing_rate*100, color="#24788c")
    ax.set(xlabel="Hours without a measurement (%)", title="Missingness in eligible hospital A training hours", xlim=(0, 100))
    fig.tight_layout()
    fig.savefig(output / "training_missingness.png", dpi=160)
    plt.close(fig)
    print(f"Preparation complete in {checks['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
