"""Prepare controlled data variants and run three models on identical subsets."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import (FEATURE_NAMES, COLUMNS, TrainPreprocessor, assess_patient,
                                causal_features, config, save_json)
from sepsis_warning.history_study import settings, sha_file, digest, sample_patients, child_environment


def main():
    started = time.monotonic()
    cfg, study = config(), settings()
    processed = ROOT/"processed_data"
    cache = processed/"history_study"
    output = ROOT/"outputs/history_study"
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    reference_files = [processed/f"{split}_{part}.npy" for split in ["train", "validation"] for part in ["X", "y", "features"]]
    reference_files += [processed/"splits/patient_splits.csv", ROOT/"outputs/preprocessor.joblib",
                        ROOT/"outputs/lr_validation_predictions.csv.gz"]
    initial = {str(path.relative_to(ROOT)): sha_file(path) for path in reference_files}
    save_json(output/"input_fingerprints.json", initial)
    train_rows = pd.read_csv(processed/"train_rows.csv.gz")
    validation_rows = pd.read_csv(processed/"validation_rows.csv.gz")
    labels = np.load(processed/"train_y.npy")
    assert np.array_equal(labels, train_rows.label.to_numpy())
    patients = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    raw = np.load(processed/"raw_records.npy", mmap_mode="r")
    # Build only A train/validation variants. Test and hospital B remain untouched.
    variants = {}
    availability = []
    for retention in study["study"]["lab_retention_hours"]:
        signature = digest({"retention": retention, "inputs": initial, "algorithm": "lab_only_latest_expiry_v1"})
        variant_dir = cache/f"history{retention}h"
        variant_dir.mkdir(exist_ok=True)
        completion = variant_dir/"variant.json"
        if retention == 6:
            paths = {split: processed/f"{split}_features.npy" for split in ["train", "validation"]}
        else:
            paths = {split: variant_dir/f"{split}_features.npy" for split in ["train", "validation"]}
            valid_cache = completion.exists() and json.loads(completion.read_text()).get("signature") == signature
            if not valid_cache:
                print(f"Building {retention}h lab-retention features; other features stay fixed", flush=True)
                for split, rows in [("train", train_rows), ("validation", validation_rows)]:
                    matrix = np.lib.format.open_memmap(paths[split], mode="w+", dtype=np.float32, shape=(len(rows), len(FEATURE_NAMES)))
                    position = 0
                    for patient_id in rows.patient_id.drop_duplicates():
                        patient = patients.loc[patient_id]
                        values = raw[int(patient.raw_start):int(patient.raw_stop)]
                        reason, eligible, _ = assess_patient(values, cfg)
                        assert reason == "included"
                        features = causal_features(values, cfg, lab_retention_hours=retention)[eligible]
                        matrix[position:position+len(features)] = features
                        position += len(features)
                    assert position == len(rows)
                    matrix.flush()
                    baseline = np.load(processed/f"{split}_features.npy", mmap_mode="r")
                    unchanged = np.r_[np.arange(8), np.arange(34, len(FEATURE_NAMES))]
                    for start in range(0, len(rows), 20000):
                        np.testing.assert_allclose(matrix[start:start+20000, unchanged], baseline[start:start+20000, unchanged], equal_nan=True, rtol=0, atol=0)
                    del matrix
        variants[retention] = paths
        matrix = np.load(paths["train"], mmap_mode="r")
        for j, feature in enumerate(COLUMNS[:34]):
            availability.append({"retention_hours": retention, "feature": feature,
                                 "latest_missing_rate": float(np.isnan(matrix[:, j]).mean())})
        save_json(completion, {"signature": signature, "retention_hours": retention,
                              "only_changed": "latest lab values; vital expiry and all rolling statistics remain 6h"})
    pd.DataFrame(availability).to_csv(output/"history_availability.csv", index=False)
    trial_specs = []
    for fraction in [1.0, *[f for f in study["study"]["patient_fractions"] if f != 1]]:
        seeds = [None] if fraction == 1 else study["study"]["sample_seeds"]
        for seed in seeds:
            ids = train_rows.patient_id.unique().tolist() if fraction == 1 else sample_patients(train_rows, fraction, seed)
            indices = np.flatnonzero(train_rows.patient_id.isin(ids).to_numpy())
            assert not (set(ids) & set(validation_rows.patient_id))
            for retention in study["study"]["lab_retention_hours"]:
                suffix = "full" if fraction == 1 else f"fraction{int(fraction*100)}_seed{seed}"
                name = f"history{retention}h_{suffix}"
                trial_specs.append((name, retention, fraction, seed, ids, indices))
    for index, (name, retention, fraction, seed, ids, indices) in enumerate(trial_specs):
        trial_dir = cache/name
        result_dir = output/name
        trial_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
        signature = digest({"inputs": initial, "retention": retention, "fraction": fraction,
                            "patient_ids": ids, "preprocessing": cfg["features"], "algorithm": "history_study_v1"})
        meta_path = trial_dir/"trial.json"
        valid_cache = meta_path.exists() and json.loads(meta_path.read_text()).get("signature") == signature
        if not valid_cache:
            selected_rows = train_rows.iloc[indices].reset_index(drop=True)
            selected_rows.to_csv(trial_dir/"train_rows.csv.gz", index=False, compression="gzip")
            pd.DataFrame({"patient_id": ids}).to_csv(trial_dir/"patients.csv", index=False)
            np.save(trial_dir/"train_y.npy", labels[indices])
            if retention == 6 and fraction == 1:
                processor = joblib.load(ROOT/"outputs/preprocessor.joblib")
                train_path, validation_path = processed/"train_X.npy", processed/"validation_X.npy"
            else:
                raw_train = np.load(variants[retention]["train"], mmap_mode="r")
                selected = raw_train if fraction == 1 else raw_train[indices]
                processor = TrainPreprocessor(FEATURE_NAMES, cfg["features"]["clip_lower_quantile"], cfg["features"]["clip_upper_quantile"]).fit(selected)
                train_path = trial_dir/"train_X.npy"
                matrix = np.lib.format.open_memmap(train_path, mode="w+", dtype=np.float32, shape=(len(indices), len(processor.names_)))
                for start in range(0, len(indices), 20000):
                    matrix[start:start+20000] = processor.transform(selected[start:start+20000])
                matrix.flush()
                del matrix, selected, raw_train
                raw_validation = np.load(variants[retention]["validation"], mmap_mode="r")
                validation_path = trial_dir/"validation_X.npy"
                matrix = np.lib.format.open_memmap(validation_path, mode="w+", dtype=np.float32, shape=(len(validation_rows), len(processor.names_)))
                for start in range(0, len(validation_rows), 20000):
                    matrix[start:start+20000] = processor.transform(raw_validation[start:start+20000])
                matrix.flush()
                del matrix, raw_validation
            joblib.dump(processor, trial_dir/"preprocessor.joblib", compress=3)
            metadata = {"name": name, "signature": signature, "retention_hours": retention,
                        "fraction": fraction, "sample_seed": seed, "patients": len(ids),
                        "event_patients": int(selected_rows.groupby("patient_id").label.max().sum()),
                        "hours": len(indices), "positive_hours": int(labels[indices].sum()),
                        "features": len(processor.names_), "train_X": str(train_path.relative_to(ROOT)),
                        "validation_X": str(validation_path.relative_to(ROOT)),
                        "preprocessor_fit_on": "selected A training patients only", "model_hyperparameters_fixed": True}
            save_json(meta_path, metadata)
        print(f"Trial {index+1}/{len(trial_specs)}: {name}", flush=True)
        active = []
        try:
            for model in study["study"]["models"]:
                log = (result_dir/f"{model}.log").open("w")
                process = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/fit_history_model.py"),
                                            "--trial", name, "--model", model], cwd=ROOT,
                                           env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
                active.append((model, process, log))
            last_notice = time.monotonic()
            while active:
                for item in active.copy():
                    model, process, log = item
                    if process.poll() is not None:
                        log.close()
                        active.remove(item)
                        if process.returncode:
                            raise RuntimeError(f"{name}/{model} failed:\n"+(result_dir/f"{model}.log").read_text()[-6000:])
                        result = json.loads((result_dir/f"{model}_results.json").read_text())
                        print(f"  {model}: AP={result['validation']['auprc_average_precision']:.4f}, AUROC={result['validation']['auroc']:.4f}", flush=True)
                if active and time.monotonic()-last_notice > 35:
                    print(f"  Still fitting: {', '.join(item[0] for item in active)}", flush=True)
                    last_notice = time.monotonic()
                if active:
                    time.sleep(2)
        finally:
            for _, process, log in active:
                process.terminate()
                log.close()
    assert initial == {str(path.relative_to(ROOT)): sha_file(path) for path in reference_files}
    save_json(output/"completed.json", {"trials": len(trial_specs), "model_results": len(trial_specs)*3,
              "input_fingerprints_unchanged": True, "test_sets_evaluated": [],
              "model_settings_fixed_across_windows_and_sample_sizes": True,
              "elapsed_seconds": round(time.monotonic()-started, 2)})
    print("History-retention and patient learning-curve experiments complete.", flush=True)


if __name__ == "__main__":
    main()
