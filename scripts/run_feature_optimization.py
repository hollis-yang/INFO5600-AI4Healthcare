"""Screen causal feature alternatives inside A training, then evaluate selected models."""
from __future__ import annotations

import hashlib
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
from sepsis_warning.data import config, save_json, assess_patient
from sepsis_warning.feature_engineering import PROFILES, FeaturePreprocessor, alternative_features, feature_names
from sepsis_warning.history_study import digest, sha_file, child_environment
from sepsis_warning.optimization import (PROCESSED, OUTPUT, settings, frozen_models, patient_folds,
                                         representation, representation_key, raw_features, choose_profiles)


def array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def build_combined(split, fingerprint, study):
    target = PROCESSED/f"{split}_combined_features.npy"
    marker = PROCESSED/f"{split}_features.json"
    signature = digest({"inputs": fingerprint, "features": study["features"],
                        "implementation": sha_file(ROOT/"src/sepsis_warning/feature_engineering.py")})
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature and target.exists():
        return
    rows = pd.read_csv(ROOT/f"processed_data/{split}_rows.csv.gz")
    patients = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    raw = np.load(ROOT/"processed_data/raw_records.npy", mmap_mode="r")
    names = feature_names("combined")
    matrix = np.lib.format.open_memmap(target, mode="w+", dtype=np.float32, shape=(len(rows), len(names)))
    position = 0
    print(f"Building causal features for {split}: {rows.patient_id.nunique()} patients", flush=True)
    for count, identifier in enumerate(rows.patient_id.drop_duplicates(), 1):
        patient = patients.loc[identifier]
        values = raw[int(patient.raw_start):int(patient.raw_stop)]
        reason, eligible, _ = assess_patient(values, config())
        assert reason == "included"
        part = alternative_features(values, config(), study["features"])[eligible]
        matrix[position:position+len(part)] = part
        position += len(part)
        if count % 3000 == 0:
            print(f"  Features: {count} patients complete", flush=True)
    assert position == len(rows)
    matrix.flush()
    save_json(marker, {"signature": signature, "patients": int(rows.patient_id.nunique()), "rows": len(rows),
                      "feature_names": names, "test_sets_evaluated": []})


def prepare_case(spec, fold, row_folds, inputs, study):
    key = representation_key(spec, fold)
    directory = PROCESSED/"cases"/key
    directory.mkdir(parents=True, exist_ok=True)
    train_rows = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    if fold == "full":
        train_indices = np.arange(len(train_rows))
        evaluation_rows = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
        evaluation_indices = np.arange(len(evaluation_rows))
        evaluation_split, stage = "validation", "validation"
    else:
        train_indices, evaluation_indices = np.flatnonzero(row_folds != fold), np.flatnonzero(row_folds == fold)
        evaluation_rows, evaluation_split, stage = train_rows, "train", "cv"
    signature = digest({"inputs": inputs, "representation": spec, "fold": fold,
                        "train_indices": array_digest(train_indices), "evaluation_indices": array_digest(evaluation_indices),
                        "preprocessor": sha_file(ROOT/"src/sepsis_warning/feature_engineering.py"),
                        "spline_parameters": study["splines"] if spec["splines"] else None})
    path = directory/"case.json"
    if path.exists() and json.loads(path.read_text()).get("signature") == signature:
        return key
    raw_train, names = raw_features("train", spec)
    selected = raw_train[train_indices]
    processor = FeaturePreprocessor(names, native=spec["native"], clip=spec["clip"],
                                    spline_cfg=study["splines"] if spec["splines"] else None).fit(selected)
    train_x = np.lib.format.open_memmap(directory/"train_X.npy", mode="w+", dtype=np.float32,
                                       shape=(len(selected), len(processor.names_)))
    for start in range(0, len(selected), 15000):
        train_x[start:start+15000] = processor.transform(selected[start:start+15000])
    train_x.flush()
    del train_x, selected, raw_train
    raw_evaluation, evaluation_names = raw_features(evaluation_split, spec)
    assert names == evaluation_names
    evaluation_x = np.lib.format.open_memmap(directory/"evaluation_X.npy", mode="w+", dtype=np.float32,
                                             shape=(len(evaluation_indices), len(processor.names_)))
    for start in range(0, len(evaluation_indices), 15000):
        evaluation_x[start:start+15000] = processor.transform(raw_evaluation[evaluation_indices[start:start+15000]])
    evaluation_x.flush()
    del evaluation_x, raw_evaluation
    selected_rows, held_rows = train_rows.iloc[train_indices].reset_index(drop=True), evaluation_rows.iloc[evaluation_indices].reset_index(drop=True)
    assert not set(selected_rows.patient_id) & set(held_rows.patient_id)
    selected_rows.to_csv(directory/"train_rows.csv.gz", index=False, compression="gzip")
    held_rows.to_csv(directory/"evaluation_rows.csv.gz", index=False, compression="gzip")
    for name, values in [("train_indices", train_indices), ("evaluation_indices", evaluation_indices),
                         ("train_y", selected_rows.label.to_numpy()), ("evaluation_y", held_rows.label.to_numpy())]:
        np.save(directory/f"{name}.npy", values)
    joblib.dump(processor, directory/"preprocessor.joblib", compress=3)
    save_json(path, {"key": key, "signature": signature, "stage": stage, "fold": fold, "representation": spec,
                    "training_patients": int(selected_rows.patient_id.nunique()), "training_hours": len(selected_rows),
                    "training_event_patients": int(selected_rows.groupby("patient_id").label.max().sum()),
                    "evaluation_patients": int(held_rows.patient_id.nunique()), "evaluation_hours": len(held_rows),
                    "features": len(processor.names_), "preprocessing_fit_on": "this case's training patients only"})
    return key


def run_jobs(jobs, row_folds, inputs, study):
    unique = {}
    for job in jobs:
        unique.setdefault((job["case"], job["model"]), job)
    pending = list(unique.values())
    active, completed = [], 0
    total = len(pending)
    costs = {"lr": 4, "ebm": 3, "lightgbm": 3}
    budget = study["study"]["worker_cpu_budget"]
    last_notice = time.monotonic()
    try:
        while pending or active:
            for item in active.copy():
                job, process, log = item
                if process.poll() is not None:
                    log.close()
                    active.remove(item)
                    if process.returncode:
                        raise RuntimeError(f"Worker failed: {job}\n"+(OUTPUT/job["stage"]/job["case"]/f"{job['model']}.log").read_text()[-6000:])
                    result = json.loads((OUTPUT/job["stage"]/job["case"]/f"{job['model']}_results.json").read_text())
                    completed += 1
                    print(f"  Done {completed}/{total}: {job['profile']} fold={job['fold']} {job['model']} AP={result['evaluation']['auprc_average_precision']:.4f}", flush=True)
            used = sum(costs[item[0]["model"]] for item in active)
            for job in pending.copy():
                if len(active) >= study["study"]["maximum_workers"]:
                    break
                cost = costs[job["model"]]
                if used+cost > budget:
                    continue
                prepare_case(job["representation"], job["fold"], row_folds, inputs, study)
                output = OUTPUT/job["stage"]/job["case"]
                output.mkdir(parents=True, exist_ok=True)
                log = (output/f"{job['model']}.log").open("w")
                process = subprocess.Popen([sys.executable, "-u", str(ROOT/"scripts/fit_feature_model.py"),
                                            "--case", job["case"], "--model", job["model"]], cwd=ROOT,
                                           env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
                active.append((job, process, log))
                pending.remove(job)
                used += cost
                print(f"  Started: {job['profile']} fold={job['fold']} {job['model']}", flush=True)
            if time.monotonic()-last_notice > 35:
                print(f"Progress: {completed}/{total}; active: "+", ".join(f"{j['profile']}/{j['fold']}/{j['model']}" for j, _, _ in active), flush=True)
                last_notice = time.monotonic()
            if active:
                time.sleep(2)
    finally:
        for _, process, log in active:
            process.terminate()
            log.close()


def make_job(profile, model, fold):
    spec = representation(profile, model)
    return {"profile": profile, "model": model, "fold": fold, "representation": spec,
            "case": representation_key(spec, fold), "stage": "validation" if fold == "full" else "cv"}


def main():
    started = time.monotonic()
    study = settings()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT/"reports/history_study/input_fingerprints.json").read_text())
    files = [ROOT/name for name in reference]
    files += [ROOT/"processed_data/raw_records.npy", ROOT/"processed_data/train_rows.csv.gz",
              ROOT/"processed_data/validation_rows.csv.gz", ROOT/"outputs/patient_summary.csv"]
    inputs = {str(path.relative_to(ROOT)): sha_file(path) for path in files}
    assert all(inputs[name] == value for name, value in reference.items())
    save_json(OUTPUT/"input_fingerprints.json", inputs)
    protocol = {"study": study, "profiles": PROFILES, "model_settings": frozen_models(),
                "selection_rule": "Highest mean matched-fold AP among profiles gaining >=0.001 and improving >=2/3 folds; otherwise baseline.",
                "selection_uses_outer_validation_metrics": False, "target_horizon_hours": 6, "latest_value_expiry_hours": 6,
                "preprocessing": "Fitted afresh on each CV-training partition; internal stopping bags share that training-fitted transform.",
                "clinical_combinations": "HR/SBP and SBP-DBP only when readings are <=3h old and <=2h apart.",
                "test_sets_evaluated": [],
                "source_fingerprints": {str(path.relative_to(ROOT)): sha_file(path) for path in
                  [ROOT/"src/sepsis_warning/feature_engineering.py", ROOT/"src/sepsis_warning/optimization.py",
                   ROOT/"scripts/run_feature_optimization.py", ROOT/"scripts/fit_feature_model.py"]}}
    save_json(OUTPUT/"protocol.json", protocol)
    rows = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    patients, row_folds = patient_folds(rows, study["study"]["folds"], study["study"]["fold_seed"])
    patients.to_csv(OUTPUT/"patient_cv_folds.csv", index=False)
    np.save(PROCESSED/"row_folds.npy", row_folds)
    build_combined("train", inputs, study)
    jobs = []
    for profile in study["study"]["profiles"]:
        for fold in range(study["study"]["folds"]):
            for model in PROFILES[profile].get("models", study["study"]["models"]):
                jobs.append(make_job(profile, model, fold))
    save_json(OUTPUT/"cv_manifest.json", jobs)
    print(f"Internal patient CV: {len(jobs)} comparisons, {len({(j['case'], j['model']) for j in jobs})} distinct fits", flush=True)
    run_jobs(jobs, row_folds, inputs, study)
    fold_scores, result_signatures = [], {}
    for job in jobs:
        path = OUTPUT/"cv"/job["case"]/f"{job['model']}_results.json"
        result = json.loads(path.read_text())
        fold_scores.append({"model": job["model"], "profile": job["profile"], "fold": job["fold"],
                            "auprc": result["evaluation"]["auprc_average_precision"]})
        result_signatures[str(path.relative_to(ROOT))] = result["signature"]
    selected, comparisons = choose_profiles(fold_scores, study["study"])
    # This file is written before any new outer-validation features or model metrics are used.
    save_json(OUTPUT/"selection.json", {"selected_profiles": selected, "cv_comparisons": comparisons,
              "cv_result_signatures": result_signatures, "outer_validation_metrics_used": False,
              "source_protocol_digest": digest(protocol), "test_sets_evaluated": []})
    print("Frozen CV selections: "+json.dumps(selected), flush=True)
    if any(PROFILES[profile]["source"] != "baseline" for profile in selected.values()):
        build_combined("validation", inputs, study)
    final_jobs = [make_job(profile, model, "full") for model, profile in selected.items()]
    save_json(OUTPUT/"validation_manifest.json", final_jobs)
    run_jobs(final_jobs, row_folds, inputs, study)
    assert inputs == {str(path.relative_to(ROOT)): sha_file(path) for path in files}
    for name, expected in protocol["source_fingerprints"].items():
        assert sha_file(ROOT/name) == expected
    save_json(OUTPUT/"completed.json", {"cv_comparisons": len(jobs), "distinct_cv_models": len(result_signatures),
              "selected_validation_models": len(final_jobs), "selected_profiles": selected,
              "canonical_inputs_unchanged": True, "test_sets_evaluated": [],
              "elapsed_seconds": round(time.monotonic()-started, 2)})
    print("Feature optimization complete: internal patient CV and selected-model validation.", flush=True)


if __name__ == "__main__":
    main()
