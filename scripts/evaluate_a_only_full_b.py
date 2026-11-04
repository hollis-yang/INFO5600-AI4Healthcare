"""Evaluate frozen A-only baselines on all eligible B patients, without fitting."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.data import assess_patient, causal_features, config
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.utility import normalized_utility, patient_utility

MODELS = ["lr", "ebm", "lightgbm"]
OUT = ROOT / "outputs/a_only_full_b"
DATA = ROOT / "processed_data/a_only_full_b"
REPORT = ROOT / "reports/a_only_full_b"


def main():
    started = time.monotonic()
    for directory in [OUT, DATA, REPORT]:
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "processed_data/raw_records.npy"
    audit_path = ROOT / "outputs/patient_summary.csv"
    split_path = ROOT / "processed_data/splits/patient_splits.csv"
    references = ROOT / "outputs/ab_expansion/references"
    model_paths = {m: references / f"common_{m}_model.joblib" for m in MODELS}
    result_paths = {m: references / f"common_{m}_results.json" for m in MODELS}
    paths = [raw_path, audit_path, split_path, *model_paths.values(), *result_paths.values()]
    for path in paths:
        assert not (getattr(path.stat(), "st_flags", 0) & 0x40000000), path
    fingerprints = {str(p.relative_to(ROOT)): content_hash(p) for p in paths}
    bundles = {m: joblib.load(p) for m, p in model_paths.items()}
    results = {m: json.loads(p.read_text()) for m, p in result_paths.items()}
    thresholds = {}
    for model, bundle in bundles.items():
        assert len(bundle["preprocessor"].names_) == 312
        thresholds[model] = float(results[model]["alerts"]["threshold"])
        assert bundle["threshold"] == thresholds[model]
    split = pd.read_csv(split_path)
    a_train = split.loc[split.hospital.eq("A") & split.split.eq("train")]
    assert len(a_train) == 14091
    identifiers = set(split.loc[split.hospital.eq("B"), "patient_id"])
    assert len(identifiers) == 19743 and not (identifiers & set(a_train.patient_id))
    audit = pd.read_csv(audit_path)
    selected = audit.loc[audit.patient_id.isin(identifiers)].copy()
    assert len(selected) == len(identifiers) and selected.included.all() and selected.hospital.eq("B").all()
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": "Frozen A-only, 14,091 patients; original A 70/15/15 split preserved",
        "evaluation": "All 19,743 included B patients, irrespective of later B split",
        "features": 312,
        "models_refit": False,
        "threshold_fit_on": "Original A_validation only; restored frozen history6h_full thresholds",
        "thresholds": thresholds,
        "ranking_scope": "Eligible pre-onset future-six-hour task",
        "utility_scope": "Full original trajectories of the same included patients; hourly decisions; no suppression",
        "labels_shifted_again": False,
        "A_test_evaluated": False,
        "interpretation": "Exploratory cross-hospital transfer: B has been examined in earlier experiments",
        "input_fingerprints": fingerprints,
        "script_sha256": content_hash(Path(__file__)),
    }
    write_json(OUT / "protocol.json", protocol)
    raw = np.load(raw_path, mmap_mode="r")
    n = int(selected.n_hours.sum())
    labels = np.empty(n, dtype=np.int8)
    eligible = np.empty(n, dtype=bool)
    probabilities = {
        m: np.lib.format.open_memmap(DATA / f"{m}_probability.npy", mode="w+", dtype=np.float64, shape=(n,))
        for m in MODELS
    }
    cfg = config()
    records, blocks, batch_start, position, prefix_checks = [], [], 0, 0, 0

    def predict_pending():
        nonlocal batch_start
        matrix = np.concatenate(blocks)
        assert len(matrix) == position - batch_start
        for name, bundle in bundles.items():
            x = bundle["preprocessor"].transform(matrix)
            assert x.shape == (len(matrix), 312)
            classifier = bundle["model"]
            with threadpool_limits(limits=2):
                p = (classifier.booster_.predict(x, num_threads=2) if name == "lightgbm"
                     else classifier.predict_proba(x)[:, 1])
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            probabilities[name][batch_start:position] = p
        blocks.clear()
        batch_start = position

    for i, patient in enumerate(selected.itertuples(index=False), 1):
        values = raw[int(patient.raw_start):int(patient.raw_stop)]
        reason, mask, onset = assess_patient(values, cfg)
        assert reason == "included" and len(values) == patient.n_hours
        features = causal_features(values, cfg)
        if i in [1, 5000, 10000, 15000, 19743]:
            cutoff = max(1, len(values) // 2)
            np.testing.assert_allclose(causal_features(values[:cutoff], cfg), features[:cutoff], rtol=0, atol=0, equal_nan=True)
            prefix_checks += 1
        end = position + len(values)
        labels[position:end] = values[:, 40]
        eligible[position:end] = mask
        records.append({"patient_id": patient.patient_id, "hospital": "B", "start": position,
                        "stop": end, "hours": len(values), "first_hour": int(values[0, 39]),
                        "has_event": bool(values[:, 40].any())})
        blocks.append(features)
        position = end
        if position - batch_start >= 10000:
            predict_pending()
        if i % 2000 == 0:
            print(f"Predicted {i:,}/19,743 B patients ({position:,} full hours), {time.monotonic()-started:.1f}s", flush=True)
    if blocks:
        predict_pending()
    assert position == n and int(eligible.sum()) == 641470
    patients = pd.DataFrame(records)
    assert int(patients.has_event.sum()) == 905 and int(labels[eligible].sum()) == 5430
    patients.to_csv(DATA / "patients.csv", index=False)
    np.save(DATA / "labels.npy", labels)
    np.save(DATA / "task_eligible.npy", eligible)
    for p in probabilities.values():
        p.flush()

    # Verify all previously scored eligible B-validation/B-test rows, not just samples.
    index = patients.set_index("patient_id")
    checks = []
    for subset in ["B_validation", "B_test"]:
        old_rows = pd.read_csv(ROOT / f"processed_data/ab_expansion/{subset}_rows.csv.gz")
        locations = (old_rows.patient_id.map(index.start).to_numpy()
                     + old_rows.hour.to_numpy() - old_rows.patient_id.map(index.first_hour).to_numpy()).astype(int)
        assert eligible[locations].all()
        np.testing.assert_array_equal(labels[locations], old_rows.label.to_numpy())
        for model in MODELS:
            old = np.load(ROOT / f"outputs/ab_expansion/models/A_only__common/{model}_{subset}_probability.npy")
            np.testing.assert_allclose(probabilities[model][locations], old, rtol=0, atol=1e-12)
            checks.append({"model": model, "subset": subset, "rows": len(old),
                           "max_absolute_difference": float(np.max(np.abs(probabilities[model][locations] - old)))})

    metrics = []
    for model in MODELS:
        probability = probabilities[model]
        primary = ranking_metrics(labels[eligible], probability[eligible])
        complete = ranking_metrics(labels, probability)
        components = []
        for patient in patients.itertuples(index=False):
            start, stop = int(patient.start), int(patient.stop)
            components.append({"patient_id": patient.patient_id,
                               **patient_utility(labels[start:stop], probability[start:stop] >= thresholds[model])})
        frame = pd.DataFrame(components)
        frame.to_csv(OUT / f"{model}_patient_utilities.csv", index=False)
        score = normalized_utility(frame)
        metrics.append({"model": model, "patients": len(patients), "event_patients": 905,
                        "eligible_hours": int(eligible.sum()), "full_hours": n,
                        "auroc": primary["auroc"], "auprc": primary["auprc_average_precision"],
                        "utility_score": score["normalized_utility"], "A_validation_threshold": thresholds[model],
                        "full_trajectory_auroc": complete["auroc"],
                        "full_trajectory_auprc": complete["auprc_average_precision"],
                        "utility_components": score})
        print(json.dumps(metrics[-1]), flush=True)
    # Inputs and frozen references must remain byte-for-byte unchanged.
    for name, expected in fingerprints.items():
        assert content_hash(ROOT / name) == expected
    verification = {"prefix_causality_checks": prefix_checks, "previous_prediction_checks": checks,
                    "input_fingerprints_unchanged": True, "model_training_performed": False,
                    "A_test_evaluated": False}
    write_json(OUT / "verification.json", verification)
    write_json(OUT / "metrics.json", metrics)
    table = pd.DataFrame([{k: v for k, v in row.items() if k != "utility_components"} for row in metrics])
    table.to_csv(REPORT / "metrics.csv", index=False)
    for name, value in [("protocol.json", protocol), ("metrics.json", metrics), ("verification.json", verification)]:
        write_json(REPORT / name, value)
    lines = ["# Frozen A-only models evaluated on all eligible hospital B patients", "",
             "Training uses the original A 70/15/15 split: 14,091 A training patients. No model or preprocessor was fitted again.", "",
             "Evaluation includes all 19,743 eligible B patients (905 events), with 641,470 eligible pre-onset hours. "
             f"Official utility uses all {n:,} original hours of those same patients and the original A-validation-only thresholds.", "",
             "| Model | Pre-onset AUROC | Pre-onset AUPRC | Official utility |", "|---|---:|---:|---:|"]
    lines += [f"| {r['model']} | {r['auroc']:.6f} | {r['auprc']:.6f} | {r['utility_score']:.6f} |" for r in metrics]
    lines += ["", "The original raw labels are not shifted again. Utility uses hourly binary outputs without alarm suppression. "
              "The full-trajectory AUROC/AUPRC are also saved in metrics.csv for a separate complete-record view.", "",
              "B has already been examined in earlier experiments; these are exploratory transfer results, not a new untouched external test. "
              "The A test set remains unevaluated.", "",
              "All saved predictions on the previously evaluated B-validation/B-test subsets match within 1e-12. "
              "Inputs and models remain unchanged; details are recorded in verification.json."]
    (REPORT / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Completed in {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
