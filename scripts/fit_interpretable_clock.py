"""Fit LR or an additive EBM on fixed patient roles without reading test data."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.history_study import digest
from sepsis_warning.friend_563 import DATA as BASE_DATA
from sepsis_warning.clock_interpretable import (fit_interpretable, predict_bundle,
                                               preprocessor_metadata, valid_matrix)
import sepsis_warning.clock_interpretable as implementation


def ranking(labels, scores):
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def utilities(scores, delta, threshold):
    scaled = np.rint(np.asarray(delta)*180).astype(np.int64)
    denominator = int(np.maximum(scaled, 0).sum())
    assert denominator > 0
    return float(scaled[np.asarray(scores) >= threshold].sum()/denominator)


def threshold(scores, delta):
    # Exact tie groups, including the inactive decision; integer utilities
    # preserve true ties in the pinned official default reward curve.
    scores, delta = np.asarray(scores), np.asarray(delta)
    assert scores.shape == delta.shape and np.isfinite(scores).all()
    scaled = np.rint(delta*180).astype(np.int64)
    np.testing.assert_allclose(scaled, delta*180, rtol=0, atol=1e-9)
    denominator = int(np.maximum(scaled, 0).sum())
    assert denominator > 0
    order = np.argsort(-scores, kind="stable")
    values = scores[order]
    ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(scores)-1]
    gains = np.r_[0, np.cumsum(scaled[order])[ends]]
    thresholds = np.r_[np.nextafter(float(values[0]), np.inf), values[ends]]
    best = int(np.argmax(gains))
    return {"threshold": float(thresholds[best]),
            "max_validation_utility": float(gains[best]/denominator),
            "candidate_groups": len(thresholds),
            "positive_hours_predicted": int((scores >= thresholds[best]).sum()),
            "tie_rule": "Highest threshold / fewest alarms among equal-utility groups"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=[-1, 0, 1, 2])
    args = parser.parse_args()
    import sepsis_warning.clock_models as study

    started = time.monotonic()
    protocol = study.load_protocol()
    candidate = study.candidate_by_id(args.candidate)
    assert candidate["model"] in {"lr", "ebm"} and candidate.get("target", "binary") == "binary"
    view = candidate["feature_view"]
    names = protocol["feature_names"][view]
    assert len(names) == len(set(names))
    final = args.fold == -1
    out = study.OUT / "final" / args.candidate if final else study.OUT / "cv" / f"fold{args.fold}" / args.candidate
    out.mkdir(parents=True, exist_ok=True)
    signature = digest({"candidate": candidate, "feature_names": names,
                        "protocol_sha256": content_hash(study.OUT / "protocol.json"),
                        "worker_sha256": content_hash(__file__),
                        "implementation_sha256": content_hash(implementation.__file__),
                        "shared_module_sha256": content_hash(study.__file__), "fold": args.fold})
    model_path, result_path = out / "model.joblib", out / "results.json"
    score_path = out / ("validation_score.npy" if final else "held_score.npy")
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("signature") == signature:
            for path in (model_path, score_path, out / "preprocessor.json"):
                study.available(path)
            assert content_hash(model_path) == previous["model_sha256"]
            assert content_hash(score_path) == previous["score_sha256"]
            assert content_hash(out / "preprocessor.json") == previous["preprocessor_sha256"]
            print(f"Reusing {args.candidate}, fold={args.fold}", flush=True)
            return
    people = pd.read_csv(study.available(BASE_DATA / "train_patients.csv"))
    labels = np.load(study.available(BASE_DATA / "train_labels.npy"), mmap_mode="r")
    delta = np.load(study.available(ROOT / "processed_data/challenge_improvement/train_delta.npy"), mmap_mode="r")
    assert len(labels) == len(delta) == int(people.hours.sum())
    if final:
        fit_people, cal_people, held_people = people, None, None
        fit_indices = np.arange(len(labels))
    else:
        fit_people, cal_people, held_people = study.cv_roles(args.fold)
        sets = [set(part.patient_id) for part in (fit_people, cal_people, held_people)]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
        assert set.union(*sets) == set(people.patient_id)
        fit_indices = study.row_indices(fit_people)
        cal_indices, held_indices = study.row_indices(cal_people), study.row_indices(held_people)
    fit_id_hash = digest(fit_people.patient_id.tolist())
    matrix = study.load_features("train", view, indices=fit_indices)
    valid_matrix(matrix, len(names))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle, selected, details, fit_seconds = fit_interpretable(
            candidate["model"], candidate["parameters"], names, matrix,
            np.asarray(labels[fit_indices]), fit_id_hash)
        preprocessing = preprocessor_metadata(bundle, len(fit_indices))
        write_json(out / "preprocessor.json", preprocessing)
        temporary = model_path.with_name("model.tmp.joblib")
        joblib.dump(bundle, temporary, compress=0)
        os.replace(temporary, model_path)
        del matrix
        gc.collect()
        calibration = None
        if not final:
            cal_matrix = study.load_features("train", view, indices=cal_indices)
            cal_scores = predict_bundle(bundle, cal_matrix)
            np.save(out / "cal_score.npy", cal_scores)
            study.local_patients(cal_people).to_csv(out / "cal_patients.csv", index=False)
            calibration = threshold(cal_scores, delta[cal_indices])
            calibration.update(ranking(labels[cal_indices], cal_scores))
            del cal_matrix
            gc.collect()
            evaluation_matrix = study.load_features("train", view, indices=held_indices)
            scores = predict_bundle(bundle, evaluation_matrix)
            evaluation = ranking(labels[held_indices], scores)
            evaluation["utility_score"] = utilities(scores, delta[held_indices], calibration["threshold"])
            evaluated_people = study.local_patients(held_people)
            evaluated_people.to_csv(out / "held_patients.csv", index=False)
        else:
            evaluated_people = pd.read_csv(study.available(BASE_DATA / "validation_patients.csv"))
            assert not set(people.patient_id)&set(evaluated_people.patient_id)
            evaluation_matrix = study.load_features("validation", view)
            evaluation_labels = np.load(study.available(BASE_DATA / "validation_labels.npy"))
            scores = predict_bundle(bundle, evaluation_matrix)
            evaluation = ranking(evaluation_labels, scores)
        np.save(score_path, scores)

    def role_record(part):
        if part is None:
            return None
        return {"patients": len(part), "event_patients": int(part.has_event.sum()),
                "hours": int(part.hours.sum()), "patient_ids_sha256": digest(part.patient_id.tolist())}

    record = {"signature": signature, "candidate": args.candidate, "model": candidate["model"],
              "target": "binary", "feature_view": view, "feature_set": view, "fold": args.fold,
              "parameters": candidate["parameters"], "selected_rounds": selected,
              "maximum_rounds": candidate["parameters"].get("max_rounds"),
              "selection_metric": "fixed_configuration", "fit": role_record(fit_people),
              "inner_calibration": role_record(cal_people), "evaluation_patients": len(evaluated_people),
              "evaluation_hours": len(scores), "evaluation": evaluation, "calibration": calibration,
              "feature_count": len(names), "fit_seconds": round(fit_seconds, 3),
              "execution_seconds": round(time.monotonic()-started, 3),
              "model_sha256": content_hash(model_path), "score_sha256": content_hash(score_path),
              "preprocessor_sha256": content_hash(out / "preprocessor.json"),
              "test_read": False, "validation_used_for_fitting": False,
              "outer_holdout_used_for_selection": False, "training_details": details,
              "rounds_source": ("Fixed EBM max_rounds candidate; no native early stopping" if candidate["model"] == "ebm"
                                else "LR solver convergence; C fixed, iteration count is not a tuning budget"),
              "warnings": [str(item.message) for item in caught],
              "environment": {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "interpret-core"]}}
    if not final:
        for name in ("cal_score.npy", "cal_patients.csv", "held_patients.csv"):
            record[name.removesuffix(".npy").removesuffix(".csv") + "_sha256"] = content_hash(out / name)
    write_json(result_path, record)
    summary = f"Completed {args.candidate}, fold={args.fold}, AUROC={evaluation['auroc']:.6f}, AP={evaluation['auprc']:.6f}"
    if not final:
        summary += f", US={evaluation['utility_score']:.6f}"
    print(summary + f", fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
