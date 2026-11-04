"""Fit one binary native tree on a fixed clock-ablation feature view.

Fitting, early stopping and threshold calibration use separate patient roles.
The final fit uses frozen CV-derived rounds, predicts validation only, and has
no test-data path. Native training helpers are reused from the frozen previous
experiment so its learning and stopping settings retain their exact behavior.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.friend_563 import DATA as BASE_DATA
from sepsis_warning.history_study import digest

DELTA_PATH = ROOT / "processed_data/challenge_improvement/train_delta.npy"
HELPER_PATH = ROOT / "scripts/fit_challenge_improvement.py"


def fitting_helpers():
    specification = importlib.util.spec_from_file_location("frozen_clock_tree_helpers", HELPER_PATH)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def role_record(people):
    if people is None:
        return None
    return {"patients": len(people), "event_patients": int(people.has_event.sum()),
            "hours": int(people.hours.sum()), "patient_ids_sha256": digest(people.patient_id.tolist())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, choices=[-1, 0, 1, 2], required=True)
    args = parser.parse_args()
    # XGBoost requires the same pinned package/OpenMP environment as the parent
    # coordinator. Keep direct CLI invocation consistent with scheduled fits.
    if os.environ.get("INFO5600_CLOCK_NATIVE_RUNTIME_READY") != "1":
        from sepsis_warning.friend_563 import runtime_environment
        environment = runtime_environment()
        environment["INFO5600_CLOCK_NATIVE_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", "-u", __file__, *sys.argv[1:]], environment)

    import sepsis_warning.clock_models as study

    started = time.monotonic()
    protocol = study.load_protocol()
    candidate = dict(study.candidate_by_id(args.candidate))
    assert candidate["model"] in {"lightgbm", "xgboost"}
    assert candidate["target"] == "binary" and candidate["selection_metric"] == "auroc"
    candidate.setdefault("auc_patience_rounds", int(protocol.get(
        "auc_early_stopping_rounds", protocol.get("early_stopping_rounds", 50))))
    candidate.setdefault("min_delta", float(protocol.get("min_delta", 1e-5)))
    view = candidate["feature_view"]
    names = protocol["feature_names"][view]
    assert len(names) == len(set(names))
    final = args.fold == -1
    out = (study.OUT / "final" / args.candidate if final else
           study.OUT / "cv" / f"fold{args.fold}" / args.candidate)
    out.mkdir(parents=True, exist_ok=True)
    maximum_rounds = int(candidate.get("max_rounds", protocol.get("maximum_rounds", 1000)))
    rounds = (int(json.loads(study.available(study.OUT / "rounds.json").read_text())[args.candidate])
              if final else maximum_rounds)
    assert 1 <= rounds <= maximum_rounds
    study.available(HELPER_PATH)
    signature = digest({"candidate": candidate, "feature_names": names,
                        "protocol_sha256": content_hash(study.OUT / "protocol.json"),
                        "worker_sha256": content_hash(__file__),
                        "shared_module_sha256": content_hash(study.__file__),
                        "native_helper_sha256": content_hash(HELPER_PATH),
                        "fold": args.fold, "rounds": rounds if final else None})
    model_path = out / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt")
    score_path = out / ("validation_score.npy" if final else "held_score.npy")
    result_path = out / "results.json"
    if result_path.exists():
        previous = json.loads(study.available(result_path).read_text())
        if previous.get("signature") == signature:
            for path, key in [(model_path, "model_sha256"), (score_path, "score_sha256")]:
                assert content_hash(study.available(path)) == previous[key]
            if not final:
                for name, key in (("cal_score.npy", "cal_score_sha256"),
                                  ("cal_patients.csv", "cal_patients_sha256"),
                                  ("held_patients.csv", "held_patients_sha256")):
                    assert content_hash(study.available(out / name)) == previous[key]
            print(f"Reusing {args.candidate}, fold={args.fold}", flush=True)
            return

    fitting = fitting_helpers()
    all_patients = pd.read_csv(study.available(BASE_DATA / "train_patients.csv"))
    original_y = np.load(study.available(BASE_DATA / "train_labels.npy"), mmap_mode="r")
    all_delta = np.load(study.available(DELTA_PATH), mmap_mode="r")
    assert all_patients.patient_id.is_unique
    assert len(original_y) == len(all_delta) == int(all_patients.hours.sum())
    assert np.isin(original_y, [0, 1]).all()
    if final:
        fit_patients = all_patients
        cal_patients = held_patients = None
        fit_indices = cal_indices = held_indices = None
        y_fit = np.asarray(original_y)
    else:
        fit_patients, cal_patients, held_patients = study.cv_roles(args.fold)
        identifiers = [set(part.patient_id) for part in (fit_patients, cal_patients, held_patients)]
        assert not identifiers[0] & identifiers[1]
        assert not identifiers[0] & identifiers[2]
        assert not identifiers[1] & identifiers[2]
        assert set.union(*identifiers) == set(all_patients.patient_id)
        assert held_patients.cv_fold.eq(args.fold).all()
        assert fit_patients.cv_fold.ne(args.fold).all() and cal_patients.cv_fold.ne(args.fold).all()
        fit_indices, cal_indices, held_indices = [study.row_indices(part) for part in
                                                (fit_patients, cal_patients, held_patients)]
        y_fit = original_y[fit_indices]

    x_fit = study.load_features("train", view, indices=fit_indices)
    fitting.valid_matrix(x_fit, len(names))
    assert len(x_fit) == len(y_fit) == int(fit_patients.hours.sum())
    x_cal = None if final else study.load_features("train", view, indices=cal_indices)
    if x_cal is not None:
        fitting.valid_matrix(x_cal, len(names))
        assert len(x_cal) == int(cal_patients.hours.sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, selected, fit_seconds, parameters, history = fitting.native_fit(
            candidate, names, x_fit, y_fit, None, rounds, x_cal,
            None if final else original_y[cal_indices],
            None if final else all_delta[cal_indices])
        temporary = model_path.with_name("model.tmp.ubj" if candidate["model"] == "xgboost" else "model.tmp.txt")
        if candidate["model"] == "xgboost":
            model.save_model(str(temporary))
        else:
            model.save_model(str(temporary), num_iteration=selected)
        os.replace(temporary, model_path)
        del x_fit, y_fit
        gc.collect()
        artifact_hashes = {}
        if final:
            assert selected == rounds
            validation_patients = pd.read_csv(study.available(BASE_DATA / "validation_patients.csv"))
            assert not set(all_patients.patient_id) & set(validation_patients.patient_id)
            x_evaluation = study.load_features("validation", view)
            y_evaluation = np.load(study.available(BASE_DATA / "validation_labels.npy"))
            assert len(x_evaluation) == len(y_evaluation) == int(validation_patients.hours.sum())
            scores = fitting.predict(model, candidate["model"], x_evaluation, selected)
            evaluation, calibration = fitting.ranking(y_evaluation, scores), None
            evaluation_patients = validation_patients
        else:
            cal_scores = fitting.predict(model, candidate["model"], x_cal, selected)
            calibration = fitting.exact_utility_threshold(cal_scores, all_delta[cal_indices])
            calibration.update(fitting.ranking(original_y[cal_indices], cal_scores))
            np.save(out / "cal_score.npy", cal_scores)
            study.local_patients(cal_patients).to_csv(out / "cal_patients.csv", index=False)
            del x_cal
            gc.collect()
            x_evaluation = study.load_features("train", view, indices=held_indices)
            y_evaluation = np.asarray(original_y[held_indices])
            assert len(x_evaluation) == len(y_evaluation) == int(held_patients.hours.sum())
            scores = fitting.predict(model, candidate["model"], x_evaluation, selected)
            evaluation = fitting.ranking(y_evaluation, scores)
            evaluation["utility_score"] = fitting.utility_at_threshold(
                scores, all_delta[held_indices], calibration["threshold"])
            evaluation_patients = study.local_patients(held_patients)
            evaluation_patients.to_csv(out / "held_patients.csv", index=False)
            artifact_hashes = {"cal_score_sha256": content_hash(out / "cal_score.npy"),
                               "cal_patients_sha256": content_hash(out / "cal_patients.csv"),
                               "held_patients_sha256": content_hash(out / "held_patients.csv")}
        assert scores.shape == y_evaluation.shape and np.isfinite(scores).all()
        assert ((scores >= 0) & (scores <= 1)).all()
        np.save(score_path, scores)

    record = {"signature": signature, "candidate": args.candidate,
              "model": candidate["model"], "target": "binary", "feature_view": view, "feature_set": view,
              "fold": args.fold, "parameters": parameters, "selected_rounds": selected,
              "maximum_rounds": maximum_rounds, "selection_metric": "auroc",
              "fit": role_record(fit_patients), "inner_calibration": role_record(cal_patients),
              "evaluation_patients": len(evaluation_patients), "evaluation_hours": len(scores),
              "evaluation": evaluation, "calibration": calibration,
              "feature_count": len(names), "fit_seconds": round(fit_seconds, 3),
              "execution_seconds": round(time.monotonic()-started, 3),
              "model_sha256": content_hash(model_path), "score_sha256": content_hash(score_path),
              **artifact_hashes,
              "source_labels_sha256": content_hash(BASE_DATA / "train_labels.npy"),
              "source_patient_map_sha256": content_hash(BASE_DATA / "train_patients.csv"),
              "native_helper_sha256": content_hash(HELPER_PATH),
              "auc_patience_rounds": candidate["auc_patience_rounds"], "min_delta": candidate["min_delta"],
              "test_read": False, "validation_used_for_fitting": False,
              "outer_holdout_used_for_selection": False,
              "rounds_source": ("Frozen median of three inner-selected CV iteration counts" if final else
                               "Inner patient calibration AUROC; outer holdout selects neither rounds nor threshold"),
              "utility_iteration_history": history, "class_weights": None, "preprocessing": "Native NaN, feature-column selection only",
              "warnings": [str(item.message) for item in caught],
              "environment": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm")}}
    write_json(result_path, record)
    utility = f", US={evaluation['utility_score']:.6f}" if not final else ""
    print(f"Completed {args.candidate}, fold={args.fold}, rounds={selected}, "
          f"AUROC={evaluation['auroc']:.6f}, AP={evaluation['auprc']:.6f}"
          f"{utility}, fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
