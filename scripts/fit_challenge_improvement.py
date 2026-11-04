"""Fit one competition-inspired candidate without reading the test partition.

An outer training fold is scored only after an inner patient calibration set
has selected the stopping iteration and the alarm threshold.  The inner set is
never included in gradient fitting.  Final fits use all training patients and
frozen, CV-derived iteration counts; their validation predictions are for the
coordinator's final threshold calibration only.
"""
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

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.history_study import digest
from sepsis_warning.friend_563 import DATA as BASE_DATA


def ranking(labels, scores):
    """Ranking remains against original SepsisLabel, even for utility targets."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    assert labels.shape == scores.shape and np.isfinite(scores).all()
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def exact_utility_threshold(scores, delta):
    """Exact tie-grouped optimum for the pinned default utility curves.

    Delta is computed before training and remains outside the feature matrix.
    All default marginal utilities are integer multiples of 1/180.  Using
    integer cumulative sums therefore preserves exact ties, including the
    all-negative decision.  Highest threshold wins utility ties.
    """
    scores, delta = np.asarray(scores), np.asarray(delta)
    assert scores.ndim == delta.ndim == 1 and scores.shape == delta.shape
    assert len(scores) and np.isfinite(scores).all() and np.isfinite(delta).all()
    scaled = np.rint(delta * 180).astype(np.int64)
    np.testing.assert_allclose(delta * 180, scaled, rtol=0, atol=1e-9)
    denominator = int(np.maximum(scaled, 0).sum())
    assert denominator > 0, "Utility calibration needs evaluable septic patients"
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores)-1]
    gains = np.r_[0, np.cumsum(scaled[order])[ends]]
    thresholds = np.r_[np.nextafter(float(sorted_scores[0]), np.inf), sorted_scores[ends]]
    best = int(np.argmax(gains))
    return {"threshold": float(thresholds[best]),
            "max_validation_utility": float(gains[best] / denominator),
            "candidate_groups": len(thresholds),
            "positive_hours_predicted": int((scores >= thresholds[best]).sum()),
            "tie_rule": "Highest threshold / fewest alarms among equal-utility groups"}


def utility_at_threshold(scores, delta, threshold):
    scaled = np.rint(np.asarray(delta) * 180).astype(np.int64)
    denominator = int(np.maximum(scaled, 0).sum())
    assert denominator > 0
    return float(scaled[np.asarray(scores) >= threshold].sum() / denominator)


def valid_matrix(matrix, feature_count):
    assert matrix.ndim == 2 and matrix.shape[1] == feature_count
    assert matrix.dtype == np.dtype(np.float32)
    for start in range(0, len(matrix), 20000):
        assert not np.isinf(matrix[start:start+20000]).any()


def predict(model, model_name, matrix, rounds):
    scores = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), 10000):
        batch = matrix[start:start+10000]
        with threadpool_limits(limits=4):
            value = (model.inplace_predict(batch, iteration_range=(0, rounds))
                     if model_name == "xgboost"
                     else model.predict(batch, num_iteration=rounds, num_threads=4))
        scores[start:start+len(batch)] = value
    assert np.isfinite(scores).all()
    return scores


def targets(original_labels, delta, target):
    """Return the fitting target and optional weights, fitted nowhere else."""
    original_labels, delta = np.asarray(original_labels), np.asarray(delta)
    if target == "utility_regression":
        return delta, None, None
    if target == "utility_classification":
        raw_weight = np.abs(delta)
        mean = float(raw_weight.mean())
        assert mean > 0
        # Mean-one weights keep the common L1/L2 strength on a useful scale.
        # The scalar normalization does not change the weighted-loss optimum
        # without regularization.  Record it because regularization is present.
        return (delta > 0).astype(np.int8), raw_weight / mean, mean
    assert target in {"binary", "class_weighted"}, target
    return original_labels, None, None


def configured_parameters(candidate):
    parameters = dict(candidate["parameters"])
    regression = candidate["target"] == "utility_regression"
    utility_selection = candidate["selection_metric"] == "utility"
    if candidate["model"] == "xgboost":
        assert int(parameters.get("nthread", 4)) <= 4
        parameters["objective"] = "reg:squarederror" if regression else "binary:logistic"
        if utility_selection:
            parameters.pop("eval_metric", None)
            parameters["disable_default_eval_metric"] = 1
        else:
            parameters["eval_metric"] = "auc"
        if candidate["target"] == "class_weighted":
            if float(parameters.get("scale_pos_weight", 1.)) == 1.:
                parameters["scale_pos_weight"] = 5.
    else:
        assert int(parameters.get("num_threads", 4)) <= 4
        parameters["objective"] = "regression_l2" if regression else "binary"
        parameters["metric"] = "None" if utility_selection else "auc"
        if candidate["target"] == "class_weighted":
            if float(parameters.get("scale_pos_weight", 1.)) == 1.:
                parameters["scale_pos_weight"] = 5.
    if candidate["target"] in {"utility_regression", "utility_classification"}:
        assert float(parameters.get("scale_pos_weight", 1)) == 1
        assert not parameters.get("is_unbalance", False)
    return parameters


def native_fit(candidate, feature_names, x_train, fit_target, fit_weight,
               maximum_rounds, x_cal=None, original_cal_y=None, cal_delta=None):
    """Native trees; held-out rows cannot influence histogram bin construction."""
    started = time.monotonic()
    model_name = candidate["model"]
    parameters = configured_parameters(candidate)
    has_calibration = x_cal is not None
    utility_selection = candidate["selection_metric"] == "utility"
    history = []
    check_every = int(candidate.get("utility_check_every", 10))
    patience = int(candidate.get("utility_patience_checks", 8))
    min_delta = float(candidate.get("min_delta", 1e-5))

    if model_name == "lightgbm":
        import lightgbm as lgb

        training = lgb.Dataset(x_train, label=fit_target, weight=fit_weight,
                               feature_name=feature_names,
                               params={"max_bin": parameters["max_bin"]},
                               free_raw_data=True)
        options = {}
        if has_calibration and utility_selection:
            class UtilityStopping:
                order = 30
                before_iteration = False

                def __init__(self):
                    self.best_round = 0
                    self.best_score = -np.inf
                    self.stale = 0

                def __call__(self, env):
                    current = env.iteration + 1
                    if current % check_every and current != maximum_rounds:
                        return
                    scores = env.model.predict(x_cal, num_iteration=current,
                                               num_threads=4)
                    calibration = exact_utility_threshold(scores, cal_delta)
                    value = calibration["max_validation_utility"]
                    history.append({"rounds": current, "inner_utility": value,
                                    "threshold": calibration["threshold"]})
                    print(f"[{current}] inner_cal utility={value:.6f}", flush=True)
                    if value > self.best_score + min_delta:
                        self.best_round, self.best_score = current, value
                        self.stale = 0
                    else:
                        self.stale += 1
                    if self.stale >= patience or current == maximum_rounds:
                        raise lgb.callback.EarlyStopException(
                            self.best_round - 1,
                            [("inner_cal", "utility", self.best_score, True)])

            stopping = UtilityStopping()
            options["callbacks"] = [stopping]
        elif has_calibration:
            calibration = lgb.Dataset(x_cal, label=original_cal_y,
                                      reference=training,
                                      feature_name=feature_names, free_raw_data=True)
            options = {"valid_sets": [calibration], "valid_names": ["inner_cal"],
                       "callbacks": [lgb.early_stopping(
                           int(candidate.get("auc_patience_rounds", 50)),
                           first_metric_only=True, verbose=True, min_delta=min_delta),
                           lgb.log_evaluation(50)]}
        with threadpool_limits(limits=4):
            model = lgb.train(parameters, training,
                              num_boost_round=maximum_rounds, **options)
        selected = int(model.best_iteration) if has_calibration else maximum_rounds
        assert selected > 0 and model.current_iteration() >= selected
        del training
    else:
        import xgboost as xgb

        bins = int(parameters["max_bin"])
        training = xgb.QuantileDMatrix(
            x_train, fit_target, weight=fit_weight, missing=np.nan,
            feature_names=feature_names, nthread=4, max_bin=bins)
        callbacks, evaluations = [], []
        if has_calibration and utility_selection:
            class UtilityStopping(xgb.callback.TrainingCallback):
                def __init__(self):
                    self.best_round = 0
                    self.best_score = -np.inf
                    self.stale = 0

                def after_iteration(self, model, epoch, evals_log):
                    current = epoch + 1
                    if current % check_every and current != maximum_rounds:
                        return False
                    scores = model.inplace_predict(x_cal,
                                                   iteration_range=(0, current))
                    calibration = exact_utility_threshold(scores, cal_delta)
                    value = calibration["max_validation_utility"]
                    history.append({"rounds": current, "inner_utility": value,
                                    "threshold": calibration["threshold"]})
                    print(f"[{current}] inner_cal utility={value:.6f}", flush=True)
                    if value > self.best_score + min_delta:
                        self.best_round, self.best_score = current, value
                        self.stale = 0
                    else:
                        self.stale += 1
                    return self.stale >= patience

                def after_training(self, model):
                    assert self.best_round > 0
                    best_model = model[:self.best_round]
                    # Booster slicing deliberately drops early-stopping attrs.
                    # Reattach them to the actual returned, truncated model.
                    best_model.set_attr(best_iteration=str(self.best_round - 1),
                                        best_score=str(self.best_score))
                    return best_model

            callbacks = [UtilityStopping()]
        elif has_calibration:
            calibration = xgb.QuantileDMatrix(
                x_cal, original_cal_y, missing=np.nan,
                feature_names=feature_names, nthread=4, max_bin=bins,
                ref=training)
            evaluations = [(calibration, "inner_cal")]
            callbacks = [xgb.callback.EarlyStopping(
                rounds=int(candidate.get("auc_patience_rounds", 50)),
                metric_name="auc", data_name="inner_cal", maximize=True,
                save_best=True, min_delta=min_delta)]
        with threadpool_limits(limits=4):
            model = xgb.train(parameters, training, num_boost_round=maximum_rounds,
                              evals=evaluations, callbacks=callbacks,
                              verbose_eval=50 if has_calibration else False)
        selected = int(model.best_iteration) + 1 if has_calibration else maximum_rounds
        assert model.num_boosted_rounds() == selected
        del training
    gc.collect()
    return model, selected, time.monotonic() - started, parameters, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--fold", type=int, choices=[-1, 0, 1, 2], required=True)
    args = parser.parse_args()
    # Import after argparse so --help is usable before the coordinator prepares
    # data or writes a protocol.  This worker has no test-file code path.
    import sepsis_warning.challenge_improvement as study

    started = time.monotonic()
    protocol = study.load_protocol()
    candidate = study.candidate_by_id(args.candidate)
    names = protocol["feature_names"][candidate["feature_set"]]
    assert len(names) == len(set(names))
    final = args.fold == -1
    out = (study.OUT / "final" / args.candidate if final else
           study.OUT / "cv" / f"fold{args.fold}" / args.candidate)
    out.mkdir(parents=True, exist_ok=True)
    maximum_rounds = int(candidate.get("max_rounds", protocol.get("maximum_rounds", 1000)))
    rounds = (int(json.loads(study.available(study.OUT / "rounds.json").read_text())[args.candidate])
              if final else maximum_rounds)
    assert 1 <= rounds <= maximum_rounds
    signature = digest({"candidate": candidate, "feature_names": names,
                        "protocol_sha256": content_hash(study.OUT / "protocol.json"),
                        "worker_sha256": content_hash(__file__),
                        "shared_module_sha256": content_hash(study.__file__),
                        "fold": args.fold, "rounds": rounds if final else None})
    model_path = out / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt")
    result_path = out / "results.json"
    score_path = out / ("validation_score.npy" if final else "held_score.npy")
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("signature") == signature:
            study.available(model_path); study.available(score_path)
            assert content_hash(model_path) == previous["model_sha256"]
            assert content_hash(score_path) == previous["score_sha256"]
            print(f"Reusing {args.candidate}, fold={args.fold}", flush=True)
            return

    all_patients = pd.read_csv(study.available(BASE_DATA / "train_patients.csv"))
    original_y = np.load(study.available(BASE_DATA / "train_labels.npy"), mmap_mode="r")
    all_delta = np.load(study.available(study.DATA / "train_delta.npy"), mmap_mode="r")
    assert len(original_y) == len(all_delta) == int(all_patients.hours.sum())
    if final:
        fit_patients = all_patients
        cal_patients = held_patients = None
        fit_indices = np.arange(len(original_y))
        cal_indices = held_indices = None
    else:
        fit_patients, cal_patients, held_patients = study.cv_roles(args.fold)
        ids = [set(part.patient_id) for part in [fit_patients, cal_patients, held_patients]]
        assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
        assert set.union(*ids) == set(all_patients.patient_id)
        assert held_patients.cv_fold.eq(args.fold).all()
        assert fit_patients.cv_fold.ne(args.fold).all() and cal_patients.cv_fold.ne(args.fold).all()
        fit_indices, cal_indices, held_indices = [study.row_indices(part) for part in
                                                [fit_patients, cal_patients, held_patients]]

    x_fit = study.load_features("train", candidate["feature_set"], fit_indices)
    y_fit, fit_weight, weight_normalizer = targets(
        original_y[fit_indices], all_delta[fit_indices], candidate["target"])
    valid_matrix(x_fit, len(names))
    x_cal = None if final else study.load_features("train", candidate["feature_set"], cal_indices)
    if x_cal is not None:
        valid_matrix(x_cal, len(names))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, selected, fit_seconds, parameters, history = native_fit(
            candidate, names, x_fit, y_fit, fit_weight, rounds, x_cal,
            None if final else original_y[cal_indices],
            None if final else all_delta[cal_indices])
        temporary = model_path.with_name("model.tmp.ubj" if candidate["model"] == "xgboost" else "model.tmp.txt")
        if candidate["model"] == "xgboost":
            model.save_model(str(temporary))
        else:
            model.save_model(str(temporary), num_iteration=selected)
        os.replace(temporary, model_path)
        del x_fit, y_fit, fit_weight
        gc.collect()
        if final:
            assert selected == rounds
            validation_patients = pd.read_csv(study.available(BASE_DATA / "validation_patients.csv"))
            assert not set(all_patients.patient_id) & set(validation_patients.patient_id)
            x_evaluation = study.load_features("validation", candidate["feature_set"])
            y_evaluation = np.load(study.available(BASE_DATA / "validation_labels.npy"))
            scores = predict(model, candidate["model"], x_evaluation, selected)
            evaluation, calibration = ranking(y_evaluation, scores), None
            evaluation_patients = validation_patients
        else:
            cal_scores = predict(model, candidate["model"], x_cal, selected)
            calibration = exact_utility_threshold(cal_scores, all_delta[cal_indices])
            calibration["auroc"] = ranking(original_y[cal_indices], cal_scores)["auroc"]
            np.save(out / "cal_score.npy", cal_scores)
            study.local_patients(cal_patients).to_csv(out / "cal_patients.csv", index=False)
            del x_cal
            gc.collect()
            x_evaluation = study.load_features("train", candidate["feature_set"], held_indices)
            y_evaluation = np.asarray(original_y[held_indices])
            scores = predict(model, candidate["model"], x_evaluation, selected)
            evaluation = ranking(y_evaluation, scores)
            evaluation["utility_score"] = utility_at_threshold(
                scores, all_delta[held_indices], calibration["threshold"])
            evaluation_patients = study.local_patients(held_patients)
            evaluation_patients.to_csv(out / "held_patients.csv", index=False)
        if candidate["target"] != "utility_regression":
            assert ((scores >= 0) & (scores <= 1)).all()
        np.save(score_path, scores)

    def role_record(people):
        if people is None:
            return None
        return {"patients": len(people), "event_patients": int(people.has_event.sum()),
                "hours": int(people.hours.sum()),
                "patient_ids_sha256": digest(people.patient_id.tolist())}

    record = {"signature": signature, "candidate": args.candidate,
              "model": candidate["model"], "target": candidate["target"],
              "feature_set": candidate["feature_set"], "fold": args.fold,
              "parameters": parameters, "selected_rounds": selected,
              "maximum_rounds": maximum_rounds, "selection_metric": candidate["selection_metric"],
              "fit": role_record(fit_patients), "inner_calibration": role_record(cal_patients),
              "evaluation_patients": len(evaluation_patients), "evaluation_hours": len(scores),
              "evaluation": evaluation, "calibration": calibration,
              "weight_mean_before_normalization": weight_normalizer,
              "weight_scale": None if weight_normalizer is None else 1. / weight_normalizer,
              "weight_normalization": "Utility classification weights divided by fitting-subset mean only",
              "feature_count": len(names), "fit_seconds": round(fit_seconds, 3),
              "execution_seconds": round(time.monotonic()-started, 3),
              "model_sha256": content_hash(model_path), "score_sha256": content_hash(score_path),
              "test_read": False, "validation_used_for_fitting": False,
              "outer_holdout_used_for_selection": False,
              "score_interpretation": ("Marginal alarm utility regression score; not a probability"
                                       if candidate["target"] == "utility_regression" else
                                       "Target-specific probability; utility classification uses changed labels"),
              "rounds_source": "Frozen median of three inner-selected CV iteration counts" if final else
                               "Inner patient calibration set; outer holdout never selects rounds or threshold",
              "utility_iteration_history": history,
              "warnings": [str(item.message) for item in caught],
              "environment": {name: version(name) for name in
                              ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm"]}}
    write_json(result_path, record)
    print(f"Completed {args.candidate}, fold={args.fold}, rounds={selected}, "
          f"AUROC={evaluation['auroc']:.6f}, AP={evaluation['auprc']:.6f}, "
          f"US={evaluation.get('utility_score', float('nan')):.6f}, fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
