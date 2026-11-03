"""Fit one native tree model on the friend's frozen full-record 563 features.

CV early stopping uses its held-out training fold, so those CV scores are
development estimates. Final models use frozen CV-derived round counts and
never read validation labels for fitting, early stopping, or model selection.
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
from threadpoolctl import threadpool_limits

import sepsis_warning.friend_563 as study
from sepsis_warning.friend_563 import DATA, OUT, REPORT
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.history_study import digest


def available(path):
    path = Path(path)
    assert path.exists() and path.stat().st_size > 0, path
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Required cached input is a cloud placeholder: {path}")
    return path


def valid_matrix(matrix):
    assert matrix.ndim == 2 and matrix.shape[1] == 563
    assert matrix.dtype == np.dtype(np.float32)
    for start in range(0, len(matrix), 20000):
        assert not np.isinf(matrix[start:start+20000]).any()


def predict(model, model_name, matrix):
    probability = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), 10000):
        batch = matrix[start:start+10000]
        with threadpool_limits(limits=4):
            value = (model.inplace_predict(batch) if model_name == "xgboost"
                     else model.predict(batch, num_threads=4))
        probability[start:start+len(batch)] = value
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    return probability


def native_fit(model_name, parameters, feature_names, x_train, y_train,
               maximum_rounds, stopping_rounds, min_delta, x_held=None, y_held=None):
    """Native missing-value handling; bin cuts are fitted on training data only."""
    started = time.monotonic()
    cv = x_held is not None
    if model_name == "xgboost":
        import xgboost as xgb

        assert parameters["tree_method"] == "hist"
        assert parameters["eval_metric"] == "auc"
        assert int(parameters.get("nthread", 4)) <= 4
        bins = int(parameters["max_bin"])
        training = xgb.QuantileDMatrix(x_train, y_train, missing=np.nan,
                                     feature_names=feature_names, nthread=4,
                                     max_bin=bins)
        callbacks, evaluations = [], []
        if cv:
            held = xgb.QuantileDMatrix(x_held, y_held, missing=np.nan,
                                      feature_names=feature_names, nthread=4,
                                      max_bin=bins, ref=training)
            evaluations = [(held, "held")]
            callbacks = [xgb.callback.EarlyStopping(
                rounds=stopping_rounds, metric_name="auc", data_name="held",
                maximize=True, save_best=True, min_delta=min_delta)]
        with threadpool_limits(limits=4):
            model = xgb.train(parameters, training, num_boost_round=maximum_rounds,
                              evals=evaluations, callbacks=callbacks,
                              verbose_eval=50 if cv else False)
        selected = int(model.best_iteration)+1 if cv else maximum_rounds
        assert int(model.num_boosted_rounds()) == selected
        del training
        if cv:
            del held
    else:
        import lightgbm as lgb

        assert parameters["metric"] == "auc"
        assert int(parameters.get("num_threads", 4)) <= 4
        training = lgb.Dataset(x_train, label=y_train, feature_name=feature_names,
                               params={"max_bin": parameters["max_bin"]},
                               free_raw_data=True)
        options = {}
        if cv:
            held = lgb.Dataset(x_held, label=y_held, reference=training,
                               feature_name=feature_names, free_raw_data=True)
            options = {"valid_sets": [held], "valid_names": ["held"],
                       "callbacks": [lgb.early_stopping(
                           stopping_rounds, first_metric_only=True,
                           verbose=True, min_delta=min_delta), lgb.log_evaluation(50)]}
        with threadpool_limits(limits=4):
            model = lgb.train(parameters, training,
                              num_boost_round=maximum_rounds, **options)
        selected = int(model.best_iteration) if cv else maximum_rounds
        assert selected > 0 and model.current_iteration() >= selected
        del training
        if cv:
            del held
    gc.collect()
    return model, selected, time.monotonic()-started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], required=True)
    parser.add_argument("--fold", type=int, choices=[-1, 0, 1, 2], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(available(OUT / "protocol.json").read_text())
    parameters = dict(protocol["parameters"][args.model])
    names = protocol["feature_names"]
    assert len(names) == len(set(names)) == 563
    final = args.fold == -1
    out = OUT / "final" / args.model if final else OUT / "cv" / f"fold{args.fold}" / args.model
    out.mkdir(parents=True, exist_ok=True)
    rounds = (int(json.loads(available(OUT / "rounds.json").read_text())[args.model])
              if final else int(protocol["maximum_rounds"]))
    assert 1 <= rounds <= int(protocol["maximum_rounds"])
    signature = digest({"protocol": protocol, "worker": content_hash(__file__),
                        "shared_module": content_hash(study.__file__),
                        "model": args.model, "fold": args.fold,
                        "round_count": rounds if final else None})
    model_path = out / ("model.ubj" if args.model == "xgboost" else "model.txt")
    result_path = out / "results.json"
    if result_path.exists():
        record = json.loads(result_path.read_text())
        if record.get("signature") == signature:
            probability_path = out / ("validation_probability.npy" if final else "probability.npy")
            available(model_path); available(probability_path)
            assert content_hash(model_path) == record["model_sha256"]
            print(f"Reusing completed {args.model}, fold={args.fold}", flush=True)
            return
    patients = pd.read_csv(available(DATA / "train_patients.csv"))
    raw = np.load(available(DATA / "train_features.npy"), mmap_mode="r")
    labels = np.load(available(DATA / "train_labels.npy"), mmap_mode="r")
    assert patients.patient_id.is_unique and set(patients.hospital) == {"A", "B"}
    assert set(patients.cv_fold) == {0, 1, 2}
    assert len(raw) == len(labels) == int(patients.hours.sum())
    assert raw.shape[1] == 563 and np.isin(labels, [0, 1]).all()
    np.testing.assert_array_equal(patients.start, np.r_[0, patients.stop.to_numpy()[:-1]])
    np.testing.assert_array_equal(patients.stop-patients.start, patients.hours)
    assert int(patients.stop.iloc[-1]) == len(raw)
    if final:
        training_patients = patients
        x_train, y_train = raw, labels
        x_held = y_held = held_patients = None
    else:
        row_folds = np.repeat(patients.cv_fold.to_numpy(), patients.hours.to_numpy())
        train_indices = np.flatnonzero(row_folds != args.fold)
        held_indices = np.flatnonzero(row_folds == args.fold)
        training_patients = patients.loc[patients.cv_fold != args.fold]
        held_patients = patients.loc[patients.cv_fold == args.fold].copy()
        assert not set(training_patients.patient_id) & set(held_patients.patient_id)
        assert len(training_patients)+len(held_patients) == len(patients)
        x_train, y_train = raw[train_indices], labels[train_indices]
        x_held, y_held = raw[held_indices], labels[held_indices]
        held_patients["stop"] = held_patients.hours.cumsum()
        held_patients["start"] = held_patients.stop-held_patients.hours
        held_patients.to_csv(out / "held_patients.csv", index=False)
        np.save(out / "labels.npy", y_held)
        del train_indices, held_indices, row_folds
    valid_matrix(x_train)
    if x_held is not None:
        valid_matrix(x_held)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, selected, fit_seconds = native_fit(
            args.model, parameters, names, x_train, y_train, rounds,
            int(protocol["early_stopping_rounds"]), float(protocol["min_delta"]),
            x_held, y_held)
        if final:
            assert selected == rounds
        temporary = model_path.with_name("model.tmp")
        if args.model == "xgboost":
            model.save_model(str(temporary.with_suffix(".ubj")))
            os.replace(temporary.with_suffix(".ubj"), model_path)
        else:
            model.save_model(str(temporary), num_iteration=selected)
            os.replace(temporary, model_path)
        if final:
            del raw, labels, x_train, y_train
            gc.collect()
            validation = np.load(available(DATA / "validation_features.npy"), mmap_mode="r")
            validation_labels = np.load(available(DATA / "validation_labels.npy"), mmap_mode="r")
            validation_patients = pd.read_csv(available(DATA / "validation_patients.csv"))
            assert not set(patients.patient_id) & set(validation_patients.patient_id)
            assert len(validation) == len(validation_labels) == int(validation_patients.hours.sum())
            probability = predict(model, args.model, validation)
            np.save(out / "validation_probability.npy", probability)
            evaluation = {"validation": ranking_metrics(validation_labels, probability)}
            evaluation_patients = len(validation_patients)
            evaluation_hours = len(validation_labels)
        else:
            probability = predict(model, args.model, x_held)
            np.save(out / "probability.npy", probability)
            evaluation = ranking_metrics(y_held, probability)
            evaluation_patients, evaluation_hours = len(held_patients), len(y_held)
    record = {"signature": signature, "model": args.model, "fold": args.fold,
              "parameters": parameters, "selected_rounds": selected,
              "maximum_rounds": int(protocol["maximum_rounds"]),
              "training_patients": len(training_patients),
              "training_event_patients": int(training_patients.has_event.sum()),
              "training_hours": int(training_patients.hours.sum()),
              "evaluation_patients": evaluation_patients, "evaluation_hours": evaluation_hours,
              "evaluation": evaluation, "fit_seconds": round(fit_seconds, 3),
              "execution_seconds": round(time.monotonic()-started, 3),
              "feature_count": 563, "model_sha256": content_hash(model_path),
              "test_read": False, "validation_used_for_fitting": False,
              "training_hospitals": sorted(training_patients.hospital.unique().tolist()),
              "preprocessing": "Unchanged causal 563 features; native NaN; no clipping/imputation/scaling",
              "rounds_source": "Frozen median of three training CV best-round counts" if final
                               else "Held-out training fold AUROC early stopping",
              "warnings": [str(w.message) for w in caught],
              "environment": {name: version(name) for name in [
                  "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm"]}}
    write_json(result_path, record)
    score = evaluation["validation"] if final else evaluation
    print(f"Completed {args.model}, fold={args.fold}, rounds={selected}, "
          f"AUROC={score['auroc']:.6f}, AP={score['auprc_average_precision']:.6f}, "
          f"fit={fit_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
