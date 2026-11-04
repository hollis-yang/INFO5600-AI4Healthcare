"""Memory-bounded, fit-patient-only LR preprocessing and additive EBM fits."""
from __future__ import annotations

import gc
import hashlib
import time

import numpy as np
from joblib import parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def valid_matrix(matrix, width):
    assert matrix.ndim == 2 and matrix.shape[1] == width
    assert matrix.dtype == np.dtype(np.float32)
    for start in range(0, len(matrix), 20000):
        assert not np.isinf(matrix[start:start+20000]).any()


def fill_missing_inplace(matrix, statistics, batch_size=20000):
    """Same median replacement as fitted SimpleImputer; retain all columns."""
    for start in range(0, len(matrix), batch_size):
        block = matrix[start:start+batch_size]
        np.copyto(block, statistics[None, :], where=np.isnan(block))
    return matrix


def fit_lr_preprocessor_inplace(matrix, batch_size=20000):
    """Fit medians and scaling on this fit matrix only, then transform in place.

    Fitting each sklearn imputer on one actual feature column avoids a masked
    copy of the complete matrix. Vectorized replacement uses those exact
    statistics. All-missing columns are retained with the sklearn
    keep_empty_features=True value zero. Scaling uses fit rows only, in
    bounded batches, with float64 accumulated mean/variance.
    """
    assert matrix.dtype == np.dtype(np.float32) and matrix.flags.writeable
    n, width = matrix.shape
    statistics = np.empty(width, dtype=np.float64)
    empty = np.empty(width, dtype=bool)
    for feature in range(width):
        column = matrix[:, feature:feature+1]
        empty[feature] = bool(np.isnan(column).all())
        imputer = SimpleImputer(strategy="median", keep_empty_features=True, copy=False)
        imputer.fit(column)
        statistics[feature] = float(imputer.statistics_[0])
    assert np.isfinite(statistics).all()
    fill_missing_inplace(matrix, statistics, batch_size)
    scaler = StandardScaler(copy=False)
    for start in range(0, n, batch_size):
        scaler.partial_fit(matrix[start:start+batch_size])
    assert int(scaler.n_samples_seen_) == n
    for start in range(0, n, batch_size):
        scaler.transform(matrix[start:start+batch_size], copy=False)
    assert np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all()
    imputer = {"statistics_": statistics, "strategy": "median",
               "keep_empty_features": True, "all_missing_features": empty,
               "fit_method": "Per-column sklearn SimpleImputer.fit on fitting patients only"}
    return imputer, scaler


def transform_lr(matrix, imputer, scaler):
    """Return an owned float32 batch, preserving callers' cached raw inputs."""
    result = np.array(matrix, dtype=np.float32, order="C", copy=True)
    fill_missing_inplace(result, imputer["statistics_"])
    scaler.transform(result, copy=False)
    assert np.isfinite(result).all()
    return result


def preprocessor_metadata(bundle, fit_rows):
    if bundle["model_name"] == "ebm":
        return {"type": "native_NaN", "fit_rows": fit_rows,
                "learned_binning": "EBM fit patients only", "imputation": None,
                "standardization": None, "native_random_hour_validation": False}
    imputer, scaler = bundle["imputer"], bundle["scaler"]
    arrays = {"imputer_statistics": imputer["statistics_"],
              "all_missing_features": imputer["all_missing_features"],
              "scaler_mean": scaler.mean_, "scaler_variance": scaler.var_,
              "scaler_scale": scaler.scale_}
    return {"type": "fit_only_median_then_standard_scaler", "fit_rows": fit_rows,
            "statistics_fit_on": "Only fitting patient rows; no calibration or held rows",
            "keep_empty_features": True, "empty_fill_value": 0.,
            "fit_method": imputer["fit_method"], "scaler_samples_seen": int(scaler.n_samples_seen_),
            **{key: value.tolist() for key, value in arrays.items()},
            "array_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                             for key, value in arrays.items()}}


def fit_interpretable(model_name, parameters, feature_names, matrix, labels, fit_patient_ids_sha256):
    started = time.monotonic()
    valid_matrix(matrix, len(feature_names))
    assert len(matrix) == len(labels) and np.isin(labels, [0, 1]).all()
    assert model_name in {"lr", "ebm"}
    parameters = dict(parameters)
    details = {}
    if model_name == "lr":
        assert parameters.get("class_weight") is None
        assert parameters.get("penalty", "l2") == "l2"
        assert parameters.get("solver") == "lbfgs"
        preprocessing_started = time.monotonic()
        imputer, scaler = fit_lr_preprocessor_inplace(matrix)
        details["preprocessing_seconds"] = time.monotonic()-preprocessing_started
        model = LogisticRegression(**parameters)
        with threadpool_limits(limits=4):
            model.fit(matrix, labels)
        details["solver_iterations"] = int(model.n_iter_[0])
        details["solver_converged"] = details["solver_iterations"] < int(parameters["max_iter"])
        if not details["solver_converged"]:
            raise RuntimeError("LR reached max_iter; inspect convergence before reporting it")
        details["threads"] = "BLAS capped at four; lbfgs converts to float64 internally"
        selected_rounds = details["solver_iterations"]
    else:
        from interpret.glassbox import ExplainableBoostingClassifier

        assert parameters["interactions"] == 0
        assert parameters["validation_size"] == 0
        assert parameters["early_stopping_rounds"] == 0
        assert parameters["outer_bags"] == 1 and parameters["inner_bags"] == 0
        imputer = scaler = None
        model = ExplainableBoostingClassifier(feature_names=feature_names,
                                               feature_types=["continuous"]*len(feature_names),
                                               **parameters)
        with parallel_backend("threading", n_jobs=int(parameters.get("n_jobs", 4))), threadpool_limits(limits=1):
            model.fit(matrix, labels)
        assert all(len(term) == 1 for term in model.term_features_)
        assert model.n_features_in_ == len(feature_names)
        selected_rounds = int(parameters["max_rounds"])
        details["executed_boosting_steps"] = np.asarray(model.best_iteration_).tolist()
        details["configured_cycles"] = selected_rounds
        details["native_random_hour_validation"] = False
        details["internal_holdout_rows"] = 0
        details["cyclic_progress_behavior"] = (
            "interpret-core0.7.8 forces cyclic_progress=1.0 when greedy_ratio=0, "
            "including when the public constructor default is False")
        details["threads"] = "n_jobs dispatches bags; one outer bag may train serially; BLAS/native pools capped at one"
    bundle = {"model": model, "model_name": model_name, "feature_names": list(feature_names),
              "imputer": imputer, "scaler": scaler,
              "fit_patient_ids_sha256": fit_patient_ids_sha256,
              "parameters": parameters}
    gc.collect()
    return bundle, selected_rounds, details, time.monotonic()-started


def predict_bundle(bundle, matrix, batch_size=10000):
    model_name, model = bundle["model_name"], bundle["model"]
    assert list(model.classes_) == [0, 1]
    assert matrix.ndim == 2 and matrix.shape[1] == len(bundle["feature_names"])
    result = np.empty(len(matrix), dtype=np.float64)
    for start in range(0, len(matrix), batch_size):
        block = matrix[start:start+batch_size]
        if model_name == "lr":
            block = transform_lr(block, bundle["imputer"], bundle["scaler"])
        with threadpool_limits(limits=4 if model_name == "lr" else 1):
            result[start:start+len(block)] = model.predict_proba(block)[:, 1]
    assert np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all()
    return result
