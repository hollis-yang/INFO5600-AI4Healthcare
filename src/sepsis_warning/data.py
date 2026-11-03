"""Auditable cohort rules and causal hourly features."""
from __future__ import annotations

import json
import platform
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    "SepsisLabel",
]
PHYSIOLOGY = COLUMNS[:34]
CONTEXT = COLUMNS[34:40]
STATISTICS = ["latest", "mean_6h", "min_6h", "max_6h", "delta_6h",
              "missing_now", "never_measured", "hours_since", "count_6h"]
FEATURE_NAMES = [f"{name}__{stat}" for stat in STATISTICS for name in PHYSIOLOGY]
FEATURE_NAMES += [f"{name}__current" for name in CONTEXT]
FEATURE_NAMES += [f"{name}__missing_now" for name in CONTEXT]


def config():
    with (ROOT / "configs/baseline.toml").open("rb") as stream:
        return tomllib.load(stream)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    from importlib.metadata import version
    return {"python": sys.version, "platform": platform.platform(), "packages": {
        name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn",
                                        "matplotlib", "joblib", "threadpoolctl"]}}


def assess_patient(values, cfg):
    """Return the primary exclusion reason, eligibility mask, and benchmark onset."""
    n = len(values)
    eligible = np.zeros(n, dtype=bool)
    times, labels = values[:, 39], values[:, 40]
    onset = np.nan
    horizon = cfg["cohort"]["prediction_horizon_hours"]
    reason = "included"
    if not np.isfinite(times).all() or np.any(times < 1) or np.any(times != np.floor(times)):
        reason = "invalid_hour"
    elif np.any(np.diff(times) != 1):
        reason = "nonconsecutive_or_duplicate_hours"
    elif not np.isin(labels, [0, 1]).all():
        reason = "invalid_label"
    elif not np.isfinite(values[:, 34]).all() or np.ptp(values[:, 34]) > 0.01:
        reason = "missing_or_inconsistent_age"
    elif values[0, 34] < cfg["cohort"]["adult_minimum_age"]:
        reason = "age_under_18"
    elif np.any(np.diff(labels) < 0):
        reason = "nonmonotonic_label"
    elif labels[0] == 1:
        reason = "positive_at_record_start"
    elif labels.any():
        onset = float(times[np.flatnonzero(labels == 1)[0]] + horizon)
        if cfg["cohort"]["exclude_onset_beyond_observation"] and onset > times[-1]:
            reason = "onset_beyond_observation"
        else:
            eligible = times < onset
    else:
        eligible = times + horizon <= times[-1] if cfg["cohort"]["require_complete_negative_window"] else np.ones(n, bool)
        if not eligible.any():
            reason = "no_complete_negative_window"
    if reason != "included":
        eligible[:] = False
    return reason, eligible, onset


def causal_features(values, cfg, lab_retention_hours=None):
    """Current values and trailing observed-only statistics; never backfill."""
    physiology = values[:, :34].astype(np.float32, copy=True)
    physiology[~np.isfinite(physiology)] = np.nan
    n, d = physiology.shape
    hours = values[:, 39]
    observed = np.isfinite(physiology)
    last_index = np.maximum.accumulate(np.where(observed, np.arange(n)[:, None], -1), axis=0)
    never = last_index < 0
    safe_index = np.maximum(last_index, 0)
    latest = physiology[safe_index, np.arange(d)]
    since = hours[:, None] - hours[safe_index]
    since = since.astype(np.float32)
    since[never] = np.nan
    expiry = np.full(d, cfg["features"]["forward_fill_hours"], dtype=float)
    if lab_retention_hours is not None:
        expiry[8:34] = lab_retention_hours
    latest[never | (since > expiry)] = np.nan

    window = cfg["features"]["lookback_hours"]
    history = np.full((window, n, d), np.nan, np.float32)
    for lag in range(window):
        if lag < n:
            history[lag, lag:] = physiology[:n-lag]
    valid = np.isfinite(history)
    count = valid.sum(axis=0).astype(np.float32)
    mean = np.divide(np.nansum(history, axis=0), count,
                     out=np.full((n, d), np.nan, np.float32), where=count > 0)
    minimum = np.where(valid, history, np.inf).min(axis=0)
    maximum = np.where(valid, history, -np.inf).max(axis=0)
    minimum[count == 0] = np.nan
    maximum[count == 0] = np.nan
    recent_lag = valid.argmax(axis=0)
    oldest_lag = window - 1 - valid[::-1].argmax(axis=0)
    recent = np.take_along_axis(history, recent_lag[None], axis=0)[0]
    oldest = np.take_along_axis(history, oldest_lag[None], axis=0)[0]
    delta = recent - oldest
    delta[count < 2] = np.nan
    context = values[:, 34:40].copy()
    context[~np.isfinite(context)] = np.nan
    blocks = [latest, mean, minimum, maximum, delta, (~observed).astype(np.float32),
              never.astype(np.float32), since, count, context,
              (~np.isfinite(context)).astype(np.float32)]
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


class TrainPreprocessor:
    """Training-only clipping, imputation and scaling shared by every split."""
    def __init__(self, names, lower=0.005, upper=0.995):
        self.names = list(names)
        self.lower_quantile, self.upper_quantile = lower, upper

    def fit(self, features):
        from sklearn.preprocessing import StandardScaler
        self.keep_ = np.isfinite(features).any(axis=0)
        self.names_ = np.asarray(self.names)[self.keep_].tolist()
        self.dropped_ = np.asarray(self.names)[~self.keep_].tolist()
        selected = features[:, self.keep_]
        self.fill_ = np.nanmedian(selected, axis=0).astype(np.float32)
        self.lower_ = np.full(selected.shape[1], -np.inf, np.float32)
        self.upper_ = np.full(selected.shape[1], np.inf, np.float32)
        self.categorical_ = np.array([name in [f"{c}__current" for c in ["Gender", "Unit1", "Unit2"]]
                                      for name in self.names_])
        for j, name in enumerate(self.names_):
            observed = selected[np.isfinite(selected[:, j]), j]
            if self.categorical_[j]:
                vals, counts = np.unique(observed, return_counts=True)
                self.fill_[j] = vals[counts.argmax()]
            elif name.rsplit("__", 1)[1] in {"latest", "mean_6h", "min_6h", "max_6h", "delta_6h", "hours_since", "current"}:
                self.lower_[j], self.upper_[j] = np.quantile(observed, [self.lower_quantile, self.upper_quantile])
        imputed = self._impute(features)
        self.scaler_ = StandardScaler().fit(imputed)
        return self

    def _impute(self, features):
        selected = features[:, self.keep_].copy()
        np.clip(selected, self.lower_, self.upper_, out=selected)
        missing = np.isnan(selected)
        selected[missing] = np.broadcast_to(self.fill_, selected.shape)[missing]
        assert np.isfinite(selected).all()
        return selected

    def transform(self, features):
        return self.scaler_.transform(self._impute(features), copy=False)


def create_splits(patients, cfg):
    from sklearn.model_selection import train_test_split
    included = patients.loc[patients.included].copy()
    seed = cfg["split"]["random_seed"]
    a = included.loc[included.hospital == cfg["split"]["development_hospital"]]
    fractions = [cfg["split"][name] for name in ["train_fraction", "validation_fraction", "test_fraction"]]
    assert np.isclose(sum(fractions), 1.0) and min(fractions) > 0
    remainder = fractions[1] + fractions[2]
    train_ids, rest_ids = train_test_split(a.index, test_size=remainder, random_state=seed,
                                         stratify=a.has_positive_label)
    validation_ids, test_ids = train_test_split(rest_ids, test_size=fractions[2]/remainder, random_state=seed,
                                              stratify=a.loc[rest_ids, "has_positive_label"])
    included["split"] = "external_test"
    included.loc[train_ids, "split"] = "train"
    included.loc[validation_ids, "split"] = "validation"
    included.loc[test_ids, "split"] = "test"
    assert included.patient_id.is_unique
    assert set(included.loc[included.hospital == cfg["split"]["external_hospital"], "split"]) == {"external_test"}
    return included
