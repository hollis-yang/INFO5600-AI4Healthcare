"""Causal, observed-only alternatives to the original six-hour features."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sepsis_warning.data import COLUMNS, FEATURE_NAMES, TrainPreprocessor, causal_features

LABS = COLUMNS[8:34]
VITALS = COLUMNS[:7]  # EtCO2 is entirely unavailable in the original A training data.
LAB_SUFFIXES = ["last2_delta_24h", "last2_slope_24h", "last2_interval_24h", "count_24h"]
VITAL_SUFFIXES = ["slope_3h", "slope_12h", "std_12h", "deviation_prior12h", "count_12h"]
COMBINATIONS = ["HR_SBP__ratio", "SBP_DBP__difference"]
DROP_CONTEXT = {f"{variable}__{suffix}" for variable in ["Unit1", "Unit2", "HospAdmTime", "ICULOS"]
                for suffix in ["current", "missing_now"]}
PROFILES = {
    "baseline": {"source": "baseline", "native": False, "clip": True},
    "native_missing": {"source": "baseline", "native": True, "clip": True},
    "lab_trends": {"source": "labs", "native": True, "clip": True},
    "vital_dynamics": {"source": "dynamic", "native": True, "clip": True},
    "clinical_combinations": {"source": "combined", "native": True, "clip": True},
    "without_context_time": {"source": "combined", "native": True, "clip": True, "drop_context": True},
    "without_clipping": {"source": "combined", "native": True, "clip": False},
    "lr_splines": {"source": "combined", "native": False, "clip": True, "splines": True, "models": ["lr"]},
}


def feature_names(source):
    if source == "baseline":
        return list(FEATURE_NAMES)
    removed = {f"{variable}__{suffix}" for variable in LABS
               for suffix in ["mean_6h", "min_6h", "max_6h", "delta_6h"]}
    names = [name for name in FEATURE_NAMES if name not in removed]
    names += [f"{variable}__{suffix}" for suffix in LAB_SUFFIXES for variable in LABS]
    if source in ["dynamic", "combined"]:
        names += [f"{variable}__{suffix}" for suffix in VITAL_SUFFIXES for variable in VITALS]
    if source == "combined":
        names += COMBINATIONS
    return names


def rolling_observations(values, window):
    """Stack original hourly observations in (t-window, t], never filled values."""
    n, d = values.shape
    result = np.full((window, n, d), np.nan, np.float32)
    for lag in range(min(window, n)):
        result[lag, lag:] = values[:n-lag]
    return result


def observed_slope(history):
    valid = np.isfinite(history)
    count = valid.sum(axis=0)
    time = -np.arange(len(history), dtype=float)[:, None, None]
    sx = np.sum(np.where(valid, time, 0), axis=0)
    sy = np.nansum(history, axis=0, dtype=float)
    sxx = np.sum(np.where(valid, time*time, 0), axis=0)
    sxy = np.sum(np.where(valid, time*history, 0), axis=0)
    denominator = count*sxx-sx*sx
    result = np.full(count.shape, np.nan, dtype=np.float32)
    np.divide(count*sxy-sx*sy, denominator, out=result, where=(count >= 2) & (denominator > 0))
    return result


def alternative_features(values, cfg, feature_cfg, source="combined"):
    base = causal_features(values, cfg)  # Original six-hour latest-value expiry is frozen.
    if source == "baseline":
        return base
    names = feature_names(source)
    physiology = values[:, :34].astype(np.float32, copy=True)
    physiology[~np.isfinite(physiology)] = np.nan
    n = len(values)
    hours = values[:, 39]
    labs = physiology[:, 8:34]
    obs = np.isfinite(labs)
    last = np.maximum.accumulate(np.where(obs, np.arange(n)[:, None], -1), axis=0)
    before = np.vstack([np.full((1, len(LABS)), -1), last[:-1]])
    previous = before[np.maximum(last, 0), np.arange(len(LABS))]
    previous[last < 0] = -1
    safe_last, safe_previous = np.maximum(last, 0), np.maximum(previous, 0)
    interval = hours[safe_last]-hours[safe_previous]
    available = (previous >= 0) & (hours[:, None]-hours[safe_previous] < feature_cfg["lab_trend_window_hours"])
    delta = labs[safe_last, np.arange(len(LABS))]-labs[safe_previous, np.arange(len(LABS))]
    delta[~available] = np.nan
    rate = np.divide(delta, interval, out=np.full_like(delta, np.nan), where=available & (interval > 0))
    interval = np.where(available, interval, np.nan).astype(np.float32)
    counts = np.cumsum(obs, axis=0)
    width = feature_cfg["lab_trend_window_hours"]
    if n > width:
        counts[width:] -= np.cumsum(obs, axis=0)[:-width]
    removed = {f"{variable}__{suffix}" for variable in LABS
               for suffix in ["mean_6h", "min_6h", "max_6h", "delta_6h"]}
    blocks = [base[:, [name not in removed for name in FEATURE_NAMES]], delta, rate, interval, counts]
    if source in ["dynamic", "combined"]:
        vitals = physiology[:, :7]
        histories = {window: rolling_observations(vitals, window) for window in feature_cfg["vital_slope_windows"]}
        slopes = [observed_slope(histories[window]) for window in feature_cfg["vital_slope_windows"]]
        history = histories[feature_cfg["vital_variability_window"]]
        count = np.isfinite(history).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std = np.nanstd(history, axis=0, ddof=0)
            prior = rolling_observations(vitals, feature_cfg["personal_baseline_window"])
            prior = np.concatenate([np.full_like(prior[:, :1], np.nan), prior[:, :-1]], axis=1)
            reference = np.nanmedian(prior, axis=0)
        std[count < 2] = np.nan
        reference[np.isfinite(prior).sum(axis=0) < feature_cfg["personal_baseline_minimum_observations"]] = np.nan
        deviation = base[:, :7]-reference
        blocks.extend([*slopes, std, deviation, count])
    if source == "combined":
        latest = base[:, :7]
        since = base[:, [FEATURE_NAMES.index(f"{variable}__hours_since") for variable in VITALS]]
        combinations = []
        for left, right, ratio in [(0, 3, True), (3, 5, False)]:
            eligible = np.isfinite(latest[:, left]) & np.isfinite(latest[:, right])
            eligible &= np.maximum(since[:, left], since[:, right]) <= feature_cfg["combination_maximum_age_hours"]
            eligible &= np.abs(since[:, left]-since[:, right]) <= feature_cfg["combination_maximum_age_difference_hours"]
            result = np.full(n, np.nan, np.float32)
            if ratio:
                eligible &= latest[:, right] > 0
                np.divide(latest[:, left], latest[:, right], out=result, where=eligible)
            else:
                result[eligible] = latest[eligible, left]-latest[eligible, right]
            combinations.append(result[:, None])
        blocks.extend(combinations)
    matrix = np.concatenate(blocks, axis=1).astype(np.float32)
    assert matrix.shape == (n, len(names)) and not np.isinf(matrix).any()
    return matrix


class FeaturePreprocessor(TrainPreprocessor):
    """Fold-fitted transforms; native missingness changes no finite scaled values."""
    def __init__(self, names, native=False, clip=True, spline_cfg=None):
        super().__init__(names)
        self.native, self.clip, self.spline_cfg = native, clip, spline_cfg

    def fit(self, features):
        self.keep_ = np.isfinite(features).any(axis=0)
        self.base_names_ = np.asarray(self.names)[self.keep_].tolist()
        self.names_ = list(self.base_names_)
        self.dropped_ = np.asarray(self.names)[~self.keep_].tolist()
        selected = features[:, self.keep_]
        self.fill_ = np.nanmedian(selected, axis=0).astype(np.float32)
        self.lower_ = np.full(selected.shape[1], -np.inf, np.float32)
        self.upper_ = np.full(selected.shape[1], np.inf, np.float32)
        self.categorical_ = np.array([name in [f"{v}__current" for v in ["Gender", "Unit1", "Unit2"]]
                                      for name in self.base_names_])
        old_continuous = {"latest", "mean_6h", "min_6h", "max_6h", "delta_6h", "hours_since", "current"}
        new_continuous = set(LAB_SUFFIXES[:3]+VITAL_SUFFIXES[:4]+["ratio", "difference"])
        for j, name in enumerate(self.base_names_):
            observed = selected[np.isfinite(selected[:, j]), j]
            if self.categorical_[j]:
                vals, counts = np.unique(observed, return_counts=True)
                self.fill_[j] = vals[counts.argmax()]
            elif self.clip and name.rsplit("__", 1)[1] in old_continuous | new_continuous:
                self.lower_[j], self.upper_[j] = np.quantile(observed, [.005, .995])
        transformed = self._impute(features)
        self.scaler_ = StandardScaler().fit(transformed)
        self.spline_ = None
        if self.spline_cfg:
            assert not self.native
            self.spline_indices_ = [self.base_names_.index(f"{v}__latest") for v in self.spline_cfg["variables"]]
            parameters = {key: value for key, value in self.spline_cfg.items() if key != "variables"}
            self.spline_ = SplineTransformer(**parameters)
            basis = self.spline_.fit_transform(self.scaler_.transform(transformed)[:, self.spline_indices_])
            self.spline_scaler_ = StandardScaler().fit(basis)
            self.names_ += self.spline_.get_feature_names_out([self.base_names_[i] for i in self.spline_indices_]).tolist()
        return self

    def transform(self, features):
        result = self.scaler_.transform(self._impute(features), copy=False)
        if self.native:
            result[np.isnan(features[:, self.keep_])] = np.nan
        if self.spline_ is not None:
            basis = self.spline_.transform(result[:, self.spline_indices_])
            result = np.concatenate([result, self.spline_scaler_.transform(basis)], axis=1)
        return result.astype(np.float32, copy=False)
