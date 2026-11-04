"""Causal additions to the frozen 563-feature PhysioNet representation.

Every input row is an original hourly observation with the 40 challenge input
columns, without SepsisLabel.  No imputation or learned transformation is fitted
here.  ``lag`` preserves the ordering of actual recent observations;
``observed`` summarizes actual measurements rather than filled values; and
``clinical`` combines recent, sufficiently contemporaneous measurements.

The last-two-observation features do not expire.  Their measurement interval and
uncapped latest-measurement age accompany them so a model can distinguish a
recent change from an old one.  All output features depend on the input prefix
ending at the corresponding row, including its current observation.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from sepsis_warning.data import COLUMNS

BLOCKS = ("lag", "observed", "clinical")
VITALS = tuple(COLUMNS[:7])  # EtCO2 is unavailable in the original A data.
PHYSIOLOGY = tuple(COLUMNS[:34])
LAGS = (1, 2, 3, 5)
OBSERVED_STATISTICS = ("last2_observed_delta", "last2_observed_interval_h",
                       "last_observed_age_h", "observed_count_6h", "observed_count_12h")
CLINICAL_MAXIMUM_AGE_H = 2
CLINICAL_MAXIMUM_AGE_DIFFERENCE_H = 1

FEATURE_NAMES_BY_BLOCK = {
    "lag": tuple(f"challenge__{name}__lag{lag}h_{kind}"
                 for lag in LAGS for kind in ("value", "observed") for name in VITALS),
    "observed": tuple(f"challenge__{name}__{statistic}"
                      for statistic in OBSERVED_STATISTICS for name in PHYSIOLOGY),
    "clinical": ("challenge__HR_SBP__shock_index", "challenge__SBP_DBP__pulse_pressure"),
}


def _selected_blocks(blocks: Iterable[str] | str | None) -> tuple[str, ...]:
    selected = BLOCKS if blocks is None else ((blocks,) if isinstance(blocks, str) else tuple(blocks))
    if len(selected) != len(set(selected)) or any(block not in BLOCKS for block in selected):
        raise ValueError(f"Choose distinct feature blocks from {BLOCKS}; got {selected}")
    return selected


def feature_names(blocks: Iterable[str] | str | None = None) -> list[str]:
    """Names in matrix order; a string selects one block, None selects all."""
    return [name for block in _selected_blocks(blocks) for name in FEATURE_NAMES_BY_BLOCK[block]]


def block_slices(blocks: Iterable[str] | str | None = None) -> dict[str, slice]:
    """Column slices for the requested blocks in their requested order."""
    result, position = {}, 0
    for block in _selected_blocks(blocks):
        stop = position + len(FEATURE_NAMES_BY_BLOCK[block])
        result[block] = slice(position, stop)
        position = stop
    return result


def _measurement_history(values: np.ndarray, hours: np.ndarray):
    """Latest/previous real observation indices and their current ages."""
    n, d = values.shape
    observed = np.isfinite(values)
    latest = np.maximum.accumulate(np.where(observed, np.arange(n)[:, None], -1), axis=0)
    prior_latest = np.vstack([np.full((1, d), -1, dtype=np.int64), latest[:-1]])
    previous = prior_latest[np.maximum(latest, 0), np.arange(d)]
    previous[latest < 0] = -1
    age = hours[:, None] - hours[np.maximum(latest, 0)]
    age[latest < 0] = np.nan
    latest_values = values[np.maximum(latest, 0), np.arange(d)]
    latest_values[latest < 0] = np.nan
    return observed, latest, previous, age, latest_values


def engineer_patient(raw40: np.ndarray, blocks: Iterable[str] | str | None = None) -> np.ndarray:
    """Return n×228 float32 causal features, or only the requested blocks.

    Input must contain exactly the 40 original input columns, with finite,
    consecutive integer ICULOS.  Missing and infinite physiological inputs are
    treated as unavailable.  Before-record lag values are NaN and their
    observation flags are zero.  Window counts include the current hour and
    the preceding 5/11 hours.  Patients must be passed separately.
    """
    selected = _selected_blocks(blocks)
    raw = np.asarray(raw40, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 40:
        raise ValueError("raw40 must have shape (hours, 40), excluding SepsisLabel")
    n = len(raw)
    hours = raw[:, 39]
    if (not np.isfinite(hours).all() or np.any(hours != np.floor(hours))
            or np.any(np.diff(hours) != 1)):
        raise ValueError("ICULOS must contain finite consecutive integer hours")
    if not n or not selected:
        return np.empty((n, len(feature_names(selected))), dtype=np.float32)
    values = raw[:, :34].copy()
    values[~np.isfinite(values)] = np.nan
    matrices = {}

    if "lag" in selected:
        parts = []
        for lag in LAGS:
            snapshot = np.full((n, len(VITALS)), np.nan)
            if n > lag:
                snapshot[lag:] = values[:-lag, :len(VITALS)]
            parts.extend([snapshot, np.isfinite(snapshot).astype(np.float64)])
        matrices["lag"] = np.concatenate(parts, axis=1)

    if "observed" in selected or "clinical" in selected:
        observed, latest, previous, age, latest_values = _measurement_history(values, hours)

    if "observed" in selected:
        safe_latest, safe_previous = np.maximum(latest, 0), np.maximum(previous, 0)
        delta = latest_values - values[safe_previous, np.arange(len(PHYSIOLOGY))]
        interval = hours[safe_latest] - hours[safe_previous]
        delta[previous < 0] = np.nan
        interval[previous < 0] = np.nan
        cumulative = np.vstack([np.zeros((1, len(PHYSIOLOGY)), dtype=np.int64),
                                np.cumsum(observed, axis=0)])
        end = np.arange(1, n + 1)
        counts = [cumulative[end] - cumulative[np.maximum(0, end-window)] for window in (6, 12)]
        matrices["observed"] = np.concatenate([delta, interval, age, *counts], axis=1)

    if "clinical" in selected:
        combinations = np.full((n, 2), np.nan)
        for target, left, right in ((0, 0, 3), (1, 3, 5)):
            eligible = (np.isfinite(latest_values[:, left]) & np.isfinite(latest_values[:, right])
                        & (latest_values[:, left] > 0) & (latest_values[:, right] > 0)
                        & (np.maximum(age[:, left], age[:, right]) <= CLINICAL_MAXIMUM_AGE_H)
                        & (np.abs(age[:, left]-age[:, right]) <= CLINICAL_MAXIMUM_AGE_DIFFERENCE_H))
            if target == 0:
                np.divide(latest_values[:, left], latest_values[:, right],
                          out=combinations[:, target], where=eligible)
            else:
                eligible &= latest_values[:, left] >= latest_values[:, right]
                combinations[eligible, target] = latest_values[eligible, left]-latest_values[eligible, right]
        matrices["clinical"] = combinations

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.concatenate([matrices[block] for block in selected], axis=1).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result
