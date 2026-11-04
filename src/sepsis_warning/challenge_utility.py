"""Outcome-derived training targets for the pinned PhysioNet 2019 utility.

These arrays are supervised targets, never input features.  The original
SepsisLabel is already shifted six hours before reconstructed onset.  AUROC/AP
must still be evaluated against that original label, including for regressors.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def _labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Expected a one-dimensional original binary SepsisLabel vector")
    return labels


def patient_utility_delta(labels):
    """Return U(alarm) - U(no alarm) at each hour of one complete record.

    This reproduces the default pinned scorer's marginal decision utility,
    including the avoided false-negative penalty.  In particular, it is not
    the scorer's true-positive reward curve.  Integer multiples of 1/180
    express the default reward differences exactly before float conversion.
    A record ending before inferred onset is also supported, as by the scorer.
    """
    labels = _labels(labels)
    scaled = np.full(len(labels), -9, dtype=np.int64)
    positive = np.flatnonzero(labels)
    if len(positive):
        relative_hour = np.arange(len(labels)) - (int(positive[0]) + 6)
        early = (relative_hour >= -12) & (relative_hour <= -6)
        late = (relative_hour > -6) & (relative_hour <= 3)
        scaled[early] = 30 * (relative_hour[early] + 12)
        # (3-r)/9 - [-2*(r+6)/9] = (r+15)/9, not (3-r)/9.
        scaled[late] = 20 * (relative_hour[late] + 15)
        scaled[relative_hour > 3] = 0
    return scaled.astype(np.float64) / 180.0


def patient_utility_targets(labels):
    """Return regression target and SBU-style weighted classification targets.

    The zero-utility hours receive weight zero (and an arbitrary label zero).
    Absolute weights are deliberately unnormalized.  Any global rescaling
    needed by a learner must be fitted on its training patients and recorded,
    because rescaling can change the effective regularization.
    """
    delta = patient_utility_delta(labels)
    return {
        "delta": delta,
        "classification_label": (delta > 0).astype(np.int8),
        "classification_weight": np.abs(delta),
    }


def utility_targets(labels, patients):
    """Build targets aligned to all rows of a concatenated patient matrix.

    ``patients`` is a DataFrame or an iterable of records with integer ``start``
    and ``stop`` fields.  Rows must partition the complete label vector in
    matrix order without gaps or overlaps.  Construct these targets for the
    full training matrix first, then index the targets with the CV row masks.
    Only labels and patient boundaries are consumed; no feature matrix is read.
    """
    labels = _labels(labels)
    delta = np.empty(len(labels), dtype=np.float64)
    records = patients.itertuples(index=False) if hasattr(patients, "itertuples") else iter(patients)
    position = 0
    for patient in records:
        if isinstance(patient, Mapping):
            start, stop = patient["start"], patient["stop"]
        else:
            start, stop = patient.start, patient.stop
        try:
            a, b = int(start), int(stop)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Patient boundaries must be finite integers") from error
        if a != start or b != stop or a != position or b <= a or b > len(labels):
            raise ValueError("Patient boundaries must partition labels in row order without gaps/overlaps")
        delta[a:b] = patient_utility_delta(labels[a:b])
        position = b
    if position != len(labels):
        raise ValueError("Patient boundaries do not cover every label row")
    return {
        "delta": delta,
        "classification_label": (delta > 0).astype(np.int8),
        "classification_weight": np.abs(delta),
    }
