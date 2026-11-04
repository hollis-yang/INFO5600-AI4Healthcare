"""Causal score transformations for training-only alarm-policy development."""
from __future__ import annotations

import numpy as np

POLICIES = ("raw", "mean2", "mean3", "ema50", "ema80", "min2", "max2", "max3")


def transform_patient(scores, policy):
    """Use only the current/past scores; short prefixes use available history.

    min2 keeps the first score unchanged. It requires two high scores from
    hour two onward, rather than silently declaring the first hour negative.
    max2/max3 retain a recent risk peak; these are hourly challenge outputs,
    not an episode cooldown or a change to the underlying fitted model.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or policy not in POLICIES:
        raise ValueError("Expected finite one-dimensional scores and a known policy")
    result = values.copy()
    if policy == "raw" or not len(values):
        return result
    if policy.startswith("ema"):
        alpha = .5 if policy == "ema50" else .8
        for t in range(1, len(values)):
            result[t] = alpha*values[t] + (1-alpha)*result[t-1]
    else:
        window = int(policy[-1])
        function = {"mean": np.mean, "min": np.min, "max": np.max}[policy[:-1]]
        for t in range(len(values)):
            result[t] = function(values[max(0, t-window+1):t+1])
    return result


def transform_scores(scores, patients, policy):
    values = np.asarray(scores, dtype=np.float64)
    result = np.empty_like(values)
    position = 0
    for p in patients.itertuples(index=False):
        a, b = int(p.start), int(p.stop)
        if a != position or b <= a or b > len(values):
            raise ValueError("Patient boundaries must partition the score vector")
        result[a:b] = transform_patient(values[a:b], policy)
        position = b
    if position != len(values):
        raise ValueError("Patient map does not cover scores")
    return result


def exact_threshold(scores, delta):
    values, delta = np.asarray(scores), np.asarray(delta)
    if values.ndim != 1 or values.shape != delta.shape or not len(values) or not np.isfinite(values).all():
        raise ValueError("Aligned finite scores and utility differences required")
    scaled = np.rint(delta*180).astype(np.int64)
    np.testing.assert_allclose(delta*180, scaled, rtol=0, atol=1e-9)
    denominator = int(np.maximum(scaled, 0).sum())
    if denominator <= 0:
        raise ValueError("Calibration requires utility-evaluable event patients")
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(values)-1]
    gains = np.r_[0, np.cumsum(scaled[order])[ends]]
    thresholds = np.r_[np.nextafter(float(sorted_scores[0]), np.inf), sorted_scores[ends]]
    best = int(np.argmax(gains))
    return {"threshold": float(thresholds[best]), "utility": float(gains[best]/denominator)}


def decision_counts(labels, scores, patients, threshold, delta):
    alarm = np.asarray(scores) >= threshold
    scaled = np.rint(np.asarray(delta)*180).astype(np.int64)
    result = {"utility_gain_scaled": int(scaled[alarm].sum()),
              "utility_denominator_scaled": int(np.maximum(scaled, 0).sum()),
              "event_patients": 0, "detected_pre6": 0, "nonseptic_patients": 0,
              "nonseptic_alarmed_patients": 0, "nonseptic_alarm_hours": 0,
              "nonseptic_hours": 0}
    for p in patients.itertuples(index=False):
        a, b = int(p.start), int(p.stop)
        y, pred = labels[a:b], alarm[a:b]
        positive = np.flatnonzero(y)
        if len(positive):
            result["event_patients"] += 1
            first = int(positive[0])
            result["detected_pre6"] += int(pred[first:first+6].any())
        else:
            result["nonseptic_patients"] += 1
            result["nonseptic_alarmed_patients"] += int(pred.any())
            result["nonseptic_alarm_hours"] += int(pred.sum())
            result["nonseptic_hours"] += len(y)
    return result
