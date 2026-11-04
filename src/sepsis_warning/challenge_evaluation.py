"""Full-record ranking, official utility, and paired patient uncertainty.

Scores are finite ranking outputs and need not be sepsis probabilities.  In
particular, marginal-utility regression scores are never passed to Brier loss.
Bootstrap resamples whole patients and keeps both fitted models' thresholds
fixed.  These intervals quantify evaluation-sample variation conditional on
the fitted models; they do not include training or model-selection variation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis_warning.challenge_utility import utility_targets
from sepsis_warning.utility import patient_utility


def _inputs(labels, scores, patients):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Expected nonempty, aligned one-dimensional labels and scores")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Original labels must be binary and scores finite")
    patients = pd.DataFrame(patients).copy().reset_index(drop=True)
    if not {"start", "stop"}.issubset(patients.columns):
        raise ValueError("Patient map requires start and stop")
    delta = utility_targets(labels, patients)["delta"]  # validates full, ordered coverage
    if "patient_id" in patients and not patients.patient_id.is_unique:
        raise ValueError("Patient IDs must be unique")
    lengths = patients.stop.to_numpy(dtype=int) - patients.start.to_numpy(dtype=int)
    codes = np.repeat(np.arange(len(patients)), lengths)
    return labels.astype(np.int8, copy=False), scores, patients, delta, codes


def _ranking(labels, scores):
    if len(np.unique(labels)) != 2:
        return {"auroc": None, "auprc": None}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "auprc": float(average_precision_score(labels, scores))}


def _utility(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _counts(labels, alarms):
    positive, negative = labels == 1, labels == 0
    tp, fp = int(np.sum(positive & alarms)), int(np.sum(negative & alarms))
    fn, tn = int(np.sum(positive & ~alarms)), int(np.sum(negative & ~alarms))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    fpr = fp / (fp + tn) if fp + tn else 0.
    return {"hour_precision": precision, "hour_recall": recall, "hour_fpr": fpr,
            "hour_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "hour_accuracy": (tp + tn) / len(labels), "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate_scores(labels, scores, patients, threshold):
    """Evaluate the original full trajectories using a previously frozen threshold.

    Six-hour detection is reported separately: at least one alarm during
    reconstructed onset-6 through onset-1 (strictly before onset).  Lead time
    uses the earliest alarm inside that window.  It is not official utility.
    Phase utility is each phase's marginal gain over never alarming, divided
    by the full evaluation cohort's normalization denominator.
    """
    labels, scores, patients, delta, codes = _inputs(labels, scores, patients)
    if not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    alarms = scores >= threshold
    scaled = np.rint(delta * 180).astype(np.int64)
    numerator = int(scaled[alarms].sum())
    denominator = int(np.maximum(scaled, 0).sum())
    fast_utility = _utility(numerator, denominator)
    contributions, phase_rows = [], []
    phases = ("nonseptic", "before_onset_minus12", "onset_minus12_to_minus6",
              "onset_minus6_to_before_onset", "onset_through_plus3", "after_onset_plus3")
    phase_hours = dict.fromkeys(phases, 0)
    phase_alarms = dict.fromkeys(phases, 0)
    phase_gain = dict.fromkeys(phases, 0)
    event_patients, detected, leads, nonseptic, nonseptic_alarmed = 0, 0, [], 0, 0
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        y, alarm, gain = labels[a:b], alarms[a:b], scaled[a:b]
        values = patient_utility(y, alarm)
        contributions.append(values)
        if y.any():
            event_patients += 1
            onset = int(np.flatnonzero(y)[0]) + 6
            relative = np.arange(len(y)) - onset
            masks = (("before_onset_minus12", relative < -12),
                     ("onset_minus12_to_minus6", (relative >= -12) & (relative < -6)),
                     ("onset_minus6_to_before_onset", (relative >= -6) & (relative < 0)),
                     ("onset_through_plus3", (relative >= 0) & (relative <= 3)),
                     ("after_onset_plus3", relative > 3))
            hits = np.flatnonzero(alarm & (relative >= -6) & (relative < 0))
            if len(hits):
                detected += 1
                leads.append(int(onset - hits[0]))
        else:
            nonseptic += 1
            nonseptic_alarmed += int(alarm.any())
            masks = (("nonseptic", np.ones(len(y), dtype=bool)),)
        for name, mask in masks:
            phase_hours[name] += int(mask.sum())
            phase_alarms[name] += int((mask & alarm).sum())
            phase_gain[name] += int(gain[mask & alarm].sum())
    contributions = pd.DataFrame(contributions)
    totals = {key: float(contributions[key].sum()) for key in ("observed", "best", "inaction", "worst")}
    official_denominator = totals["best"] - totals["inaction"]
    official_utility = _utility(totals["observed"] - totals["inaction"], official_denominator)
    if denominator > 0:
        np.testing.assert_allclose(official_utility, fast_utility, rtol=0, atol=1e-12)
    for name in phases:
        phase_rows.append({"phase": name, "hours": phase_hours[name], "alarm_hours": phase_alarms[name],
                           "gain_over_inaction": phase_gain[name] / 180.,
                           "normalized_utility_contribution": _utility(phase_gain[name], denominator)})
    per_hospital = {}
    if "hospital" in patients:
        hospitals = patients.hospital.astype(str).to_numpy()
        for hospital in sorted(set(hospitals)):
            person_mask = hospitals == hospital
            rows = person_mask[codes]
            part = contributions.loc[person_mask]
            part_denominator = float((part.best - part.inaction).sum())
            per_hospital[hospital] = {**_ranking(labels[rows], scores[rows]),
                                     "utility_score": _utility(float((part.observed - part.inaction).sum()), part_denominator),
                                     "patients": int(person_mask.sum()), "hours": int(rows.sum()),
                                     "positive_hours": int(labels[rows].sum()), **_counts(labels[rows], alarms[rows])}
    return {**_ranking(labels, scores), "utility_score": official_utility,
            "threshold": float(threshold), "patients": len(patients), "hours": len(labels),
            "positive_hours": int(labels.sum()), "event_patients": event_patients,
            **_counts(labels, alarms), "official_utility_totals": totals,
            "normalization_denominator": official_denominator,
            "utility_check_absolute_difference": None if denominator == 0 else abs(official_utility - fast_utility),
            "per_hospital": per_hospital, "alarm_utility_by_phase": phase_rows,
            "preonset_six_hour_detection": {
                "definition": "Any alarm in onset-6 through onset-1; earliest alarm within that window determines lead",
                "event_patients": event_patients, "detected_event_patients": detected,
                "event_sensitivity": detected / event_patients if event_patients else None,
                "median_lead_hours": float(np.median(leads)) if leads else None,
                "nonseptic_patients": nonseptic, "nonseptic_patients_with_any_alarm": nonseptic_alarmed,
                "nonseptic_patient_alarm_fraction": nonseptic_alarmed / nonseptic if nonseptic else None}}


class _PreparedRanking:
    """Tie-aware weighted AUROC/AP with one sort for all patient resamples."""

    def __init__(self, labels, scores, codes):
        order = np.argsort(-np.asarray(scores), kind="stable")
        values = np.asarray(scores)[order]
        self.ends = np.r_[np.flatnonzero(values[1:] != values[:-1]), len(order) - 1]
        self.y = np.asarray(labels, dtype=np.float64)[order]
        self.codes = np.asarray(codes, dtype=int)[order]

    def evaluate(self, patient_counts):
        weights = np.asarray(patient_counts, dtype=np.float64)[self.codes]
        tp = np.cumsum(weights * self.y)[self.ends]
        fp = np.cumsum(weights * (1. - self.y))[self.ends]
        positive, negative = tp[-1], fp[-1]
        if positive <= 0 or negative <= 0:
            return None
        previous_tp = np.r_[0., tp[:-1]]
        previous_fp = np.r_[0., fp[:-1]]
        auc = np.sum((fp - previous_fp) * (tp + previous_tp)) / (2. * positive * negative)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
        ap = np.sum((tp - previous_tp) * precision) / positive
        return {"auroc": float(auc), "auprc": float(ap)}


def bootstrap_differences(labels, patients, reference_scores, candidate_scores,
                          reference_threshold, candidate_threshold, repetitions=500, seed=2026):
    """Paired patient-bootstrap differences with frozen thresholds and models.

    No threshold search or refitting occurs inside the resamples.  The same
    patient multiplicities are applied to both models and all three metrics.
    Degenerate resamples without both original classes or utility denominator
    are skipped and explicitly counted.
    """
    labels, reference, patients, delta, codes = _inputs(labels, reference_scores, patients)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    if candidate.shape != labels.shape or not np.isfinite(candidate).all():
        raise ValueError("Candidate scores must be finite and aligned")
    if not np.isfinite([reference_threshold, candidate_threshold]).all():
        raise ValueError("Thresholds must be finite")
    if int(repetitions) != repetitions or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    scaled = np.rint(delta * 180).astype(np.int64)
    denominator = np.bincount(codes, weights=np.maximum(scaled, 0), minlength=len(patients))
    reference_gain = np.bincount(codes, weights=scaled * (reference >= reference_threshold), minlength=len(patients))
    candidate_gain = np.bincount(codes, weights=scaled * (candidate >= candidate_threshold), minlength=len(patients))
    prepared_reference = _PreparedRanking(labels, reference, codes)
    prepared_candidate = _PreparedRanking(labels, candidate, codes)
    full_counts = np.ones(len(patients), dtype=int)
    reference_point = prepared_reference.evaluate(full_counts)
    candidate_point = prepared_candidate.evaluate(full_counts)
    if reference_point is None or candidate_point is None or denominator.sum() <= 0:
        raise ValueError("Evaluation needs both original classes and at least one utility-evaluable event")
    reference_point["utility_score"] = float(reference_gain.sum() / denominator.sum())
    candidate_point["utility_score"] = float(candidate_gain.sum() / denominator.sum())
    metric_names = ("auroc", "auprc", "utility_score")
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(int(repetitions)):
        counts = np.bincount(rng.integers(len(patients), size=len(patients)), minlength=len(patients))
        norm = float(np.dot(counts, denominator))
        if norm <= 0:
            continue
        a, b = prepared_reference.evaluate(counts), prepared_candidate.evaluate(counts)
        if a is None or b is None:
            continue
        a["utility_score"] = float(np.dot(counts, reference_gain) / norm)
        b["utility_score"] = float(np.dot(counts, candidate_gain) / norm)
        for name in metric_names:
            samples[name].append(b[name] - a[name])
    valid = len(samples["auroc"])
    if not valid:
        raise ValueError("No nondegenerate bootstrap resamples")
    return {"reference": reference_point, "candidate": candidate_point,
            "difference": {name: {"point": candidate_point[name] - reference_point[name],
                                   "lower": float(np.quantile(samples[name], .025)),
                                   "upper": float(np.quantile(samples[name], .975))}
                           for name in metric_names},
            "requested_resamples": int(repetitions), "valid_resamples": valid,
            "seed": int(seed), "thresholds_fixed": True,
            "uncertainty": "Patient-paired percentile interval, conditional on fitted models, selected candidates and frozen thresholds; excludes training and selection variation."}
