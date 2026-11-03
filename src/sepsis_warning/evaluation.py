"""Validation metrics and alert replay with explicit episode definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_score, recall_score,
                             roc_auc_score)


def ranking_metrics(labels, probabilities, sample_weight=None):
    return {"auroc": float(roc_auc_score(labels, probabilities, sample_weight=sample_weight)),
            "auprc_average_precision": float(average_precision_score(labels, probabilities, sample_weight=sample_weight)),
            "brier_score": float(brier_score_loss(labels, probabilities, sample_weight=sample_weight)),
            "positive_hour_prevalence": float(np.average(labels, weights=sample_weight))}


def threshold_metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "hour_sensitivity": float(recall_score(labels, predictions, zero_division=0)),
            "hour_precision": float(precision_score(labels, predictions, zero_division=0)),
            "hour_specificity": float(tn / (tn + fp)), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def alert_replay(rows, scores, threshold, suppression=6):
    codes, identifiers = pd.factorize(rows.patient_id, sort=False)
    hours = rows.hour.to_numpy()
    labels = rows.label.to_numpy()
    onset = rows.onset_hour.to_numpy()
    patient_hours = np.bincount(codes, minlength=len(identifiers))
    is_event = np.bincount(codes, weights=labels, minlength=len(identifiers)) > 0
    false_alerts = np.zeros(len(identifiers), int)
    detected = np.zeros(len(identifiers), int)
    last_alert = {}
    episodes = []
    for i in np.flatnonzero(scores >= threshold):
        patient, hour = int(codes[i]), int(hours[i])
        if hour < last_alert.get(patient, -np.inf) + suppression:
            continue
        last_alert[patient] = hour
        true_alert = bool(labels[i])
        if true_alert:
            detected[patient] = 1
        else:
            false_alerts[patient] += 1
        lead = float(onset[i]-hour) if true_alert else np.nan
        episodes.append((identifiers[patient], hour, float(scores[i]), true_alert, lead))
    episode_table = pd.DataFrame(episodes, columns=["patient_id", "hour", "probability", "true_alert", "lead_hours"])
    true_episodes = int(episode_table.true_alert.sum()) if len(episode_table) else 0
    true_leads = episode_table.loc[episode_table.true_alert, "lead_hours"] if len(episode_table) else []
    result = {"threshold": float(threshold), "suppression_hours": suppression,
              "eligible_patient_hours": int(patient_hours.sum()), "event_patients": int(is_event.sum()),
              "detected_event_patients": int(detected.sum()),
              "event_sensitivity": float(detected.sum()/is_event.sum()),
              "false_alert_episodes": int(false_alerts.sum()),
              "false_alerts_per_100_patient_hours": float(100*false_alerts.sum()/patient_hours.sum()),
              "total_alert_episodes": len(episode_table), "true_alert_episodes": true_episodes,
              "alert_ppv": float(true_episodes/len(episode_table)) if len(episode_table) else 0.0,
              "median_true_alert_lead_hours": float(np.median(true_leads)) if len(true_leads) else None}
    per_patient = pd.DataFrame({"patient_id": identifiers, "hours": patient_hours,
                                "is_event": is_event, "detected": detected, "false_alerts": false_alerts})
    return result, episode_table, per_patient


def select_alert_threshold(rows, scores, cfg):
    # Deterministic validation-only grid; replay each candidate after suppression.
    quantiles = np.r_[0.0, np.linspace(0.50, 0.99, 50), np.linspace(0.991, 0.9999, 50)]
    candidates = np.unique(np.r_[np.quantile(scores, quantiles), 0.5, 1.0])
    results = []
    for threshold in candidates:
        result, _, _ = alert_replay(rows, scores, threshold, cfg["alerts"]["suppression_hours"])
        results.append(result)
    table = pd.DataFrame(results)
    allowed = table.loc[table.false_alerts_per_100_patient_hours <= cfg["alerts"]["false_alert_budget_per_100_patient_hours"]]
    chosen = allowed.sort_values(["event_sensitivity", "alert_ppv", "false_alerts_per_100_patient_hours", "threshold"],
                                  ascending=[False, False, True, False]).iloc[0]
    return float(chosen.threshold), table


def patient_bootstrap(rows, scores, repetitions, seed):
    codes, identifiers = pd.factorize(rows.patient_id, sort=False)
    labels = rows.label.to_numpy()
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        weights = np.bincount(rng.integers(len(identifiers), size=len(identifiers)), minlength=len(identifiers))[codes]
        if np.sum(weights*labels) == 0:
            continue
        samples.append(ranking_metrics(labels, scores, weights))
    frame = pd.DataFrame(samples)
    return {name: {"lower": float(frame[name].quantile(0.025)), "upper": float(frame[name].quantile(0.975))}
            for name in frame.columns}


def paired_patient_bootstrap(rows, reference_scores, candidate_scores, repetitions, seed):
    """Use identical patient resamples for model metrics and candidate-reference differences."""
    codes, identifiers = pd.factorize(rows.patient_id, sort=False)
    labels = rows.label.to_numpy()
    rng = np.random.default_rng(seed)
    reference, candidate, differences = [], [], []
    for _ in range(repetitions):
        frequencies = np.bincount(rng.integers(len(identifiers), size=len(identifiers)), minlength=len(identifiers))
        weights = frequencies[codes]
        if np.sum(weights*labels) == 0:
            continue
        a = ranking_metrics(labels, reference_scores, weights)
        b = ranking_metrics(labels, candidate_scores, weights)
        reference.append(a)
        candidate.append(b)
        differences.append({key: b[key]-a[key] for key in a if key != "positive_hour_prevalence"})
    result = {}
    for name, values in [("lr", reference), ("ebm", candidate), ("ebm_minus_lr", differences)]:
        frame = pd.DataFrame(values)
        result[name] = {column: {"lower": float(frame[column].quantile(0.025)),
                                "upper": float(frame[column].quantile(0.975))} for column in frame}
    result["valid_resamples"] = len(differences)
    return result
