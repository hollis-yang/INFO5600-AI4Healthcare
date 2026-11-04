"""Develop simple causal policies on train CV only; never replay them on test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from sepsis_warning import challenge_improvement as study
from sepsis_warning import friend_563 as baseline
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.alarm_policy_analysis import POLICIES, transform_scores, exact_threshold, decision_counts
from sepsis_warning.challenge_utility import utility_targets

MODEL = "lgb_bce_lag_observed_leaf63"
REPORT = ROOT / "reports/us_loss_analysis"


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    labels = np.load(study.available(baseline.DATA / "train_labels.npy"))
    patients = pd.read_csv(study.available(baseline.DATA / "train_patients.csv"))
    delta = utility_targets(labels, patients)["delta"]
    protocol = {"model": MODEL, "policies": list(POLICIES),
                "data": "Original training patients only, same fit/calibration/outer-held roles",
                "thresholds": "Each policy independently maximizes inner-calibration US; held labels never set thresholds",
                "min2_start": "First hour unchanged; from hour two both current and prior score must exceed threshold",
                "calibration_selection": "Base model iteration counts were also selected on inner calibration; outer fold remains separate",
                "test_policy": "No validation or test policy replay; development sensitivity analysis, not a new test result",
                "source_sha256": content_hash(ROOT / "src/sepsis_warning/alarm_policy_analysis.py")}
    write_json(REPORT / "policy_protocol.json", protocol)
    fold_rows, scores_by_policy, counts_by_policy = [], {}, {}
    for policy in POLICIES:
        scores_by_policy[policy] = np.full(len(labels), np.nan)
        counts_by_policy[policy] = []
    for fold in range(3):
        _, cal, held = study.cv_roles(fold)
        cal_index, held_index = study.row_indices(cal), study.row_indices(held)
        cal_local, held_local = study.local_patients(cal), study.local_patients(held)
        directory = study.OUT / "cv" / f"fold{fold}" / MODEL
        np.testing.assert_array_equal(pd.read_csv(study.available(directory / "cal_patients.csv")).patient_id, cal.patient_id)
        np.testing.assert_array_equal(pd.read_csv(study.available(directory / "held_patients.csv")).patient_id, held.patient_id)
        original_cal = np.load(study.available(directory / "cal_score.npy"))
        original_held = np.load(study.available(directory / "held_score.npy"))
        record = json.loads(study.available(directory / "results.json").read_text())
        assert content_hash(directory / "held_score.npy") == record["score_sha256"]
        for policy in POLICIES:
            cal_scores = transform_scores(original_cal, cal_local, policy)
            held_scores = transform_scores(original_held, held_local, policy)
            selected = exact_threshold(cal_scores, delta[cal_index])
            if policy == "raw":
                assert selected["threshold"] == record["calibration"]["threshold"]
            counts = decision_counts(labels[held_index], held_scores, held_local, selected["threshold"], delta[held_index])
            ranking = study.ranking_scores(labels[held_index], held_scores)
            scores_by_policy[policy][held_index] = held_scores
            counts_by_policy[policy].append(counts)
            fold_rows.append({"model": MODEL, "policy": policy, "fold": fold,
                              **ranking, **counts, "threshold": selected["threshold"],
                              "inner_calibration_utility": selected["utility"],
                              "utility_score": counts["utility_gain_scaled"]/counts["utility_denominator_scaled"]})
        print(f"Completed causal policies on training fold {fold}", flush=True)
    folds = pd.DataFrame(fold_rows)
    rows = []
    for policy in POLICIES:
        total = {key: sum(item[key] for item in counts_by_policy[policy]) for key in counts_by_policy[policy][0]}
        part = folds.loc[folds.policy.eq(policy)]
        rows.append({"policy": policy, **study.ranking_scores(labels, scores_by_policy[policy]),
                     "mean_fold_auroc": float(part.auroc.mean()), "mean_fold_auprc": float(part.auprc.mean()),
                     "utility_score": total["utility_gain_scaled"]/total["utility_denominator_scaled"],
                     "pre6_event_sensitivity": total["detected_pre6"]/total["event_patients"],
                     "nonseptic_patient_alarm_fraction": total["nonseptic_alarmed_patients"]/total["nonseptic_patients"],
                     "nonseptic_hour_alarm_fraction": total["nonseptic_alarm_hours"]/total["nonseptic_hours"], **total})
    summary = pd.DataFrame(rows)
    reference = summary.loc[summary.policy.eq("raw")].iloc[0]
    for metric in ("mean_fold_auroc", "auprc", "utility_score", "pre6_event_sensitivity", "nonseptic_hour_alarm_fraction"):
        summary[f"delta_{metric}"] = summary[metric]-reference[metric]
    summary.to_csv(REPORT / "policy_cv_summary.csv", index=False)
    folds.to_csv(REPORT / "policy_cv_folds.csv", index=False)
    write_json(REPORT / "policy_summary.json", {"protocol": protocol, "results": summary.to_dict(orient="records"),
                                               "test_evaluated": False, "new_models_trained": 0})
    print(summary[["policy", "mean_fold_auroc", "utility_score", "delta_utility_score", "pre6_event_sensitivity",
                   "nonseptic_hour_alarm_fraction"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
