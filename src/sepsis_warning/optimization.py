"""Prespecified patient-CV protocol for feature optimization."""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sepsis_warning.data import ROOT
from sepsis_warning.feature_engineering import PROFILES, DROP_CONTEXT, feature_names
from sepsis_warning.history_study import digest

PROCESSED = ROOT/"processed_data/feature_optimization"
OUTPUT = ROOT/"outputs/feature_optimization"
REPORT = ROOT/"reports/feature_optimization"


def settings():
    with (ROOT/"configs/feature_optimization.toml").open("rb") as stream:
        result = tomllib.load(stream)
    f = result["features"]
    assert f["lab_trend_window_hours"] == 24 and f["vital_slope_windows"] == [3, 12]
    assert f["vital_variability_window"] == f["personal_baseline_window"] == 12
    return result


def frozen_models():
    return json.loads((ROOT/"reports/history_study/model_settings.json").read_text())


def patient_folds(rows, n_splits=3, seed=2026):
    patients = rows.groupby("patient_id", sort=True).label.max().rename("has_event").reset_index()
    patients["fold"] = -1
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, held) in enumerate(splitter.split(patients.patient_id, patients.has_event)):
        patients.loc[held, "fold"] = fold
    mapping = patients.set_index("patient_id").fold
    return patients, rows.patient_id.map(mapping).to_numpy()


def representation(profile_name, model_name):
    profile = PROFILES[profile_name]
    result = {"source": profile["source"], "native": profile["native"] and model_name != "lr",
              "clip": profile["clip"], "drop_context": profile.get("drop_context", False),
              "splines": profile.get("splines", False)}
    assert not result["splines"] or model_name == "lr"
    return result


def representation_key(spec, fold):
    prefix = "full" if fold == "full" else f"fold{fold}"
    return prefix+"_"+spec["source"]+"_"+digest(spec)[:12]


def raw_features(split, spec):
    if spec["source"] == "baseline":
        matrix = np.load(ROOT/f"processed_data/{split}_features.npy", mmap_mode="r")
    else:
        matrix = np.load(PROCESSED/f"{split}_combined_features.npy", mmap_mode="r")
        matrix = matrix[:, :len(feature_names(spec["source"]))]
    names = feature_names(spec["source"])
    if spec["drop_context"]:
        columns = [i for i, name in enumerate(names) if name not in DROP_CONTEXT]
        return matrix[:, columns], [names[i] for i in columns]
    return matrix, names


def choose_profiles(fold_results, study):
    """Use only matched CV folds; no outer-validation score enters this rule."""
    frame = pd.DataFrame(fold_results)
    selected, comparisons = {}, []
    for model in study["models"]:
        part = frame.loc[frame.model == model]
        baseline = part.loc[part.profile == "baseline"].set_index("fold").auprc
        candidates = []
        for profile, group in part.groupby("profile", sort=False):
            score = group.set_index("fold").auprc.reindex(baseline.index)
            assert len(score) == study["folds"] and score.notna().all()
            difference = score-baseline
            eligible = (difference.mean() >= study["minimum_mean_ap_gain"] and
                        int((difference > 0).sum()) >= study["minimum_improved_folds"])
            row = {"model": model, "profile": profile, "mean_fold_auprc": float(score.mean()),
                   "mean_gain_vs_baseline": float(difference.mean()),
                   "improved_folds": int((difference > 0).sum()), "eligible": bool(eligible)}
            comparisons.append(row)
            if eligible:
                candidates.append(row)
        selected[model] = max(candidates, key=lambda r: (r["mean_fold_auprc"], r["profile"]))["profile"] if candidates else "baseline"
    return selected, comparisons
