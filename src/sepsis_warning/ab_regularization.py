"""Hospital/event-stratified patient CV and frozen regularization candidates."""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sepsis_warning.data import ROOT,FEATURE_NAMES
from sepsis_warning.history_study import child_environment
from sepsis_warning.ab_expansion import content_hash,write_json

PROCESSED = ROOT/"processed_data/ab_regularization"
OUTPUT = ROOT/"outputs/ab_regularization"
REPORT = ROOT/"reports/ab_regularization"
AB_CASE = ROOT/"processed_data/ab_expansion/cases/A_B_full__common"
AB_MODELS = ROOT/"outputs/ab_expansion/models/A_B_full__common"
VALIDATION_SPLITS = ["A_validation","B_validation"]


def settings():
    with (ROOT/"configs/ab_regularization.toml").open("rb") as stream:return tomllib.load(stream)


def model_candidates(study=None):
    study = settings() if study is None else study
    base = json.loads((ROOT/"reports/history_study/model_settings.json").read_text())
    result = []
    for value in study["lr"]["C_values"]:
        name = "baseline" if value == base["models"]["lr"]["C"] else "C_"+str(value).replace(".","p")
        result.append({"model":"lr","candidate":name,"parameters":{**base["models"]["lr"],"C":value}})
    for model in ["ebm","lightgbm"]:
        for name,overrides in study[model].items():
            result.append({"model":model,"candidate":name,"parameters":{**base["models"][model],**overrides}})
    return result


def patient_folds(rows,n_splits=3,seed=2026):
    patients = rows.groupby("patient_id",sort=True).agg(hospital=("hospital","first"),has_event=("label","max")).reset_index()
    assert set(patients.hospital) == {"A","B"}
    assert rows.groupby("patient_id").hospital.nunique().max() == 1
    strata = patients.hospital+"_"+patients.has_event.astype(str)
    splitter = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    patients["fold"] = -1
    for fold,(_,held) in enumerate(splitter.split(patients.patient_id,strata)):patients.loc[held,"fold"] = fold
    mapping = patients.set_index("patient_id").fold
    return patients,rows.patient_id.map(mapping).to_numpy()


def choose_candidates(fold_results,study):
    """Require a material macro-hospital AP gain on at least two matched folds."""
    frame = pd.DataFrame(fold_results)
    selected,comparisons = {},[]
    for model in study["models"]:
        part = frame.loc[frame.model.eq(model)]
        baseline = part.loc[part.candidate.eq("baseline")].set_index("fold").macro_hospital_auprc
        assert len(baseline) == study["folds"] and baseline.index.is_unique
        qualifying = []
        for candidate,group in part.groupby("candidate",sort=True):
            score = group.set_index("fold").macro_hospital_auprc.reindex(baseline.index)
            assert len(group) == study["folds"] and group.fold.is_unique and score.notna().all()
            difference = score-baseline
            eligible = bool(difference.mean() >= study["minimum_mean_ap_gain"] and
                            (difference > 0).sum() >= study["minimum_improved_folds"])
            row = {"model":model,"candidate":candidate,"mean_macro_hospital_auprc":float(score.mean()),
                   "mean_gain_vs_baseline":float(difference.mean()),"improved_folds":int((difference > 0).sum()),"eligible":eligible}
            comparisons.append(row)
            if eligible:qualifying.append(row)
        selected[model] = max(qualifying,key=lambda r:(r["mean_macro_hospital_auprc"],r["candidate"]))["candidate"] if qualifying else "baseline"
    return selected,comparisons


def runtime_environment():
    env = child_environment()
    # Identical pinned wheels, kept outside Desktop's evictable cloud directory.
    local = Path("/private/tmp/info5600_ab_runtime")
    if sys.platform == "darwin" and (local/"interpret").exists() and (local/"lightgbm").exists():
        env["PYTHONPATH"] = str(local)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def raw_pool():
    return np.load(PROCESSED/"training_features.npy",mmap_mode="r")


def output_directory(stage,fold,model,candidate):
    return OUTPUT/stage/(f"fold{fold}" if stage == "cv" else "full")/f"{model}_{candidate}"


def model_metrics(rows,probability):
    from sepsis_warning.evaluation import ranking_metrics
    result = {"pooled":ranking_metrics(rows.label.to_numpy(),probability)}
    for hospital in ["A","B"]:
        mask = rows.hospital.eq(hospital).to_numpy()
        result[hospital] = ranking_metrics(rows.label.to_numpy()[mask],probability[mask])
    result["macro_hospital_auprc"] = float(np.mean([result[h]["auprc_average_precision"] for h in ["A","B"]]))
    return result
