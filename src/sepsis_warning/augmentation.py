"""Preserve original feature information while adding the new causal dynamics."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import ROOT, FEATURE_NAMES, save_json
from sepsis_warning.feature_engineering import feature_names, FeaturePreprocessor
from sepsis_warning.history_study import digest, sha_file
from sepsis_warning.optimization import PROCESSED, representation_key

ADDED_DATA = ROOT/"processed_data/feature_augmentation"
ADDED_OUTPUT = ROOT/"outputs/feature_augmentation"
ADDED_REPORT = ROOT/"reports/feature_augmentation"
ADDED_PROFILES = {"preserve_baseline": {"splines": False, "models": ["lr","ebm","lightgbm"]},
                  "preserve_baseline_splines": {"splines": True, "models": ["lr"]}}


def added_names():
    # The new-lab, vital-dynamic and combination blocks follow the retained base block.
    combined = feature_names("combined")
    start = combined.index("BaseExcess__last2_delta_24h")
    return list(FEATURE_NAMES)+combined[start:]


def build_added(split):
    ADDED_DATA.mkdir(parents=True,exist_ok=True)
    destination = ADDED_DATA/f"{split}_features.npy"
    source = PROCESSED/f"{split}_combined_features.npy"
    marker = ADDED_DATA/f"{split}_features.json"
    signature = digest({"combined_source":sha_file(source),"baseline":sha_file(ROOT/f"processed_data/{split}_features.npy"),
                        "names":added_names(),"implementation":sha_file(__file__)})
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature and destination.exists():
        return
    original = np.load(ROOT/f"processed_data/{split}_features.npy",mmap_mode="r")
    combined = np.load(source,mmap_mode="r")
    start = feature_names("combined").index("BaseExcess__last2_delta_24h")
    matrix = np.lib.format.open_memmap(destination,mode="w+",dtype=np.float32,shape=(len(original),len(added_names())))
    for first in range(0,len(original),15000):
        last = first+15000
        matrix[first:last,:len(FEATURE_NAMES)] = original[first:last]
        matrix[first:last,len(FEATURE_NAMES):] = combined[first:last,start:]
    matrix.flush()
    save_json(marker,{"signature":signature,"features":added_names(),"rows":len(original),"preserves_all_original_features":True})


def added_job(profile,model,fold):
    spec = {"source":"augmented","native":model != "lr","clip":True,"drop_context":False,
            "splines":ADDED_PROFILES[profile]["splines"]}
    return {"profile":profile,"model":model,"fold":fold,"representation":spec,
            "case":representation_key(spec,fold),"stage":"validation" if fold == "full" else "cv"}


def prepare_added_case(job,study,inputs,row_folds):
    directory = PROCESSED/"cases"/job["case"]
    directory.mkdir(parents=True,exist_ok=True)
    train = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    fold,spec = job["fold"],job["representation"]
    train_indices = np.arange(len(train)) if fold == "full" else np.flatnonzero(row_folds != fold)
    evaluation_rows = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz") if fold == "full" else train
    evaluation_indices = np.arange(len(evaluation_rows)) if fold == "full" else np.flatnonzero(row_folds == fold)
    signature = digest({"inputs":inputs,"fold":fold,"representation":spec,
                        "row_folds":digest(row_folds.tolist()),"implementation":sha_file(__file__),
                        "preprocessor":sha_file(ROOT/"src/sepsis_warning/feature_engineering.py"),"splines":study["splines"]})
    marker = directory/"case.json"
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:
        return
    raw = np.load(ADDED_DATA/"train_features.npy",mmap_mode="r")
    selected = raw[train_indices]
    processor = FeaturePreprocessor(added_names(),native=spec["native"],clip=True,
                                    spline_cfg=study["splines"] if spec["splines"] else None).fit(selected)
    matrix = np.lib.format.open_memmap(directory/"train_X.npy",mode="w+",dtype=np.float32,shape=(len(selected),len(processor.names_)))
    for first in range(0,len(selected),15000):
        matrix[first:first+15000] = processor.transform(selected[first:first+15000])
    matrix.flush()
    del matrix,selected,raw
    raw = np.load(ADDED_DATA/("validation_features.npy" if fold == "full" else "train_features.npy"),mmap_mode="r")
    matrix = np.lib.format.open_memmap(directory/"evaluation_X.npy",mode="w+",dtype=np.float32,shape=(len(evaluation_indices),len(processor.names_)))
    for first in range(0,len(evaluation_indices),15000):
        matrix[first:first+15000] = processor.transform(raw[evaluation_indices[first:first+15000]])
    matrix.flush()
    selected_rows,held_rows = train.iloc[train_indices].reset_index(drop=True),evaluation_rows.iloc[evaluation_indices].reset_index(drop=True)
    assert not set(selected_rows.patient_id) & set(held_rows.patient_id)
    selected_rows.to_csv(directory/"train_rows.csv.gz",index=False,compression="gzip")
    held_rows.to_csv(directory/"evaluation_rows.csv.gz",index=False,compression="gzip")
    for name,values in [("train_indices",train_indices),("evaluation_indices",evaluation_indices),
                        ("train_y",selected_rows.label.to_numpy()),("evaluation_y",held_rows.label.to_numpy())]:
        np.save(directory/f"{name}.npy",values)
    joblib.dump(processor,directory/"preprocessor.joblib",compress=3)
    save_json(marker,{"key":job["case"],"signature":signature,"stage":job["stage"],"fold":fold,"representation":spec,
                      "training_patients":int(selected_rows.patient_id.nunique()),"training_hours":len(selected_rows),
                      "training_event_patients":int(selected_rows.groupby("patient_id").label.max().sum()),
                      "evaluation_patients":int(held_rows.patient_id.nunique()),"evaluation_hours":len(held_rows),
                      "features":len(processor.names_),"preprocessing_fit_on":"this case's training patients only"})
