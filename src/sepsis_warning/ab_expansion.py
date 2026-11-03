"""Frozen patient splits and matched-size controls for A+B data expansion."""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sepsis_warning.data import ROOT
from sepsis_warning.feature_engineering import FeaturePreprocessor

PROCESSED = ROOT/"processed_data/ab_expansion"
OUTPUT = ROOT/"outputs/ab_expansion"
REPORT = ROOT/"reports/ab_expansion"
EVALUATION_SPLITS = ["A_validation", "B_validation", "B_test"]


def settings():
    with (ROOT/"configs/ab_expansion.toml").open("rb") as stream:
        return tomllib.load(stream)


def content_hash(path):
    """Detect unavailable file-provider content instead of hashing an empty read."""
    path = Path(path)
    expected_size = path.stat().st_size
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
            count += len(block)
    if count != expected_size:
        raise OSError(f"File content unavailable: {path} (read {count}/{expected_size} bytes)")
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    os.replace(temporary,path)


def split_b_patients(rows,study):
    patients = rows.groupby("patient_id",sort=True).agg(has_event=("label","max"),hospital=("hospital","first")).reset_index()
    assert set(patients.hospital) == {"B"} and patients.patient_id.is_unique
    assert np.isclose(study["b_train_fraction"]+study["b_validation_fraction"]+study["b_test_fraction"],1)
    train,held = train_test_split(patients.index,test_size=1-study["b_train_fraction"],
                                  random_state=study["b_split_seed"],stratify=patients.has_event)
    validation,test = train_test_split(held,test_size=study["b_test_fraction"]/(study["b_validation_fraction"]+study["b_test_fraction"]),
                                      random_state=study["b_split_seed"],stratify=patients.loc[held,"has_event"])
    patients["expansion_split"] = "B_test"
    patients.loc[train,"expansion_split"] = "B_train"
    patients.loc[validation,"expansion_split"] = "B_validation"
    assert len(test)+len(validation)+len(train) == len(patients)
    return patients


def matched_mixture(rows,target_patients,seed):
    """Keep the full pool's hospital/event proportions at a fixed patient count."""
    patients = rows.groupby("patient_id",sort=True).agg(has_event=("label","max"),hospital=("hospital","first")).reset_index()
    assert 0 < target_patients <= len(patients)
    groups = list(patients.groupby(["hospital","has_event"],sort=True))
    expected = np.array([len(group) for _,group in groups])*target_patients/len(patients)
    allocations = np.floor(expected).astype(int)
    remaining = target_patients-int(allocations.sum())
    for index in np.argsort(-(expected-allocations),kind="stable")[:remaining]:
        allocations[index] += 1
    rng = np.random.default_rng(seed)
    identifiers = []
    for (_,group),count in zip(groups,allocations):
        ids = group.patient_id.to_numpy().copy()
        rng.shuffle(ids)
        identifiers.extend(ids[:count])
    assert len(identifiers) == len(set(identifiers)) == target_patients
    return sorted(identifiers)


class FixedColumnsPreprocessor:
    """Training-only statistics on the original A-defined feature set."""
    def __init__(self,raw_names,allowed_names,native=False,clip=True):
        self.raw_names,self.names_ = list(raw_names),list(allowed_names)
        self.native,self.clip = native,clip
        assert len(set(self.names_)) == len(self.names_)
        self.columns_ = np.array([self.raw_names.index(name) for name in self.names_])

    def fit(self,features):
        selected = features[:,self.columns_]
        self.inner_ = FeaturePreprocessor(self.names_,native=self.native,clip=self.clip).fit(selected)
        self.active_ = self.inner_.keep_.copy()
        self.fit_rows_ = len(features)
        return self

    def transform(self,features):
        transformed = self.inner_.transform(features[:,self.columns_])
        if self.active_.all():return transformed
        result = np.full((len(features),len(self.names_)),np.nan if self.native else 0,np.float32)
        result[:,self.active_] = transformed
        return result


def reference_path(profile,model):
    if profile == "common":
        return ROOT/f"outputs/history_study/history6h_full/{model}_model.joblib"
    assert profile == "enhanced_lightgbm" and model == "lightgbm"
    return ROOT/"outputs/feature_optimization/validation/full_combined_f3a8818e555b/lightgbm_model.joblib"


def case_key(cohort,profile):
    return cohort+"__"+profile


def result_directory(case):
    return OUTPUT/"models"/case
