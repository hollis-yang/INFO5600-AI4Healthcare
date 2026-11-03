"""Verify A+B patient isolation, train-only transforms and serialized model results."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_expansion import (OUTPUT,PROCESSED,REPORT,EVALUATION_SPLITS,FixedColumnsPreprocessor,
    content_hash,settings,split_b_patients,matched_mixture,result_directory,write_json)
from sepsis_warning.data import FEATURE_NAMES,config,assess_patient
from sepsis_warning.feature_engineering import feature_names,alternative_features
from sepsis_warning.optimization import settings as feature_settings
from sepsis_warning.regularized import patient_bags
from sepsis_warning.evaluation import ranking_metrics,alert_replay,select_alert_threshold


def raw_sources(profile):
    if profile == "common":
        paths = [ROOT/"processed_data"/f"{s}_features.npy" for s in ["train","validation","external_test"]]
        names = list(FEATURE_NAMES)
    else:
        paths = [ROOT/"processed_data/feature_optimization"/f"{s}_combined_features.npy" for s in ["train","validation"]]
        paths += [PROCESSED/"B_combined_features.npy"]
        names = feature_names("combined")
    return [np.load(path,mmap_mode="r") for path in paths],names


def training_probe(a,b,a_indices,b_indices,indices):
    result = np.empty((len(indices),a.shape[1]),np.float32)
    a_mask = indices < len(a_indices)
    result[a_mask] = a[a_indices[indices[a_mask]]]
    result[~a_mask] = b[b_indices[indices[~a_mask]-len(a_indices)]]
    return result


def verify_statistics(processor,a,b,a_indices,b_indices,names,nrows):
    fixed = isinstance(processor,FixedColumnsPreprocessor)
    inner = processor.inner_ if fixed else processor
    assert int(inner.scaler_.n_samples_seen_) == nrows
    if fixed:
        assert processor.fit_rows_ == nrows
        assert processor.raw_names == names
    active_names = inner.base_names_ if hasattr(inner,"base_names_") else inner.names_
    checked = []
    for name in ["HR__latest","Creatinine__latest","ICULOS__current","Creatinine__last2_slope_24h","HR__slope_12h"]:
        if name not in active_names:continue
        j,raw_j = active_names.index(name),names.index(name)
        values = np.concatenate([a[a_indices,raw_j],b[b_indices,raw_j]])
        observed = values[np.isfinite(values)]
        np.testing.assert_allclose(np.median(observed),inner.fill_[j],rtol=0,atol=1e-6)
        if np.isfinite(inner.lower_[j]):
            np.testing.assert_allclose(np.quantile(observed,[.005,.995]),[inner.lower_[j],inner.upper_[j]],rtol=1e-6,atol=1e-6)
        np.clip(values,inner.lower_[j],inner.upper_[j],out=values)
        values[np.isnan(values)] = inner.fill_[j]
        np.testing.assert_allclose(np.mean(values,dtype=float),inner.scaler_.mean_[j],rtol=1e-10,atol=1e-9)
        std = np.std(values,dtype=float)
        np.testing.assert_allclose(std if std else 1.,inner.scaler_.scale_[j],rtol=1e-8,atol=1e-8)
        checked.append(name)
    return checked


def main():
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    completion = json.loads((OUTPUT/"completed.json").read_text())
    report_manifest = json.loads((OUTPUT/"report_manifest.json").read_text())
    for path,expected in report_manifest["analysis_source_fingerprints"].items():assert content_hash(ROOT/path) == expected
    assert content_hash(OUTPUT/"cohort_membership.csv") == report_manifest["cohort_membership_sha256"]
    assert content_hash(OUTPUT/"protocol.json") == report_manifest["protocol_sha256"]
    assert completion["model_results"] == 14 and completion["new_fits"] == 10
    assert not protocol["A_test_evaluated"] and not protocol["B_test_is_unseen_hospital"] and not protocol["hyperparameters_tuned"]
    for mapping in [protocol["input_fingerprints"],protocol["source_fingerprints"],protocol["reference_model_fingerprints"]]:
        for path,expected in mapping.items():assert content_hash(ROOT/path) == expected,path
    secondary_inputs = json.loads((OUTPUT/"secondary_input_fingerprints.json").read_text())
    assert secondary_inputs["recorded_before_secondary_model_fit"]
    for path,expected in secondary_inputs["fingerprints"].items():assert content_hash(ROOT/path) == expected,path
    assert protocol["model_settings"] == json.loads((ROOT/"reports/history_study/model_settings.json").read_text())
    original = pd.read_csv(ROOT/"processed_data/splits/patient_splits.csv")
    a_test_ids = set(original.loc[original.hospital.eq("A")&original.split.eq("test"),"patient_id"])
    a_rows = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    a_val = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    b_rows = pd.read_csv(ROOT/"processed_data/external_test_rows.csv.gz")
    b_split = pd.read_csv(PROCESSED/"B_patient_splits.csv")
    pd.testing.assert_frame_equal(b_split,split_b_patients(b_rows,settings()["study"]))
    versioned = pd.read_csv(PROCESSED/"patient_splits.csv")
    pd.testing.assert_frame_equal(original,versioned[original.columns])
    assert set(b_split.patient_id) == set(b_rows.patient_id)
    b_train = b_rows.loc[b_rows.patient_id.isin(b_split.loc[b_split.expansion_split.eq("B_train"),"patient_id"])]
    full = pd.concat([a_rows,b_train],ignore_index=True)
    evaluation = {split:pd.read_csv(PROCESSED/f"{split}_rows.csv.gz") for split in EVALUATION_SPLITS}
    pd.testing.assert_frame_equal(evaluation["A_validation"],a_val)
    eval_ids = set()
    for split,rows in evaluation.items():
        assert not eval_ids&set(rows.patient_id) and not a_test_ids&set(rows.patient_id)
        eval_ids.update(rows.patient_id)
        indices = np.load(PROCESSED/f"{split}_indices.npy")
        source = a_val if split == "A_validation" else b_rows
        pd.testing.assert_frame_equal(rows,source.iloc[indices].reset_index(drop=True))
        if split != "A_validation":assert set(rows.patient_id) == set(b_split.loc[b_split.expansion_split.eq(split),"patient_id"])
    members = pd.read_csv(OUTPUT/"cohort_membership.csv")
    for seed in [2026,2027]:
        actual = set(members.loc[members.cohort.eq(f"A_B_matched_seed{seed}"),"patient_id"])
        assert actual == set(matched_mixture(full,a_rows.patient_id.nunique(),seed))
    jobs = json.loads((OUTPUT/"model_manifest.json").read_text())
    checked,model_checks,raw_cache = {},[],{}
    for job in jobs:
        case,model_name,profile = job["case"],job["model"],job["profile"]
        directory = PROCESSED/"cases"/case
        meta = json.loads((directory/"case.json").read_text())
        rows = pd.read_csv(directory/"train_rows.csv.gz")
        assert not set(rows.patient_id)&(eval_ids|a_test_ids)
        expected_ids = set(members.loc[members.cohort.eq(job["cohort"]),"patient_id"])
        assert set(rows.patient_id) == expected_ids
        a_indices,b_indices = np.load(directory/"A_train_indices.npy"),np.load(directory/"B_train_indices.npy")
        pd.testing.assert_frame_equal(rows,pd.concat([a_rows.iloc[a_indices],b_rows.iloc[b_indices]],ignore_index=True))
        np.testing.assert_array_equal(rows.label,np.load(directory/"train_y.npy"))
        out = result_directory(case)
        result = json.loads((out/f"{model_name}_results.json").read_text())
        bundle = joblib.load(out/f"{model_name}_model.joblib")
        assert result["case"] == bundle["case"] == meta
        assert bundle["parameters"] == result["parameters"] == protocol["model_settings"]["models"][model_name]
        assert result["reused_A_only_model"] == (job["cohort"] == "A_only")
        assert not result["A_test_evaluated"] and not result["B_test_is_unseen_hospital"]
        processor,model = bundle["preprocessor"],bundle["model"]
        assert processor.names_ == protocol["feature_columns"][profile]
        assert not set(processor.names_)&{"hospital","patient_id","label","onset_hour"}
        assert len(processor.names_) == (312 if profile == "common" else 349)
        if profile not in raw_cache:raw_cache[profile] = raw_sources(profile)
        (raw_a,raw_val,raw_b),names = raw_cache[profile]
        if case not in checked:
            probes = np.unique(np.linspace(0,len(rows)-1,64,dtype=int))
            transformed = processor.transform(training_probe(raw_a,raw_b,a_indices,b_indices,probes))
            if job["cohort"] != "A_only":
                matrix = np.load(directory/"train_X.npy",mmap_mode="r")
                assert matrix.shape == (len(rows),len(processor.names_))
                np.testing.assert_allclose(matrix[probes],transformed,equal_nan=True,atol=0,rtol=0)
            stats = verify_statistics(processor,raw_a,raw_b,a_indices,b_indices,names,len(rows))
            for split,table in evaluation.items():
                indices = np.load(PROCESSED/f"{split}_indices.npy")
                sample = np.unique(np.linspace(0,len(table)-1,64,dtype=int))
                raw = raw_val if split == "A_validation" else raw_b
                expected = processor.transform(raw[indices[sample]])
                cached = np.load(directory/f"{split}_X.npy",mmap_mode="r")
                assert cached.shape == (len(table),len(processor.names_))
                np.testing.assert_allclose(expected,cached[sample],equal_nan=True,atol=0,rtol=0)
            checked[case] = {"statistics_probes":stats,"matrix_probe_rows":64,"training_rows":len(rows),"features":len(processor.names_)}
        probabilities = {}
        for split,table in evaluation.items():
            p = np.load(out/f"{model_name}_{split}_probability.npy")
            probabilities[split] = p
            assert len(p) == len(table) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            for metric,value in ranking_metrics(table.label.to_numpy(),p).items():
                np.testing.assert_allclose(value,result["evaluation"][split][metric],rtol=0,atol=1e-12)
            sample = np.unique(np.linspace(0,len(table)-1,64,dtype=int))
            matrix = np.load(directory/f"{split}_X.npy",mmap_mode="r")[sample]
            with threadpool_limits(limits=4):
                predicted = model.booster_.predict(matrix) if model_name == "lightgbm" else model.predict_proba(matrix)[:,1]
            np.testing.assert_allclose(predicted,p[sample],rtol=0,atol=1e-12)
            replay,_,_ = alert_replay(table,p,result["threshold"],config()["alerts"]["suppression_hours"])
            assert replay == result["alerts"][split]
        pooled_rows = pd.concat([evaluation["A_validation"],evaluation["B_validation"]],ignore_index=True)
        pooled_p = np.r_[probabilities["A_validation"],probabilities["B_validation"]]
        threshold,_ = select_alert_threshold(pooled_rows,pooled_p,config())
        assert threshold == result["threshold"] == bundle["threshold"]
        assert result["threshold_fit_on"] == ["A_validation","B_validation"]
        assert result["alerts"]["pooled_validation"]["false_alerts_per_100_patient_hours"] <= 1
        for metric,value in ranking_metrics(pooled_rows.label.to_numpy(),pooled_p).items():
            np.testing.assert_allclose(value,result["evaluation"]["pooled_validation"][metric],rtol=0,atol=1e-12)
        if job["cohort"] == "A_only":
            reference = json.loads((OUTPUT/"references"/f"{profile}_{model_name}_results.json").read_text())
            key = "validation" if profile == "common" else "evaluation"
            assert result["evaluation"]["A_validation"] == reference[key]
        elif model_name != "lr":
            _,expected = patient_bags(rows,n_bags=3 if model_name == "ebm" else 1,seed=2026)
            pd.testing.assert_frame_equal(pd.read_csv(out/f"{model_name}_internal_patients.csv"),expected,check_dtype=False,check_exact=True)
        model_checks.append({"case":case,"model":model_name,"signature":result["signature"]})
        print(f"Verified {len(model_checks)}/{len(jobs)} {case}/{model_name}",flush=True)
    # Verify dynamic B feature construction and absence of future dependence on real patients.
    audit = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    raw = np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    combined = np.load(PROCESSED/"B_combined_features.npy",mmap_mode="r")
    for event in [0,1]:
        identifier = b_split.loc[b_split.has_event.eq(event),"patient_id"].iloc[0]
        patient = audit.loc[identifier]
        values = np.array(raw[int(patient.raw_start):int(patient.raw_stop)],copy=True)
        whole = alternative_features(values,config(),feature_settings()["features"])
        prefix = min(12,len(values))
        np.testing.assert_allclose(whole[:prefix],alternative_features(values[:prefix],config(),feature_settings()["features"]),equal_nan=True,rtol=0,atol=0)
        _,eligible,_ = assess_patient(values,config())
        np.testing.assert_allclose(combined[b_rows.patient_id.eq(identifier)],whole[eligible],equal_nan=True,rtol=0,atol=0)
    evidence = {"model_results_verified":len(model_checks),"preprocessing_cases_verified":len(checked),
                "patient_partitions_and_matched_draws_reproduced":True,"canonical_inputs_and_sources_unchanged":True,
                "A_test_patient_count_reserved":len(a_test_ids),"A_test_evaluated":False,"B_test_unseen_hospital":False,
                "fixed_A_defined_feature_columns":True,"training_only_statistics_recomputed":True,
                "secondary_raw_feature_inputs_unchanged":True,
                "serialized_predictions_and_ranking_metrics_verified":True,"pooled_validation_thresholds_reproduced":True,
                "patient_grouped_internal_stopping_assignments_reproduced":True,"real_B_patient_prefix_causality_verified":True,
                "cases":checked,"models":model_checks}
    write_json(OUTPUT/"verification.json",evidence)
    write_json(REPORT/"verification.json",evidence)
    print("All A+B verification checks passed.",flush=True)


if __name__ == "__main__":main()
