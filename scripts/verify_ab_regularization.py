"""Audit hospital folds, train-only transforms, penalty selection and missingness results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import (OUTPUT,PROCESSED,REPORT,AB_CASE,AB_MODELS,VALIDATION_SPLITS,
    patient_folds,model_candidates,choose_candidates,model_metrics,output_directory)
from sepsis_warning.ab_expansion import content_hash,write_json
from sepsis_warning.history_study import digest
from sepsis_warning.data import FEATURE_NAMES,COLUMNS,config
from sepsis_warning.regularized import patient_bags
from sepsis_warning.evaluation import ranking_metrics,select_alert_threshold,alert_replay


def assert_dict_close(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for key in actual:assert_dict_close(actual[key],expected[key])
    else:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)


def verify_statistics(processor,raw,indices):
    inner=processor.inner_;assert processor.fit_rows_==len(indices)==int(inner.scaler_.n_samples_seen_)
    assert processor.names_==json.loads((OUTPUT/"protocol.json").read_text())["feature_columns"]
    for name in ["HR__latest","DBP__latest","Creatinine__latest","ICULOS__current","Gender__current"]:
        j=inner.base_names_.index(name);source=FEATURE_NAMES.index(name)
        values=np.array(raw[indices,source],copy=True);observed=values[np.isfinite(values)]
        if inner.categorical_[j]:
            levels,counts=np.unique(observed,return_counts=True);fill=levels[counts.argmax()]
        else:fill=np.median(observed)
        np.testing.assert_allclose(fill,inner.fill_[j],atol=1e-6,rtol=0)
        if np.isfinite(inner.lower_[j]):np.testing.assert_allclose(np.quantile(observed,[.005,.995]),[inner.lower_[j],inner.upper_[j]],rtol=1e-6,atol=1e-6)
        np.clip(values,inner.lower_[j],inner.upper_[j],out=values);values[np.isnan(values)]=inner.fill_[j]
        np.testing.assert_allclose(np.mean(values,dtype=float),inner.scaler_.mean_[j],rtol=1e-10,atol=1e-9)
        sd=np.std(values,dtype=float);np.testing.assert_allclose(sd if sd else 1.,inner.scaler_.scale_[j],rtol=1e-8,atol=1e-8)


def verify_missingness(rows,patients,raw):
    out=OUTPUT/"missingness";manifest=json.loads((out/"manifest.json").read_text())
    assert manifest["test_sets_used"]==[] and manifest["training_patients"]==len(patients)
    assert content_hash(PROCESSED/"training_features.npy")==manifest["input_training_features_sha256"]
    assert content_hash(OUTPUT/"protocol.json")==manifest["protocol_sha256"]
    for path,expected in manifest["source_fingerprints"].items():assert content_hash(ROOT/path)==expected
    summary=pd.read_csv(out/"by_hospital.csv").set_index(["hospital","variable"])
    event_table=pd.read_csv(out/"by_event.csv").set_index(["hospital","has_event","variable"])
    hour_table=pd.read_csv(out/"by_hour_band.csv").set_index(["hospital","hour_band","variable"])
    codes=pd.Categorical(rows.patient_id,categories=patients.patient_id).codes
    counts=np.bincount(codes,minlength=len(patients));event=patients.has_event.to_numpy(bool)
    hospital=patients.hospital.to_numpy();hours_hospital=rows.hospital.to_numpy()
    target=event.mean();prefix=rows.hour.le(24).to_numpy()
    prefix_counts=np.bincount(codes[prefix],minlength=len(patients))
    missing_fractions=np.empty((len(patients),40))
    for j,name in enumerate(COLUMNS[:40]):
        flags=raw[:,FEATURE_NAMES.index(name+"__missing_now")].astype(bool)
        latest=np.isnan(raw[:,FEATURE_NAMES.index(name+("__latest" if j<34 else "__current"))])
        patient_raw=np.bincount(codes,weights=flags,minlength=len(patients))/counts
        patient_latest=np.bincount(codes,weights=latest,minlength=len(patients))/counts
        missing_fractions[:,j]=patient_raw
        sums=np.bincount(codes[prefix],weights=flags[prefix],minlength=len(patients))
        patient_prefix=np.divide(sums,prefix_counts,out=np.full(len(patients),np.nan),where=prefix_counts>0)
        for h in ["A","B"]:
            pmask=hospital==h;hmask=hours_hospital==h;record=summary.loc[(h,name)]
            expected={"raw_hour_missing_rate":flags[hmask].mean(),"latest_hour_missing_rate":latest[hmask].mean(),
                "raw_patient_mean_missing_rate":patient_raw[pmask].mean(),"latest_patient_mean_missing_rate":patient_latest[pmask].mean(),
                "patients_never_observed_rate":(patient_raw[pmask]==1).mean(),"raw_first24_patient_mean_missing_rate":np.nanmean(patient_prefix[pmask]),
                "raw_event_standardized_patient_missing_rate":(1-target)*patient_raw[pmask&~event].mean()+target*patient_raw[pmask&event].mean()}
            for metric,value in expected.items():np.testing.assert_allclose(value,record[metric],atol=1e-12,rtol=0)
            assert record.patients==pmask.sum() and record.hours==hmask.sum()
            for status in [False,True]:
                pgroup=pmask&(event==status);hgroup=hmask&(event[codes]==status)
                record=event_table.loc[(h,status,name)]
                for metric,value in {"raw_hour_missing_rate":flags[hgroup].mean(),"latest_hour_missing_rate":latest[hgroup].mean(),
                    "raw_patient_mean_missing_rate":patient_raw[pgroup].mean()}.items():
                    np.testing.assert_allclose(value,record[metric],rtol=0,atol=1e-12)
            for band,mask in [("1-6",rows.hour.le(6)),("7-12",rows.hour.between(7,12)),("13-24",rows.hour.between(13,24)),("25+",rows.hour.gt(24))]:
                selected=hmask&mask.to_numpy();record=hour_table.loc[(h,band,name)]
                np.testing.assert_allclose(flags[selected].mean(),record.raw_hour_missing_rate,rtol=0,atol=1e-12)
                np.testing.assert_allclose(latest[selected].mean(),record.latest_hour_missing_rate,rtol=0,atol=1e-12)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    predictions=pd.read_csv(out/"hospital_classifier_oof.csv")
    np.testing.assert_array_equal(predictions.patient_id,patients.patient_id)
    label=patients.hospital.eq("B").to_numpy(np.int8)
    for fold in range(3):
        held=patients.fold.eq(fold).to_numpy();train=~held
        scaler=StandardScaler().fit(missing_fractions[train])
        classifier=LogisticRegression(C=1,solver="liblinear",max_iter=2000,random_state=42)
        with threadpool_limits(limits=1):classifier.fit(scaler.transform(missing_fractions[train]),label[train])
        expected=classifier.predict_proba(scaler.transform(missing_fractions[held]))[:,1]
        np.testing.assert_allclose(expected,predictions.probability_B.to_numpy()[held],rtol=0,atol=1e-12)
    diagnostic=json.loads((out/"hospital_classifier.json").read_text())
    assert not diagnostic["sepsis_model_input"]
    np.testing.assert_allclose(roc_auc_score(label,predictions.probability_B),diagnostic["pooled_oof_hospital_auroc"],atol=1e-12,rtol=0)


def main():
    completion=json.loads((OUTPUT/"completed.json").read_text());protocol=json.loads((OUTPUT/"protocol.json").read_text())
    selection=json.loads((OUTPUT/"selection.json").read_text());manifest=json.loads((OUTPUT/"report_manifest.json").read_text())
    assert completion["cv_models"]==33 and completion["selected_validation_models"]==3
    assert completion["test_sets_evaluated"]==selection["test_sets_evaluated"]==protocol["test_sets_evaluated"]==[]
    assert not selection["outer_validation_used_for_selection"]
    assert digest(protocol)==selection["protocol_digest"]
    assert content_hash(OUTPUT/"selection.json")==manifest["selection_sha256"]
    assert protocol["candidates"]==model_candidates(protocol["settings"])
    for mapping in [protocol["input_fingerprints"],protocol["source_fingerprints"],manifest["analysis_source_fingerprints"]]:
        for name,expected in mapping.items():assert content_hash(ROOT/name)==expected,name
    original=pd.read_csv(ROOT/"processed_data/splits/patient_splits.csv")
    b_split=pd.read_csv(ROOT/"processed_data/ab_expansion/B_patient_splits.csv")
    a_train=set(original.loc[original.hospital.eq("A")&original.split.eq("train"),"patient_id"])
    b_train=set(b_split.loc[b_split.expansion_split.eq("B_train"),"patient_id"])
    excluded=set(original.loc[original.hospital.eq("A")&original.split.ne("train"),"patient_id"])|set(b_split.loc[b_split.expansion_split.ne("B_train"),"patient_id"])
    rows=pd.read_csv(PROCESSED/"training_rows.csv.gz")
    pd.testing.assert_frame_equal(rows,pd.read_csv(AB_CASE/"train_rows.csv.gz"))
    assert set(rows.patient_id)==a_train|b_train and not set(rows.patient_id)&excluded
    patients,row_folds=patient_folds(rows,3,protocol["settings"]["study"]["fold_seed"])
    pd.testing.assert_frame_equal(patients,pd.read_csv(OUTPUT/"patient_cv_folds.csv"))
    np.testing.assert_array_equal(row_folds,np.load(PROCESSED/"row_folds.npy"))
    assert content_hash(OUTPUT/"patient_cv_folds.csv")==protocol["patient_folds_sha256"]
    raw=np.load(PROCESSED/"training_features.npy",mmap_mode="r")
    assert content_hash(PROCESSED/"training_features.npy")==json.loads((PROCESSED/"raw_pool.json").read_text())["sha256"]
    a=np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r");b=np.load(ROOT/"processed_data/external_test_features.npy",mmap_mode="r")
    position=0
    for source,indices in [(a,np.load(AB_CASE/"A_train_indices.npy")),(b,np.load(AB_CASE/"B_train_indices.npy"))]:
        for start in range(0,len(indices),30000):
            values=source[indices[start:start+30000]]
            np.testing.assert_allclose(raw[position:position+len(values)],values,atol=0,rtol=0,equal_nan=True);position+=len(values)
    assert position==len(raw)
    cases={};scores=[];verified=[]
    jobs=json.loads((OUTPUT/"cv_manifest.json").read_text())+json.loads((OUTPUT/"validation_manifest.json").read_text())
    for job in jobs:
        stage,fold,model_name,candidate=job["stage"],job["fold"],job["model"],job["candidate"]
        directory=PROCESSED/(f"fold{fold}" if stage=="cv" else "full")
        case=json.loads((directory/"case.json").read_text());out=output_directory(**job)
        result=json.loads((out/"results.json").read_text());bundle=joblib.load(out/"model.joblib")
        params=next(c["parameters"] for c in protocol["candidates"] if c["model"]==model_name and c["candidate"]==candidate)
        assert result["parameters"]==bundle["parameters"]==params
        assert result["case"]==bundle["case"]==case and result["test_sets_evaluated"]==[]
        assert result["reused_full_baseline"]==(stage=="validation" and candidate=="baseline")
        signature=digest({"case":case["signature"],"parameters":params,"worker":content_hash(ROOT/"scripts/fit_ab_regularization.py"),"protocol":digest(protocol)})
        assert result["signature"]==signature
        if stage=="cv":assert selection["cv_result_signatures"][str((out/"results.json").relative_to(ROOT))]==signature
        training=pd.read_csv(directory/"train_rows.csv.gz")
        assert not set(training.patient_id)&excluded
        training_indices=np.flatnonzero(row_folds!=fold) if stage=="cv" else np.arange(len(rows))
        pd.testing.assert_frame_equal(training,rows.iloc[training_indices].reset_index(drop=True))
        np.testing.assert_array_equal(training.label,np.load(directory/"train_y.npy"))
        if stage=="cv":np.testing.assert_array_equal(training_indices,np.load(directory/"train_indices.npy"))
        processor,model=bundle["preprocessor"],bundle["model"]
        assert processor.names_==protocol["feature_columns"] and len(processor.names_)==312
        assert not set(processor.names_)&{"hospital","patient_id","label","onset_hour"}
        if str(directory) not in cases:
            verify_statistics(processor,raw,training_indices)
            matrix_path=directory/"train_X.npy" if stage=="cv" else AB_CASE/"train_X.npy"
            sample=np.unique(np.linspace(0,len(training_indices)-1,64,dtype=int))
            np.testing.assert_allclose(np.load(matrix_path,mmap_mode="r")[sample],processor.transform(raw[training_indices[sample]]),atol=0,rtol=0,equal_nan=True)
            cases[str(directory)]={"training_rows":len(training),"patient_count":int(training.patient_id.nunique()),"features":312}
        if stage=="cv":
            held_indices=np.flatnonzero(row_folds==fold);np.testing.assert_array_equal(held_indices,np.load(directory/"evaluation_indices.npy"))
            table=pd.read_csv(directory/"evaluation_rows.csv.gz")
            pd.testing.assert_frame_equal(table,rows.iloc[held_indices].reset_index(drop=True))
            np.testing.assert_array_equal(table.label,np.load(directory/"evaluation_y.npy"))
            assert not set(training.patient_id)&set(table.patient_id)
            p=np.load(out/"probability.npy");assert_dict_close(model_metrics(table,p),result["evaluation"])
            sample=np.unique(np.linspace(0,len(table)-1,64,dtype=int))
            matrix=np.load(directory/"evaluation_X.npy",mmap_mode="r")[sample]
            np.testing.assert_allclose(matrix,processor.transform(raw[held_indices[sample]]),atol=0,rtol=0,equal_nan=True)
            with threadpool_limits(limits=2):expected=model.booster_.predict(matrix) if model_name=="lightgbm" else model.predict_proba(matrix)[:,1]
            np.testing.assert_allclose(expected,p[sample],rtol=0,atol=1e-12)
            scores.append({**job,"macro_hospital_auprc":result["evaluation"]["macro_hospital_auprc"]})
            assert result["alerts"] is None and result["threshold"] is None
        else:
            assert selection["selected_candidates"][model_name]==candidate
            assert set(training.patient_id)==set(rows.patient_id)
            tables={s:pd.read_csv(PROCESSED/f"{s}_rows.csv.gz") for s in VALIDATION_SPLITS};probabilities={}
            for split,table in tables.items():
                assert not set(training.patient_id)&set(table.patient_id)
                p=np.load(out/f"{split}_probability.npy");probabilities[split]=p
                assert_dict_close(ranking_metrics(table.label.to_numpy(),p),result["evaluation"][split])
                sample=np.unique(np.linspace(0,len(table)-1,64,dtype=int))
                matrix=np.load(AB_CASE/f"{split}_X.npy",mmap_mode="r")[sample]
                with threadpool_limits(limits=2):expected=model.booster_.predict(matrix) if model_name=="lightgbm" else model.predict_proba(matrix)[:,1]
                np.testing.assert_allclose(expected,p[sample],rtol=0,atol=1e-12)
                alerts,_,_=alert_replay(table,p,result["threshold"],config()["alerts"]["suppression_hours"])
                assert alerts==result["alerts"][split]
            pooled_rows=pd.concat([tables[s] for s in VALIDATION_SPLITS],ignore_index=True)
            pooled_p=np.concatenate([probabilities[s] for s in VALIDATION_SPLITS])
            threshold,_=select_alert_threshold(pooled_rows,pooled_p,config())
            assert threshold==result["threshold"]==bundle["threshold"]
            assert result["alerts"]["pooled_validation"]["false_alerts_per_100_patient_hours"]<=1
            assert_dict_close(ranking_metrics(pooled_rows.label.to_numpy(),pooled_p),result["evaluation"]["pooled_validation"])
        for key,value in params.items():
            actual=model.get_params()[key]
            if model_name=="lightgbm" and key=="n_estimators":
                if not result["reused_full_baseline"]:assert actual==result["details"]["inner_selected_trees"]
            else:assert actual==value,(model_name,key,actual,value)
        if model_name!="lr" and not result["reused_full_baseline"]:
            _,expected=patient_bags(training,n_bags=3 if model_name=="ebm" else 1,seed=2026)
            pd.testing.assert_frame_equal(pd.read_csv(out/"internal_patients.csv"),expected,check_dtype=False,check_exact=True)
        verified.append({**job,"signature":signature})
        if len(verified)%6==0:print(f"Verified {len(verified)}/{len(jobs)} saved candidates",flush=True)
    chosen,comparisons=choose_candidates(scores,protocol["settings"]["study"])
    assert chosen==selection["selected_candidates"]==completion["selected_candidates"] and comparisons==selection["cv_comparisons"]
    reference=json.loads((OUTPUT/"baseline_validation/results.json").read_text())
    for name,r in reference.items():
        assert r["test_sets_evaluated"]==[]
        bundle=joblib.load(AB_MODELS/f"{name}_model.joblib")
        assert content_hash(AB_MODELS/f"{name}_model.joblib")==r["source_model_sha256"]
        for split in VALIDATION_SPLITS:
            table=pd.read_csv(PROCESSED/f"{split}_rows.csv.gz");p=np.load(OUTPUT/"baseline_validation"/f"{name}_{split}_probability.npy")
            assert_dict_close(ranking_metrics(table.label.to_numpy(),p),r["evaluation"][split])
            sample=np.unique(np.linspace(0,len(table)-1,64,dtype=int));matrix=np.load(AB_CASE/f"{split}_X.npy",mmap_mode="r")[sample]
            with threadpool_limits(limits=2):expected=bundle["model"].booster_.predict(matrix) if name=="lightgbm" else bundle["model"].predict_proba(matrix)[:,1]
            np.testing.assert_allclose(expected,p[sample],atol=1e-12,rtol=0)
    verify_missingness(rows,patients,raw)
    assert not list(OUTPUT.rglob("*test*probability*"))
    evidence={"cv_models_verified":33,"selected_validation_models_verified":3,"preprocessing_cases_verified":len(cases),
        "all_training_raw_rows_match_original_A_and_B_sources":True,"existing_patient_partitions_unchanged":True,
        "hospital_event_patient_folds_reproduced":True,"fold_train_only_preprocessing_statistics_recomputed":True,
        "sampled_transformed_matrices_and_serialized_predictions_verified":True,"internal_early_stopping_patients_reproduced":True,
        "candidate_parameters_and_selection_rule_reproduced":True,"validation_thresholds_reproduced":True,
        "missingness_rates_and_hospital_diagnostic_reproduced":True,"test_sets_evaluated":[],
        "A_test_patients_reserved":int(((original.hospital=="A")&(original.split=="test")).sum()),
        "B_test_patients_not_scored_this_round":int((b_split.expansion_split=="B_test").sum()),"cases":cases,"models":verified}
    write_json(OUTPUT/"verification.json",evidence);write_json(REPORT/"verification.json",evidence)
    print("All hospital-stratified CV, regularization and missingness checks passed.",flush=True)


if __name__=="__main__":main()
