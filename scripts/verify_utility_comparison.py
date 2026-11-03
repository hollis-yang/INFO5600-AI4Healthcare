"""Verify full raw trajectories, frozen-model utility and the official scoring oracle."""
from __future__ import annotations
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import OUTPUT,PROCESSED,REPORT,VALIDATION_SPLITS
from sepsis_warning.ab_expansion import content_hash,write_json
from sepsis_warning.data import causal_features,config
from sepsis_warning.utility import official_scorer,patient_utility,normalized_utility


def main():
    out=OUTPUT/"utility";data=PROCESSED/"utility"
    protocol=json.loads((out/"protocol.json").read_text());completion=json.loads((out/"completed.json").read_text())
    assert completion["models"]==9 and completion["metric_rows"]==54 and completion["test_sets_evaluated"]==[]
    assert not completion["utility_used_for_model_or_threshold_selection"] and not protocol["model_choice_uses_utility"]
    assert completion["regularization_selection_sha256"]==content_hash(OUTPUT/"selection.json")
    for mapping in [protocol["input_fingerprints"],protocol["source_fingerprints"]]:
        for path,expected in mapping.items():assert content_hash(ROOT/path)==expected,path
    for name,metadata in protocol["upstream"]["files"].items():
        path=ROOT/"third_party/physionet2019"/name;body=path.read_bytes()
        assert hashlib.sha1(b"blob "+str(len(body)).encode()+b"\0"+body).hexdigest()==metadata["git_blob_sha"]
        assert content_hash(path)==metadata["sha256"]
    raw=np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    training=pd.read_csv(PROCESSED/"training_rows.csv.gz")
    evaluation,features,labels={}, {},{}
    total_patients=total_hours=0
    for split in VALIDATION_SPLITS:
        directory=data/split;case=json.loads((directory/"case.json").read_text())
        assert case["all_raw_hours_retained"] and not case["labels_shifted"]
        assert content_hash(directory/"features.npy")==case["features_sha256"]
        assert content_hash(directory/"labels.npy")==case["labels_sha256"]
        evaluation[split]=patients=pd.read_csv(directory/"patients.csv")
        features[split]=np.load(directory/"features.npy",mmap_mode="r");labels[split]=np.load(directory/"labels.npy")
        table=pd.read_csv(directory/"rows.csv.gz")
        assert len(table)==case["full_hours"]==len(labels[split]) and len(patients)==case["patients"]
        np.testing.assert_array_equal(labels[split],table.label)
        eligible=pd.read_csv(PROCESSED/f"{split}_rows.csv.gz")
        pd.testing.assert_frame_equal(table.loc[table.task_eligible,eligible.columns].reset_index(drop=True),eligible)
        assert set(patients.patient_id)==set(eligible.patient_id) and not set(patients.patient_id)&set(training.patient_id)
        position=0
        for patient in patients.itertuples(index=False):
            assert patient.start==position and patient.stop-patient.start==patient.raw_stop-patient.raw_start==patient.hours
            np.testing.assert_array_equal(labels[split][patient.start:patient.stop],raw[patient.raw_start:patient.raw_stop,40])
            np.testing.assert_array_equal(table.hour.to_numpy()[patient.start:patient.stop],raw[patient.raw_start:patient.raw_stop,39])
            position=int(patient.stop)
        assert position==len(table)
        for status in [False,True]:
            patient=patients.loc[patients.has_event.eq(status)].iloc[0]
            values=np.array(raw[int(patient.raw_start):int(patient.raw_stop)],copy=True)
            expected=causal_features(values,config())
            np.testing.assert_allclose(expected,features[split][int(patient.start):int(patient.stop)],rtol=0,atol=0,equal_nan=True)
            prefix=min(12,len(values));np.testing.assert_allclose(expected[:prefix],causal_features(values[:prefix],config()),rtol=0,atol=0,equal_nan=True)
        total_patients+=len(patients);total_hours+=len(table)
    scores=pd.read_csv(out/"all_scores.csv").set_index(["stage","model","policy","evaluation"])
    model_manifest=json.loads((out/"model_manifest.json").read_text());verified=0
    for item in model_manifest:
        stage,model=item["stage"],item["model"];path=ROOT/item["source_model"]
        assert content_hash(path)==item["source_model_sha256"]
        bundle=joblib.load(path)
        assert bundle["parameters"]==item["parameters"] and float(bundle["threshold"])==item["thresholds"]["existing_validation_budget"]
        for policy,threshold in item["thresholds"].items():
            components=[]
            for split in VALIDATION_SPLITS:
                directory=out/stage/model;probability=np.load(directory/f"{split}_probability.npy")
                assert len(probability)==len(labels[split]) and np.isfinite(probability).all()
                sample=np.unique(np.linspace(0,len(probability)-1,64,dtype=int))
                matrix=bundle["preprocessor"].transform(features[split][sample])
                with threadpool_limits(limits=2):
                    expected=bundle["model"].booster_.predict(matrix,num_threads=2) if model=="lightgbm" else bundle["model"].predict_proba(matrix)[:,1]
                np.testing.assert_allclose(expected,probability[sample],rtol=0,atol=1e-12)
                table=pd.read_csv(directory/f"{split}_{policy}_patient_utilities.csv")
                np.testing.assert_array_equal(table.patient_id,evaluation[split].patient_id)
                for patient,record in zip(evaluation[split].itertuples(index=False),table.itertuples(index=False)):
                    result=patient_utility(labels[split][patient.start:patient.stop],probability[patient.start:patient.stop]>=threshold)
                    for key,value in result.items():np.testing.assert_allclose(value,getattr(record,key),rtol=0,atol=1e-12)
                for key,value in normalized_utility(table).items():np.testing.assert_allclose(value,scores.loc[(stage,model,policy,split),key],rtol=0,atol=1e-9)
                components.append(table)
            pooled=pd.concat(components,ignore_index=True)
            for key,value in normalized_utility(pooled).items():np.testing.assert_allclose(value,scores.loc[(stage,model,policy,"pooled_validation"),key],rtol=0,atol=1e-9)
        verified+=1;print(f"Verified official utility {verified}/9: {stage}/{model}",flush=True)
    # Independent upstream directory entry point on real event/non-event patients.
    checks=[]
    for split in VALIDATION_SPLITS:
        patients=evaluation[split]
        selected=pd.concat([patients.loc[patients.has_event.eq(status)].head(1) for status in [False,True]])
        p=np.load(out/"A_B_baseline/ebm"/f"{split}_probability.npy")
        threshold=next(m for m in model_manifest if m["stage"]=="A_B_baseline" and m["model"]=="ebm")["thresholds"]["existing_validation_budget"]
        contributions=[]
        with tempfile.TemporaryDirectory(prefix="info5600_official_utility_") as temporary:
            lab=Path(temporary)/"labels";pred=Path(temporary)/"predictions";lab.mkdir();pred.mkdir()
            for i,patient in enumerate(selected.itertuples(index=False)):
                y=labels[split][patient.start:patient.stop];prob=p[patient.start:patient.stop];binary=prob>=threshold
                np.savetxt(lab/f"p{i}.psv",y,header="SepsisLabel",comments="",fmt="%d")
                np.savetxt(pred/f"p{i}.psv",np.c_[prob,binary],header="PredictedProbability|PredictedLabel",comments="",delimiter="|")
                contributions.append(patient_utility(y,binary))
            oracle=official_scorer().evaluate_sepsis_score(str(lab),str(pred))[4]
        np.testing.assert_allclose(oracle,normalized_utility(pd.DataFrame(contributions))["normalized_utility"],rtol=0,atol=1e-12)
        checks.append(split)
    evidence={"models_verified":verified,"full_validation_patients":total_patients,"full_validation_hours":total_hours,
        "all_original_labels_and_hours_reproduced":True,"full_features_and_prefix_causality_checked":True,
        "serialized_model_predictions_and_per_patient_utilities_reproduced":True,"official_upstream_blobs_and_license_verified":True,
        "real_patient_upstream_directory_evaluator_matches_wrapper":checks,"utility_used_for_selection":False,"test_sets_evaluated":[]}
    write_json(out/"verification.json",evidence);write_json(REPORT/"utility/verification.json",evidence)
    print("All official utility comparison checks passed.",flush=True)


if __name__=="__main__":main()
