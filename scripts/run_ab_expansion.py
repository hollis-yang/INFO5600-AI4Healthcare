"""Run fixed A-only, matched mixed-hospital and full A+B training experiments."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import FEATURE_NAMES,config,assess_patient
from sepsis_warning.feature_engineering import feature_names,alternative_features
from sepsis_warning.optimization import settings as feature_settings
from sepsis_warning.history_study import digest,child_environment
from sepsis_warning.ab_expansion import (PROCESSED,OUTPUT,EVALUATION_SPLITS,settings,content_hash,write_json,
    split_b_patients,matched_mixture,FixedColumnsPreprocessor,reference_path,case_key,result_directory)


def frame_summary(name,rows):
    patients = rows.groupby("patient_id").agg(hospital=("hospital","first"),has_event=("label","max"))
    result = {"cohort":name,"patients":len(patients),"event_patients":int(patients.has_event.sum()),
              "hours":len(rows),"positive_hours":int(rows.label.sum()),"positive_hour_prevalence":float(rows.label.mean())}
    for hospital in ["A","B"]:
        group = patients[patients.hospital == hospital]
        result[f"{hospital}_patients"],result[f"{hospital}_event_patients"] = len(group),int(group.has_event.sum())
    return result


def build_b_combined(b_rows,inputs):
    target = PROCESSED/"B_combined_features.npy"
    marker = PROCESSED/"B_combined_features.json"
    signature = digest({"inputs":inputs,"features":feature_settings()["features"],"source":content_hash(ROOT/"src/sepsis_warning/feature_engineering.py")})
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:return
    raw = np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    patients = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    matrix = np.lib.format.open_memmap(target,mode="w+",dtype=np.float32,shape=(len(b_rows),len(feature_names("combined"))))
    position = 0
    print("Building B causal dynamic features for the secondary LightGBM control",flush=True)
    for number,identifier in enumerate(b_rows.patient_id.drop_duplicates(),1):
        patient = patients.loc[identifier]
        values = raw[int(patient.raw_start):int(patient.raw_stop)]
        reason,eligible,_ = assess_patient(values,config())
        assert reason == "included"
        part = alternative_features(values,config(),feature_settings()["features"])[eligible]
        np.testing.assert_array_equal(values[eligible,40],b_rows.label.to_numpy()[position:position+len(part)])
        matrix[position:position+len(part)] = part
        position += len(part)
        if number%4000 == 0:print(f"  B dynamic features: {number} patients",flush=True)
    assert position == len(b_rows)
    matrix.flush()
    write_json(marker,{"signature":signature,"rows":len(b_rows),"feature_names":feature_names("combined")})


def raw_arrays(profile):
    if profile == "common":
        return (np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r"),
                np.load(ROOT/"processed_data/validation_features.npy",mmap_mode="r"),
                np.load(ROOT/"processed_data/external_test_features.npy",mmap_mode="r"),list(FEATURE_NAMES))
    return (np.load(ROOT/"processed_data/feature_optimization/train_combined_features.npy",mmap_mode="r"),
            np.load(ROOT/"processed_data/feature_optimization/validation_combined_features.npy",mmap_mode="r"),
            np.load(PROCESSED/"B_combined_features.npy",mmap_mode="r"),feature_names("combined"))


def prepare_case(cohort,profile,rows,a_rows,b_rows,evaluation_indices,protocol):
    key = case_key(cohort,profile)
    directory = PROCESSED/"cases"/key
    directory.mkdir(parents=True,exist_ok=True)
    signature = digest({"inputs":protocol["input_fingerprints"],"cohort":cohort,"profile":profile,
                        "patient_ids":sorted(rows.patient_id.unique().tolist()),"protocol":protocol})
    marker = directory/"case.json"
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:return key
    a_raw,a_validation,b_raw,names = raw_arrays(profile)
    a_index = np.flatnonzero(a_rows.patient_id.isin(rows.patient_id))
    b_index = np.flatnonzero(b_rows.patient_id.isin(rows.patient_id))
    selected_rows = pd.concat([a_rows.iloc[a_index],b_rows.iloc[b_index]],ignore_index=True)
    pd.testing.assert_frame_equal(selected_rows,rows.reset_index(drop=True))
    np.save(directory/"A_train_indices.npy",a_index)
    np.save(directory/"B_train_indices.npy",b_index)
    selected_rows.to_csv(directory/"train_rows.csv.gz",index=False,compression="gzip")
    np.save(directory/"train_y.npy",selected_rows.label.to_numpy(np.int8))
    allowed = protocol["feature_columns"][profile]
    if cohort == "A_only":
        model_name = "lr" if profile == "common" else "lightgbm"
        processor = joblib.load(OUTPUT/"references"/f"{profile}_{model_name}_model.joblib")["preprocessor"]
        assert processor.names_ == allowed
    else:
        selected = np.concatenate([a_raw[a_index],b_raw[b_index]],axis=0)
        processor = FixedColumnsPreprocessor(names,allowed,native=profile != "common",clip=profile == "common").fit(selected)
        matrix = np.lib.format.open_memmap(directory/"train_X.npy",mode="w+",dtype=np.float32,shape=(len(rows),len(allowed)))
        for start in range(0,len(selected),15000):matrix[start:start+15000] = processor.transform(selected[start:start+15000])
        matrix.flush()
        del selected,matrix
    for split in EVALUATION_SPLITS:
        source,indices = (a_validation,evaluation_indices[split]) if split == "A_validation" else (b_raw,evaluation_indices[split])
        matrix = np.lib.format.open_memmap(directory/f"{split}_X.npy",mode="w+",dtype=np.float32,shape=(len(indices),len(allowed)))
        for start in range(0,len(indices),15000):matrix[start:start+15000] = processor.transform(source[indices[start:start+15000]])
        matrix.flush()
        del matrix
    joblib.dump(processor,directory/"preprocessor.joblib",compress=3)
    metadata = {"key":key,"signature":signature,"cohort":cohort,"profile":profile,"features":len(allowed),
                "training_patients":int(rows.patient_id.nunique()),"training_event_patients":int(rows.groupby("patient_id").label.max().sum()),
                "training_hours":len(rows),"normalizer_fitted_on":"this training cohort only", "feature_columns_frozen_from_A":True}
    write_json(marker,metadata)
    print(f"Prepared {key}: {metadata['training_patients']} patients, {metadata['training_event_patients']} events",flush=True)
    return key


def run_jobs(jobs,study):
    pending,active = jobs.copy(),[]
    costs = {"lr":4,"ebm":3,"lightgbm":3}
    completed = 0
    last = time.monotonic()
    try:
        while pending or active:
            for item in active.copy():
                job,process,log = item
                if process.poll() is not None:
                    active.remove(item);log.close()
                    if process.returncode:
                        raise RuntimeError(f"Expansion worker failed: {job}\n"+(result_directory(job["case"])/f"{job['model']}.log").read_text()[-6000:])
                    r = json.loads((result_directory(job["case"])/f"{job['model']}_results.json").read_text())
                    completed += 1
                    print(f"Done {completed}/{len(jobs)} {job['case']}/{job['model']}: A-val AP={r['evaluation']['A_validation']['auprc_average_precision']:.4f}; B-test AP={r['evaluation']['B_test']['auprc_average_precision']:.4f}",flush=True)
            used = sum(costs[j["model"]] for j,_,_ in active)
            ebms = sum(j["model"] == "ebm" for j,_,_ in active)
            for job in pending.copy():
                if len(active) >= study["maximum_workers"]:break
                if used+costs[job["model"]] > study["cpu_budget"]:continue
                if job["model"] == "ebm" and ebms >= study["maximum_ebm_workers"]:continue
                directory = result_directory(job["case"])
                directory.mkdir(parents=True,exist_ok=True)
                log = (directory/f"{job['model']}.log").open("w")
                process = subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/fit_ab_model.py"),"--case",job["case"],"--model",job["model"]],
                                           cwd=ROOT,env=child_environment(),stdout=log,stderr=subprocess.STDOUT)
                active.append((job,process,log));pending.remove(job)
                used += costs[job["model"]];ebms += job["model"] == "ebm"
                print("Started "+job["case"]+"/"+job["model"],flush=True)
            if time.monotonic()-last > 35:
                print(f"Expansion progress {completed}/{len(jobs)}; active "+", ".join(j["case"]+"/"+j["model"] for j,_,_ in active),flush=True)
                last = time.monotonic()
            if active:time.sleep(2)
    finally:
        for _,process,log in active:process.terminate();log.close()


def main():
    started = time.monotonic()
    study = settings()
    OUTPUT.mkdir(parents=True,exist_ok=True);PROCESSED.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,"-c","from interpret.glassbox import ExplainableBoostingClassifier; import lightgbm; print('Pinned model imports ready')"],env=child_environment(),check=True)
    sources = [ROOT/f"processed_data/{name}" for name in ["train_features.npy","validation_features.npy","external_test_features.npy",
               "train_rows.csv.gz","validation_rows.csv.gz","external_test_rows.csv.gz","splits/patient_splits.csv","raw_records.npy"]]
    sources += [ROOT/"outputs/patient_summary.csv",ROOT/"outputs/preprocessor.joblib",ROOT/"data/manifest.csv"]
    inputs = {}
    for path in sources:
        print("Fingerprinting "+str(path.relative_to(ROOT)),flush=True)
        inputs[str(path.relative_to(ROOT))] = content_hash(path)
    a_rows = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    a_validation = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    b_rows = pd.read_csv(ROOT/"processed_data/external_test_rows.csv.gz")
    assert set(a_rows.hospital) == {"A"} and set(b_rows.hospital) == {"B"}
    b_split = split_b_patients(b_rows,study["study"])
    b_split.to_csv(PROCESSED/"B_patient_splits.csv",index=False)
    original_splits = pd.read_csv(ROOT/"processed_data/splits/patient_splits.csv")
    versioned_splits = original_splits.copy()
    mapping = b_split.set_index("patient_id").expansion_split
    versioned_splits["expansion_split"] = versioned_splits["split"]
    mask = versioned_splits.hospital == "B"
    versioned_splits.loc[mask,"expansion_split"] = versioned_splits.loc[mask,"patient_id"].map(mapping)
    versioned_splits.to_csv(PROCESSED/"patient_splits.csv",index=False)
    b_train_ids = set(b_split.loc[b_split.expansion_split == "B_train","patient_id"])
    b_train = b_rows[b_rows.patient_id.isin(b_train_ids)].reset_index(drop=True)
    pool = pd.concat([a_rows,b_train],ignore_index=True)
    cohorts = {"A_only":a_rows,"A_B_full":pool}
    for seed in study["study"]["matched_sample_seeds"]:
        ids = matched_mixture(pool,a_rows.patient_id.nunique(),seed)
        cohorts[f"A_B_matched_seed{seed}"] = pool[pool.patient_id.isin(ids)].reset_index(drop=True)
    summaries = [frame_summary(name,rows) for name,rows in cohorts.items()]
    pd.DataFrame(summaries).to_csv(OUTPUT/"training_cohorts.csv",index=False)
    members = []
    for name,rows in cohorts.items():
        table = rows.groupby("patient_id").agg(hospital=("hospital","first"),has_event=("label","max")).reset_index()
        table["cohort"] = name;members.append(table)
    pd.concat(members,ignore_index=True).to_csv(OUTPUT/"cohort_membership.csv",index=False)
    evaluation_indices = {"A_validation":np.arange(len(a_validation))}
    for split in ["B_validation","B_test"]:
        ids = set(b_split.loc[b_split.expansion_split == split,"patient_id"])
        indices = np.flatnonzero(b_rows.patient_id.isin(ids))
        evaluation_indices[split] = indices
        b_rows.iloc[indices].reset_index(drop=True).to_csv(PROCESSED/f"{split}_rows.csv.gz",index=False,compression="gzip")
        np.save(PROCESSED/f"{split}_indices.npy",indices)
    a_validation.to_csv(PROCESSED/"A_validation_rows.csv.gz",index=False,compression="gzip")
    np.save(PROCESSED/"A_validation_indices.npy",evaluation_indices["A_validation"])
    evaluations = [frame_summary(split,pd.read_csv(PROCESSED/f"{split}_rows.csv.gz")) for split in EVALUATION_SPLITS]
    pd.DataFrame(evaluations).to_csv(OUTPUT/"evaluation_cohorts.csv",index=False)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)
    references = OUTPUT/"references";references.mkdir(exist_ok=True)
    reference_hashes,feature_columns = {},{}
    for profile,models in [("common",study["study"]["models"]),("enhanced_lightgbm",["lightgbm"])]:
        for model in models:
            source = reference_path(profile,model)
            body = source.read_bytes();assert body
            target = references/f"{profile}_{model}_model.joblib"
            target.write_bytes(body)
            source_results = source.with_name(f"{model}_results.json")
            (references/f"{profile}_{model}_results.json").write_bytes(source_results.read_bytes())
            # LightGBM needs its OpenMP runtime at interpreter startup; inspect names via metadata
            # for that reference, while the worker verifies the serialized bundle as well.
            if model == "lr" or profile == "enhanced_lightgbm":
                if profile == "common":feature_columns[profile] = joblib.load(target)["preprocessor"].names_
                else:
                    meta = json.loads((ROOT/"processed_data/feature_optimization/cases/full_combined_f3a8818e555b/case.json").read_text())
                    p = joblib.load(ROOT/"processed_data/feature_optimization/cases"/meta["key"]/"preprocessor.joblib")
                    feature_columns[profile] = p.names_
            reference_hashes[str(source.relative_to(ROOT))] = content_hash(target)
    assert len(feature_columns["common"]) == 312 and len(feature_columns["enhanced_lightgbm"]) == 349
    protocol = {"study":study,"input_fingerprints":inputs,"reference_model_fingerprints":reference_hashes,
                "model_settings":json.loads((ROOT/"reports/history_study/model_settings.json").read_text()),
                "feature_columns":feature_columns,"model_feature_ids_excluded":["patient_id","hospital"],
                "matched_control":"Two training-patient draws at A-only N, proportional to the full mixture's hospital/event strata.",
                "threshold_calibration":"Same pooled A-validation+B-validation for every method; A-only refers to risk-model training, not calibration.",
                "A_test_evaluated":False,"B_test_is_unseen_hospital":False,"hyperparameters_tuned":False,
                "source_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in [ROOT/"configs/ab_expansion.toml",Path(__file__),
                   ROOT/"scripts/fit_ab_model.py",ROOT/"src/sepsis_warning/ab_expansion.py",ROOT/"src/sepsis_warning/data.py",ROOT/"src/sepsis_warning/feature_engineering.py"]}}
    write_json(OUTPUT/"protocol.json",protocol)
    write_json(OUTPUT/"input_fingerprints.json",inputs)
    build_b_combined(b_rows,inputs)
    jobs = []
    # Prepare all transformations before launching model workers to bound peak RAM.
    for cohort in ["A_only","A_B_full",*[f"A_B_matched_seed{s}" for s in study["study"]["matched_sample_seeds"]]]:
        key = prepare_case(cohort,"common",cohorts[cohort],a_rows,b_rows,evaluation_indices,protocol)
        for model in study["study"]["models"]:jobs.append({"case":key,"model":model,"cohort":cohort,"profile":"common"})
    for cohort in ["A_only","A_B_full"]:
        key = prepare_case(cohort,"enhanced_lightgbm",cohorts[cohort],a_rows,b_rows,evaluation_indices,protocol)
        jobs.append({"case":key,"model":"lightgbm","cohort":cohort,"profile":"enhanced_lightgbm"})
    write_json(OUTPUT/"model_manifest.json",jobs)
    run_jobs(jobs,study["study"])
    assert all(content_hash(ROOT/name) == expected for name,expected in inputs.items())
    assert all(content_hash(ROOT/name) == expected for name,expected in protocol["source_fingerprints"].items())
    write_json(OUTPUT/"completed.json",{"model_results":len(jobs),"new_fits":10,"reused_A_only_models":4,
              "training_cohorts":summaries,"canonical_inputs_unchanged":True,"A_test_evaluated":False,
              "B_test_is_unseen_hospital":False,"elapsed_seconds":round(time.monotonic()-started,2)})
    print("A+B expansion experiment complete.",flush=True)


if __name__ == "__main__":main()
