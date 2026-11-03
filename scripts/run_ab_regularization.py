"""Prepare hospital-stratified patient CV, tune only penalties, freeze selection, validate."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import joblib
import numpy as np
import pandas as pd
from sepsis_warning.ab_regularization import (PROCESSED,OUTPUT,REPORT,AB_CASE,AB_MODELS,VALIDATION_SPLITS,
    settings,patient_folds,model_candidates,choose_candidates,runtime_environment,output_directory)
from sepsis_warning.ab_expansion import content_hash,write_json,FixedColumnsPreprocessor
from sepsis_warning.data import FEATURE_NAMES
from sepsis_warning.history_study import digest


def prepare_raw_pool(protocol,rows):
    marker=PROCESSED/"raw_pool.json"
    signature=digest(protocol["input_fingerprints"])
    if marker.exists() and json.loads(marker.read_text())["signature"]==signature:return
    a=np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r")
    b=np.load(ROOT/"processed_data/external_test_features.npy",mmap_mode="r")
    ia,ib=np.load(AB_CASE/"A_train_indices.npy"),np.load(AB_CASE/"B_train_indices.npy")
    matrix=np.lib.format.open_memmap(PROCESSED/"training_features.npy",mode="w+",dtype=np.float32,shape=(len(rows),318))
    position=0
    for source,indices in [(a,ia),(b,ib)]:
        for start in range(0,len(indices),15000):
            part=source[indices[start:start+15000]];matrix[position:position+len(part)]=part;position+=len(part)
    assert position==len(rows);matrix.flush()
    write_json(marker,{"signature":signature,"shape":list(matrix.shape),"sha256":content_hash(PROCESSED/"training_features.npy")})
    print("Prepared unchanged A+B raw training features",flush=True)


def prepare_case(fold,rows,row_folds,protocol):
    directory=PROCESSED/f"fold{fold}";directory.mkdir(exist_ok=True)
    signature=digest({"protocol":protocol,"fold":fold})
    if (directory/"case.json").exists() and json.loads((directory/"case.json").read_text())["signature"]==signature:return
    training,evaluation=np.flatnonzero(row_folds!=fold),np.flatnonzero(row_folds==fold)
    source=np.load(PROCESSED/"training_features.npy",mmap_mode="r")
    selected=source[training]
    processor=FixedColumnsPreprocessor(FEATURE_NAMES,protocol["feature_columns"]).fit(selected)
    del selected
    for split,indices in [("train",training),("evaluation",evaluation)]:
        np.save(directory/f"{split}_indices.npy",indices)
        np.save(directory/f"{split}_y.npy",rows.label.to_numpy(np.int8)[indices])
        rows.iloc[indices].reset_index(drop=True).to_csv(directory/f"{split}_rows.csv.gz",index=False,compression="gzip")
        matrix=np.lib.format.open_memmap(directory/f"{split}_X.npy",mode="w+",dtype=np.float32,shape=(len(indices),312))
        for start in range(0,len(indices),15000):matrix[start:start+15000]=processor.transform(source[indices[start:start+15000]])
        matrix.flush();del matrix
    joblib.dump(processor,directory/"preprocessor.joblib",compress=3)
    write_json(directory/"case.json",{"signature":signature,"stage":"cv","fold":fold,"training_patients":int(rows.iloc[training].patient_id.nunique()),
        "training_event_patients":int(rows.iloc[training].groupby("patient_id").label.max().sum()),"training_hours":len(training),
        "evaluation_patients":int(rows.iloc[evaluation].patient_id.nunique()),"evaluation_hours":len(evaluation),"features":312,"preprocessing_fitted_on":"outer CV training patients only"})
    print(f"Prepared fold {fold}: {len(training):,} training hours; {len(evaluation):,} held-out hours",flush=True)


def run_jobs(jobs,study):
    pending,active=jobs.copy(),[]
    costs={"lr":study["lr_threads"],"ebm":3,"lightgbm":3}
    done=0;last=time.monotonic()
    try:
        while pending or active:
            for item in active.copy():
                job,process,log=item
                if process.poll() is not None:
                    active.remove(item);log.close()
                    directory=output_directory(job["stage"],job["fold"],job["model"],job["candidate"])
                    if process.returncode:raise RuntimeError(f"Candidate failed {job}:\n"+(directory/"worker.log").read_text()[-7000:])
                    result=json.loads((directory/"results.json").read_text());done+=1
                    print(f"Done {done}/{len(jobs)} {job}: macro AP={result['evaluation']['macro_hospital_auprc']:.5f}",flush=True)
            used=sum(costs[j["model"]] for j,_,_ in active);ebms=sum(j["model"]=="ebm" for j,_,_ in active)
            for job in pending.copy():
                if len(active)>=study["maximum_workers"]:break
                if used+costs[job["model"]]>study["cpu_budget"]:continue
                if job["model"]=="ebm" and ebms>=study["maximum_ebm_workers"]:continue
                directory=output_directory(job["stage"],job["fold"],job["model"],job["candidate"]);directory.mkdir(parents=True,exist_ok=True)
                log=(directory/"worker.log").open("w")
                process=subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/fit_ab_regularization.py"),"--stage",job["stage"],"--fold",str(job["fold"]),
                    "--model",job["model"],"--candidate",job["candidate"]],cwd=ROOT,env=runtime_environment(),stdout=log,stderr=subprocess.STDOUT)
                active.append((job,process,log));pending.remove(job);used+=costs[job["model"]];ebms+=job["model"]=="ebm"
                print("Started "+str(job),flush=True)
            if time.monotonic()-last>35:
                print(f"CV progress {done}/{len(jobs)}; active "+", ".join(f"{j['model']}/{j['candidate']}/fold{j['fold']}" for j,_,_ in active),flush=True);last=time.monotonic()
            if active:time.sleep(2)
    finally:
        for _,process,log in active:process.terminate();log.close()


def main():
    started=time.monotonic();study=settings()
    OUTPUT.mkdir(parents=True,exist_ok=True);PROCESSED.mkdir(parents=True,exist_ok=True)
    runtime=subprocess.check_output([sys.executable,"-c","import json,interpret,lightgbm;from importlib.metadata import version;print(json.dumps({'interpret-core':version('interpret-core'),'lightgbm':version('lightgbm'),'interpret_path':interpret.__file__,'lightgbm_path':lightgbm.__file__}))"],env=runtime_environment(),timeout=30,text=True)
    print(runtime.strip(),flush=True)
    source_paths=[ROOT/"configs/ab_regularization.toml",Path(__file__),ROOT/"scripts/fit_ab_regularization.py",
        ROOT/"src/sepsis_warning/ab_regularization.py",ROOT/"src/sepsis_warning/ab_expansion.py",ROOT/"src/sepsis_warning/data.py",
        ROOT/"src/sepsis_warning/feature_engineering.py",ROOT/"src/sepsis_warning/regularized.py",ROOT/"src/sepsis_warning/evaluation.py"]
    inputs=[ROOT/f"processed_data/{name}" for name in ["train_features.npy","external_test_features.npy","splits/patient_splits.csv"]]
    inputs += [AB_CASE/name for name in ["train_rows.csv.gz","A_train_indices.npy","B_train_indices.npy","preprocessor.joblib","train_X.npy","A_validation_X.npy","B_validation_X.npy"]]
    inputs += [ROOT/"processed_data/ab_expansion"/name for name in ["B_patient_splits.csv","patient_splits.csv","A_validation_rows.csv.gz","B_validation_rows.csv.gz"]]
    inputs += [ROOT/"reports/ab_expansion/protocol.json",ROOT/"reports/history_study/model_settings.json"]
    inputs += [AB_MODELS/f"{m}_model.joblib" for m in study["study"]["models"]]
    fingerprints={}
    for path in inputs:
        if getattr(path.stat(),"st_flags",0)&0x40000000:raise OSError(f"Required input is an unavailable macOS placeholder: {path}")
        print("Fingerprint "+str(path.relative_to(ROOT)),flush=True);fingerprints[str(path.relative_to(ROOT))]=content_hash(path)
    assert content_hash(ROOT/"processed_data/ab_expansion/B_patient_splits.csv")==content_hash(ROOT/"reports/ab_expansion/B_patient_splits.csv")
    rows=pd.read_csv(AB_CASE/"train_rows.csv.gz")
    original=pd.read_csv(ROOT/"processed_data/splits/patient_splits.csv")
    b_splits=pd.read_csv(ROOT/"processed_data/ab_expansion/B_patient_splits.csv")
    allowed=set(original.loc[original.hospital.eq("A")&original.split.eq("train"),"patient_id"])|set(b_splits.loc[b_splits.expansion_split.eq("B_train"),"patient_id"])
    assert set(rows.patient_id)==allowed and len(rows)==918004
    rows.to_csv(PROCESSED/"training_rows.csv.gz",index=False,compression="gzip")
    patients,row_folds=patient_folds(rows,study["study"]["folds"],study["study"]["fold_seed"])
    patients.to_csv(OUTPUT/"patient_cv_folds.csv",index=False);np.save(PROCESSED/"row_folds.npy",row_folds)
    counts=patients.groupby(["fold","hospital","has_event"]).size().rename("patients").reset_index()
    counts.to_csv(OUTPUT/"fold_strata.csv",index=False)
    feature_columns=json.loads((ROOT/"reports/ab_expansion/protocol.json").read_text())["feature_columns"]["common"]
    assert len(feature_columns)==312
    frozen=json.loads((ROOT/"reports/history_study/model_settings.json").read_text())
    protocol={"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":study,"candidates":model_candidates(study),"feature_columns":feature_columns,
        "input_fingerprints":fingerprints,"source_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in source_paths},
        "runtime":json.loads(runtime),"lightgbm_stopping":frozen["lightgbm_stopping"],"outer_validation_used_for_selection":False,"test_sets_evaluated":[],
        "training_patients":int(rows.patient_id.nunique()),"event_patients":int(patients.has_event.sum()),"patient_folds_sha256":content_hash(OUTPUT/"patient_cv_folds.csv")}
    # On resume the original timestamp is part of the frozen cache signatures.
    if (OUTPUT/"protocol.json").exists():
        previous=json.loads((OUTPUT/"protocol.json").read_text())
        a,b=dict(previous),dict(protocol);a.pop("created_at_utc");b.pop("created_at_utc")
        assert a==b,"Protocol changed; keep existing results separate instead of silently replacing them"
        protocol=previous
    write_json(OUTPUT/"protocol.json",protocol)
    prepare_raw_pool(protocol,rows)
    for fold in range(study["study"]["folds"]):prepare_case(fold,rows,row_folds,protocol)
    directory=PROCESSED/"full";directory.mkdir(exist_ok=True)
    shutil.copyfile(PROCESSED/"training_rows.csv.gz",directory/"train_rows.csv.gz")
    np.save(directory/"train_y.npy",rows.label.to_numpy(np.int8));shutil.copyfile(AB_CASE/"preprocessor.joblib",directory/"preprocessor.joblib")
    write_json(directory/"case.json",{"signature":digest({"protocol":protocol,"stage":"full"}),"stage":"validation","fold":-1,
        "training_patients":len(patients),"training_event_patients":int(patients.has_event.sum()),"training_hours":len(rows),"features":312,
        "preprocessing_fitted_on":"same full A+B training pool; original A+B preprocessor reused"})
    for split in VALIDATION_SPLITS:shutil.copyfile(ROOT/"processed_data/ab_expansion"/f"{split}_rows.csv.gz",PROCESSED/f"{split}_rows.csv.gz")
    jobs=[{"stage":"cv","fold":fold,"model":candidate["model"],"candidate":candidate["candidate"]}
        for candidate in sorted(protocol["candidates"],key=lambda c:({"ebm":0,"lightgbm":1,"lr":2}[c["model"]],c["candidate"])) for fold in range(3)]
    write_json(OUTPUT/"cv_manifest.json",jobs);run_jobs(jobs,study["study"])
    fold_scores=[];signatures={}
    for job in jobs:
        path=output_directory(job["stage"],job["fold"],job["model"],job["candidate"])/"results.json"
        result=json.loads(path.read_text());signatures[str(path.relative_to(ROOT))]=result["signature"]
        fold_scores.append({**job,"macro_hospital_auprc":result["evaluation"]["macro_hospital_auprc"]})
    selected,comparisons=choose_candidates(fold_scores,study["study"])
    selection={"selected_candidates":selected,"cv_comparisons":comparisons,"protocol_digest":digest(protocol),"cv_result_signatures":signatures,
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),"outer_validation_used_for_selection":False,"test_sets_evaluated":[]}
    write_json(OUTPUT/"selection.json",selection)
    print("Frozen CV selection: "+str(selected),flush=True)
    final=[{"stage":"validation","fold":-1,"model":m,"candidate":selected[m]} for m in study["study"]["models"]]
    write_json(OUTPUT/"validation_manifest.json",final);run_jobs(final,study["study"])
    for mapping in [fingerprints,protocol["source_fingerprints"]]:
        for name,expected in mapping.items():
            if getattr((ROOT/name).stat(),"st_flags",0)&0x40000000:raise OSError(f"Post-run input unavailable: {name}")
            assert content_hash(ROOT/name)==expected,name
    write_json(OUTPUT/"completed.json",{"cv_models":len(jobs),"selected_validation_models":len(final),"selected_candidates":selected,
        "canonical_inputs_and_sources_unchanged":True,"test_sets_evaluated":[],"elapsed_seconds":round(time.monotonic()-started,2)})
    print("Hospital-stratified regularization experiment completed.",flush=True)


if __name__=="__main__":main()
