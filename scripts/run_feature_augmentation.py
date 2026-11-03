"""Supplement the replacement study with prespecified information-preserving controls."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))

import numpy as np
from sepsis_warning.data import save_json
from sepsis_warning.history_study import digest,sha_file,child_environment
from sepsis_warning.optimization import OUTPUT,PROCESSED,settings,choose_profiles
from sepsis_warning.augmentation import (ADDED_OUTPUT,ADDED_PROFILES,added_job,build_added,prepare_added_case)
from run_feature_optimization import build_combined,make_job,run_jobs


def fit_ready(jobs,study,inputs,row_folds):
    pending,active = jobs.copy(),[]
    costs = {"lr":4,"ebm":3,"lightgbm":3}
    last = time.monotonic()
    completed = 0
    try:
        while pending or active:
            for item in active.copy():
                job,process,log = item
                if process.poll() is not None:
                    active.remove(item);log.close()
                    if process.returncode:
                        raise RuntimeError((OUTPUT/job["stage"]/job["case"]/f"{job['model']}.log").read_text()[-6000:])
                    result = json.loads((OUTPUT/job["stage"]/job["case"]/f"{job['model']}_results.json").read_text())
                    completed += 1
                    print(f"Added comparison {completed}/{len(jobs)}: {job['profile']}/{job['fold']}/{job['model']} AP={result['evaluation']['auprc_average_precision']:.4f}",flush=True)
            used = sum(costs[j["model"]] for j,_,_ in active)
            for job in pending.copy():
                if len(active) >= study["study"]["maximum_workers"]:break
                if used+costs[job["model"]] > study["study"]["worker_cpu_budget"]:continue
                prepare_added_case(job,study,inputs,row_folds)
                out = OUTPUT/job["stage"]/job["case"]
                out.mkdir(parents=True,exist_ok=True)
                log = (out/f"{job['model']}.log").open("w")
                process = subprocess.Popen([sys.executable,"-u",str(ROOT/"scripts/fit_feature_model.py"),"--case",job["case"],"--model",job["model"]],
                                           cwd=ROOT,env=child_environment(),stdout=log,stderr=subprocess.STDOUT)
                active.append((job,process,log));pending.remove(job)
                used += costs[job["model"]]
                print(f"Added comparison started: {job['profile']}/{job['fold']}/{job['model']}",flush=True)
            if time.monotonic()-last > 35:
                print(f"Added comparison progress: {completed}/{len(jobs)}",flush=True);last=time.monotonic()
            if active:time.sleep(2)
    finally:
        for _,process,log in active:process.terminate();log.close()


def main():
    started = time.monotonic()
    study = settings()
    ADDED_OUTPUT.mkdir(parents=True,exist_ok=True)
    design_path = ADDED_OUTPUT/"design.json"
    if not design_path.exists():
        save_json(design_path,{"created_utc":datetime.now(timezone.utc).isoformat(),
             "reason":"All three classifiers lost mean CV AP when old lab summaries were replaced; preserve original information and append dynamics as a separate control.",
             "profiles":ADDED_PROFILES,"raw_features":459,"initial_outer_validation_manifest_present_at_design":(OUTPUT/"validation_manifest.json").exists(),
             "selection_uses_outer_validation_metrics":False,"same_patient_folds":True,
             "source_fingerprints":{str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/"src/sepsis_warning/augmentation.py"]},
             "classifier_settings_unchanged":True,"test_sets_evaluated":[]})
    design = json.loads(design_path.read_text())
    assert all(sha_file(ROOT/p) == expected for p,expected in design["source_fingerprints"].items())
    build_added("train")
    while not (OUTPUT/"completed.json").exists():
        print("Added control fixed; waiting for initial study before using model CPU slots.",flush=True)
        time.sleep(35)
    inputs = json.loads((OUTPUT/"input_fingerprints.json").read_text())
    row_folds = np.load(PROCESSED/"row_folds.npy")
    jobs = [added_job(profile,model,fold) for profile,parameters in ADDED_PROFILES.items()
            for fold in range(study["study"]["folds"]) for model in parameters["models"]]
    save_json(ADDED_OUTPUT/"cv_manifest.json",jobs)
    fit_ready(jobs,study,inputs,row_folds)
    original_jobs = json.loads((OUTPUT/"cv_manifest.json").read_text())
    all_jobs = [*original_jobs,*jobs]
    scores,signatures = [],{}
    for job in all_jobs:
        path = OUTPUT/"cv"/job["case"]/f"{job['model']}_results.json"
        result = json.loads(path.read_text())
        scores.append({"model":job["model"],"profile":job["profile"],"fold":job["fold"],"auprc":result["evaluation"]["auprc_average_precision"]})
        signatures[str(path.relative_to(ROOT))] = result["signature"]
    selected,comparisons = choose_profiles(scores,study["study"])
    save_json(ADDED_OUTPUT/"selection.json",{"selected_profiles":selected,"cv_comparisons":comparisons,
              "cv_result_signatures":signatures,"outer_validation_metrics_used":False,
              "initial_protocol_digest":digest(json.loads((OUTPUT/"protocol.json").read_text())),"added_design_digest":digest(design),
              "note":"Selection pools only matched training-CV scores; initial validation results are not consulted.","test_sets_evaluated":[]})
    print("Combined CV selections frozen: "+json.dumps(selected),flush=True)
    final_jobs = [added_job(profile,model,"full") if profile in ADDED_PROFILES else make_job(profile,model,"full") for model,profile in selected.items()]
    save_json(ADDED_OUTPUT/"validation_manifest.json",final_jobs)
    if any(j["representation"]["source"] != "baseline" for j in final_jobs):
        build_combined("validation",inputs,study)
    if any(j["profile"] in ADDED_PROFILES for j in final_jobs):build_added("validation")
    old = [j for j in final_jobs if j["profile"] not in ADDED_PROFILES]
    if old:run_jobs(old,row_folds,inputs,study)
    added = [j for j in final_jobs if j["profile"] in ADDED_PROFILES]
    if added:fit_ready(added,study,inputs,row_folds)
    assert all(sha_file(ROOT/path) == expected for path,expected in inputs.items())
    assert all(sha_file(ROOT/p) == expected for p,expected in design["source_fingerprints"].items())
    save_json(ADDED_OUTPUT/"completed.json",{"cv_comparisons":len(all_jobs),"distinct_cv_models":len(signatures),
              "added_cv_models":len(jobs),"selected_validation_models":len(final_jobs),"selected_profiles":selected,
              "canonical_inputs_unchanged":True,"test_sets_evaluated":[],"elapsed_seconds":round(time.monotonic()-started,2)})
    print("Information-preserving feature control complete.",flush=True)


if __name__ == "__main__":main()
