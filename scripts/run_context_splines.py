"""One LR-specific control motivated by training-only context ablations."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))

import joblib
import numpy as np
from sepsis_warning.data import FEATURE_NAMES,save_json
from sepsis_warning.context_splines import ContextSplinePreprocessor,CONTEXT_SPLINE_COLUMNS
from sepsis_warning.history_study import digest,sha_file,child_environment
from sepsis_warning.optimization import OUTPUT,PROCESSED,settings,representation_key,choose_profiles
from sepsis_warning.augmentation import ADDED_OUTPUT
from run_feature_optimization import prepare_case,make_job

PROFILE = "baseline_context_splines"


def job_for(fold):
    spec = {"source":"baseline","native":False,"clip":True,"drop_context":False,"splines":True,"context_splines":True}
    return {"profile":PROFILE,"model":"lr","fold":fold,"representation":spec,"case":representation_key(spec,fold),
            "stage":"validation" if fold == "full" else "cv"}


def prepare(job,study,inputs,folds):
    reference_job = make_job("baseline","lr",job["fold"])
    reference = PROCESSED/"cases"/reference_job["case"]
    if not (reference/"case.json").exists():
        prepare_case(reference_job["representation"],job["fold"],folds,inputs,study)
    directory = PROCESSED/"cases"/job["case"]
    directory.mkdir(parents=True,exist_ok=True)
    parameters = {k:v for k,v in study["splines"].items() if k != "variables"}
    signature = digest({"reference":json.loads((reference/"case.json").read_text())["signature"],
                        "representation":job["representation"],"columns":CONTEXT_SPLINE_COLUMNS,"parameters":parameters,
                        "implementation":sha_file(ROOT/"src/sepsis_warning/context_splines.py")})
    marker = directory/"case.json"
    if marker.exists() and json.loads(marker.read_text()).get("signature") == signature:return
    train_indices = np.load(reference/"train_indices.npy")
    raw = np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r")
    selected = raw[train_indices]
    processor = ContextSplinePreprocessor(FEATURE_NAMES,parameters).fit(selected)
    matrix = np.lib.format.open_memmap(directory/"train_X.npy",mode="w+",dtype=np.float32,shape=(len(selected),len(processor.names_)))
    for start in range(0,len(selected),15000):matrix[start:start+15000] = processor.transform(selected[start:start+15000])
    matrix.flush()
    del matrix,selected,raw
    evaluation_indices = np.load(reference/"evaluation_indices.npy")
    split = "validation" if job["fold"] == "full" else "train"
    raw = np.load(ROOT/f"processed_data/{split}_features.npy",mmap_mode="r")
    matrix = np.lib.format.open_memmap(directory/"evaluation_X.npy",mode="w+",dtype=np.float32,shape=(len(evaluation_indices),len(processor.names_)))
    for start in range(0,len(evaluation_indices),15000):matrix[start:start+15000] = processor.transform(raw[evaluation_indices[start:start+15000]])
    matrix.flush()
    for name in ["train_indices.npy","evaluation_indices.npy","train_y.npy","evaluation_y.npy","train_rows.csv.gz","evaluation_rows.csv.gz"]:
        shutil.copy2(reference/name,directory/name)
    joblib.dump(processor,directory/"preprocessor.joblib",compress=3)
    metadata = json.loads((reference/"case.json").read_text())
    metadata.update({"key":job["case"],"signature":signature,"representation":job["representation"],"features":len(processor.names_)})
    save_json(marker,metadata)


def fit(job):
    subprocess.run([sys.executable,"-u",str(ROOT/"scripts/fit_feature_model.py"),"--case",job["case"],"--model",job["model"]],
                   cwd=ROOT,env=child_environment(),check=True)


def main():
    study = settings()
    design_path = ADDED_OUTPUT/"context_spline_design.json"
    if not design_path.exists():
        save_json(design_path,{"created_utc":datetime.now(timezone.utc).isoformat(),"profile":PROFILE,
              "reason":"Context removal reduced matched-fold AP; test whether LR benefits from nonlinear functions of available time/context inputs while preserving all baseline features.",
              "columns":CONTEXT_SPLINE_COLUMNS,"model_parameters_unchanged":True,"selection_uses_outer_validation_metrics":False,
              "initial_validation_already_evaluated":(OUTPUT/"completed.json").exists(),
              "source_fingerprints":{str(p.relative_to(ROOT)):sha_file(p) for p in [Path(__file__),ROOT/"src/sepsis_warning/context_splines.py"]},
              "test_sets_evaluated":[]})
    design = json.loads(design_path.read_text())
    assert all(sha_file(ROOT/p) == expected for p,expected in design["source_fingerprints"].items())
    while not (ADDED_OUTPUT/"completed.json").exists():
        print("Context spline control fixed; waiting for augmentation model jobs.",flush=True);time.sleep(35)
    inputs = json.loads((OUTPUT/"input_fingerprints.json").read_text())
    folds = np.load(PROCESSED/"row_folds.npy")
    jobs = [job_for(fold) for fold in range(study["study"]["folds"])]
    save_json(ADDED_OUTPUT/"context_spline_cv_manifest.json",jobs)
    for job in jobs:prepare(job,study,inputs,folds);fit(job)
    prior_jobs = json.loads((ADDED_OUTPUT/"cv_manifest.json").read_text())
    added_jobs = [j for j in prior_jobs if j["profile"] != PROFILE]+jobs
    all_jobs = json.loads((OUTPUT/"cv_manifest.json").read_text())+added_jobs
    scores,signatures = [],{}
    for job in all_jobs:
        path = OUTPUT/"cv"/job["case"]/f"{job['model']}_results.json"
        result = json.loads(path.read_text())
        scores.append({"model":job["model"],"profile":job["profile"],"fold":job["fold"],"auprc":result["evaluation"]["auprc_average_precision"]})
        signatures[str(path.relative_to(ROOT))] = result["signature"]
    selected,comparisons = choose_profiles(scores,study["study"])
    for name in ["cv_manifest.json","selection.json","completed.json","validation_manifest.json"]:
        backup = ADDED_OUTPUT/(name.removesuffix(".json")+"_before_context_splines.json")
        if not backup.exists():shutil.copy2(ADDED_OUTPUT/name,backup)
    selection = json.loads((ADDED_OUTPUT/"selection.json").read_text())
    selection.update({"selected_profiles":selected,"cv_comparisons":comparisons,"cv_result_signatures":signatures,
                      "context_spline_design_digest":digest(design),"outer_validation_metrics_used":False})
    save_json(ADDED_OUTPUT/"selection.json",selection)
    save_json(ADDED_OUTPUT/"cv_manifest.json",added_jobs)
    previous_validation = json.loads((ADDED_OUTPUT/"validation_manifest.json").read_text())
    final_jobs = [job_for("full") if selected[j["model"]] == PROFILE else j for j in previous_validation]
    save_json(ADDED_OUTPUT/"validation_manifest.json",final_jobs)
    print("Final CV choices including context splines: "+json.dumps(selected),flush=True)
    for job in final_jobs:
        if job["profile"] == PROFILE:prepare(job,study,inputs,folds)
        fit(job)
    completion = json.loads((ADDED_OUTPUT/"completed.json").read_text())
    completion.update({"cv_comparisons":len(all_jobs),"distinct_cv_models":len(signatures),"selected_profiles":selected,
                       "context_spline_cv_models":len(jobs),"context_splines_complete":True})
    assert all(sha_file(ROOT/p) == expected for p,expected in inputs.items())
    save_json(ADDED_OUTPUT/"completed.json",completion)
    print("Context spline comparison complete.",flush=True)


if __name__ == "__main__":main()
