"""Recover unavailable CV metadata from a saved model and independently saved metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import environment,save_json
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.feature_cache import load_case_raw
from sepsis_warning.history_study import child_environment,digest,sha_file
from sepsis_warning.optimization import OUTPUT,PROCESSED,REPORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",required=True)
    parser.add_argument("--model",choices=["lr","ebm","lightgbm"],required=True)
    parser.add_argument("--openmp-ready",action="store_true")
    args = parser.parse_args()
    directory = OUTPUT/"cv"/args.case
    target = directory/f"{args.model}_results.json"
    assert not target.read_bytes(),"Recovery is limited to metadata whose content is unavailable."
    old_stat = target.stat()
    case_dir = PROCESSED/"cases"/args.case
    meta = json.loads((case_dir/"case.json").read_text())
    assert meta["stage"] == "cv"
    table = pd.read_csv(REPORT/"cv_fold_results.csv")
    reference = table.loc[(table["case"] == args.case)&(table.model == args.model)].iloc[0]
    assert reference.warning_count == 0,"Original warning text is required for a nonzero warning count."
    bundle = joblib.load(directory/f"{args.model}_model.joblib")
    assert bundle["case"] == meta
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    parameters = protocol["model_settings"]["models"][args.model]
    assert bundle["parameters"] == parameters
    signature = digest({"case":meta["signature"],"parameters":parameters,
        "stopping":protocol["model_settings"]["lightgbm_stopping"],"worker":sha_file(ROOT/"scripts/fit_feature_model.py")})
    assert signature == reference.signature
    raw,_ = load_case_raw("train",meta["representation"])
    processor,model = bundle["preprocessor"],bundle["model"]
    def predict(indices):
        probabilities = []
        for start in range(0,len(indices),15000):
            x = processor.transform(raw[indices[start:start+15000]])
            probabilities.append(model.booster_.predict(x) if args.model == "lightgbm" else model.predict_proba(x)[:,1])
        return np.concatenate(probabilities)
    evaluation = predict(np.load(case_dir/"evaluation_indices.npy"))
    stored_probability = np.load(directory/f"{args.model}_probability.npy")
    np.testing.assert_allclose(evaluation,stored_probability,rtol=0,atol=1e-12)
    metrics = ranking_metrics(np.load(case_dir/"evaluation_y.npy"),evaluation)
    training = ranking_metrics(np.load(case_dir/"train_y.npy"),predict(np.load(case_dir/"train_indices.npy")))
    for field,column in [("auprc_average_precision","auprc"),("auroc","auroc"),("brier_score","brier")]:
        np.testing.assert_allclose(metrics[field],reference[column],rtol=0,atol=1e-12)
    np.testing.assert_allclose(training["auprc_average_precision"],reference.training_auprc,rtol=0,atol=1e-12)
    if args.model == "lightgbm":
        trees = model.booster_.num_trees()
        assert trees == reference.inner_selected_trees
        details = {"inner_selected_trees":trees,"early_stopping_patient_grouped":True,"refit_on_all_selected_training_patients":True}
    elif args.model == "lr":details = {"iterations":int(model.n_iter_[0])}
    else:details = {"executed_cycles_including_patience":(model.best_iteration_[0]/len(model.term_features_)).tolist()}
    recovery = {"recovered_utc":datetime.now(timezone.utc).isoformat(),"case":args.case,"model":args.model,
        "reason":"macOS compressed/dataless placeholder returned zero bytes despite nonzero stat size; cloud retrieval did not restore content.",
        "previous_stat_size":old_stat.st_size,"previous_flags":old_stat.st_flags,"model_refitted":False,
        "all_saved_evaluation_predictions_reproduced":True,"previous_summary_metrics_matched":True,
        "configuration_signature_matched":True}
    payload = {"signature":signature,"case":meta,"model":args.model,"parameters":parameters,
        "evaluation":metrics,"training":training,"alerts":None,"fit_seconds":float(reference.fit_seconds),
        "reused_baseline":False,"details":details,"warnings":[],"test_sets_evaluated":[],"environment":environment(),
        "metadata_recovery":recovery}
    temporary = target.with_suffix(".recovered.json")
    save_json(temporary,payload)
    os.replace(temporary,target)
    save_json(OUTPUT/"metadata_recovery.json",recovery)
    save_json(REPORT/"metadata_recovery.json",recovery)
    print(json.dumps(recovery,indent=2),flush=True)


if __name__ == "__main__":
    if sys.platform == "darwin" and "--openmp-ready" not in sys.argv:
        import subprocess
        raise SystemExit(subprocess.call([sys.executable,__file__,*sys.argv[1:],"--openmp-ready"],env=child_environment()))
    main()
