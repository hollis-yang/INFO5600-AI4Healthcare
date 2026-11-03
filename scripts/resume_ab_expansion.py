"""Resume already prepared, frozen A+B cases without re-reading evicted old caches."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from sepsis_warning.ab_expansion import OUTPUT,PROCESSED,settings,write_json,content_hash
from run_ab_expansion import run_jobs


def main():
    started = time.monotonic()
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    jobs = json.loads((OUTPUT/"model_manifest.json").read_text())
    assert len(jobs) == 14
    for name,expected in protocol["source_fingerprints"].items():assert content_hash(ROOT/name) == expected
    for job in jobs:
        directory = PROCESSED/"cases"/job["case"]
        case = json.loads((directory/"case.json").read_text())
        assert case["cohort"] == job["cohort"] and case["profile"] == job["profile"]
        assert all((directory/name).exists() for name in ["preprocessor.joblib","train_rows.csv.gz","train_y.npy",
                   "A_validation_X.npy","B_validation_X.npy","B_test_X.npy"])
        if job["cohort"] != "A_only":assert (directory/"train_X.npy").exists()
    run_jobs(jobs,settings()["study"])
    checks = {}
    for name,expected in protocol["input_fingerprints"].items():
        path = ROOT/name
        if path.stat().st_flags & 0x40000000:
            checks[name] = {"sha256_at_freeze":expected,"post_run_status":"macOS dataless placeholder; no post-run byte comparison"}
        else:
            assert content_hash(path) == expected
            checks[name] = {"sha256_at_freeze":expected,"post_run_status":"exact SHA-256 match"}
    write_json(OUTPUT/"post_run_inputs.json",checks)
    write_json(OUTPUT/"completed.json",{"model_results":14,"new_fits":10,"reused_A_only_models":4,
               "training_cohorts":pd.read_csv(OUTPUT/"training_cohorts.csv").to_dict("records"),
               "canonical_inputs_unchanged":all(c["post_run_status"] == "exact SHA-256 match" for c in checks.values()),
               "resumed_from_frozen_prepared_cases":True,"A_test_evaluated":False,"B_test_is_unseen_hospital":False,
               "elapsed_seconds":round(time.monotonic()-started,2)})
    print("Prepared A+B experiment completed.",flush=True)


if __name__ == "__main__":main()
