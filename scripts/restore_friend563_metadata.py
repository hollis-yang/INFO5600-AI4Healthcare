"""Restore only reproducible, cloud-evicted metadata; verify original hashes."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "processed_data/friend_563_comparison"


def main():
    people = pd.read_csv(DATA / "patient_splits.csv")
    for split in ("train", "validation", "test"):
        target = DATA / f"{split}_patients.csv"
        if not (target.stat().st_flags & 0x40000000):
            continue
        case = json.loads((DATA / f"{split}_case.json").read_text())
        records, position = [], 0
        selected = people.loc[people.split.eq(split)].sort_values(["hospital", "patient_id"])
        for p in selected.itertuples(index=False):
            stop = position + p.hours
            records.append({"patient_id": p.patient_id, "hospital": p.hospital, "has_event": p.has_event,
                            "cv_fold": p.cv_fold, "start": position, "stop": stop, "hours": p.hours,
                            "raw_start": p.raw_start, "raw_stop": p.raw_stop, "first_hour": p.first_hour})
            position = stop
        temporary = target.with_suffix(".verified_restore")
        pd.DataFrame(records).to_csv(temporary, index=False)
        assert hashlib.sha256(temporary.read_bytes()).hexdigest() == case["patients_sha256"]
        os.replace(temporary, target)
        print(f"Restored original-identical {target.name}", flush=True)
    target = DATA / "raw_case.json"
    if target.stat().st_flags & 0x40000000:
        relative = "reports/friend_563_comparison/raw_provenance.json"
        payload = (ROOT / relative).read_bytes()
        expected = subprocess.check_output(["git", "ls-files", "-s", "--", relative], cwd=ROOT, text=True).split()[1]
        assert hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest() == expected
        record = json.loads(payload)
        assert record["all_original_PSV_sha256_verified"]
        temporary = target.with_suffix(".verified_restore")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
        print("Restored raw provenance from the committed identical report", flush=True)


if __name__ == "__main__":
    main()
