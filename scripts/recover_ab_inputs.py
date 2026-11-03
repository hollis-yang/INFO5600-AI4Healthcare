"""Restore an evicted B feature cache only after reproducing its frozen SHA-256."""
from __future__ import annotations

import json
import ast
import gzip
import io
import hashlib
import struct
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
import joblib
from sepsis_warning.data import causal_features,assess_patient,config,FEATURE_NAMES
from sepsis_warning.ab_expansion import OUTPUT,REPORT,write_json,content_hash


def recover_patient_summary(expected):
    target = ROOT/"outputs/patient_summary.csv"
    if not target.stat().st_flags & 0x40000000:
        assert content_hash(target) == expected
        return
    # Reuse the original dictionary expression rather than changing serialization.
    source = ast.parse((ROOT/"scripts/prepare_data.py").read_text())
    call = next(node for node in ast.walk(source) if isinstance(node,ast.Call) and
                isinstance(node.func,ast.Attribute) and node.func.attr == "append" and
                isinstance(node.func.value,ast.Name) and node.func.value.id == "patient_rows")
    expression = compile(ast.Expression(call.args[0]),"original_patient_dictionary","eval")
    manifest = pd.read_csv(ROOT/"data/manifest.csv")
    raw = np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    records,offset = [],0
    for record in manifest.itertuples(index=False):
        values = raw[offset:offset+record.hours]
        reason,eligible,onset = assess_patient(values,config())
        positive = np.flatnonzero(values[:,40] == 1)
        hospital = "A" if "training_setA" in record.relative_path else "B"
        path = ROOT/record.relative_path
        records.append(eval(expression,{"np":np},{"hospital":hospital,"path":path,"record":record,
                       "offset":offset,"values":values,"reason":reason,"eligible":eligible,"onset":onset,"positive":positive}))
        offset += len(values)
    temporary = target.with_name("patient_summary.recovered.csv")
    pd.DataFrame(records).to_csv(temporary,index=False)
    actual = content_hash(temporary)
    assert actual == expected,"Rebuilt patient audit differs; original was not replaced"
    os.replace(temporary,target)
    write_json(OUTPUT/"patient_summary_recovery.json",{"exact_original_bytes_restored":True,
               "original_sha256":expected,"reconstructed_sha256":actual,"source":"Original prepare_data patient dictionary and unchanged raw_records/manifest"})
    print("Restored exact patient summary bytes",actual,flush=True)


def recover_hour_rows():
    fingerprints = json.loads((OUTPUT/"protocol.json").read_text())["input_fingerprints"]
    records = []
    for split in ["train","validation","external_test"]:
        name = f"processed_data/{split}_rows.csv.gz"
        target = ROOT/name
        if not target.stat().st_flags & 0x40000000:continue
        stamp = int(target.stat().st_mtime)
        if split == "train":
            rows = pd.read_csv(ROOT/"processed_data/ab_expansion/cases/A_only__common/train_rows.csv.gz")
        elif split == "validation":rows = pd.read_csv(ROOT/"processed_data/ab_expansion/A_validation_rows.csv.gz")
        else:
            train = pd.read_csv(ROOT/"processed_data/ab_expansion/cases/A_B_full__common/train_rows.csv.gz")
            train = train.loc[train.hospital.eq("B")]
            rows = pd.concat([train,*[pd.read_csv(ROOT/f"processed_data/ab_expansion/{s}_rows.csv.gz") for s in ["B_validation","B_test"]]],ignore_index=True)
            rows = rows.sort_values(["patient_id","hour"]).reset_index(drop=True)
        buffer = io.BytesIO()
        with gzip.GzipFile(filename=target.name,mode="wb",fileobj=buffer,mtime=1) as stream:
            rows.to_csv(stream,index=False)
        body = bytearray(buffer.getvalue())
        # Gzip encodes the original write second; the footer depends only on CSV bytes.
        found = None
        for candidate in [stamp,*[stamp+delta for delta in range(-120,121) if delta]]:
            body[4:8] = struct.pack("<I",candidate)
            if hashlib.sha256(body).hexdigest() == fingerprints[name]:found = candidate;break
        assert found is not None,f"Exact gzip reconstruction failed; original not replaced: {name}"
        temporary = target.with_name(target.name+".recovered")
        temporary.write_bytes(body);os.replace(temporary,target)
        records.append({"path":name,"original_sha256":fingerprints[name],"reconstructed_sha256":content_hash(target),
                        "exact_original_bytes_restored":True,"gzip_mtime":found,"source":"Frozen prepared row copies; original gzip filename and timestamp"})
        print("Restored exact hourly metadata bytes",name,flush=True)
    if records:write_json(OUTPUT/"row_cache_recovery.json",{"restored":records})


def main():
    if "--rows-only" in sys.argv:
        recover_hour_rows();return
    if "--processor-only" in sys.argv:
        name = "outputs/preprocessor.joblib"
        expected = json.loads((OUTPUT/"protocol.json").read_text())["input_fingerprints"][name]
        processor = joblib.load(OUTPUT/"references/common_lr_model.joblib")["preprocessor"]
        target = ROOT/name
        temporary = target.with_name("preprocessor.recovered.joblib")
        joblib.dump(processor,temporary,compress=3)
        actual = content_hash(temporary)
        assert actual == expected,"Serialized processor differs; original not replaced"
        os.replace(temporary,target)
        write_json(OUTPUT/"preprocessor_recovery.json",{"path":name,"original_sha256":expected,
                   "reconstructed_sha256":actual,"exact_original_bytes_restored":True,"source":"Unchanged A-only reference model preprocessor"})
        print("Restored exact original preprocessor bytes",actual,flush=True)
        return
    name = "processed_data/external_test_features.npy"
    fingerprints = json.loads((OUTPUT/"protocol.json").read_text())["input_fingerprints"]
    recover_patient_summary(fingerprints["outputs/patient_summary.csv"])
    expected = fingerprints[name]
    target = ROOT/name
    temporary = target.with_name("external_test_features.recovered.npy")
    rows = pd.read_csv(ROOT/"processed_data/external_test_rows.csv.gz")
    patients = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    raw = np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    matrix = np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float32,shape=(len(rows),len(FEATURE_NAMES)))
    position = 0
    for number,identifier in enumerate(rows.patient_id.drop_duplicates(),1):
        patient = patients.loc[identifier]
        values = raw[int(patient.raw_start):int(patient.raw_stop)]
        reason,eligible,_ = assess_patient(values,config())
        assert reason == "included"
        part = causal_features(values,config())[eligible]
        matrix[position:position+len(part)] = part
        position += len(part)
        if number%4000 == 0:print(f"Reconstructed B baseline features: {number} patients",flush=True)
    assert position == len(rows)
    matrix.flush();del matrix
    actual = content_hash(temporary)
    assert actual == expected,"Reconstructed bytes differ; original cache was not replaced"
    os.replace(temporary,target)
    record = {"path":name,"original_sha256":expected,"reconstructed_sha256":actual,
              "exact_original_bytes_restored":True,"source":"Unchanged raw_records, patient row order and causal_features",
              "experiment_settings_changed":False}
    write_json(OUTPUT/"input_recovery.json",record);write_json(REPORT/"input_recovery.json",record)
    print("Restored exact B feature bytes:",actual,flush=True)


if __name__ == "__main__":main()
