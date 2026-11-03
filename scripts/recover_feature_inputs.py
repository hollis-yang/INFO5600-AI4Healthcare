"""Rebuild evicted deterministic caches only when their original SHA-256 matches."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import save_json


def checksum(path):
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
            total += len(block)
    assert total == path.stat().st_size,f"Unavailable file content: {path}"
    return digest.hexdigest()


def main():
    expected = json.loads((ROOT/"outputs/feature_optimization/input_fingerprints.json").read_text())
    records = []
    for name in ["processed_data/train_y.npy","processed_data/validation_X.npy"]:
        path = ROOT/name
        flags = path.stat().st_flags
        if not flags & 0x40000000:
            assert checksum(path) == expected[name]
            continue
        recovered = path.with_name(path.stem+".recovered.npy")
        if path.name == "train_y.npy":
            labels = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz",usecols=["label"]).label.to_numpy(np.int8)
            np.save(recovered,labels)
            source = "Unchanged train_rows.csv.gz labels, original int8 dtype and NumPy format"
        else:
            features = np.load(ROOT/"processed_data/validation_features.npy",mmap_mode="r")
            processor = joblib.load(ROOT/"outputs/preprocessor.joblib")
            matrix = np.lib.format.open_memmap(recovered,mode="w+",dtype=np.float32,shape=(len(features),len(processor.names_)))
            for start in range(0,len(features),20000):matrix[start:start+20000] = processor.transform(features[start:start+20000])
            matrix.flush()
            del matrix,features
            source = "Unchanged validation_features.npy and original training-fitted preprocessor"
        actual = checksum(recovered)
        assert actual == expected[name],f"Reconstruction differs; original placeholder not replaced: {name}"
        os.replace(recovered,path)
        records.append({"path":name,"previous_flags":flags,"source":source,"original_sha256":expected[name],"reconstructed_sha256":actual,"exact_original_bytes_restored":True})
        print("Restored exact canonical bytes:",name,flush=True)
    if records:
        save_json(ROOT/"outputs/feature_optimization/input_cache_recovery.json",{"restored":records,"models_refitted":False})
        save_json(ROOT/"reports/feature_optimization/input_cache_recovery.json",{"restored":records,"models_refitted":False})


if __name__ == "__main__":main()
