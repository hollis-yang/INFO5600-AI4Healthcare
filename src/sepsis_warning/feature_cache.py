"""Small verification probes and reversible removal of generated CV matrices."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sepsis_warning.augmentation import ADDED_DATA, added_names
from sepsis_warning.optimization import PROCESSED, raw_features


def load_case_raw(split,spec):
    if spec["source"] == "augmented":
        return np.load(ADDED_DATA/f"{split}_features.npy",mmap_mode="r"),added_names()
    return raw_features(split,spec)


def verification_probes(directory):
    directory = Path(directory)
    meta = json.loads((directory/"case.json").read_text())
    sample = np.linspace(0,meta["training_hours"]-1,64,dtype=int)
    held_sample = np.linspace(0,meta["evaluation_hours"]-1,64,dtype=int)
    saved = directory/"verification_probes.npz"
    if (directory/"train_X.npy").exists():
        train = np.load(directory/"train_X.npy",mmap_mode="r")
        evaluation = np.load(directory/"evaluation_X.npy",mmap_mode="r")
        assert train.shape == (meta["training_hours"],meta["features"])
        assert evaluation.shape == (meta["evaluation_hours"],meta["features"])
        x,held = train[sample].copy(),evaluation[held_sample].copy()
        np.savez_compressed(saved,train=x,evaluation=held,train_indices=sample,evaluation_indices=held_sample)
    else:
        probes = np.load(saved)
        np.testing.assert_array_equal(probes["train_indices"],sample)
        np.testing.assert_array_equal(probes["evaluation_indices"],held_sample)
        x,held = probes["train"],probes["evaluation"]
    return sample,held_sample,x,held


def purge_cv_matrices(cases):
    removed = []
    for key in sorted(set(cases)):
        directory = PROCESSED/"cases"/key
        meta = json.loads((directory/"case.json").read_text())
        assert meta["stage"] == "cv" and (directory/"verification_probes.npz").exists()
        for name in ["train_X.npy","evaluation_X.npy"]:
            path = directory/name
            if path.exists():
                removed.append({"path":str(path),"bytes":path.stat().st_size})
                path.unlink()
    return removed


def restore_cv_matrices():
    restored = []
    for marker in sorted((PROCESSED/"cases").glob("*/case.json")):
        directory = marker.parent
        meta = json.loads(marker.read_text())
        if meta["stage"] != "cv":continue
        processor = joblib.load(directory/"preprocessor.joblib")
        for role in ["train","evaluation"]:
            path = directory/f"{role}_X.npy"
            if path.exists():continue
            raw,_ = load_case_raw("train",meta["representation"])
            indices = np.load(directory/f"{role}_indices.npy")
            matrix = np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(indices),len(processor.names_)))
            for start in range(0,len(indices),15000):
                matrix[start:start+15000] = processor.transform(raw[indices[start:start+15000]])
            matrix.flush()
            del matrix,raw
            restored.append(str(path))
    return restored
