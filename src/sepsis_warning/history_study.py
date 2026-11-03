"""Controlled lab-history and patient-sample experiments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from sepsis_warning.data import ROOT


def settings():
    with (ROOT/"configs/history_study.toml").open("rb") as stream:
        return tomllib.load(stream)


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sample_patients(rows, fraction, seed):
    """Nested per-class prefixes, so each smaller sample is inside the larger one."""
    patients = rows.groupby("patient_id", sort=True).label.max()
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        ids = patients.index[patients == label].to_numpy().copy()
        rng.shuffle(ids)
        selected.extend(ids[:max(1, round(len(ids)*fraction))])
    return sorted(selected)


def child_environment():
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(ROOT/"outputs/.matplotlib")
    env["XDG_CACHE_HOME"] = str(ROOT/"outputs/.cache")
    # Reuse the installed, trusted sklearn OpenMP runtime in macOS child processes.
    # Ordinary installations can instead install libomp as documented by LightGBM.
    if sys.platform == "darwin":
        spec = importlib.util.find_spec("sklearn")
        directory = Path(spec.origin).parent/".dylibs"
        if (directory/"libomp.dylib").exists():
            existing = env.get("DYLD_LIBRARY_PATH", "")
            env["DYLD_LIBRARY_PATH"] = str(directory)+(os.pathsep+existing if existing else "")
    return env


class PreparedAveragePrecision:
    """Exact weighted AP with cached score sorting for patient bootstraps."""
    def __init__(self, labels, probability):
        probability = np.asarray(probability)
        self.order = np.argsort(-probability, kind="stable")
        sorted_probability = probability[self.order]
        self.labels = np.asarray(labels)[self.order].astype(float)
        self.starts = np.r_[0, np.flatnonzero(np.diff(sorted_probability) != 0)+1]

    def evaluate(self, weights):
        weights = np.asarray(weights, dtype=float)[self.order]
        positive = np.add.reduceat(weights*self.labels, self.starts)
        totals = np.add.reduceat(weights, self.starts)
        denominator = positive.sum()
        if denominator == 0:
            return np.nan
        cumulative = np.cumsum(totals)
        precision = np.divide(np.cumsum(positive), cumulative,
                              out=np.zeros_like(cumulative), where=cumulative > 0)
        return float(np.dot(positive, precision)/denominator)


def bootstrap_ap_contrast(rows, reference_probabilities, candidate_probabilities, repetitions=300, seed=42):
    """Candidate mean AP minus reference mean AP, resampling whole patients."""
    codes, identifiers = pd.factorize(rows.patient_id, sort=False)
    labels = rows.label.to_numpy()
    references = [PreparedAveragePrecision(labels, score) for score in reference_probabilities]
    candidates = [PreparedAveragePrecision(labels, score) for score in candidate_probabilities]
    ones = np.ones(len(labels))
    point = np.mean([prepared.evaluate(ones) for prepared in candidates])
    if references:
        point -= np.mean([prepared.evaluate(ones) for prepared in references])
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        frequency = np.bincount(rng.integers(len(identifiers), size=len(identifiers)), minlength=len(identifiers))
        weights = frequency[codes]
        difference = np.mean([prepared.evaluate(weights) for prepared in candidates])
        if references:
            difference -= np.mean([prepared.evaluate(weights) for prepared in references])
        if np.isfinite(difference):
            differences.append(difference)
    return {"difference": float(point), "lower": float(np.quantile(differences, .025)),
            "upper": float(np.quantile(differences, .975)), "resamples": len(differences),
            "reference_runs": len(references), "candidate_runs": len(candidates),
            "uncertainty": "Patient-resampled validation uncertainty conditional on the chosen training subsets."}
