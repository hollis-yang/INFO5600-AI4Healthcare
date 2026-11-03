"""Official full-trajectory utility, kept separate from custom pre-onset metrics."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from sepsis_warning.data import ROOT
from sepsis_warning.ab_expansion import content_hash


@lru_cache(maxsize=1)
def official_scorer():
    directory=ROOT/"third_party/physionet2019"
    metadata=json.loads((directory/"UPSTREAM.json").read_text())
    for name,record in metadata["files"].items():assert content_hash(directory/name)==record["sha256"]
    spec=importlib.util.spec_from_file_location("physionet2019_official_score",directory/"evaluate_sepsis_score.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def patient_utility(labels,predictions):
    labels=np.asarray(labels);predictions=np.asarray(predictions)
    assert len(labels)==len(predictions)>0 and np.isin(labels,[0,1]).all() and np.isin(predictions,[0,1]).all()
    best=np.zeros(len(labels),np.int8)
    if labels.any():
        # Official labels already begin six hours before reconstructed onset.
        onset=int(np.argmax(labels))+6
        best[max(0,onset-12):min(onset+3+1,len(labels))]=1
    inactive=np.zeros(len(labels),np.int8)
    function=official_scorer().compute_prediction_utility
    return {"observed":float(function(labels,predictions)),"best":float(function(labels,best)),
            "inaction":float(function(labels,inactive)),"worst":float(function(labels,1-best))}


def normalized_utility(contributions):
    totals={key:float(np.sum(contributions[key])) for key in ["observed","best","inaction","worst"]}
    denominator=totals["best"]-totals["inaction"]
    assert denominator>0,"At least one evaluable septic patient is required for normalized utility"
    return {"normalized_utility":float((totals["observed"]-totals["inaction"])/denominator),
            "normalization_denominator":denominator,**totals}


def paired_utility_contrast(reference,candidate,repetitions=300,seed=42):
    columns=["patient_id","best","inaction","worst"]
    pd.testing.assert_frame_equal(reference[columns].reset_index(drop=True),candidate[columns].reset_index(drop=True))
    denominator=reference.best.to_numpy()-reference.inaction.to_numpy()
    delta=candidate.observed.to_numpy()-reference.observed.to_numpy()
    point=float(delta.sum()/denominator.sum())
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        count=np.bincount(rng.integers(len(reference),size=len(reference)),minlength=len(reference))
        d=float(np.dot(count,denominator))
        if d>0:samples.append(float(np.dot(count,delta)/d))
    return {"difference":point,"lower":float(np.quantile(samples,.025)),"upper":float(np.quantile(samples,.975)),
            "resamples":len(samples),"uncertainty":"Patient-paired evaluation uncertainty conditional on fitted models and frozen thresholds."}
