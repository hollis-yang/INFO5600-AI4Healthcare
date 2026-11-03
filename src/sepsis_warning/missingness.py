"""Patient-weighted missingness summaries independent of holdout outcomes."""
from __future__ import annotations
import numpy as np


def patient_means(values,codes,n_patients,mask=None):
    if mask is None:mask=np.ones(len(codes),bool)
    denominator=np.bincount(codes[mask],minlength=n_patients)
    result=np.full((n_patients,values.shape[1]),np.nan)
    for j in range(values.shape[1]):
        total=np.bincount(codes[mask],weights=values[mask,j],minlength=n_patients)
        np.divide(total,denominator,out=result[:,j],where=denominator>0)
    return result


def event_standardized(values,has_event,target_prevalence):
    """Descriptive direct standardization to one pooled event-patient proportion."""
    labels=np.asarray(has_event,dtype=bool)
    assert labels.any() and (~labels).any() and 0<=target_prevalence<=1
    return (1-target_prevalence)*np.nanmean(values[~labels],axis=0)+target_prevalence*np.nanmean(values[labels],axis=0)
