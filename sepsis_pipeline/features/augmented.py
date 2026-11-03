"""Causal features inspired by Separatrix; independently implemented adaptations.

Differences: forward fill within each prefix (no interpolation), no Unit1/Unit2,
correct run lengths, squared-energy entropy rather than KDE pseudo-entropy,
partial windows retain only observed history, and native missing values.
"""
import numpy as np
from .baseline import FEATURES as BASE, features as baseline_features
from ..schema import HEADER

VINDEX=[0,1,2,3,4,5,6,38,39]
VNAMES=[HEADER[i] for i in VINDEX]
MINDEX=list(range(34))+[38,39]
MNAMES=[HEADER[i] for i in MINDEX]
STATS=['mean','min','max','median','variance','q95','q99','q05','q01']
VALUE_NAMES=[f'sep_{v}_{s}{w}' for w in [5,11] for s in STATS for v in VNAMES]
VALUE_NAMES += [f'sep_{v}_{s}' for s in ['energy','energy_entropy','mean_diff'] for v in VNAMES]
VALUE_NAMES += ['sep_prefix_length']+[f'sep_{HEADER[i]}_last' for i in VINDEX+[34,35]]
MISSING_NAMES=[f'sep_{v}_{s}' for s in ['run_mean','run_variance','observed_fraction','observed_variance'] for v in MNAMES]
MISSING_NAMES += [f'sep_coverage_lag{k}' for k in range(5)]+[f'sep_coverage_variance_lag{k}' for k in range(5)]
FEATURES=BASE+VALUE_NAMES+MISSING_NAMES
assert len(VALUE_NAMES)==201 and len(MISSING_NAMES)==154 and len(FEATURES)==563

def forward(values):
    finite=np.isfinite(values);idx=np.where(finite,np.arange(len(values))[:,None],-1)
    idx=np.maximum.accumulate(idx,axis=0)
    out=values[np.maximum(idx,0),np.arange(values.shape[1])]
    return np.where(idx>=0,out,np.nan)

def history(values,width):
    out=np.full((width,*values.shape),np.nan)
    for k in range(min(width,len(values))):out[k,k:]=values[:len(values)-k]
    return out

def value_features(raw):
    import warnings
    x=forward(raw[:,VINDEX]);n=len(x);parts=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for w in [5,11]:
            h=history(x,w)
            parts.extend([np.nanmean(h,axis=0),np.nanmin(h,axis=0),np.nanmax(h,axis=0),np.nanmedian(h,axis=0),np.nanvar(h,axis=0)])
            parts.extend(np.nanquantile(h,[.95,.99,.05,.01],axis=0))
    z=np.where(np.isfinite(x),x,0.);squares=z*z;energy=np.cumsum(squares,axis=0)
    sl=np.cumsum(squares*np.log(np.maximum(squares,1e-30)),axis=0)
    entropy=np.log(np.maximum(energy,1e-30))-np.divide(sl,energy,out=np.zeros_like(sl),where=energy>0)
    entropy[energy==0]=0
    diffs=np.diff(x,axis=0,prepend=np.full((1,x.shape[1]),np.nan))
    dc=np.cumsum(np.isfinite(diffs),axis=0);ds=np.nancumsum(diffs,axis=0)
    dm=np.divide(ds,dc,out=np.full_like(ds,np.nan),where=dc>0)
    parts += [energy,entropy,dm,np.arange(1,n+1)[:,None],forward(raw[:,VINDEX+[34,35]])]
    return np.column_stack(parts)

def missing_features(raw):
    obs=np.isfinite(raw[:,MINDEX]);n,d=obs.shape
    run=np.ones(d);completed_square=np.zeros(d);runs=np.ones(d);means=np.zeros((n,d));variances=np.zeros((n,d))
    for t in range(n):
        if t:
            changed=obs[t]!=obs[t-1]
            completed_square += np.where(changed,run*run,0)
            runs += changed;run=np.where(changed,1,run+1)
        means[t]=(t+1)/runs
        variances[t]=np.maximum(0,(completed_square+run*run)/runs-means[t]**2)
    frac=np.cumsum(obs,axis=0)/np.arange(1,n+1)[:,None];var=frac*(1-frac)
    cover=obs.mean(axis=1);cvar=obs.var(axis=1)
    lags=history(np.column_stack([cover,cvar]),5).transpose(1,2,0).reshape(n,10)
    return np.column_stack([means,variances,frac,var,lags])

def features(raw):
    raw=np.array(raw,dtype=float,copy=True)
    if raw.ndim!=2 or raw.shape[1]!=40:raise ValueError('Expected one patient prefix, 40 input columns, no outcome')
    raw[~np.isfinite(raw)]=np.nan
    out=np.column_stack([baseline_features(raw),value_features(raw),missing_features(raw)])
    out[~np.isfinite(out)]=np.nan
    return out.astype(np.float32)

