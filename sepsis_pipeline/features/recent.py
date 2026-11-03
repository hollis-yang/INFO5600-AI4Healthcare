"""410 features with bounded recent history and no explicit stay clocks."""
import numpy as np
from .baseline import FEATURES as BASE


VARIABLES = 'HR O2Sat Temp SBP MAP DBP Resp EtCO2 BaseExcess HCO3 FiO2 pH PaCO2 SaO2 AST BUN Alkalinephos Calcium Chloride Creatinine Bilirubin_direct Glucose Lactate Magnesium Phosphate Potassium Bilirubin_total TroponinI Hct Hgb PTT WBC Fibrinogen Platelets'.split()

COMPACT = [n for n in BASE if n not in ['ICULOS', 'HospAdmTime']]

EXTRA = [f'local_{v}_{s}' for v in VARIABLES for s in ['count6', 'count12', 'mean12', 'min12', 'max12', 'slope6']]

FEATURES = COMPACT + EXTRA

def recent_features(frame):
    """Features use <=12 observed hourly bins and age/sex, never elapsed stay time."""
    n = len(frame)
    out = np.empty((n, len(FEATURES)), dtype=np.float32)
    out[:, :len(COMPACT)] = frame[COMPACT].to_numpy(dtype=np.float32)
    vals = frame[[v + '_current' for v in VARIABLES]].to_numpy(dtype=np.float32, copy=True)
    missing = frame[[v + '_missing_now' for v in VARIABLES]].to_numpy(dtype=bool)
    vals[missing] = np.nan
    ids = frame.subject_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends):
        v = vals[a:b]
        m = b - a
        hist = np.full((12, m, 34), np.nan, dtype=np.float32)
        for lag in range(min(12, m)):
            hist[lag, lag:] = v[:m - lag]
        obs = np.isfinite(hist)
        count12 = obs.sum(axis=0)
        count6 = obs[:6].sum(axis=0)
        mean12 = np.nansum(hist, axis=0) / np.maximum(count12, 1)
        min12 = np.min(np.where(obs, hist, np.inf), axis=0)
        max12 = np.max(np.where(obs, hist, -np.inf), axis=0)
        mean12[count12 == 0] = np.nan
        min12[count12 == 0] = np.nan
        max12[count12 == 0] = np.nan
        newest = obs[:6].argmax(axis=0)
        oldest = 5 - obs[:6][::-1].argmax(axis=0)
        rr = np.arange(m)[:, None]
        cc = np.arange(34)[None, :]
        slope = (hist[newest, rr, cc] - hist[oldest, rr, cc]) / np.maximum(oldest - newest, 1)
        slope[count6 < 2] = np.nan
        out[a:b, len(COMPACT):] = np.stack([count6, count12, mean12, min12, max12, slope], axis=2).reshape(m, -1)
    return out
