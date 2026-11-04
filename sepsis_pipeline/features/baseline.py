"""208 baseline features: 34 variables x 6 statistics + 4 context values."""

import numpy as np

from ..schema import VARIABLES, CONTEXT, HEADER

STATS = ['current', 'mean6', 'sd6', 'delta6', 'age6', 'missing_now']

FEATURES = [v + '_' + s for v in VARIABLES for s in STATS] + CONTEXT

def features(data):
    values = data[:, :34].astype(np.float64, copy=True)
    values[~np.isfinite(values)] = np.nan
    n, width = values.shape
    history = np.full((6, n, width), np.nan)
    for lag in range(min(6, n)):
        history[lag, lag:] = values[:n - lag]
    observed = np.isfinite(history)
    count = observed.sum(axis=0)
    safe = np.maximum(count, 1)
    mean = np.nansum(history, axis=0) / safe
    variance = np.nansum((history - mean[None, :, :]) ** 2, axis=0) / safe
    most_recent = observed.argmax(axis=0)
    oldest = 5 - observed[::-1].argmax(axis=0)
    rr = np.arange(n)[:, None]
    cc = np.arange(width)[None, :]
    current = history[most_recent, rr, cc]
    first = history[oldest, rr, cc]
    delta = np.where(count >= 2, current - first, np.nan)
    mean[count == 0] = np.nan
    variance[count == 0] = np.nan
    age = np.where(count > 0, most_recent, 6).astype(float)
    missing = ~np.isfinite(values)
    engineered = np.stack([current, mean, np.sqrt(variance), delta, age, missing], axis=2).reshape(n, -1)
    context = data[:, [34, 35, 38, 39]].copy()
    context[~np.isfinite(context)] = np.nan
    return np.column_stack([engineered, context]).astype(np.float32)
