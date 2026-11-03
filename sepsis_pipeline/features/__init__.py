"""One-patient, past-only feature transforms; never pass labels as inputs."""
import numpy as np
import pandas as pd

from . import augmented, baseline, recent

FEATURE_NAMES = {'augmented': augmented.FEATURES, 'recent': recent.FEATURES}


def transform(raw, profile='augmented'):
    """Return one feature row per input hour, in exactly the input order.

    Call separately for each patient. Native missing values are retained;
    any later imputation/selection must fit only on the current training fold.
    """
    if profile not in FEATURE_NAMES:
        raise ValueError(f'Unknown feature profile: {profile}')
    raw = np.array(raw, dtype=float, copy=True)
    if raw.ndim != 2 or raw.shape[1] != 40 or not len(raw):
        raise ValueError('Expected one nonempty patient record with 40 input columns, no label')
    raw[~np.isfinite(raw)] = np.nan
    if profile == 'augmented':
        return augmented.features(raw)
    frame = pd.DataFrame(baseline.features(raw), columns=baseline.FEATURES)
    frame.insert(0, 'subject_id', 'one_patient')
    result = recent.recent_features(frame)
    result[~np.isfinite(result)] = np.nan
    return result.astype(np.float32)
