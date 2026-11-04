"""Validate raw records, retain the established cohort, and freeze patient splits.

Run from the repository root: python -m sepsis_pipeline.preprocess --help
This step selects patients; it never truncates included patients' records.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .schema import CV_SEED, HEADER, SPLIT_SEED, STANDARD_ID


def read_record(path):
    """Read unchanged bytes and return the 41-column record and its checksum."""
    blob = Path(path).read_bytes()
    lines = blob.decode('utf-8').splitlines()
    if not lines or lines[0].strip().split('|') != HEADER:
        raise ValueError(f'Unexpected input header: {path}')
    body = '\n'.join(lines[1:])
    data = (np.loadtxt(io.StringIO(body), delimiter='|', ndmin=2)
            if body.strip() else np.empty((0, 41)))
    if data.ndim != 2 or data.shape[1] != 41:
        raise ValueError(f'Expected 41 columns: {path}')
    return data, hashlib.sha256(blob).hexdigest()


def classify_record(data):
    """Preserve the previously screened incident cohort at patient level.

    The inferred onset and complete-window check are inclusion criteria only.
    Once included, ALL rows and the supplied SepsisLabel are retained downstream.
    """
    if not len(data):
        return 'empty', np.nan
    hours, labels = data[:, 39], data[:, 40]
    if not np.isfinite(hours).all() or not np.all(np.diff(hours) == 1):
        return 'invalid_hour_sequence', np.nan
    if not np.isin(labels, [0, 1]).all() or np.any(np.diff(labels) < 0):
        return 'invalid_label_sequence', np.nan
    if not np.isfinite(data[0, 34]) or data[0, 34] < 18:
        return 'age_not_eligible', np.nan
    for col in [34, 35]:
        finite = data[np.isfinite(data[:, col]), col]
        if len(np.unique(finite)) > 1:
            return 'inconsistent_demographics', np.nan
    if labels[0] == 1:
        return 'positive_at_first_record', np.nan
    onset = np.nan
    if labels.max() == 1:
        onset = float(hours[np.flatnonzero(labels == 1)[0]] + 6)
        if onset > hours[-1]:
            return 'inferred_onset_after_record_end', onset
    positive_window = (onset > hours) & (onset <= hours + 6)
    eligible = (np.isnan(onset) | (hours < onset)) & (positive_window | (hours + 6 <= hours[-1]))
    return ('included' if eligible.any() else 'no_complete_prediction_window'), onset


def assign_splits(patients):
    """Patient-disjoint 70/15/15 within each hospital; three training-only folds."""
    if patients.empty or not patients.subject_id.is_unique or not patients.raw_sha256.is_unique:
        raise ValueError('Expected a nonempty cohort with unique patients and raw records')
    parts = []
    for _, group in patients.groupby('hospital', sort=True):
        group = group.sort_values('subject_id').reset_index(drop=True).copy()
        _, held_out = train_test_split(
            group.index.to_numpy(), test_size=.3, random_state=SPLIT_SEED, stratify=group.event)
        validation, test = train_test_split(
            held_out, test_size=.5, random_state=SPLIT_SEED, stratify=group.event.iloc[held_out])
        group['split'] = 'train'
        group.loc[validation, 'split'] = 'validation'
        group.loc[test, 'split'] = 'test'
        parts.append(group)
    people = pd.concat(parts, ignore_index=True).sort_values(['hospital', 'subject_id']).reset_index(drop=True)
    people['cv_fold'] = -1
    train = people.index[people.split == 'train'].to_numpy()
    strata = people.loc[train, 'hospital'].astype(str) + '_' + people.loc[train, 'event'].astype(str)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=CV_SEED)
    for fold, (_, held_out) in enumerate(cv.split(train, strata)):
        people.loc[train[held_out], 'cv_fold'] = fold
    return people


def prepare(data_root, output):
    data_root, output = Path(data_root), Path(output)
    jobs = []
    for hospital in ['A', 'B']:
        paths = sorted((data_root / f'training_set{hospital}').glob('*.psv'))
        if not paths:
            raise ValueError(f'No patient files in training_set{hospital}')
        jobs.extend((hospital, path) for path in paths)
    output.mkdir(parents=True, exist_ok=False)
    seen, people, exclusions = set(), [], []
    audit = {h: Counter() for h in ['A', 'B']}
    for i, (hospital, path) in enumerate(jobs, 1):
        data, digest = read_record(path)
        audit[hospital]['raw_patients'] += 1
        audit[hospital]['raw_hours'] += len(data)
        reason, onset = classify_record(data)
        if digest in seen:
            reason = 'duplicate_complete_record'
        seen.add(digest)
        audit[hospital][reason] += 1
        identity = {'subject_id': hospital + '_' + path.stem, 'hospital': hospital,
                    'raw_path': path.relative_to(data_root).as_posix(), 'raw_sha256': digest}
        if reason == 'included':
            people.append({**identity, 'event': int(data[:, 40].max()),
                           'record_hours': len(data), 'positive_hours': int(data[:, 40].sum()),
                           'onset_hour': onset})
        else:
            exclusions.append({**identity, 'reason': reason})
        if i % 5000 == 0:
            print(f'Validated {i}/{len(jobs)} patient records', flush=True)
    people = assign_splits(pd.DataFrame(people))
    people.to_parquet(output / 'patient_splits.parquet', index=False)
    pd.DataFrame(exclusions).to_csv(output / 'excluded_patients.csv', index=False)
    counts = people.groupby(['split', 'hospital']).agg(
        patients=('subject_id', 'size'), event_patients=('event', 'sum'),
        hours=('record_hours', 'sum'), positive_hours=('positive_hours', 'sum')).reset_index()
    counts.to_csv(output / 'cohort_counts.csv', index=False)
    folds = people.loc[people.split == 'train'].groupby(['cv_fold', 'hospital', 'event']).size()
    folds.rename('patients').reset_index().to_csv(output / 'cv_fold_counts.csv', index=False)
    report = {'standard_id': STANDARD_ID, 'scope': 'full', 'unit': 'patient-hour',
              'label': 'original SepsisLabel; includes post-onset and record-ending hours',
              'cohort': 'Existing screened cohort; full record does not reinstate excluded patients',
              'split_seed': SPLIT_SEED, 'cv_seed': CV_SEED, 'cv_folds': 3,
              'raw_patients': len(jobs), 'included_patients': len(people),
              'excluded_patients': len(exclusions), 'by_hospital': audit,
              'counts': counts.to_dict('records'),
              'manifest_sha256': hashlib.sha256((output / 'patient_splits.parquet').read_bytes()).hexdigest()}
    (output / 'preprocessing_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(counts.to_string(index=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'), help='Contains training_setA/ and training_setB/')
    parser.add_argument('--output', type=Path, default=Path('outputs/preprocessing'), help='A new output directory')
    args = parser.parse_args()
    prepare(args.data, args.output)


if __name__ == '__main__':
    main()
