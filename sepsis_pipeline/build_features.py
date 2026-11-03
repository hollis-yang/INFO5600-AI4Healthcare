"""Build full-record features from the frozen patient manifest in bounded batches."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .features import FEATURE_NAMES, transform
from .preprocess import read_record
from .schema import STANDARD_ID

METADATA_COLUMNS = ['subject_id', 'hospital', 'split', 'cv_fold', 'hour', 'label']


def build(data_root, preprocessing, output, profile='augmented', split='all', batch_patients=128):
    data_root, preprocessing, output = Path(data_root), Path(preprocessing), Path(output)
    if profile not in FEATURE_NAMES or split not in ['all', 'train', 'validation', 'test']:
        raise ValueError('Unknown feature profile or split')
    if batch_patients < 1:
        raise ValueError('batch_patients must be positive')
    manifest_path = preprocessing / 'patient_splits.parquet'
    report = json.loads((preprocessing / 'preprocessing_report.json').read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if report['standard_id'] != STANDARD_ID or report['manifest_sha256'] != manifest_hash:
        raise ValueError('Preprocessing standard or manifest checksum does not match')
    people = pd.read_parquet(manifest_path).sort_values(['hospital', 'subject_id'])
    if not people.subject_id.is_unique or not people.raw_sha256.is_unique:
        raise ValueError('Duplicate patients or raw records in manifest')
    if split != 'all':
        people = people.loc[people.split == split]
    if people.empty:
        raise ValueError('No patients selected')
    output.mkdir(parents=True, exist_ok=False)
    names = FEATURE_NAMES[profile]
    counts = []
    for part in ['train', 'validation', 'test']:
        selected = people.loc[people.split == part]
        if selected.empty:
            continue
        writer, pending, hours, positive_hours = None, [], 0, 0

        def flush():
            nonlocal writer, pending
            if not pending:
                return
            frame = pd.concat(pending, ignore_index=True)
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output / f'{part}.parquet', table.schema, compression='zstd')
            writer.write_table(table)
            pending = []

        try:
            for i, person in enumerate(selected.itertuples(index=False), 1):
                path = (data_root / person.raw_path).resolve()
                if not path.is_relative_to(data_root.resolve()):
                    raise ValueError('Raw record path must be inside the data directory')
                raw, digest = read_record(path)
                if digest != person.raw_sha256:
                    raise ValueError(f'Raw record changed: {person.subject_id}')
                if len(raw) != person.record_hours or int(raw[:, 40].sum()) != person.positive_hours:
                    raise ValueError(f'Record counts changed: {person.subject_id}')
                values = transform(raw[:, :40], profile)
                frame = pd.DataFrame(values, columns=names)
                meta = pd.DataFrame({'subject_id': person.subject_id, 'hospital': person.hospital,
                                     'split': part, 'cv_fold': person.cv_fold,
                                     'hour': raw[:, 39].astype(np.int32),
                                     'label': raw[:, 40].astype(np.int8)})
                pending.append(pd.concat([meta, frame], axis=1))
                hours += len(raw)
                positive_hours += int(raw[:, 40].sum())
                if len(pending) >= batch_patients:
                    flush()
                if i % 1000 == 0:
                    print(f'{profile}: {part} {i}/{len(selected)} patients', flush=True)
            flush()
        finally:
            if writer is not None:
                writer.close()
        if hours != int(selected.record_hours.sum()) or positive_hours != int(selected.positive_hours.sum()):
            raise AssertionError('Full-record denominator mismatch')
        counts.append({'split': part, 'patients': len(selected), 'hours': hours, 'positive_hours': positive_hours})
    schema = {'standard_id': STANDARD_ID, 'scope': 'full', 'profile': profile,
              'feature_count': len(names), 'features': names, 'dtype': 'float32',
              'metadata_columns': METADATA_COLUMNS, 'label_column': 'label',
              'manifest_sha256': manifest_hash, 'counts': counts,
              'input_rule': 'Use only features, never metadata columns, as model input',
              'missing_values': 'NaN preserved; any learned preprocessing fits on training folds only'}
    (output / 'feature_schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    print(json.dumps({'profile': profile, 'feature_count': len(names), 'counts': counts}), flush=True)
    return schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--preprocessing', type=Path, default=Path('outputs/preprocessing'))
    parser.add_argument('--profile', choices=list(FEATURE_NAMES), default='augmented')
    parser.add_argument('--split', choices=['all', 'train', 'validation', 'test'], default='all')
    parser.add_argument('--output', type=Path, help='New directory; defaults to outputs/features/<profile>')
    parser.add_argument('--batch-patients', type=int, default=128)
    args = parser.parse_args()
    build(args.data, args.preprocessing, args.output or Path('outputs/features') / args.profile,
          args.profile, args.split, args.batch_patients)


if __name__ == '__main__':
    main()
