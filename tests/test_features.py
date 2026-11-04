import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_pipeline.build_features import build
from sepsis_pipeline.features import FEATURE_NAMES, transform
from sepsis_pipeline.features.augmented import missing_features
from sepsis_pipeline.schema import HEADER, STANDARD_ID


def inputs(n=30):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, 40))
    x[rng.random(x.shape) < .45] = np.nan
    x[:, 34], x[:, 35], x[:, 39] = 50, 0, np.arange(1, n + 1)
    return x


class FeatureTests(unittest.TestCase):
    def test_dimensions_and_causality(self):
        raw = inputs()
        for profile, count in [('augmented', 563), ('recent', 410)]:
            full = transform(raw, profile)
            self.assertEqual(full.shape, (30, count))
            self.assertEqual(len(set(FEATURE_NAMES[profile])), count)
            for stop in [1, 2, 5, 6, 11, 12, 20]:
                np.testing.assert_allclose(transform(raw[:stop], profile), full[:stop], rtol=0, atol=0, equal_nan=True)

    def test_excluded_inputs_and_missing_values(self):
        raw = inputs()
        changed = raw.copy(); changed[:, 36:38] = 99999
        for profile in FEATURE_NAMES:
            np.testing.assert_allclose(transform(raw, profile), transform(changed, profile), rtol=0, atol=0, equal_nan=True)
            with self.assertRaises(ValueError):
                transform(np.column_stack([raw, np.ones(len(raw))]), profile)
            result = transform(np.full((5, 40), np.nan), profile)
            self.assertFalse(np.isinf(result).any())
            self.assertTrue(np.isnan(result).any())

    def test_recent_excludes_stay_clock(self):
        raw = inputs(); changed = raw.copy(); changed[:, 38:40] += 10000
        np.testing.assert_allclose(transform(raw, 'recent'), transform(changed, 'recent'), rtol=0, atol=0, equal_nan=True)
        self.assertNotIn('ICULOS', FEATURE_NAMES['recent'])
        self.assertNotIn('HospAdmTime', FEATURE_NAMES['recent'])

    def test_observation_run_lengths(self):
        raw = np.full((6, 40), np.nan); raw[[2, 3, 5], 0] = 100
        f = missing_features(raw)
        self.assertAlmostEqual(f[-1, 0], 1.5)
        self.assertAlmostEqual(f[-1, 36], .25)
        self.assertAlmostEqual(f[-1, 72], .5)

    def test_builder_keeps_all_rows_and_patient_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); prep = root / 'prep'; prep.mkdir()
            data = root / 'data'; data.mkdir(); people = []; originals = []
            for i in range(2):
                raw = np.column_stack([inputs(20), np.r_[np.zeros(5), np.ones(15)] if i == 0 else np.zeros(20)])
                raw[:, 0] += i * 1000
                path = data / f'p{i}.psv'
                np.savetxt(path, raw, delimiter='|', header='|'.join(HEADER), comments='')
                people.append({'subject_id': f'A_p{i}', 'hospital': 'A', 'raw_path': path.name,
                               'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'split': 'test', 'cv_fold': -1, 'record_hours': 20,
                               'positive_hours': int(raw[:, 40].sum())})
                originals.append(raw)
            manifest = prep / 'patient_splits.parquet'
            pd.DataFrame(people).to_parquet(manifest, index=False)
            (prep / 'preprocessing_report.json').write_text(json.dumps({
                'standard_id': STANDARD_ID, 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
            for profile in FEATURE_NAMES:
                out = root / profile
                schema = build(data, prep, out, profile, batch_patients=1)
                result = pd.read_parquet(out / 'test.parquet')
                self.assertEqual(len(result), 40)
                self.assertEqual(int(result.label.sum()), 15)
                self.assertEqual(schema['counts'][0]['hours'], 40)
                for i, raw in enumerate(originals):
                    frame = result.loc[result.subject_id == f'A_p{i}']
                    np.testing.assert_allclose(frame[FEATURE_NAMES[profile]], transform(raw[:, :40], profile), rtol=0, atol=0, equal_nan=True)
                    np.testing.assert_array_equal(frame.label, raw[:, 40])
            (data / 'p0.psv').write_text((data / 'p0.psv').read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'Raw record changed'):
                build(data, prep, root / 'changed')


if __name__ == '__main__':
    unittest.main()
