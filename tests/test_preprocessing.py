import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_pipeline.preprocess import assign_splits, classify_record, read_record
from sepsis_pipeline.schema import HEADER


def patient(n=20):
    x = np.full((n, 41), np.nan)
    x[:, 34], x[:, 35], x[:, 39], x[:, 40] = 50, 0, np.arange(1, n + 1), 0
    return x


class PreprocessingTests(unittest.TestCase):
    def test_cohort_exclusions(self):
        x = patient(); x[:, 40] = 1
        self.assertEqual(classify_record(x)[0], 'positive_at_first_record')
        x = patient(); x[-2:, 40] = 1
        self.assertEqual(classify_record(x)[0], 'inferred_onset_after_record_end')
        self.assertEqual(classify_record(patient(6))[0], 'no_complete_prediction_window')
        x = patient(); x[3, 39] = 99
        self.assertEqual(classify_record(x)[0], 'invalid_hour_sequence')
        x = patient(); x[3, 40] = 1
        self.assertEqual(classify_record(x)[0], 'invalid_label_sequence')
        x = patient(); x[:, 34] = 17
        self.assertEqual(classify_record(x)[0], 'age_not_eligible')

    def test_classification_preserves_full_record(self):
        x = patient(); x[5:, 40] = 1
        original = x.copy()
        reason, onset = classify_record(x)
        self.assertEqual((reason, onset), ('included', 12))
        np.testing.assert_array_equal(x, original)
        self.assertEqual(int(x[:, 40].sum()), 15)

    def test_schema_and_empty_record(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'record.psv'
            path.write_text('|'.join(HEADER) + '\n')
            x, digest = read_record(path)
            self.assertEqual(x.shape, (0, 41))
            self.assertEqual(classify_record(x)[0], 'empty')
            self.assertEqual(len(digest), 64)
            path.write_text('wrong|header\n')
            with self.assertRaises(ValueError):
                read_record(path)

    def test_split_reproducible_patient_disjoint_and_training_only_cv(self):
        people = pd.DataFrame([
            {'subject_id': f'{h}_{i:04}', 'raw_sha256': f'{h}_{i:04}', 'hospital': h, 'event': int(i % 5 == 0)}
            for h in ['A', 'B'] for i in range(200)])
        a = assign_splits(people)
        b = assign_splits(people.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.subject_id.is_unique)
        self.assertEqual(a.groupby('split').size().to_dict(), {'test': 60, 'train': 280, 'validation': 60})
        self.assertTrue(a.loc[a.split != 'train', 'cv_fold'].eq(-1).all())
        self.assertEqual(set(a.loc[a.split == 'train', 'cv_fold']), {0, 1, 2})


if __name__ == '__main__':
    unittest.main()
