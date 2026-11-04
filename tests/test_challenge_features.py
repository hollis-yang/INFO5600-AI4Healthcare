import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sepsis_warning.challenge_features import block_slices, engineer_patient, feature_names


class ChallengeFeatures(unittest.TestCase):
    def patient(self, n=20):
        raw = np.full((n, 40), np.nan)
        raw[:, 34:39] = [60, 1, 0, 1, -10]
        raw[:, 39] = np.arange(11, 11+n)
        return raw

    def test_lags_preserve_real_missing_measurements(self):
        raw = self.patient()
        raw[[0, 2], 0] = [80, 100]
        result = engineer_patient(raw, "lag")
        names = feature_names("lag")
        value = names.index("challenge__HR__lag1h_value")
        flag = names.index("challenge__HR__lag1h_observed")
        self.assertTrue(np.isnan(result[0, value]))
        self.assertEqual(result[0, flag], 0)
        self.assertEqual(result[1, value], 80)
        self.assertEqual(result[1, flag], 1)
        self.assertTrue(np.isnan(result[2, value]))
        self.assertEqual(result[2, flag], 0)
        self.assertEqual(result[3, value], 100)

    def test_sparse_changes_counts_and_uncapped_age(self):
        raw = self.patient(30)
        raw[[0, 8, 21], 19] = [1.0, 1.8, 1.3]
        matrix = engineer_patient(raw, "observed")
        names = feature_names("observed")
        column = lambda suffix: names.index("challenge__Creatinine__"+suffix)
        self.assertTrue(np.isnan(matrix[7, column("last2_observed_delta")]))
        self.assertAlmostEqual(matrix[8, column("last2_observed_delta")], .8, places=6)
        self.assertEqual(matrix[8, column("last2_observed_interval_h")], 8)
        self.assertEqual(matrix[20, column("last_observed_age_h")], 12)
        self.assertAlmostEqual(matrix[21, column("last2_observed_delta")], -.5, places=6)
        self.assertEqual(matrix[21, column("last2_observed_interval_h")], 13)
        self.assertEqual(matrix[8, column("observed_count_6h")], 1)
        self.assertEqual(matrix[8, column("observed_count_12h")], 2)
        self.assertEqual(matrix[12, column("observed_count_12h")], 1)
        self.assertEqual(matrix[14, column("observed_count_6h")], 0)
        never = names.index("challenge__BUN__last_observed_age_h")
        self.assertTrue(np.isnan(matrix[:, never]).all())

    def test_clinical_requires_recent_contemporaneous_values(self):
        raw = self.patient()
        raw[0, [0, 3, 5]] = [90, 120, 70]
        result = engineer_patient(raw, "clinical")
        np.testing.assert_allclose(result[:3], [[.75, 50]]*3)
        self.assertTrue(np.isnan(result[3]).all())
        raw[5, 0], raw[3, 3] = 100, 100
        self.assertTrue(np.isnan(engineer_patient(raw, "clinical")[5, 0]))
        raw[5, 3] = 0
        self.assertTrue(np.isnan(engineer_patient(raw, "clinical")[5, 0]))

    def test_future_appending_and_mutation_do_not_change_prefix(self):
        generator = np.random.default_rng(23)
        raw = self.patient()
        raw[:, :34] = generator.normal(10, 3, size=(20, 34))
        raw[:, :34][generator.random((20, 34)) < .6] = np.nan
        full = engineer_patient(raw)
        for stop in (1, 2, 6, 11, 20):
            np.testing.assert_allclose(engineer_patient(raw[:stop]), full[:stop], rtol=0, atol=0, equal_nan=True)
        raw[11:, :34] = 1e8
        np.testing.assert_allclose(engineer_patient(raw)[:11], full[:11], rtol=0, atol=0, equal_nan=True)

    def test_blocks_empty_inputs_and_invalid_hours(self):
        raw = self.patient()
        full = engineer_patient(raw)
        self.assertEqual(full.dtype, np.float32)
        self.assertEqual(full.shape, (20, 228))
        self.assertEqual(len(set(feature_names())), 228)
        slices = block_slices()
        for block in slices:
            np.testing.assert_array_equal(engineer_patient(raw, block), full[:, slices[block]])
        order = ("clinical", "lag")
        np.testing.assert_array_equal(engineer_patient(raw, order), np.concatenate(
            [engineer_patient(raw, name) for name in order], axis=1))
        self.assertEqual(engineer_patient(raw[:0]).shape, (0, 228))
        self.assertEqual(engineer_patient(raw, ()).shape, (20, 0))
        with self.assertRaises(ValueError):
            engineer_patient(np.column_stack([raw, np.zeros(len(raw))]))
        raw[4, 39] += 1
        with self.assertRaises(ValueError):
            engineer_patient(raw)
        with self.assertRaises(ValueError):
            feature_names(("lag", "lag"))


if __name__ == "__main__":
    unittest.main()
