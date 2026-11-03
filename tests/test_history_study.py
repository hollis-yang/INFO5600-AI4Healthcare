import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from sepsis_warning.data import FEATURE_NAMES, causal_features, config
from sepsis_warning.history_study import sample_patients, PreparedAveragePrecision


class HistoryControls(unittest.TestCase):
    def make_patient(self):
        x = np.full((30, 41), np.nan, np.float32)
        x[:, 34:40] = [60, 0, 1, 0, -10, 0]
        x[:, 39] = np.arange(1, 31)
        x[:, 40] = 0
        x[1, 0], x[1, 19] = 80, 1.2
        return x

    def test_only_latest_lab_values_change_with_retention(self):
        x = self.make_patient()
        base = causal_features(x, config())
        same = causal_features(x, config(), lab_retention_hours=6)
        longer = causal_features(x, config(), lab_retention_hours=24)
        np.testing.assert_allclose(base, same, equal_nan=True)
        unchanged = np.r_[np.arange(8), np.arange(34, len(FEATURE_NAMES))]
        np.testing.assert_allclose(base[:, unchanged], longer[:, unchanged], equal_nan=True)
        self.assertTrue(np.isnan(longer[15, FEATURE_NAMES.index("HR__latest")]))
        self.assertAlmostEqual(longer[15, FEATURE_NAMES.index("Creatinine__latest")], 1.2, places=5)
        self.assertTrue(np.isnan(longer[26, FEATURE_NAMES.index("Creatinine__latest")]))

    def test_longer_retention_is_still_causal(self):
        x = self.make_patient()
        prefix = causal_features(x[:10], config(), lab_retention_hours=24)
        x[10:, :34] = 1e6
        np.testing.assert_allclose(prefix, causal_features(x, config(), lab_retention_hours=24)[:10], equal_nan=True)

    def test_learning_samples_are_nested_and_patient_stratified(self):
        rows = pd.DataFrame({"patient_id": np.repeat(np.arange(100), 4),
                             "label": np.repeat(np.arange(100) % 10 == 0, 4).astype(int)})
        quarter = sample_patients(rows, .25, 2026)
        half = sample_patients(rows, .5, 2026)
        full = sample_patients(rows, 1, 2026)
        self.assertTrue(set(quarter) < set(half) < set(full))
        self.assertEqual(quarter, sample_patients(rows, .25, 2026))
        self.assertEqual(sum(patient % 10 == 0 for patient in half), 5)

    def test_cached_weighted_ap_matches_sklearn_with_ties_and_zero_weights(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.default_rng(41)
        labels = rng.integers(0, 2, 100)
        for probability in [np.round(rng.random(100), 1), np.full(100, .1)]:
            prepared = PreparedAveragePrecision(labels, probability)
            for _ in range(10):
                weights = rng.integers(0, 4, 100)
                self.assertAlmostEqual(prepared.evaluate(weights),
                    average_precision_score(labels, probability, sample_weight=weights), places=12)


if __name__ == "__main__":
    unittest.main()
