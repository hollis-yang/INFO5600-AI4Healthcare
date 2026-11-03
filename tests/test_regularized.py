import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sepsis_warning.regularized import patient_bags, prior_correct


class PatientLevelControls(unittest.TestCase):
    def test_patient_rows_stay_together_inside_each_bag(self):
        rows = pd.DataFrame({"patient_id": np.repeat(np.arange(50), 4),
                             "label": np.repeat(np.arange(50) % 5 == 0, 4).astype(int)})
        bags, patients = patient_bags(rows)
        for bag in range(3):
            for _, index in rows.groupby("patient_id").groups.items():
                self.assertEqual(len(np.unique(bags[list(index), bag])), 1)
            self.assertEqual(int((patients[f"bag_{bag}"] < 0).sum()), 10)
        repeated, _ = patient_bags(rows)
        np.testing.assert_array_equal(bags, repeated)

    def test_prior_correction_preserves_rank_and_unweighted_probabilities(self):
        p = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
        np.testing.assert_allclose(prior_correct(p, 1), p)
        weighted = expit(logit(p)+np.log(5))
        np.testing.assert_allclose(prior_correct(weighted, 5), p)
        self.assertTrue(np.all(np.diff(prior_correct(weighted, 5)) > 0))

    def test_individual_bag_predictions_and_frozen_weighted_cuts(self):
        from interpret.glassbox import ExplainableBoostingClassifier
        from joblib import parallel_backend
        from sepsis_warning.regularized import oob_probabilities, frozen_cut_types
        rng = np.random.default_rng(11)
        x = rng.normal(size=(200, 3))
        x[:, 2] = 0
        rows = pd.DataFrame({"patient_id": np.repeat(np.arange(50), 4),
                             "label": np.repeat(np.arange(50) % 5 == 0, 4).astype(int)})
        bags, _ = patient_bags(rows)
        parameters = dict(interactions=0, outer_bags=3, n_jobs=3, max_rounds=4,
                          smoothing_rounds=1, early_stopping_rounds=2, max_bins=16, random_state=42)
        with parallel_backend("threading", n_jobs=3):
            model = ExplainableBoostingClassifier(**parameters).fit(x, rows.label, bags=bags)
        oob, covered = oob_probabilities(model, x, bags)
        np.testing.assert_array_equal(covered, (bags < 0).any(axis=1))
        self.assertTrue(np.isfinite(oob[covered]).all())
        cuts = frozen_cut_types(model)
        with parallel_backend("threading", n_jobs=3):
            weighted = ExplainableBoostingClassifier(feature_types=cuts, **parameters).fit(
                x, rows.label, sample_weight=np.where(rows.label, 5.0, 1.0), bags=bags)
        for a, b in zip(model.bins_, weighted.bins_):
            np.testing.assert_array_equal(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
