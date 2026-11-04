import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from sepsis_warning.alarm_policy_analysis import POLICIES, transform_patient, transform_scores, exact_threshold


class AlarmPolicyAnalysis(unittest.TestCase):
    def test_prefixes_and_patient_reset(self):
        values = np.array([.2, .8, .3, .1, .9, .4])
        people = pd.DataFrame({"start": [0, 3], "stop": [3, 6]})
        for policy in POLICIES:
            full = transform_patient(values, policy)
            for n in range(1, len(values)+1):
                np.testing.assert_allclose(transform_patient(values[:n], policy), full[:n], rtol=0, atol=1e-15)
            grouped = transform_scores(values, people, policy)
            expected = np.r_[transform_patient(values[:3], policy), transform_patient(values[3:], policy)]
            np.testing.assert_array_equal(grouped, expected)

    def test_risk_hold_and_consecutive_semantics(self):
        values = [.1, .8, .2, .1]
        np.testing.assert_array_equal(transform_patient(values, "max2"), [.1, .8, .8, .2])
        np.testing.assert_array_equal(transform_patient(values, "min2"), [.1, .1, .2, .1])
        self.assertEqual(transform_patient([.7], "min2")[0], .7)

    def test_threshold_ties_against_all_candidates(self):
        scores = np.array([.1, .9, .9, .2])
        delta = np.array([-.05, 1., -.05, .5])
        result = exact_threshold(scores, delta)
        thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)[::-1]]
        utilities = [delta[scores >= t].sum()/np.maximum(delta, 0).sum() for t in thresholds]
        best = int(np.argmax(utilities))
        self.assertAlmostEqual(result["utility"], utilities[best])
        self.assertEqual(result["threshold"], thresholds[best])


if __name__ == "__main__":
    unittest.main()
