import unittest

import numpy as np
import pandas as pd

from sepsis_warning.challenge_utility import (
    patient_utility_delta,
    patient_utility_targets,
    utility_targets,
)
from sepsis_warning.utility import official_scorer


class ChallengeUtilityTargets(unittest.TestCase):
    def test_known_boundaries_and_original_label_mismatch(self):
        labels = np.r_[np.zeros(20, dtype=np.int8), np.ones(20, dtype=np.int8)]
        onset = 26
        delta = patient_utility_delta(labels)
        expected = {-13: -.05, -12: 0., -9: .5, -6: 1., -5: 10/9,
                    0: 5/3, 3: 2., 4: 0., 6: 0.}
        for relative_hour, value in expected.items():
            self.assertAlmostEqual(delta[onset + relative_hour], value)
        self.assertEqual(labels[onset - 9], 0)
        self.assertEqual(labels[onset + 6], 1)
        target = patient_utility_targets(labels)
        self.assertEqual(target["classification_label"][onset - 9], 1)
        self.assertEqual(target["classification_weight"][onset + 6], 0.)
        self.assertEqual(target["classification_label"].dtype, np.dtype("int8"))
        self.assertEqual(target["delta"].dtype, np.dtype("float64"))

    def test_every_hour_flip_matches_pinned_official_scorer(self):
        score = official_scorer().compute_prediction_utility
        records = [np.zeros(1, np.int8), np.zeros(64, np.int8), np.ones(1, np.int8),
                   np.ones(24, np.int8)]
        for first_positive, hours in [(1, 6), (6, 50), (20, 60), (29, 30)]:
            labels = np.zeros(hours, dtype=np.int8)
            labels[first_positive:] = 1
            records.append(labels)
        for labels in records:
            inactive = np.zeros(len(labels), dtype=np.int8)
            baseline = score(labels, inactive)
            actual = []
            for hour in range(len(labels)):
                alarm = inactive.copy()
                alarm[hour] = 1
                actual.append(score(labels, alarm) - baseline)
            np.testing.assert_allclose(patient_utility_delta(labels), actual, rtol=0, atol=1e-13)

    def test_arbitrary_alarm_sequences_and_weighted_loss_equivalence(self):
        score = official_scorer().compute_prediction_utility
        rng = np.random.default_rng(2026)
        for labels in [np.zeros(50, dtype=np.int8), np.r_[np.zeros(20), np.ones(30)]]:
            targets = patient_utility_targets(labels)
            inactive_utility = score(labels, np.zeros(len(labels), np.int8))
            oracle = targets["classification_label"]
            oracle_utility = score(labels, oracle)
            for _ in range(30):
                alarm = rng.integers(2, size=len(labels), dtype=np.int8)
                utility = score(labels, alarm)
                self.assertAlmostEqual(utility - inactive_utility, np.dot(alarm, targets["delta"]), places=12)
                weighted_loss = np.dot(alarm != oracle, targets["classification_weight"])
                self.assertAlmostEqual(oracle_utility - utility, weighted_loss, places=12)

    def test_patient_map_preserves_boundaries_and_row_alignment(self):
        records = [np.zeros(16, np.int8), np.r_[np.zeros(4), np.ones(20)], np.ones(8, np.int8)]
        labels = np.concatenate(records)
        stops = np.cumsum([len(x) for x in records])
        patients = pd.DataFrame({"start": np.r_[0, stops[:-1]], "stop": stops})
        expected = np.concatenate([patient_utility_delta(x) for x in records])
        for mapping in [patients, patients.to_dict("records"), list(patients.itertuples(index=False))]:
            target = utility_targets(labels, mapping)
            np.testing.assert_array_equal(target["delta"], expected)
            np.testing.assert_array_equal(target["classification_weight"], np.abs(expected))
            np.testing.assert_array_equal(target["classification_label"], expected > 0)

    def test_invalid_labels_and_maps_fail_explicitly(self):
        for labels in [[0, 2], [0, np.nan], [[0, 1]]]:
            with self.assertRaises(ValueError):
                patient_utility_delta(labels)
        for mapping in [[], [{"start": 1, "stop": 4}], [{"start": 0, "stop": 3}],
                        [{"start": 0, "stop": 2}, {"start": 1, "stop": 4}],
                        [{"start": 0, "stop": 5}], [{"start": 0, "stop": 3.5}],
                        [{"start": 0, "stop": np.inf}]]:
            with self.assertRaises(ValueError):
                utility_targets(np.zeros(4, np.int8), mapping)


if __name__ == "__main__":
    unittest.main()
