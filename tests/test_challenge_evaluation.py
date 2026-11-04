import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sepsis_warning.challenge_evaluation import (_PreparedRanking, bootstrap_differences,
                                                evaluate_scores)


class ChallengeEvaluation(unittest.TestCase):
    def data(self):
        records = [np.zeros(40, dtype=np.int8), np.r_[np.zeros(12, np.int8), np.ones(28, np.int8)],
                   np.zeros(24, dtype=np.int8), np.r_[np.zeros(20, np.int8), np.ones(24, np.int8)]]
        stops = np.cumsum(list(map(len, records)))
        patients = pd.DataFrame({"patient_id": ["a", "b", "c", "d"],
                                 "hospital": ["A", "A", "B", "B"],
                                 "start": np.r_[0, stops[:-1]], "stop": stops})
        return np.concatenate(records), patients

    def test_prepared_ranking_matches_sklearn_ties_and_zero_weights(self):
        rng = np.random.default_rng(42)
        labels = rng.integers(2, size=150)
        scores = rng.integers(-3, 4, size=150).astype(float)
        codes = np.repeat(np.arange(15), 10)
        prepared = _PreparedRanking(labels, scores, codes)
        for _ in range(40):
            count = rng.integers(0, 4, size=15)
            weights = count[codes]
            result = prepared.evaluate(count)
            if result is None:
                continue
            self.assertAlmostEqual(result["auroc"], roc_auc_score(labels, scores, sample_weight=weights), places=13)
            self.assertAlmostEqual(result["auprc"], average_precision_score(labels, scores, sample_weight=weights), places=13)
        self.assertIsNone(prepared.evaluate(np.zeros(15)))

    def test_official_utility_phase_decomposition_and_unbounded_scores(self):
        labels, patients = self.data()
        scores = np.linspace(-2., 3., len(labels))
        result = evaluate_scores(labels, scores, patients, .2)
        self.assertAlmostEqual(result["auroc"], roc_auc_score(labels, scores))
        self.assertAlmostEqual(result["auprc"], average_precision_score(labels, scores))
        self.assertLess(result["utility_check_absolute_difference"], 1e-12)
        phase = result["alarm_utility_by_phase"]
        self.assertEqual(sum(row["hours"] for row in phase), len(labels))
        self.assertEqual(sum(row["alarm_hours"] for row in phase), int((scores >= .2).sum()))
        self.assertAlmostEqual(sum(row["normalized_utility_contribution"] for row in phase), result["utility_score"])
        self.assertEqual(set(result["per_hospital"]), {"A", "B"})
        self.assertNotIn("brier_score", result)

    def test_preonset_detection_uses_strictly_before_onset_window(self):
        labels, patients = self.data()
        scores = np.full(len(labels), -1.)
        # b's onset is 18 local hours.  A first alarm at -6 detects it.
        scores[int(patients.start.iloc[1]) + 12] = 2.
        # d's onset is 26: onset itself is not an early detection.
        scores[int(patients.start.iloc[3]) + 26] = 2.
        result = evaluate_scores(labels, scores, patients, 0.)["preonset_six_hour_detection"]
        self.assertEqual(result["event_patients"], 2)
        self.assertEqual(result["detected_event_patients"], 1)
        self.assertEqual(result["median_lead_hours"], 6.)

    def test_paired_bootstrap_identical_scores_have_exact_zero_difference(self):
        labels, patients = self.data()
        scores = np.arange(len(labels)) % 5 - 2.
        result = bootstrap_differences(labels, patients, scores, scores, 0., 0., repetitions=40)
        self.assertTrue(result["thresholds_fixed"])
        self.assertGreater(result["valid_resamples"], 0)
        for metric in result["difference"].values():
            self.assertEqual(metric, {"point": 0., "lower": 0., "upper": 0.})

    def test_bootstrap_differences_match_brute_patient_weighting(self):
        labels, patients = self.data()
        rng = np.random.default_rng(61)
        reference = rng.integers(-3, 4, size=len(labels)).astype(float)
        candidate = rng.integers(-2, 6, size=len(labels)).astype(float)
        result = bootstrap_differences(labels, patients, reference, candidate, 0., 1., repetitions=30, seed=9)
        self.assertAlmostEqual(result["difference"]["auroc"]["point"], roc_auc_score(labels, candidate)-roc_auc_score(labels, reference))
        # Reproduce the exact patient resamples and compare percentile endpoints.
        codes = np.repeat(np.arange(len(patients)), patients.stop-patients.start)
        raw = {"auroc": [], "auprc": []}
        sampler = np.random.default_rng(9)
        for _ in range(30):
            counts = np.bincount(sampler.integers(len(patients), size=len(patients)), minlength=len(patients))
            weights = counts[codes]
            if not np.dot(weights, labels) or not np.dot(weights, 1-labels):
                continue
            raw["auroc"].append(roc_auc_score(labels, candidate, sample_weight=weights)-roc_auc_score(labels, reference, sample_weight=weights))
            raw["auprc"].append(average_precision_score(labels, candidate, sample_weight=weights)-average_precision_score(labels, reference, sample_weight=weights))
        for metric, values in raw.items():
            self.assertAlmostEqual(result["difference"][metric]["lower"], np.quantile(values, .025), places=12)
            self.assertAlmostEqual(result["difference"][metric]["upper"], np.quantile(values, .975), places=12)

    def test_invalid_inputs_fail(self):
        labels, patients = self.data()
        scores = np.zeros(len(labels))
        with self.assertRaises(ValueError):
            evaluate_scores(labels, scores[:-1], patients, 0.)
        with self.assertRaises(ValueError):
            evaluate_scores(labels, scores, patients, np.nan)
        with self.assertRaises(ValueError):
            bootstrap_differences(labels, patients, scores, scores, 0., 0., repetitions=0)


if __name__ == "__main__":
    unittest.main()
