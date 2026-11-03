import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sepsis_warning.data import FEATURE_NAMES, assess_patient, causal_features, config
from sepsis_warning.evaluation import alert_replay


def patient(n=12):
    x = np.full((n, 41), np.nan, np.float32)
    x[:, 34:40] = [60, 0, 1, 0, -10, 0]
    x[:, 39] = np.arange(1, n+1)
    x[:, 40] = 0
    return x


class TemporalRules(unittest.TestCase):
    def test_shifted_labels_and_post_onset_exclusion(self):
        x = patient()
        x[3:, 40] = 1  # First positive at hour 4; benchmark onset at hour 10.
        reason, keep, onset = assess_patient(x, config())
        self.assertEqual(reason, "included")
        self.assertEqual(onset, 10)
        np.testing.assert_array_equal(x[keep, 39], np.arange(1, 10))
        np.testing.assert_array_equal(x[keep, 40], [0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_ambiguous_and_unobserved_onsets(self):
        x = patient()
        x[:, 40] = 1
        self.assertEqual(assess_patient(x, config())[0], "positive_at_record_start")
        x[:, 40] = 0
        x[-1, 40] = 1
        self.assertEqual(assess_patient(x, config())[0], "onset_beyond_observation")

    def test_negative_followup_boundary(self):
        x = patient(12)
        _, keep, _ = assess_patient(x, config())
        np.testing.assert_array_equal(x[keep, 39], np.arange(1, 7))
        self.assertEqual(assess_patient(patient(6), config())[0], "no_complete_negative_window")

    def test_prefix_invariance_with_large_future_values(self):
        x = patient(12)
        x[1, 0], x[4, 0], x[8, 0] = 80, 90, 100
        prefix = causal_features(x[:6], config())
        x[6:, :34] = 1e6
        np.testing.assert_allclose(prefix, causal_features(x, config())[:6], equal_nan=True)

    def test_forward_fill_expiry_and_no_backfill(self):
        x = patient(12)
        x[2, 0] = 80
        f = causal_features(x, config())
        j = FEATURE_NAMES.index("HR__latest")
        self.assertTrue(np.isnan(f[0, j]))
        self.assertEqual(f[8, j], 80)  # Six hours since measurement.
        self.assertTrue(np.isnan(f[9, j]))

    def test_alert_suppression_uses_patient_and_time(self):
        rows = pd.DataFrame({"patient_id": ["a"]*8+["b"]*2,
                             "hour": list(range(1, 9))+[1, 2],
                             "label": [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                             "onset_hour": [9]*8+[np.nan]*2})
        result, episodes, _ = alert_replay(rows, np.ones(10), 0.5, suppression=6)
        self.assertEqual(episodes.hour.tolist(), [1, 7, 1])
        self.assertEqual(result["false_alert_episodes"], 2)
        self.assertEqual(result["detected_event_patients"], 1)


if __name__ == "__main__":
    unittest.main()
