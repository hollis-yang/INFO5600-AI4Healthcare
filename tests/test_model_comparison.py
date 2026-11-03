import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sepsis_warning.evaluation import paired_patient_bootstrap


class PairedComparison(unittest.TestCase):
    def test_identical_models_have_zero_paired_difference(self):
        rows = pd.DataFrame({"patient_id": np.repeat(np.arange(12), 3),
                             "label": np.tile([0, 0, 1], 12)})
        scores = np.tile([0.1, 0.3, 0.8], 12)
        result = paired_patient_bootstrap(rows, scores, scores, 20, 42)
        self.assertEqual(result["valid_resamples"], 20)
        for interval in result["ebm_minus_lr"].values():
            self.assertEqual(interval, {"lower": 0.0, "upper": 0.0})


if __name__ == "__main__":
    unittest.main()
