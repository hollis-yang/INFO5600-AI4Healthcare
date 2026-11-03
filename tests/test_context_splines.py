import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from sepsis_warning.context_splines import ContextSplinePreprocessor,CONTEXT_SPLINE_COLUMNS
from sepsis_warning.optimization import settings


class FoldFittedContext(unittest.TestCase):
    def test_context_splines_freeze_training_knots_and_handle_quantile_ties(self):
        names = CONTEXT_SPLINE_COLUMNS
        x = np.arange(600,dtype=np.float32).reshape(100,6)
        x[:85,-1] = -1
        parameters = {k:v for k,v in settings()["splines"].items() if k != "variables"}
        processor = ContextSplinePreprocessor(names,parameters).fit(x[:90])
        self.assertIn("HospAdmTime__current",processor.uniform_fallback_columns_)
        before = [s.t.copy() for s in processor.spline_.bsplines_]
        transformed = processor.transform(x[90:]*100)
        self.assertEqual(transformed.shape[1],len(processor.names_))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(processor.spline_training_rows_,90)
        for knot,spline in zip(before,processor.spline_.bsplines_):
            np.testing.assert_array_equal(knot,spline.t)


if __name__ == "__main__":unittest.main()
