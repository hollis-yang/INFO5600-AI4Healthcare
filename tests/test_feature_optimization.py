import sys
import unittest
import tempfile
import json
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from sepsis_warning.data import FEATURE_NAMES, TrainPreprocessor, config
from sepsis_warning.feature_engineering import alternative_features, feature_names, FeaturePreprocessor
from sepsis_warning.optimization import patient_folds, settings, choose_profiles


class CausalFeatureOptimization(unittest.TestCase):
    def make_patient(self):
        x = np.full((35, 41), np.nan, np.float32)
        x[:, 34:40] = [60, 0, 1, 0, -10, 0]
        x[:, 39], x[:, 40] = np.arange(1, 36), 0
        x[:, 0] = 70+2*np.arange(35)
        x[:, 3], x[:, 5] = 120, 70
        x[0, 19], x[18, 19] = 1., 1.9
        return x

    def test_lab_rates_use_real_elapsed_time_and_expire(self):
        x = self.make_patient()
        result = alternative_features(x, config(), settings()["features"], "labs")
        names = feature_names("labs")
        rate = names.index("Creatinine__last2_slope_24h")
        self.assertTrue(np.isnan(result[17, rate]))
        self.assertAlmostEqual(result[18, rate], .9/18, places=6)
        self.assertAlmostEqual(result[23, rate], .9/18, places=6)
        self.assertTrue(np.isnan(result[24, rate]))
        self.assertEqual(result[18, names.index("Creatinine__count_24h")], 2)
        self.assertEqual(result[24, names.index("Creatinine__count_24h")], 1)
        self.assertTrue(np.isnan(result[25, names.index("Creatinine__latest")]))

    def test_vital_slopes_and_baselines_use_only_past_observations(self):
        result = alternative_features(self.make_patient(), config(), settings()["features"])
        names = feature_names("combined")
        self.assertAlmostEqual(result[15, names.index("HR__slope_3h")], 2)
        self.assertAlmostEqual(result[15, names.index("HR__slope_12h")], 2)
        self.assertAlmostEqual(result[15, names.index("HR__deviation_prior12h")], 13)
        self.assertTrue(np.isnan(result[1, names.index("HR__deviation_prior12h")]))
        self.assertAlmostEqual(result[15, names.index("SBP_DBP__difference")], 50)

    def test_future_append_and_zero_denominators(self):
        x = self.make_patient()
        before = alternative_features(x[:20], config(), settings()["features"])
        x[20:, :34] = 1e5
        np.testing.assert_allclose(before, alternative_features(x, config(), settings()["features"])[:20], equal_nan=True)
        x[:, 3] = 0
        result = alternative_features(x, config(), settings()["features"])
        self.assertTrue(np.isnan(result[:, feature_names("combined").index("HR_SBP__ratio")]).all())

    def test_new_preprocessor_matches_original_and_preserves_native_mask(self):
        from sepsis_warning.data import causal_features
        x = causal_features(self.make_patient(), config())
        old = TrainPreprocessor(FEATURE_NAMES).fit(x)
        new = FeaturePreprocessor(FEATURE_NAMES).fit(x)
        native = FeaturePreprocessor(FEATURE_NAMES, native=True).fit(x)
        np.testing.assert_allclose(old.transform(x), new.transform(x), atol=0, rtol=0)
        a, b = new.transform(x), native.transform(x)
        np.testing.assert_array_equal(np.isnan(b), np.isnan(x[:, native.keep_]))
        np.testing.assert_allclose(a[np.isfinite(b)], b[np.isfinite(b)], atol=0, rtol=0)
        original = new.scaler_.mean_.copy()
        new.transform(np.full_like(x, 1e4))
        np.testing.assert_array_equal(new.scaler_.mean_, original)

    def test_splines_are_fitted_and_frozen_on_training_only(self):
        x = alternative_features(self.make_patient(), config(), settings()["features"])
        # Ensure all four prescribed spline inputs vary.
        for j, variable in enumerate(["HR", "Resp", "Temp", "SBP"]):
            x[:, feature_names("combined").index(variable+"__latest")] = np.linspace(j, j+2, len(x))
        p = FeaturePreprocessor(feature_names("combined"), spline_cfg=settings()["splines"]).fit(x[:20])
        knots = [s.t.copy() for s in p.spline_.bsplines_]
        out = p.transform(x[20:])
        self.assertEqual(out.shape[1], len(p.names_))
        self.assertTrue(np.isfinite(out).all())
        for before, after in zip(knots, p.spline_.bsplines_):
            np.testing.assert_array_equal(before, after.t)

    def test_grouped_folds_and_selection_do_not_use_outer_validation(self):
        rows = pd.DataFrame({"patient_id": np.repeat(np.arange(120), 3), "label": np.repeat(np.arange(120)%10 == 0, 3).astype(int)})
        patients, folds = patient_folds(rows)
        self.assertEqual(len(patients), 120)
        self.assertTrue(pd.DataFrame({"id": rows.patient_id, "fold": folds}).groupby("id").fold.nunique().eq(1).all())
        records = []
        for model in ["lr", "ebm", "lightgbm"]:
            for profile, scores in [("baseline", [.07,.07,.07]), ("good", [.08,.08,.069]), ("one_fold_only", [.12,.06,.06])]:
                records.extend({"model": model, "profile": profile, "fold": i, "auprc": score} for i, score in enumerate(scores))
        selected, _ = choose_profiles(records, settings()["study"])
        self.assertEqual(set(selected.values()), {"good"})

    def test_verified_cv_cache_purge_is_exactly_reversible(self):
        import joblib
        from sepsis_warning import feature_cache
        raw = np.arange(90,dtype=np.float32).reshape(30,3)
        raw[::3,1] = np.nan
        names = ["HR__latest","Temp__latest","Resp__latest"]
        processor = FeaturePreprocessor(names,native=True).fit(raw[:20])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root/"cases/test"
            directory.mkdir(parents=True)
            train,evaluation = processor.transform(raw[:20]),processor.transform(raw[20:])
            np.save(directory/"train_X.npy",train)
            np.save(directory/"evaluation_X.npy",evaluation)
            np.save(directory/"train_indices.npy",np.arange(20))
            np.save(directory/"evaluation_indices.npy",np.arange(20,30))
            joblib.dump(processor,directory/"preprocessor.joblib")
            (directory/"case.json").write_text(json.dumps({"stage":"cv","training_hours":20,"evaluation_hours":10,
                "features":3,"representation":{"source":"baseline"}}))
            feature_cache.verification_probes(directory)
            with patch.object(feature_cache,"PROCESSED",root),patch.object(feature_cache,"load_case_raw",return_value=(raw,names)):
                self.assertEqual(len(feature_cache.purge_cv_matrices(["test"])),2)
                self.assertFalse((directory/"train_X.npy").exists())
                self.assertEqual(len(feature_cache.restore_cv_matrices()),2)
            np.testing.assert_allclose(np.load(directory/"train_X.npy"),train,equal_nan=True,atol=0,rtol=0)
            np.testing.assert_allclose(np.load(directory/"evaluation_X.npy"),evaluation,equal_nan=True,atol=0,rtol=0)


if __name__ == "__main__":
    unittest.main()
