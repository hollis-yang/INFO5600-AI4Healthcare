import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from sepsis_warning.ab_expansion import split_b_patients,matched_mixture,FixedColumnsPreprocessor,settings
from sepsis_warning.data import TrainPreprocessor


class ExpansionControls(unittest.TestCase):
    def rows(self):
        ids = [f"{hospital}_{i:03d}" for hospital in ["A","B"] for i in range(200)]
        return pd.DataFrame({"patient_id":np.repeat(ids,3),"hospital":np.repeat(["A"]*200+["B"]*200,3),
                             "label":np.tile(np.repeat(np.arange(200)%10 == 0,3),2).astype(int)})

    def test_b_patient_partition_is_stratified_and_reproducible(self):
        rows = self.rows().query("hospital == 'B'")
        split = split_b_patients(rows,settings()["study"])
        pd.testing.assert_frame_equal(split,split_b_patients(rows,settings()["study"]))
        self.assertEqual(len(split),200)
        self.assertTrue(split.patient_id.is_unique)
        self.assertEqual(set(split.expansion_split),{"B_train","B_validation","B_test"})
        self.assertEqual(split.groupby("expansion_split").has_event.sum().to_dict(),{"B_test":3,"B_train":14,"B_validation":3})

    def test_size_match_preserves_hospital_and_event_counts(self):
        rows = self.rows()
        first,second = matched_mixture(rows,200,2026),matched_mixture(rows,200,2027)
        self.assertEqual(len(first),200)
        self.assertNotEqual(first,second)
        for selected in [first,second]:
            patients = rows[rows.patient_id.isin(selected)].groupby("patient_id").agg(hospital=("hospital","first"),label=("label","max"))
            self.assertEqual(patients.groupby(["hospital","label"]).size().to_dict(),{("A",0):90,("A",1):10,("B",0):90,("B",1):10})

    def test_fixed_columns_match_original_and_exclude_new_b_only_variables(self):
        names = ["HR__latest","Temp__latest","EtCO2__latest"]
        x = np.array([[70,36,np.nan],[80,np.nan,np.nan],[90,38,np.nan]],np.float32)
        original = TrainPreprocessor(names).fit(x)
        frozen = FixedColumnsPreprocessor(names,original.names_).fit(x)
        np.testing.assert_allclose(frozen.transform(x),original.transform(x),atol=0,rtol=0)
        observed_in_b = x.copy();observed_in_b[:,2] = [30,40,50]
        np.testing.assert_allclose(frozen.transform(x),frozen.transform(observed_in_b),atol=0,rtol=0)

    def test_missing_in_sample_field_is_constant_and_holdout_does_not_change_stats(self):
        names = ["HR__latest","Temp__latest"]
        x = np.array([[70,np.nan],[80,np.nan],[90,np.nan]],np.float32)
        p = FixedColumnsPreprocessor(names,names).fit(x)
        mean = p.inner_.scaler_.mean_.copy()
        held = np.array([[100,40]],np.float32)
        self.assertEqual(p.transform(held)[0,1],0)
        np.testing.assert_array_equal(mean,p.inner_.scaler_.mean_)


if __name__ == "__main__":unittest.main()
