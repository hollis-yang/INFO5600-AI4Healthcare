import unittest
import numpy as np
from sepsis_warning.missingness import patient_means,event_standardized


class MissingnessControls(unittest.TestCase):
    def test_patient_weighting_prevents_long_records_from_dominating(self):
        values=np.array([[0],[1],[1],[1]],bool);codes=np.array([0,1,1,1])
        result=patient_means(values,codes,2)
        self.assertEqual(float(result.mean()),.5)
        self.assertEqual(float(values.mean()),.75)
        prefix=patient_means(values,codes,2,np.array([True,False,False,False]))
        self.assertEqual(prefix[0,0],0);self.assertTrue(np.isnan(prefix[1,0]))

    def test_event_standardization_changes_only_common_case_weights(self):
        values=np.array([[.2],[.2],[.8]])
        result=event_standardized(values,np.array([0,0,1]),.25)
        np.testing.assert_allclose(result,[.35])


if __name__=='__main__':unittest.main()
