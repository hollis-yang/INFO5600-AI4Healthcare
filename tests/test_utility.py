import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sepsis_warning.utility import official_scorer,patient_utility,normalized_utility,paired_utility_contrast


class OfficialUtility(unittest.TestCase):
    def test_upstream_example_and_false_positive_cost(self):
        scorer=official_scorer()
        self.assertAlmostEqual(scorer.compute_prediction_utility([0,0,0,0,1,1],[0,0,1,1,1,1]),3.388888888888889)
        self.assertAlmostEqual(patient_utility(np.zeros(10),np.ones(10))['observed'],-.5)

    def test_inaction_and_oracle_normalization_and_unshifted_labels(self):
        labels=np.r_[np.zeros(12),np.ones(28)]
        inactive=patient_utility(labels,np.zeros(40))
        self.assertEqual(normalized_utility(pd.DataFrame([inactive]))['normalized_utility'],0)
        best=np.zeros(40);best[6:22]=1  # first label 12 -> onset 18; [-12,+3], inclusive.
        self.assertEqual(normalized_utility(pd.DataFrame([patient_utility(labels,best)]))['normalized_utility'],1)
        changed=best.copy();changed[22:]=1
        self.assertEqual(patient_utility(labels,best)['observed'],patient_utility(labels,changed)['observed'])

    def test_wrapper_matches_unmodified_directory_evaluator(self):
        labels=[np.r_[np.zeros(15),np.ones(25)],np.zeros(32)]
        predictions=[np.array([int(i in range(7,21)) for i in range(40)]),np.array([int(i in [3,4]) for i in range(32)])]
        components=pd.DataFrame([patient_utility(y,p) for y,p in zip(labels,predictions)])
        with tempfile.TemporaryDirectory() as temporary:
            lab=Path(temporary)/'labels';pred=Path(temporary)/'predictions';lab.mkdir();pred.mkdir()
            for i,(y,p) in enumerate(zip(labels,predictions)):
                np.savetxt(lab/f'p{i}.psv',y,header='SepsisLabel',comments='',fmt='%d')
                probability=np.where(p==1,.8,.2)
                np.savetxt(pred/f'p{i}.psv',np.c_[probability,p],header='PredictedProbability|PredictedLabel',comments='',delimiter='|')
            result=official_scorer().evaluate_sepsis_score(str(lab),str(pred))
        self.assertAlmostEqual(result[4],normalized_utility(components)['normalized_utility'])

    def test_identical_models_have_zero_paired_contrast(self):
        frame=pd.DataFrame([{'patient_id':'a',**patient_utility(np.r_[np.zeros(12),np.ones(24)],np.ones(36))},
            {'patient_id':'b',**patient_utility(np.zeros(20),np.ones(20))}])
        result=paired_utility_contrast(frame,frame,repetitions=20)
        self.assertEqual(result['difference'],0);self.assertEqual(result['lower'],0);self.assertEqual(result['upper'],0)


if __name__=='__main__':unittest.main()
