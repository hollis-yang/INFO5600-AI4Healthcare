import unittest
import numpy as np
import pandas as pd
from sepsis_warning.ab_regularization import patient_folds,choose_candidates,model_candidates,settings,model_metrics


class HospitalStratifiedCV(unittest.TestCase):
    def test_each_patient_stays_in_one_hospital_event_stratified_fold(self):
        records=[]
        for hospital in ["A","B"]:
            for i in range(60):
                for hour in range(1,4):records.append((f"{hospital}_{i}",hospital,hour,int(i<12 and hour>1)))
        rows=pd.DataFrame(records,columns=["patient_id","hospital","hour","label"])
        patients,folds=patient_folds(rows)
        self.assertTrue((pd.DataFrame({"id":rows.patient_id,"fold":folds}).groupby("id").fold.nunique()==1).all())
        counts=patients.groupby(["hospital","has_event","fold"]).size()
        self.assertTrue((counts.xs(1,level="has_event")==4).all())
        self.assertTrue((counts.xs(0,level="has_event")==16).all())
        np.testing.assert_array_equal(folds,patient_folds(rows)[1])

    def test_selection_uses_macro_hospital_scores_and_requires_two_folds(self):
        study={"models":["lr"],"folds":3,"minimum_mean_ap_gain":.001,"minimum_improved_folds":2}
        rows=[]
        for candidate,values in [("baseline",[.05,.06,.07]),("one_lucky_fold",[.049,.059,.09]),("consistent",[.053,.063,.073])]:
            for fold,value in enumerate(values):rows.append({"model":"lr","candidate":candidate,"fold":fold,"macro_hospital_auprc":value,"outer_validation_auprc":100 if candidate=="one_lucky_fold" else 0})
        self.assertEqual(choose_candidates(rows,study)[0],{"lr":"consistent"})
        self.assertEqual(choose_candidates([r for r in rows if r['candidate']!='consistent'],study)[0],{"lr":"baseline"})

    def test_candidates_only_change_prespecified_penalties(self):
        candidates=model_candidates()
        self.assertEqual(len(candidates),11)
        allowed={"lr":{"C"},"ebm":{"min_samples_leaf","reg_lambda"},"lightgbm":{"min_child_samples","reg_lambda"}}
        for model in ["lr","ebm","lightgbm"]:
            base=next(c['parameters'] for c in candidates if c['model']==model and c['candidate']=='baseline')
            for c in [c for c in candidates if c['model']==model]:
                self.assertLessEqual({k for k in base if base[k]!=c['parameters'][k]},allowed[model])

    def test_macro_metric_does_not_equal_pooled_metric_in_general(self):
        rows=pd.DataFrame({'hospital':['A']*4+['B']*4,'label':[1,0,0,0,1,0,0,0]})
        p=np.array([.4,.3,.2,.1,.6,.9,.8,.7])
        result=model_metrics(rows,p)
        self.assertAlmostEqual(result['macro_hospital_auprc'],(.25+1)/2)
        self.assertNotAlmostEqual(result['macro_hospital_auprc'],result['pooled']['auprc_average_precision'])


if __name__=='__main__':unittest.main()
