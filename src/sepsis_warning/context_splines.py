"""Fold-fitted nonlinear context for LR, retaining every original feature."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sepsis_warning.feature_engineering import FeaturePreprocessor

CONTEXT_SPLINE_COLUMNS = ["HR__latest","Resp__latest","Temp__latest","SBP__latest",
                          "ICULOS__current","HospAdmTime__current"]


class ContextSplinePreprocessor(FeaturePreprocessor):
    def __init__(self,names,parameters):
        super().__init__(names,native=False,clip=True)
        self.context_parameters = dict(parameters)

    def fit(self,features):
        super().fit(features)
        self.spline_indices_ = [self.base_names_.index(name) for name in CONTEXT_SPLINE_COLUMNS]
        standardized = self.scaler_.transform(self._impute(features))[:,self.spline_indices_]
        parameters = dict(self.context_parameters)
        n_knots = parameters.pop("n_knots")
        parameters.pop("knots",None)
        knots = np.quantile(standardized,np.linspace(0,1,n_knots),axis=0)
        self.uniform_fallback_columns_ = []
        for j,name in enumerate(CONTEXT_SPLINE_COLUMNS):
            if (np.diff(knots[:,j]) <= 0).any():
                assert knots[-1,j] > knots[0,j], f"Constant spline input: {name}"
                knots[:,j] = np.linspace(knots[0,j],knots[-1,j],n_knots)
                self.uniform_fallback_columns_.append(name)
        self.spline_ = SplineTransformer(knots=knots,**parameters)
        basis = self.spline_.fit_transform(standardized)
        self.spline_scaler_ = StandardScaler().fit(basis)
        self.names_ += self.spline_.get_feature_names_out(CONTEXT_SPLINE_COLUMNS).tolist()
        self.spline_training_rows_ = len(features)
        return self
