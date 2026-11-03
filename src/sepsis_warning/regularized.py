"""Patient-grouped early stopping, bag-specific predictions and prior adjustment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split


def patient_bags(rows, n_bags=3, holdout_fraction=0.2, seed=2026):
    codes, identifiers = pd.factorize(rows.patient_id, sort=False)
    labels = np.bincount(codes, weights=rows.label, minlength=len(identifiers)) > 0
    per_patient = np.ones((len(identifiers), n_bags), np.int8)
    for bag in range(n_bags):
        _, validation = train_test_split(np.arange(len(identifiers)), test_size=holdout_fraction,
                                          random_state=seed+31*bag, stratify=labels)
        per_patient[validation, bag] = -1
    assignments = pd.DataFrame({"patient_id": identifiers, "has_event": labels})
    for bag in range(n_bags):
        assignments[f"bag_{bag}"] = per_patient[:, bag]
    return per_patient[codes], assignments


def prior_correct(probability, positive_weight):
    """Analytic class-weight prior correction; not a guarantee of calibration."""
    if positive_weight == 1:
        return np.asarray(probability).copy()
    return expit(logit(np.clip(probability, 1e-15, 1-1e-15))-np.log(positive_weight))


def bag_model(model, bag):
    individual = model.copy()
    individual.term_scores_ = [scores[bag].copy() for scores in model.bagged_scores_]
    individual.intercept_ = np.array([model.bagged_intercept_[bag]])
    return individual


def oob_probabilities(model, features, bags):
    """Predict each row using only bags that excluded that patient's gradient updates."""
    score_sum = np.zeros(len(features))
    weight_sum = np.zeros(len(features))
    reconstruction = []
    for bag, weight in enumerate(model.bag_weights_):
        individual = bag_model(model, bag)
        reconstruction.append(individual.decision_function(features[:32]))
        holdout = bags[:, bag] < 0
        score_sum[holdout] += weight*individual.decision_function(features[holdout])
        weight_sum[holdout] += weight
    np.testing.assert_allclose(np.average(reconstruction, axis=0, weights=model.bag_weights_),
                               model.decision_function(features[:32]), atol=1e-10)
    covered = weight_sum > 0
    probability = np.full(len(features), np.nan)
    probability[covered] = expit(score_sum[covered]/weight_sum[covered])
    return probability, covered


def frozen_cut_types(model):
    """Reuse unweighted bin cuts so held-out class weights do not alter binning."""
    return [cuts[0].tolist() if len(cuts[0]) else "continuous" for cuts in model.bins_]
