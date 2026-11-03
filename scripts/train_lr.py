"""Train one prespecified LR baseline and report hospital A validation results."""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from threadpoolctl import threadpool_limits
from sepsis_warning.data import config, environment, save_json
from sepsis_warning.evaluation import (alert_replay, patient_bootstrap, ranking_metrics,
                                      select_alert_threshold, threshold_metrics)


def main():
    started = time.monotonic()
    cfg = config()
    processed = ROOT / cfg["data"]["processed_directory"]
    output = ROOT / cfg["data"]["output_directory"]
    train_x = np.load(processed / "train_X.npy", mmap_mode="r")
    train_y = np.load(processed / "train_y.npy")
    val_x = np.load(processed / "validation_X.npy", mmap_mode="r")
    val_y = np.load(processed / "validation_y.npy")
    rows = pd.read_csv(processed / "validation_rows.csv.gz")
    assert np.array_equal(rows.label.to_numpy(), val_y)
    assert train_x.shape[1] == val_x.shape[1]
    model = LogisticRegression(C=cfg["lr"]["C"], solver="lbfgs", penalty="l2", class_weight=None,
                               max_iter=cfg["lr"]["max_iter"], tol=cfg["lr"]["tolerance"],
                               random_state=cfg["split"]["random_seed"])
    print(f"Fitting LR on {len(train_y):,} hours, {train_x.shape[1]} features; no resampling or class weighting", flush=True)
    with threadpool_limits(limits=cfg["lr"]["threads"]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        fit_seconds = time.monotonic()-started
        scores = model.predict_proba(val_x)[:, 1]
        train_scores = model.predict_proba(train_x)[:, 1]
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    metrics = ranking_metrics(val_y, scores)
    print(f"Validation: {metrics}; iterations={model.n_iter_.tolist()}; fit={fit_seconds:.1f}s", flush=True)
    save_json(output / "lr_validation_metrics_early.json", metrics)
    threshold, threshold_table = select_alert_threshold(rows, scores, cfg)
    alert_metrics, episodes, per_patient = alert_replay(rows, scores, threshold, cfg["alerts"]["suppression_hours"])
    threshold_table.to_csv(output / "lr_validation_threshold_search.csv", index=False)
    episodes.to_csv(output / "lr_validation_alert_episodes.csv", index=False)
    per_patient.to_csv(output / "lr_validation_patient_alerts.csv", index=False)
    predictions = rows.copy()
    predictions["probability"] = scores
    predictions.to_csv(output / "lr_validation_predictions.csv.gz", index=False, compression="gzip")
    print(f"Validation alert operating point: {alert_metrics}", flush=True)
    intervals = patient_bootstrap(rows, scores, cfg["lr"]["bootstrap_samples"], cfg["split"]["random_seed"])
    constant = np.full(len(val_y), train_y.mean())
    results = {"evaluation_split": "hospital_A_validation", "test_sets_evaluated": [],
               "patients": int(rows.patient_id.nunique()), "hours": len(rows),
               "training_patients": int(pd.read_csv(ROOT/cfg["data"]["split_file"]).query("split == 'train'").shape[0]),
               "training_hours": len(train_y), "features": int(train_x.shape[1]),
               "training_positive_hour_prevalence": float(train_y.mean()),
               "validation": metrics, "training": ranking_metrics(train_y, train_scores),
               "constant_probability_reference": ranking_metrics(val_y, constant),
               "patient_bootstrap_95_ci": intervals, "bootstrap_samples": cfg["lr"]["bootstrap_samples"],
               "default_threshold_0_5": threshold_metrics(val_y, scores, 0.5),
               "validation_selected_threshold_hour_metrics": threshold_metrics(val_y, scores, threshold),
               "validation_selected_threshold_alert_metrics": alert_metrics,
               "threshold_selection": "maximize event sensitivity subject to validation false-alert budget; tie-break on PPV then lower burden",
               "threshold_result_limitation": "threshold selected and reported on the same validation set; not a held-out estimate",
               "model": {"type": "L2 logistic regression", "C": cfg["lr"]["C"], "solver": "lbfgs",
                         "class_weight": None, "iterations": int(model.n_iter_[0]),
                         "converged": not bool(convergence), "convergence_warnings": convergence},
               "fit_seconds": round(fit_seconds, 2), "environment": environment()}
    processor = joblib.load(output / "preprocessor.joblib")
    joblib.dump({"model": model, "preprocessor": processor, "threshold": threshold,
                 "configuration": cfg, "feature_names": processor.names_}, output / "lr_baseline.joblib", compress=3)
    coefficients = pd.DataFrame({"feature": processor.names_, "standardized_coefficient": model.coef_[0]})
    coefficients["absolute_coefficient"] = coefficients.standardized_coefficient.abs()
    coefficients.sort_values("absolute_coefficient", ascending=False).to_csv(output / "lr_coefficients.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    precision, recall, _ = precision_recall_curve(val_y, scores)
    axes[0].plot(recall, precision, color="#24788c", label=f"LR AP = {metrics['auprc_average_precision']:.3f}")
    axes[0].axhline(val_y.mean(), color="gray", linestyle="--", label=f"Prevalence = {val_y.mean():.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[0].legend()
    fpr, tpr, _ = roc_curve(val_y, scores)
    axes[1].plot(fpr, tpr, color="#24788c", label=f"AUROC = {metrics['auroc']:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].legend()
    observed, predicted = calibration_curve(val_y, scores, n_bins=10, strategy="quantile")
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_event_fraction": observed}).to_csv(output / "lr_calibration_bins.csv", index=False)
    axes[2].plot(predicted, observed, marker="o", color="#24788c", label="LR, quantile bins")
    lim = max(predicted.max(), observed.max()) * 1.1
    axes[2].plot([0, lim], [0, lim], color="gray", linestyle="--")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration", xlim=(0, lim), ylim=(0, lim))
    axes[2].legend()
    fig.suptitle("Hospital A validation - preliminary LR baseline")
    fig.tight_layout()
    fig.savefig(output / "lr_validation_curves.png", dpi=180)
    plt.close(fig)
    top = coefficients.nlargest(20, "absolute_coefficient").sort_values("standardized_coefficient")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.feature, top.standardized_coefficient, color=np.where(top.standardized_coefficient >= 0, "#bf6546", "#24788c"))
    ax.set(xlabel="Coefficient per training-standardized feature", title="LR: 20 largest absolute coefficients (correlated features)")
    fig.tight_layout()
    fig.savefig(output / "lr_coefficients.png", dpi=160)
    plt.close(fig)
    results["total_seconds"] = round(time.monotonic()-started, 2)
    save_json(output / "lr_results.json", results)
    print(f"Saved baseline results in {results['total_seconds']:.1f}s. Test sets remain unevaluated.", flush=True)
    if convergence:
        raise RuntimeError("LR did not converge; inspect the saved convergence warnings.")


if __name__ == "__main__":
    main()
