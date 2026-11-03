# Interpretable Early Warning of Sepsis in ICU Patients

AI for Healthcare course project.

Team: Dennis Wang, Haoyang Song, Wenzhuo Zhang.

## Data

The complete public PhysioNet Challenge 2019 dataset is included in `data/`:

- `data/training_setA/`: 20,336 patient files.
- `data/training_setB/`: 20,000 patient files.
- Total: 40,336 patient files and 1,552,210 hourly records.

Each patient has one pipe-delimited `.psv` file with hourly measurements, 40 input variables and `SepsisLabel`.

See [data/README.md](data/README.md) for the source, label definition, checksums and license.

## Status

The local pipeline audits the raw data, constructs a conservative incident-onset cohort, fixes patient splits, builds causal features, and runs a logistic regression baseline on hospital A validation data. Hospital A test and hospital B external-test results remain reserved.

## Run locally

Use Python 3.11 or newer:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/prepare_data.py
python scripts/train_lr.py
```

To compare a pure additive EBM with the saved LR baseline on the same cached data:

```bash
python scripts/train_ebm.py
python scripts/compare_models.py
```

EBM settings are in `configs/ebm.toml`. The first EBM run uses one fixed configuration (`interactions=0`, 64 bins, 1,000 rounds) and all A-training rows, with internal random-hour validation/early stopping disabled. It reuses LR's training-fitted preprocessor and exact matrices. Comparison intervals use paired patient resamples. Both test sets remain reserved.

Run commands from the repository root. Settings are recorded in `configs/baseline.toml`. A complete preparation run writes approximately 3.6 GB of processed arrays. The raw files in `data/` remain unchanged.

## Local outputs

- `processed_data/`: fixed patient split list, raw numeric cache, causal features, transformed matrices and labels. This directory is separate from `data/` and ignored by Git.
- `outputs/patient_summary.csv`, `cohort_flow.csv`, `split_summary.csv`: patient audit, mutually exclusive exclusions and data split counts.
- `outputs/training_missingness.csv`, `training_measurement_intervals.csv`, `training_missingness.png`: training-only data quality summaries.
- `outputs/preprocessor.joblib`, `preprocessing_parameters.csv`, `feature_dictionary.csv`: training-fitted preprocessing and feature definitions.
- `outputs/lr_baseline.joblib`, `lr_results.json`, `lr_validation_predictions.csv.gz`: fitted baseline, metrics and validation predictions.
- `outputs/lr_validation_curves.png`, `lr_coefficients.png`: preliminary performance and coefficient plots.
- `outputs/baseline_report.txt`: human-readable summary of this local run.
- `outputs/ebm_baseline.joblib`, `ebm_results.json`, `ebm_validation_predictions.csv.gz`: fitted additive EBM and validation results.
- `outputs/lr_ebm_comparison.json`, `lr_ebm_comparison.csv`, `ebm_vs_lr_report.txt`: paired validation comparison, including operating points under the same false-alert ceiling.
- `outputs/lr_ebm_validation_comparison.png`, `ebm_feature_importance.png`, `ebm_shape_functions.png`: model comparison and EBM explanations.
- `outputs/ebm_input_fingerprint.json`: fingerprints of reused data, split list, preprocessor and LR reference predictions; checked unchanged after EBM training.

## Analysis protocol

1. Verify every raw file against the committed manifest (MD5/SHA-256), header and row count.
2. Require adult age, consecutive integer ICU hours, binary monotonic labels, an observed 0-to-1 transition for event cases, and a reconstructed onset within the observation period. Exclude start-positive and ambiguous cases, post-onset hours, and negative windows without 6 hours of follow-up. Record each exclusion; these conservative choices restrict the target population.
3. Split eligible hospital A patients 70/15/15 by event status, seed 42. Reserve all eligible B patients for external evaluation. The realized split list is saved under `processed_data/splits/`.
4. Build 318 causal features: for each of the 34 physiological variables, latest value (6-hour expiry), trailing 6-hour observed mean/min/max/change, current missingness, never-measured flag, hours since measurement, and measurement count; plus the 6 current context variables and their missingness flags. The rolling window is `(t-6, t]`; filling permits a value exactly 6 hours old. Changes require at least two observed values.
5. Drop features entirely missing in A training; impute numeric medians and categorical modes, clip continuous features at training 0.5th/99.5th percentiles, and standardize using training statistics. Flags/counts are not quantile-clipped. Apply the same frozen transform to every split.
6. Fit one prespecified L2 LR (`C=1`, no class weighting or resampling); report validation AUROC, average precision (called AUPRC here), Brier score and patient-bootstrap intervals. This first baseline does not perform hyperparameter selection or three-fold CV.
7. Replay validation alarms with 6-hour suppression. A true alert is in the final 6 pre-onset hours; earlier alerts and alerts in non-event cases count as false. Select the threshold on validation to maximize event sensitivity under 1 false episode per 100 eligible patient-hours. The same-set operating-point results are exploratory and may be optimistic.

No patient/hospital IDs, labels, reconstructed onsets, future measurements or final record lengths are model inputs. Raw hourly bins approximate information availability; exact laboratory result-release times are unavailable. Missingness features may capture clinical workflow. Coefficients describe model associations among correlated inputs, not causal effects.

This is a custom pre-onset six-hour prediction task, not the official full-trajectory PhysioNet utility-score evaluation.
