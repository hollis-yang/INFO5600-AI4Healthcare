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

The local pipeline audits the raw data, constructs a conservative incident-onset cohort, and compares interpretable models. Historical studies include retention, patient learning curves, A-only feature screening, A+B expansion and hospital-stratified regularization. The new [563-feature matched comparison](reports/friend_563_comparison/report.md) uses the original pooled A+B patient split and complete records to compare XGBoost and LightGBM. Training CV determines iteration counts, validation selects utility thresholds, and frozen models are evaluated on test patients. The source data were inspected during earlier development; results remain exploratory. Full-record recognition is distinct from the original pre-onset warning task.

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

To investigate overfitting and class imbalance while preserving the outer split:

```bash
python scripts/tune_ebm.py
python scripts/report_regularized.py
```

`configs/ebm_regularized.toml` prespecifies two stronger complexity controls and positive-class weights 1, 2 and 5. Each model has three patient-grouped bags, each with an internal 80/20 training/early-stopping split. Internal patient OOB average precision selects complexity first and then weight; the outer A validation set is read for reporting after those choices are recorded. Native EBM returns the best-loss checkpoint; reported iteration counts include patience steps, not the exact best iteration.

All preprocessing and initial binning remain fitted within A training; this unsupervised transformation is shared across internal bags. OOB results are development estimates used for stopping and selection. Weighted models reuse the selected unweighted model's cuts, preventing held-out class weights from altering bin boundaries. The final outer validation and both test sets retain natural class proportions, with no over/undersampling.

For weights greater than 1, probabilities are reported both raw and after subtracting `log(weight)` from the model's log-odds. This analytic prior correction is not fitted calibration and does not guarantee calibration. It preserves ranking metrics. To reproduce corrected risk from a weighted saved bundle, apply `sepsis_warning.regularized.prior_correct` to the raw positive-class probabilities using the bundle's `positive_weight`.

Experiments are cached under `outputs/ebm_regularized/`; matching completed configurations are reused on rerun. Curated metrics, reports and figures are copied into `reports/ebm_regularized/`. Large models, predictions and processed matrices remain local.

For the controlled history-retention and patient-count study across all three models:

```bash
python scripts/run_history_study.py
python scripts/report_history_study.py
python scripts/verify_history_study.py
```

The protocol in `configs/history_study.toml` changes only the latest laboratory value expiry (6/12/24 hours). Vital-sign expiry, rolling statistics and the target horizon remain 6 hours. LR and the previously selected unweighted 16-bin EBM retain their settings. LightGBM uses one fixed regularized baseline, with a patient-grouped internal early-stopping split and then a refit on all selected training patients. Fitted stopping iterations can differ; the stopping rules and hyperparameters are fixed.

Learning curves use nested, event-stratified 25%/50% training-patient subsets with two sample seeds, plus the shared full A-training cohort. The exact patients are matched across retention strategies and models. Each trial fits its imputer, clipping bounds and scaler using that trial's training subset only. Both test sets remain reserved. There are 15 data conditions and 45 model results; full-data 6h LR/EBM models are verified and reused.

Workers run in separate local processes and log to `outputs/history_study/<trial>/`. On macOS they reuse sklearn's bundled OpenMP runtime when available; otherwise follow LightGBM's documented `libomp` setup. Matching completed models are cached for resumption. Data variants and matrices are in `processed_data/history_study/`; curated tables, patient membership, uncertainty intervals and plots are in `reports/history_study/`.

The verification command checks all 45 result metrics and saved-model predictions, matched and nested patient membership, unchanged canonical inputs, preprocessing fit counts, sampled matrix reconstruction, fixed model settings, and internal patient holdouts. It writes `reports/history_study/final_checks.json`.

The learning-curve bands show the range of two training subsamples, not confidence intervals. Separate patient-bootstrap intervals quantify validation uncertainty conditional on those draws. These curves vary event and non-event patient counts together; they cannot isolate positive-case scarcity or establish a final sample-size ceiling.

To screen preprocessing and feature improvements within the original A training set:

```bash
python scripts/run_feature_optimization.py
python scripts/report_feature_optimization.py
python scripts/verify_feature_optimization.py
```

The information-preserving supplement retains all original features and appends the new dynamics. It was added after observing losses on the initial training CV folds and was fixed before viewing this run's initial outer-validation results:

```bash
python scripts/run_feature_augmentation.py
python scripts/run_context_splines.py
python scripts/report_feature_optimization.py --include-augmentation
python scripts/verify_feature_optimization.py --include-augmentation --purge-cv-matrices
```

This adds 12 CV models, including a spline-expanded LR, for 75 distinct CV fits and 78 comparison rows overall. Final selection still uses only the same three training-patient folds and the original eligibility rule. The consolidated results are under `reports/feature_augmentation/`; its `added_design.json` records when the supplement was fixed and whether initial validation artifacts existed at that time. The supplement waits for the initial study before taking model CPU slots.

The LR-specific context-spline control adds three further CV fits (78 distinct fits and 81 comparison rows overall). It retains the original features and adds fold-fitted splines for four vital signs plus ICULOS and HospAdmTime, testing whether LR's linear context effects are too restrictive. This control was designed from the training-CV ablation after initial validation had been evaluated; its timestamp and scope are recorded in `context_spline_design.json`. Selection still reads only training-CV metrics, and both the earlier selection and the final selection are retained.

`configs/feature_optimization.toml` prespecifies three patient-stratified CV folds and the candidate profiles. The original cohort, outer split, six-hour target and six-hour latest-value expiry remain fixed. Classifier settings and internal stopping rules come directly from the previous three-model study. Every CV fold fits its own preprocessing using only its training patients; neither CV holdout patients nor outer validation patients enter that fit. Internal early-stopping bags share the preprocessing fitted within the current CV training partition.

The study compares median imputation with native NaN handling for EBM/LightGBM; replaces sparse six-hour lab summaries with changes between the last two real observations within 24 hours; adds vital slopes, variability and deviations from each patient's earlier observations; and tests two measurement-age-aware combinations. Separate controls remove time/unit context or quantile clipping. LR also has a small fold-fitted spline expansion. These are cumulative profiles, with the context/clipping/spline controls branching from the combined features. LR's native-missing profile is identical to its baseline and reuses those three fits.

There are 66 CV comparison rows from 63 distinct model fits. For each classifier, a candidate must improve mean fold AUPRC by at least 0.001 and improve at least two folds; the qualifying candidate with the highest mean is selected, otherwise the original baseline is retained. `selection.json` is frozen before evaluating the three selected full-training models on A validation. Both test sets remain reserved, and no model hyperparameter tuning is mixed into this feature study.

Raw alternative features and fold-specific matrices are under `processed_data/feature_optimization/`; fitted models, predictions and worker logs are under `outputs/feature_optimization/`. The scheduler limits simultaneous model workers to ten requested CPU threads across three processes. Matching completed models and preprocessing cases are cached for resumption. Curated CV tables, selected validation results, uncertainty intervals, patient folds, plots and verification evidence are under `reports/feature_optimization/`. Large cached matrices remain local and require approximately 25 GB of additional disk space.

`--purge-cv-matrices` removes only generated CV train/evaluation matrices after all requested checks pass. It retains verification probes, raw features, patient indices, fitted preprocessors, models and predictions, as well as the full-training selected-model matrices. To reconstruct the CV matrices before forcing model refits, run `python scripts/restore_feature_cv_matrices.py`. The cache round trip is covered by a test with exact numerical reconstruction. The supplement's raw feature cache is separate under `processed_data/feature_augmentation/`.

CV-fold variation is not an independent confidence interval. Outer validation has been viewed during earlier development, so this study remains exploratory; the previously reserved tests are still required for a final generalization claim. Different classifiers can select different preprocessing profiles, while using the same patient splits and information available at prediction time.

The selected-profile table compares complete model pipelines; different feature profiles mean these differences cannot be attributed solely to classifier architecture. The original shared-feature baselines remain available as controls. Full small result records are retained in `model_metadata_archive.json` alongside the summaries.

Recovery records document macOS eviction of one result JSON, two deterministic input caches and some virtual-environment files during this run. Metadata was recovered from saved models and predictions; input caches were restored only after matching their original SHA-256; InterpretML 0.7.8 and LightGBM 4.6.0 wheels were restored with official PyPI digest verification. These recoveries did not alter patient assignments or classifier settings. The optional `recover_feature_result.py`, `recover_feature_inputs.py` and `repair_pinned_libraries.py` maintenance scripts record this local recovery procedure.

Run commands from the repository root. Settings are recorded in `configs/baseline.toml`. A complete preparation run writes approximately 3.6 GB of processed arrays. The raw files in `data/` remain unchanged.

## A+B expansion experiment

This separate experiment changes the earlier role of hospital B. It preserves the original A split, stratifies B patients 70/15/15 with seed 42, and compares A-only training, full A+B training, and two A+B samples with exactly the original A training patient count. The matched samples preserve the full mixture's hospital/event proportions using seeds 2026 and 2027. They do not match A-only's higher event prevalence. B test patients remain excluded from fitting and threshold selection, but B is now represented in pooled training: this is no longer unseen-hospital validation. A test remains reserved.

```sh
python scripts/run_ab_expansion.py
python scripts/report_ab_expansion.py
python scripts/verify_ab_expansion.py
```

On macOS, launch commands with the existing OpenMP environment helper (use the same wrapper for reporting and verification):

```sh
python - <<'PY'
import subprocess, sys
sys.path.insert(0, 'src')
from sepsis_warning.history_study import child_environment
subprocess.run([sys.executable, '-u', 'scripts/run_ab_expansion.py'],
               env=child_environment(), check=True)
PY
```

The main comparison freezes the original A-defined 312 features for all three classifiers and retains classifier settings from `reports/history_study/model_settings.json`. Each new training cohort fits its own preprocessing. A secondary LightGBM control uses the previous 349-feature, native-missing, no-clipping pipeline. Four original A-only models are reused, and ten new models are fitted. A single threshold per model is selected on the same pooled A+B validation patients, then applied unchanged to the component validation sets and B test. Thus A-only describes risk-model fitting; its threshold calibration also uses B validation.

Settings are in `configs/ab_expansion.toml`. The runner freezes source/input fingerprints before fitting and evaluating B test. It allows two workers, at most one EBM, and seven requested CPU threads. Generated arrays, the versioned split and case-specific transforms are under `processed_data/ab_expansion/`; models and predictions are under `outputs/ab_expansion/`. Small result snapshots, paired patient-bootstrap intervals, hospital distribution diagnostics, fixed B patient assignments and a Chinese report are under `reports/ab_expansion/`. The experiment needs approximately 7 GB of additional local storage. The original data and split artifacts remain unchanged.

`resume_ab_expansion.py` resumes model jobs only after every case and the model manifest have already been prepared under the frozen protocol. It checks frozen source hashes and records post-run input availability. This macOS execution used this route after Desktop cloud eviction stalled redundant reads of old inputs. `recover_ab_inputs.py` rebuilt the evicted B features and patient audit; its `--rows-only` and `--processor-only` options recover hourly metadata and the redundant original processor. Each replacement must reproduce its pre-experiment SHA-256 exactly. Identical pinned InterpretML/LightGBM wheels were imported from a non-cloud temporary directory; wheel digests, input recovery and post-run comparisons are recorded with the report. These maintenance steps change neither the data content nor model versions.

Bootstrap intervals condition on the fitted models and chosen training subsets. Two matched draws do not capture all training uncertainty, and the exploratory comparisons are not adjusted for multiplicity. Compare AUPRC changes within the same evaluation cohort. Full A+B versus A-only changes source distribution and sample size together; full versus matched A+B provides a more focused mixed-pool size control. No final model is selected from B test.

## Hospital-stratified A+B regularization and missingness study

This study keeps all existing A/B patient assignments and the common 312-feature pipeline. It creates three internal folds from the same 27,911 A+B training patients, stratifying by hospital and event status while keeping each patient's complete hourly trajectory together. Every fold fits its own imputation, clipping and scaling using only its outer training partition. Existing event-stratified patient bags provide internal early stopping within that partition.

`configs/ab_regularization.toml` fixes 11 candidates: five LR L2 penalties (`C=0.001,0.01,0.1,1,10`); three joint EBM leaf-size/L2 settings; and three joint LightGBM child-size/L2 settings. Tree size, bins, learning rate and maximum iterations are unchanged. The 33 CV fits use the mean of A/B-specific AUPRC, averaged over three folds. A candidate must gain at least 0.001 and improve at least two folds to replace the original setting. Otherwise the original setting is retained. The selection file is written before scoring the selected full-training models on A/B validation. Neither test set is evaluated in this round.

```sh
python scripts/run_ab_regularization.py
python scripts/analyze_ab_missingness.py
python scripts/report_ab_regularization.py
python scripts/verify_ab_regularization.py
```

On macOS, use the study's environment helper at process startup for report/verification commands that load LightGBM (the runner already uses it for model workers):

```sh
python - <<'PY'
import subprocess, sys
sys.path.insert(0, 'src')
from sepsis_warning.ab_regularization import runtime_environment
for script in ['report_ab_regularization.py', 'verify_ab_regularization.py']:
    subprocess.run([sys.executable, '-u', 'scripts/' + script],
                   env=runtime_environment(), check=True)
PY
```

The scheduler requests at most eight CPU threads across three workers, including at most two concurrent EBMs; LR uses two threads. Runtime package versions and input/source hashes are saved before fitting. On this machine the helper reuses the same verified pinned InterpretML/LightGBM wheels outside the Desktop cloud directory when that cache exists. Matching completed candidates are cached for resumption; changing a frozen protocol requires a separate study directory.

The missingness analysis uses only fixed training patients. It separates observations absent in an hourly bin from latest measurements unavailable after six-hour expiry, reports hour- and patient-weighted proportions, standardizes both hospitals to the pooled event-patient proportion, and includes first-24-hour and ICU-hour-band controls. A separate patient-level hospital classifier measures source separability using 40 whole-eligible-episode missing fractions with fold-fitted scaling. Those full-episode diagnostic features never enter hourly sepsis prediction. Missingness differences and source separability describe distribution shift; they do not establish that missingness caused model errors.

New raw/processed caches are under `processed_data/ab_regularization/`, fitted candidates and predictions under `outputs/ab_regularization/`, and small reports, fixed fold assignments and verification evidence under `reports/ab_regularization/`. Fold caches need approximately 4.6 GB; the full-training normalization and matrices reuse the unchanged A+B training artifacts. Both original and selected full models calibrate alarms on the same pooled validation patients. Validation and paired-bootstrap results remain exploratory because validation has been viewed during earlier development; CV fold variation is not a confidence interval, and the reported validation intervals condition on selected models.

The consolidated [complete metrics report](reports/ab_regularization/all_metrics.md) includes cohort counts, every regularization candidate, training/CV/OOF scores, validation AUROC/AUPRC/Brier, hourly classification and confusion matrices, patient-level alerts, both official utility threshold policies, and all 40 variables' missingness rates. [all_model_metrics.csv](reports/ab_regularization/all_model_metrics.csv) combines the selected validation results into one row per model and evaluation cohort; the hourly-only table is [hourly_classification_metrics.csv](reports/ab_regularization/hourly_classification_metrics.csv). These additional classification metrics use the saved predictions and existing thresholds, without model refitting or new threshold selection. Rebuild with `python scripts/report_all_metrics.py` and check reproducibility with `python scripts/report_all_metrics.py --verify`; the latter requires the local cached rows and predictions.

### Official utility supplement

`run_utility_comparison.py` evaluates A-only, original A+B and CV-selected A+B models on the complete raw trajectories of the same A/B validation patients. It retains the original SepsisLabel without another shift and applies each model's frozen pooled-validation budget threshold to every hour, without the custom alarm suppression rule. A fixed 0.5 threshold is a secondary sensitivity check. Utility is never used to change the frozen CV selection or tune thresholds. Neither test set is evaluated.

The unmodified official [PhysioNet/CinC 2019 scorer](https://github.com/physionetchallenges/evaluation-2019) is pinned under `third_party/physionet2019/` with its original BSD 2-Clause license, Git blob IDs and SHA-256. Utility is normalized after summing patient contributions: 0 means inaction and 1 means the cohort oracle; negative scores are possible. This full-trajectory supplement and its selected cohort differ from the custom pre-onset task and from the official hidden Challenge test.

After the regularization run, use the same macOS environment wrapper for `run_utility_comparison.py` and `verify_utility_comparison.py`. `run_utility_comparison.py --prepare-only` can prepare full trajectories while CV is running. The supplement adds about 0.3 GB under `processed_data/ab_regularization/utility/`; results and patient-level contributions remain under `outputs/ab_regularization/utility/`, and small snapshots under `reports/ab_regularization/utility/`. Unit fixtures and real patient subsets are checked against the upstream directory-based evaluator.

## 563-feature full-record XGBoost and LightGBM comparison

The separate [563-feature full-record comparison](reports/friend_563_comparison/report.md) follows the preprocessing and unchanged augmented feature code published on `main` at `32ee355b9373e81a9d3bdb1f1b4a22883e04678e`. It compares native XGBoost and LightGBM using the same original A+B patient splits and hospital/event-stratified training folds. Original PSV bytes are read from that Git commit, verified against the manifest and parsed as float64 before the published code produces float32 features. It does not reuse the earlier float32 raw cache. Features retain NaN; there is no imputation, scaling, clipping, class weighting or resampling.

Both models use a fixed shallow-tree configuration and equal early-stopping budgets. Training CV selects round counts by AUROC, final fits use the median selected count, and validation selects one utility-maximizing threshold per model. Models and thresholds are frozen before test feature construction or evaluation. The CV holdout also provides early stopping, so CV is a development estimate; validation utility is a calibration result. Test results use frozen choices, but the source patients have been examined historically. Full-record performance includes post-onset recognition and is distinct from the original pre-onset warning task.

Install the additional XGBoost dependency with `python -m pip install -r requirements-friend563.txt`. On this machine a verified official wheel is isolated under `/private/tmp/info5600_xgb_runtime`; the helper includes it and the existing macOS OpenMP settings:

```sh
git fetch origin main
python - <<'PY'
import subprocess, sys
sys.path.insert(0, 'src')
from sepsis_warning.friend_563 import runtime_environment
for script in ['run_friend_563.py', 'verify_friend_563.py']:
    subprocess.run([sys.executable, '-u', 'scripts/' + script],
                   env=runtime_environment(), check=True)
PY
```

Generated matrices and source snapshots are isolated in `processed_data/friend_563_comparison/`; models, saved predictions, calibration and freeze records are in `outputs/friend_563_comparison/`. Small reports are in `reports/friend_563_comparison/`. Parameters and package versions are recorded; the friend's unpublished training configuration is not claimed to be reproduced.

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
- `outputs/ebm_regularized/selected_model.joblib`: internally selected regularized model, preprocessing, class weight and validation alarm threshold.
- `reports/baseline_comparison/`, `reports/ebm_regularized/`: small, versioned result snapshots for comparing experiments.

## Original A-only analysis protocol

1. Verify every raw file against the committed manifest (MD5/SHA-256), header and row count.
2. Require adult age, consecutive integer ICU hours, binary monotonic labels, an observed 0-to-1 transition for event cases, and a reconstructed onset within the observation period. Exclude start-positive and ambiguous cases, post-onset hours, and negative windows without 6 hours of follow-up. Record each exclusion; these conservative choices restrict the target population.
3. Split eligible hospital A patients 70/15/15 by event status, seed 42. Reserve all eligible B patients for external evaluation. The realized split list is saved under `processed_data/splits/`.
4. Build 318 causal features: for each of the 34 physiological variables, latest value (6-hour expiry), trailing 6-hour observed mean/min/max/change, current missingness, never-measured flag, hours since measurement, and measurement count; plus the 6 current context variables and their missingness flags. The rolling window is `(t-6, t]`; filling permits a value exactly 6 hours old. Changes require at least two observed values.
5. Drop features entirely missing in A training; impute numeric medians and categorical modes, clip continuous features at training 0.5th/99.5th percentiles, and standardize using training statistics. Flags/counts are not quantile-clipped. Apply the same frozen transform to every split.
6. Fit one prespecified L2 LR (`C=1`, no class weighting or resampling); report validation AUROC, average precision (called AUPRC here), Brier score and patient-bootstrap intervals. This first baseline does not perform hyperparameter selection or three-fold CV.
7. Replay validation alarms with 6-hour suppression. A true alert is in the final 6 pre-onset hours; earlier alerts and alerts in non-event cases count as false. Select the threshold on validation to maximize event sensitivity under 1 false episode per 100 eligible patient-hours. The same-set operating-point results are exploratory and may be optimistic.

No patient/hospital IDs, labels, reconstructed onsets, future measurements or final record lengths are model inputs. Raw hourly bins approximate information availability; exact laboratory result-release times are unavailable. Missingness features may capture clinical workflow. Coefficients describe model associations among correlated inputs, not causal effects.

This is a custom pre-onset six-hour prediction task, not the official full-trajectory PhysioNet utility-score evaluation.
