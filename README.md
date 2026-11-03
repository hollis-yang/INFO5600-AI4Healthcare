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

## Reproducible pipeline

The code is published in modeling order. The first two stages are runnable from the repository root with Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 1. Validate records, screen patients, and freeze splits.
python -m sepsis_pipeline.preprocess --data data --output outputs/preprocessing

# 2. Generate the main model's 563 causal features for every included hour.
python -m sepsis_pipeline.build_features --profile augmented

# Optional 410-feature comparison without explicit length-of-stay clocks.
python -m sepsis_pipeline.build_features --profile recent

python -m unittest discover -s tests -v
```

On Windows, activate the environment with `.venv\Scripts\activate`.
Output directories must be new; use a different `--output` for a new run.
Generated patient-level tables, feature matrices and models are ignored by Git.

| Stage | Code | Explanation |
| --- | --- | --- |
| 01 · Preprocessing and patient split | [`sepsis_pipeline/preprocess.py`](sepsis_pipeline/preprocess.py) | [规则、分母、运行方式](docs/01_preprocessing.md) |
| 02 · Feature engineering | [`sepsis_pipeline/features/`](sepsis_pipeline/features/), [`build_features.py`](sepsis_pipeline/build_features.py) | [563 / 410 维特征及防泄漏约束](docs/02_feature_engineering.md) |
| Validation and provenance | [`tests/`](tests/), [`reports/`](reports/) | [来源和复现核验](docs/reproducibility.md) |

## Fixed cohort and evaluation contract

`SEPSIS_FULL_RECORD_V2` uses **all hours of the 39,874 included patients**, including post-onset and record-ending hours, with the original `SepsisLabel`. Full records do not reinstate patients excluded by the established cohort rules. See the preprocessing document for exclusions.

Split patients within each hospital, stratified by event, into 70% training / 15% validation / 15% test (seed 42); then pool corresponding sets. Training-only three-fold cross-validation is stratified by hospital and event (seed 2026). All models must use the same manifest.

| Split | Patients | Event patients | Full-record hours |
| --- | ---: | ---: | ---: |
| Training | 27,911 | 1,742 | 1,081,518 |
| Validation | 5,981 | 374 | 233,738 |
| Test | 5,982 | 374 | 230,710 |

The sole primary comparison metric is pooled full-record hourly **AUROC**. Future model selection uses training out-of-fold AUROC; validation selects one threshold per model by maximizing full-record Utility. Test data must not select models, features, or thresholds. AP, Utility and hourly classification metrics are fixed secondary metrics. No notification cooldown is applied to this hourly evaluation.

Full-record performance includes recognition after onset. It does not establish pre-onset warning performance. This split is a retrospective patient holdout within two known hospitals; the source data were previously inspected during development.

## Publication status

Data preprocessing and feature engineering are now included. Training, model weights and model evaluation scripts will be separate stages; this update does not retrain or replace the established models.
