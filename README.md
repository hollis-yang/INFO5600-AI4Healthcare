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

## Stage 01: preprocessing and patient split

The first modeling stage is now available as a runnable Python package. Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m sepsis_pipeline.preprocess --data data --output outputs/preprocessing
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate`. The output directory must be new.

See [the preprocessing specification](docs/01_preprocessing.md), [the implementation](sepsis_pipeline/preprocess.py), and [full-cohort validation](reports/preprocessing_validation.json).

`SEPSIS_FULL_RECORD_V2` retains all hours of the established 39,874-patient screened cohort, with original SepsisLabel, including post-onset and record-ending hours. Within each hospital, patient/event-stratified 70/15/15 splits use seed 42; training-only three-fold CV uses hospital/event strata and seed 2026.

| Split | Patients | Event patients | Full hours |
| --- | ---: | ---: | ---: |
| Training | 27,911 | 1,742 | 1,081,518 |
| Validation | 5,981 | 374 | 233,738 |
| Test | 5,982 | 374 | 230,710 |

The sole primary metric is full-record hourly AUROC. Future model selection uses training out-of-fold AUROC; validation chooses one threshold per model by maximizing full-record Utility. Test data do not select models or thresholds. This known-hospital retrospective holdout includes post-onset recognition; it is not an untouched external cohort or an isolated test of early warning.

Feature engineering will be published as the next stage. Generated patient manifests and feature/model artifacts stay in ignored output directories.
