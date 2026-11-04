# Frozen A-only models evaluated on all eligible hospital B patients

Training uses the original A 70/15/15 split: 14,091 A training patients. No model or preprocessor was fitted again.

Evaluation includes all 19,743 eligible B patients (905 events), with 641,470 eligible pre-onset hours. Official utility uses all 757,924 original hours of those same patients and the original A-validation-only thresholds.

| Model | Pre-onset AUROC | Pre-onset AUPRC | Official utility |
|---|---:|---:|---:|
| lr | 0.723262 | 0.031581 | 0.099126 |
| ebm | 0.791807 | 0.049039 | 0.131498 |
| lightgbm | 0.816010 | 0.051077 | 0.225675 |

The original raw labels are not shifted again. Utility uses hourly binary outputs without alarm suppression. The full-trajectory AUROC/AUPRC are also saved in metrics.csv for a separate complete-record view.

B has already been examined in earlier experiments; these are exploratory transfer results, not a new untouched external test. The A test set remains unevaluated.

All saved predictions on the previously evaluated B-validation/B-test subsets match within 1e-12. Inputs and models remain unchanged; details are recorded in verification.json.
