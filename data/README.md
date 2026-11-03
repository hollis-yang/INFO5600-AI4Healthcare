# PhysioNet Challenge 2019

Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019, version 1.0.0.

- [Official dataset description](https://physionet.org/content/challenge-2019/1.0.0/)
- [Dataset DOI](https://doi.org/10.13026/v64v-d857)
- Original download source: `https://physionet-open.s3.amazonaws.com/challenge-2019/1.0.0/training/`
- Database license: [ODC Open Database License (ODbL)](LICENSE.txt).

## Contents

| Directory | Patient files |
| --- | ---: |
| `training_setA/` | 20,336 |
| `training_setB/` | 20,000 |
| Total | 40,336 |

The complete raw data contains 1,552,210 hourly records and 255,057,484 bytes. Both partitions are public training data; they are not a predefined train/test split.

Each `.psv` file contains one patient's hourly records. The columns are 40 inputs (8 vital signs, 26 laboratory variables and 6 context variables) and `SepsisLabel`. `NaN` indicates an unrecorded measurement.

For sepsis cases, `SepsisLabel` is 1 from the dataset-defined onset time minus 6 hours onward, including after onset. It is 0 for non-sepsis cases. The supplied labels have already been shifted 6 hours early.

The raw patient files are preserved unchanged. `manifest.csv` lists repository-relative paths, byte sizes, MD5 and SHA-256 checksums, hourly counts and positive-label indicators. `download_report.json` preserves the original download summary, not modeling results.

## Citation

Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B., Sharma, A., Nemati, S., & Clifford, G. D. (2019). *Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019* (version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857

Retain the dataset license and attribution when redistributing the data.
