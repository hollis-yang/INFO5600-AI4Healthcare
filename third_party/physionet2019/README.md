# Official PhysioNet/CinC 2019 scoring code

`evaluate_sepsis_score.py` and `LICENSE.txt` are unmodified copies from
[physionetchallenges/evaluation-2019](https://github.com/physionetchallenges/evaluation-2019)
at commit `467c49b514542be7a4a0bafe40fa2c3b064dda2e`.
The upstream BSD 2-Clause license is retained. `UPSTREAM.json` records the
original Git blob IDs and SHA-256 digests.

The project imports the official per-patient utility function and uses the
official cohort normalization. Its wrapper is tested against the upstream
directory-based evaluator. Full hourly trajectories and the original
`SepsisLabel` are required: this supplement is separate from the project's
custom pre-onset six-hour task. The underlying models and alert thresholds
are not optimized using utility scores.
