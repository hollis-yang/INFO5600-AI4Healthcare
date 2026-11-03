"""Verify matched cohorts, frozen controls, causal matrices and saved predictions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import save_json
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.history_study import child_environment, digest, sha_file
from sepsis_warning.regularized import patient_bags


def main():
    processed, output = ROOT/"processed_data", ROOT/"outputs/history_study"
    completion = json.loads((output/"completed.json").read_text())
    fingerprints = json.loads((output/"input_fingerprints.json").read_text())
    assert completion["model_results"] == 45 and completion["trials"] == 15
    assert all(sha_file(ROOT/name) == expected for name, expected in fingerprints.items())
    train = pd.read_csv(processed/"train_rows.csv.gz")
    validation = pd.read_csv(processed/"validation_rows.csv.gz")
    split = pd.read_csv(processed/"splits/patient_splits.csv")
    reserved = set(split.loc[~split.patient_id.isin(train.patient_id), "patient_id"])
    parameters, members, checked = {}, {}, []
    for trial_path in sorted((processed/"history_study").glob("*/trial.json")):
        trial = json.loads(trial_path.read_text())
        directory = trial_path.parent
        rows = pd.read_csv(directory/"train_rows.csv.gz")
        identifiers = set(pd.read_csv(directory/"patients.csv").patient_id)
        assert identifiers == set(rows.patient_id)
        assert identifiers <= set(train.patient_id) and not identifiers & reserved
        expected_rows = train.loc[train.patient_id.isin(identifiers)].reset_index(drop=True)
        pd.testing.assert_frame_equal(rows, expected_rows)
        assert len(identifiers) == trial["patients"]
        assert int(rows.groupby("patient_id").label.max().sum()) == trial["event_patients"]
        np.testing.assert_array_equal(np.load(directory/"train_y.npy"), rows.label)
        key = (trial["fraction"], trial["sample_seed"])
        if key in members:
            assert members[key] == identifiers
        members[key] = identifiers
        processor = joblib.load(directory/"preprocessor.joblib")
        assert processor.scaler_.n_samples_seen_ == len(rows)
        x = np.load(ROOT/trial["train_X"], mmap_mode="r")
        val_x = np.load(ROOT/trial["validation_X"], mmap_mode="r")
        assert x.shape == (len(rows), trial["features"])
        assert val_x.shape == (len(validation), trial["features"])
        prefix = processed if trial["retention_hours"] == 6 else processed/"history_study"/f"history{trial['retention_hours']}h"
        raw_train = np.load(prefix/"train_features.npy", mmap_mode="r")
        raw_validation = np.load(prefix/"validation_features.npy", mmap_mode="r")
        indices = np.flatnonzero(train.patient_id.isin(identifiers).to_numpy())
        sample = np.linspace(0, len(rows)-1, 64, dtype=int)
        val_sample = np.linspace(0, len(validation)-1, 64, dtype=int)
        np.testing.assert_allclose(x[sample], processor.transform(raw_train[indices[sample]]), atol=0, rtol=0)
        np.testing.assert_allclose(val_x[val_sample], processor.transform(raw_validation[val_sample]), atol=0, rtol=0)
        for model_name in ["lr", "ebm", "lightgbm"]:
            result_dir = output/trial["name"]
            result = json.loads((result_dir/f"{model_name}_results.json").read_text())
            assert result["trial"] == trial and result["test_sets_evaluated"] == []
            signature = digest(result["parameters"])
            if model_name in parameters:
                assert parameters[model_name] == signature
            parameters[model_name] = signature
            scores = np.load(result_dir/f"{model_name}_validation_probability.npy")
            assert len(scores) == len(validation) and np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            for name, value in ranking_metrics(validation.label.to_numpy(), scores).items():
                np.testing.assert_allclose(value, result["validation"][name], atol=1e-12, rtol=0)
            bundle = joblib.load(result_dir/f"{model_name}_model.joblib")
            assert bundle["parameters"] == result["parameters"]
            assert bundle["preprocessor"].names_ == processor.names_
            model = bundle["model"]
            predicted = model.booster_.predict(val_x[val_sample]) if model_name == "lightgbm" else model.predict_proba(val_x[val_sample])[:, 1]
            np.testing.assert_allclose(predicted, scores[val_sample], atol=1e-12, rtol=0)
            if model_name != "lr" and not result["cached_reference_model"]:
                bags = result_dir/("ebm_patient_bags.csv" if model_name == "ebm" else "lightgbm_inner_patients.csv")
                actual = pd.read_csv(bags)
                _, expected = patient_bags(rows, n_bags=3 if model_name == "ebm" else 1, seed=2026)
                # CSV restores int8 signs as int64; compare their exact values.
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
            checked.append(f"{trial['name']}/{model_name}")
    for seed in [2026, 2027]:
        assert members[(.25, seed)] < members[(.5, seed)] < members[(1., None)]
    assert len(checked) == 45
    result = {"model_results_checked": len(checked), "data_conditions_checked": 15,
              "canonical_inputs_unchanged": True, "matched_patients_across_history_windows": True,
              "nested_stratified_patient_subsets": True, "outer_split_isolation": True,
              "subset_only_preprocessing_fit_counts": True, "sampled_matrix_transform_reconstruction": True,
              "fixed_model_parameter_signatures": parameters,
              "all_validation_metrics_recomputed": True, "all_saved_models_prediction_checked": True,
              "internal_patient_bags_reconstructed": True, "test_sets_evaluated": [],
              "source_fingerprints": {str(p.relative_to(ROOT)): sha_file(p) for p in
                  [*sorted((ROOT/"scripts").glob("*history*.py")), ROOT/"src/sepsis_warning/history_study.py",
                   ROOT/"src/sepsis_warning/data.py", ROOT/"configs/history_study.toml"]}}
    save_json(output/"final_checks.json", result)
    save_json(ROOT/"reports/history_study/final_checks.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    # macOS resolves OpenMP at process startup, before a saved LightGBM is loaded.
    if sys.platform == "darwin" and "--openmp-ready" not in sys.argv:
        raise SystemExit(subprocess.call([sys.executable, __file__, "--openmp-ready"], env=child_environment()))
    main()
