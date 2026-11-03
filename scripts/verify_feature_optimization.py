"""Audit patient isolation, fold-only transforms, model controls and saved predictions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

import joblib
import numpy as np
import pandas as pd
from sepsis_warning.data import FEATURE_NAMES, TrainPreprocessor, assess_patient, config, save_json
from sepsis_warning.feature_engineering import alternative_features, FeaturePreprocessor
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.history_study import child_environment, digest, sha_file
from sepsis_warning.optimization import (OUTPUT, PROCESSED, REPORT, patient_folds, raw_features, settings, choose_profiles)
from sepsis_warning.regularized import patient_bags
from sepsis_warning.augmentation import ADDED_DATA, ADDED_OUTPUT, ADDED_REPORT, added_names
from sepsis_warning.feature_cache import load_case_raw, verification_probes, purge_cv_matrices


def case_raw_features(split,spec):
    return load_case_raw(split,spec)


def verify_statistics(processor, raw, indices):
    probes = [name for name in ["HR__latest", "Creatinine__latest", "ICULOS__current", "Creatinine__last2_slope_24h", "HR__slope_12h"]
              if name in processor.base_names_]
    for name in probes:
        j, raw_j = processor.base_names_.index(name), processor.names.index(name)
        values = np.array(raw[indices, raw_j], copy=True)
        observed = values[np.isfinite(values)]
        np.testing.assert_allclose(np.median(observed), processor.fill_[j], rtol=0, atol=1e-6)
        if processor.clip:
            bounds = np.quantile(observed, [.005,.995])
            np.testing.assert_allclose(bounds, [processor.lower_[j],processor.upper_[j]], rtol=1e-6, atol=1e-6)
        else:
            assert np.isneginf(processor.lower_[j]) and np.isposinf(processor.upper_[j])
        np.clip(values, processor.lower_[j], processor.upper_[j], out=values)
        values[np.isnan(values)] = processor.fill_[j]
        np.testing.assert_allclose(np.mean(values,dtype=float), processor.scaler_.mean_[j], rtol=1e-10, atol=1e-9)
        std = np.std(values,dtype=float)
        np.testing.assert_allclose(std if std else 1., processor.scaler_.scale_[j], rtol=1e-8, atol=1e-8)


def main():
    include_added = "--include-augmentation" in sys.argv
    target,snapshot = (ADDED_OUTPUT,ADDED_REPORT) if include_added else (OUTPUT,REPORT)
    completion = json.loads((target/"completed.json").read_text())
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    selection = json.loads((target/"selection.json").read_text())
    initial_selection = json.loads((OUTPUT/"selection.json").read_text())
    assert not selection["outer_validation_metrics_used"]
    assert digest(protocol) == initial_selection["source_protocol_digest"]
    if include_added:
        design = json.loads((ADDED_OUTPUT/"design.json").read_text())
        assert not design["selection_uses_outer_validation_metrics"]
        assert digest(design) == selection["added_design_digest"] and digest(protocol) == selection["initial_protocol_digest"]
        assert all(sha_file(ROOT/path) == expected for path,expected in design["source_fingerprints"].items())
        if "context_spline_design_digest" in selection:
            context_design = json.loads((ADDED_OUTPUT/"context_spline_design.json").read_text())
            assert not context_design["selection_uses_outer_validation_metrics"]
            assert digest(context_design) == selection["context_spline_design_digest"]
            assert all(sha_file(ROOT/path) == expected for path,expected in context_design["source_fingerprints"].items())
    assert protocol["model_settings"] == json.loads((ROOT/"reports/history_study/model_settings.json").read_text())
    assert all(sha_file(ROOT/path) == expected for path, expected in protocol["source_fingerprints"].items())
    fingerprints = json.loads((OUTPUT/"input_fingerprints.json").read_text())
    assert all(sha_file(ROOT/path) == expected for path, expected in fingerprints.items())
    study = settings()
    train = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    validation = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    patients, folds = patient_folds(train, study["study"]["folds"], study["study"]["fold_seed"])
    pd.testing.assert_frame_equal(patients,pd.read_csv(OUTPUT/"patient_cv_folds.csv"))
    np.testing.assert_array_equal(folds,np.load(PROCESSED/"row_folds.npy"))
    assert not set(train.patient_id) & set(validation.patient_id)
    cv_jobs = json.loads((OUTPUT/"cv_manifest.json").read_text())
    validation_jobs = json.loads((OUTPUT/"validation_manifest.json").read_text())
    initial_cv_jobs = cv_jobs.copy()
    final_validation_jobs = json.loads((target/"validation_manifest.json").read_text())
    if include_added:
        cv_jobs += json.loads((ADDED_OUTPUT/"cv_manifest.json").read_text())
        validation_jobs += final_validation_jobs
    final_keys = {(j["case"],j["model"]) for j in final_validation_jobs}
    snapshot.mkdir(parents=True,exist_ok=True)
    unique = {(j["case"],j["model"]):j for j in [*cv_jobs,*validation_jobs]}
    checked_cases, scores, model_count = set(), [], 0
    for (case,model_name), job in unique.items():
        directory = PROCESSED/"cases"/case
        meta = json.loads((directory/"case.json").read_text())
        assert meta["stage"] == job["stage"] and meta["representation"] == job["representation"]
        train_indices, evaluation_indices = np.load(directory/"train_indices.npy"),np.load(directory/"evaluation_indices.npy")
        rows = pd.read_csv(directory/"train_rows.csv.gz")
        evaluation_rows = pd.read_csv(directory/"evaluation_rows.csv.gz")
        pd.testing.assert_frame_equal(rows,train.iloc[train_indices].reset_index(drop=True))
        if job["stage"] == "cv":
            assert (folds[train_indices] != job["fold"]).all() and (folds[evaluation_indices] == job["fold"]).all()
            assert len(train_indices)+len(evaluation_indices) == len(train)
            expected_eval, raw_eval_split = train, "train"
        else:
            assert set(rows.patient_id) == set(train.patient_id)
            expected_eval, raw_eval_split = validation, "validation"
        pd.testing.assert_frame_equal(evaluation_rows,expected_eval.iloc[evaluation_indices].reset_index(drop=True))
        assert not set(rows.patient_id) & set(evaluation_rows.patient_id)
        np.testing.assert_array_equal(np.load(directory/"train_y.npy"),rows.label)
        np.testing.assert_array_equal(np.load(directory/"evaluation_y.npy"),evaluation_rows.label)
        out = OUTPUT/job["stage"]/case
        bundle = joblib.load(out/f"{model_name}_model.joblib")
        assert bundle["case"] == meta
        # The fitted bundle is authoritative; refresh the redundant standalone cache
        # without reading a possibly evicted macOS placeholder.
        processor = bundle["preprocessor"]
        if case not in checked_cases:
            temporary = directory/"preprocessor.refreshed.joblib"
            joblib.dump(processor,temporary,compress=3)
            os.replace(temporary,directory/"preprocessor.joblib")
        sample,held_sample,x_probe,evaluation_probe = verification_probes(directory)
        if case not in checked_cases:
            assert processor.scaler_.n_samples_seen_ == len(rows)
            raw,names = case_raw_features("train",job["representation"])
            assert names == processor.names
            np.testing.assert_allclose(x_probe,processor.transform(raw[train_indices[sample]]),equal_nan=True,atol=0,rtol=0)
            verify_statistics(processor,raw,train_indices)
            del raw
            raw,_ = case_raw_features(raw_eval_split,job["representation"])
            np.testing.assert_allclose(evaluation_probe,processor.transform(raw[evaluation_indices[held_sample]]),equal_nan=True,atol=0,rtol=0)
            del raw
            checked_cases.add(case)
        result = json.loads((out/f"{model_name}_results.json").read_text())
        assert result["parameters"] == protocol["model_settings"]["models"][model_name]
        assert result["test_sets_evaluated"] == [] and result["case"] == meta
        if job["stage"] == "cv":
            assert selection["cv_result_signatures"][str((out/f"{model_name}_results.json").relative_to(ROOT))] == result["signature"]
        probability = np.load(out/f"{model_name}_probability.npy")
        assert np.isfinite(probability).all() and ((probability >= 0)&(probability <= 1)).all()
        for name,value in ranking_metrics(evaluation_rows.label.to_numpy(),probability).items():
            np.testing.assert_allclose(value,result["evaluation"][name],atol=1e-12,rtol=0)
        assert bundle["parameters"] == result["parameters"] and bundle["preprocessor"].names_ == processor.names_
        model = bundle["model"]
        predicted = model.booster_.predict(evaluation_probe) if model_name == "lightgbm" else model.predict_proba(evaluation_probe)[:,1]
        np.testing.assert_allclose(predicted,probability[held_sample],atol=1e-12,rtol=0)
        if model_name != "lr" and not result["reused_baseline"]:
            _,expected = patient_bags(rows,n_bags=3 if model_name == "ebm" else 1,seed=2026)
            actual = pd.read_csv(out/f"{model_name}_internal_patients.csv")
            pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
        if (case,model_name) in final_keys:
            if model_name == "lr":
                importance = pd.DataFrame({"feature":processor.names_,"importance":np.abs(model.coef_[0]),"coefficient":model.coef_[0],"metric":"absolute standardized coefficient"})
            elif model_name == "ebm":
                importance = pd.DataFrame({"feature":model.term_names_,"importance":model.term_importances(),
                                           "missing_bin_logit": [float(term[0]) for term in model.term_scores_],"metric":"mean absolute contribution"})
            else:
                importance = pd.DataFrame({"feature":processor.names_,"importance":model.booster_.feature_importance("gain"),"metric":"split gain"})
            importance.sort_values("importance",ascending=False).to_csv(target/f"{model_name}_feature_importance.csv",index=False)
            importance.sort_values("importance",ascending=False).to_csv(snapshot/f"{model_name}_feature_importance.csv",index=False)
        model_count += 1
        if model_count%12 == 0:
            print(f"Verified {model_count}/{len(unique)} saved models",flush=True)
    for job in cv_jobs:
        result = json.loads((OUTPUT/"cv"/job["case"]/f"{job['model']}_results.json").read_text())
        scores.append({"model":job["model"],"profile":job["profile"],"fold":job["fold"],"auprc":result["evaluation"]["auprc_average_precision"]})
    chosen,comparisons = choose_profiles(scores,study["study"])
    assert chosen == selection["selected_profiles"] == completion["selected_profiles"]
    assert comparisons == selection["cv_comparisons"]
    initial_profiles = {j["profile"] for j in initial_cv_jobs}
    initial_scores = [r for r in scores if r["profile"] in initial_profiles]
    initial_chosen,initial_comparisons = choose_profiles(initial_scores,study["study"])
    assert initial_chosen == initial_selection["selected_profiles"] and initial_comparisons == initial_selection["cv_comparisons"]
    # Verify new raw features on real event and non-event patient prefixes.
    audit = pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    raw = np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    combined = np.load(PROCESSED/"train_combined_features.npy",mmap_mode="r")
    if include_added:
        added = np.load(ADDED_DATA/"train_features.npy",mmap_mode="r")
        original_features = np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r")
        assert added.shape == (len(train),459) and len(set(added_names())) == 459
        for start in range(0,len(train),10000):
            np.testing.assert_allclose(added[start:start+10000,:318],original_features[start:start+10000],equal_nan=True,atol=0,rtol=0)
            np.testing.assert_allclose(added[start:start+10000,318:],combined[start:start+10000,214:],equal_nan=True,atol=0,rtol=0)
    for event in [0,1]:
        identifier = patients.loc[patients.has_event == event,"patient_id"].iloc[0]
        patient = audit.loc[identifier]
        values = np.array(raw[int(patient.raw_start):int(patient.raw_stop)],copy=True)
        whole = alternative_features(values,config(),study["features"])
        prefix = min(12,len(values))
        np.testing.assert_allclose(whole[:prefix],alternative_features(values[:prefix],config(),study["features"]),equal_nan=True,atol=0,rtol=0)
        _,eligible,_ = assess_patient(values,config())
        np.testing.assert_allclose(combined[train.patient_id.eq(identifier)],whole[eligible],equal_nan=True,atol=0,rtol=0)
    # Full baseline transform equivalence is independent of the winning profile.
    base_raw = np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r")
    original = joblib.load(ROOT/"outputs/preprocessor.joblib")
    current = FeaturePreprocessor(FEATURE_NAMES).fit(base_raw)
    for name in ["keep_","fill_","lower_","upper_"]:
        np.testing.assert_array_equal(getattr(original,name),getattr(current,name))
    for name in ["mean_","scale_"]:
        np.testing.assert_array_equal(getattr(original.scaler_,name),getattr(current.scaler_,name))
    distinct_cv_count = len({(j["case"],j["model"]) for j in cv_jobs})
    cleanup = []
    if "--purge-cv-matrices" in sys.argv:
        cleanup = purge_cv_matrices([j["case"] for j in cv_jobs])
    checks = {"cv_comparisons":len(cv_jobs),"distinct_cv_models":distinct_cv_count,
              "final_selected_validation_models":len(final_validation_jobs),"distinct_validation_models_verified":len(unique)-distinct_cv_count,
              "saved_model_predictions_verified":model_count,
              "preprocessing_cases_checked":len(checked_cases),"canonical_inputs_unchanged":True,
              "standalone_preprocessor_caches_refreshed_from_fitted_bundles":True,
              "patient_isolation_and_exact_row_mappings":True,"fold_only_statistics_recomputed_on_probe_features":True,
              "sampled_processed_matrices_reconstructed":True,"internal_stopping_patients_reconstructed":True,
              "original_full_baseline_preprocessor_reproduced_exactly":True,"real_patient_prefix_causality_and_feature_cache_checked":True,
              "all_metrics_recomputed":True,"fixed_classifier_settings_verified":True,"selection_reproduced_without_outer_validation":True,
              "information_preserving_addition_verified":include_added,
              "supplement_designed_before_initial_validation":not design["initial_outer_validation_manifest_present_at_design"] if include_added else None,
              "cv_cache_bytes_purged_after_verification":sum(item["bytes"] for item in cleanup),
              "processed_verification_probes_retained":True,
              "test_sets_evaluated":[],"verification_sources":{str(path.relative_to(ROOT)):sha_file(path) for path in
                [Path(__file__),ROOT/"scripts/report_feature_optimization.py",ROOT/"tests/test_feature_optimization.py",ROOT/"src/sepsis_warning/feature_cache.py"]}}
    save_json(target/"final_checks.json",checks)
    save_json(snapshot/"final_checks.json",checks)
    print(json.dumps(checks,indent=2),flush=True)


if __name__ == "__main__":
    if sys.platform == "darwin" and "--openmp-ready" not in sys.argv:
        raise SystemExit(subprocess.call([sys.executable,__file__,*sys.argv[1:],"--openmp-ready"],env=child_environment()))
    main()
