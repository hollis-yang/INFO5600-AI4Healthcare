"""Clock-feature ablation and matched single-model comparison on fixed patients."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sepsis_warning.data import ROOT
from sepsis_warning import friend_563 as baseline
from sepsis_warning import challenge_improvement as previous
from sepsis_warning.ab_expansion import content_hash, write_json

DATA = ROOT / "processed_data/clock_models"
OUT = ROOT / "outputs/clock_models"
REPORT = ROOT / "reports/clock_models"
VIEWS = ("baseline_789", "raw_clocks_only_736", "explicit_clock_free_734")


def available(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Cloud-evicted input needs identical restoration: {path}")
    return path


def load_protocol():
    return json.loads(available(OUT / "protocol.json").read_text())


def candidate_by_id(identifier):
    return load_protocol()["candidates"][identifier]


def cv_roles(fold):
    return previous.cv_roles(fold)


def row_indices(patients):
    return previous.row_indices(patients)


def local_patients(patients):
    return previous.local_patients(patients)


def load_features(split, feature_view, indices=None):
    """Build one view from existing caches, avoiding a duplicate 789-column array."""
    protocol = load_protocol()
    columns = np.asarray(protocol["feature_views"][feature_view]["kept_column_indices"], dtype=int)
    assert np.all(np.diff(columns) > 0)
    base = np.load(available(baseline.DATA / f"{split}_features.npy"), mmap_mode="r")
    extra = np.load(available(previous.DATA / f"{split}_extra.npy"), mmap_mode="r")
    n = len(base) if indices is None else len(indices)
    result = np.empty((n, len(columns)), dtype=np.float32)
    base_columns = columns[columns < 563]
    extra_columns = columns[columns >= 563]-563
    for start in range(0, n, 10000):
        stop = min(start+10000, n)
        selected = slice(start, stop) if indices is None else indices[start:stop]
        result[start:stop, :len(base_columns)] = base[selected][:, base_columns]
        result[start:stop, len(base_columns):] = extra[selected][:, extra_columns]
    return result


def resolve_directory(identifier, fold):
    manifest = json.loads(available(OUT / "reuse_manifest.json").read_text())["models"]
    if identifier in manifest:
        record = manifest[identifier]["final"] if fold == -1 else manifest[identifier]["cv"][str(fold)]
        return ROOT / record["directory"]
    return OUT / "final" / identifier if fold == -1 else OUT / "cv" / f"fold{fold}" / identifier


def prepare():
    for directory in (DATA, OUT, REPORT):
        directory.mkdir(parents=True, exist_ok=True)
    old_protocol = previous.load_protocol()
    plan = json.loads(available(ROOT / "configs/us_next_experiments.json").read_text())
    names = old_protocol["feature_names"]["lag_observed"]
    assert len(names) == 789
    feature_views = plan["feature_views"]
    assert {key: value["feature_count"] for key, value in feature_views.items()} == dict(zip(VIEWS, (789, 736, 734)))
    for key, view in feature_views.items():
        selected = view["kept_column_indices"]
        assert len(selected) == len(set(selected)) == view["feature_count"]
        assert [n for i,n in enumerate(names) if i not in selected] == view["dropped_features"]
    assert set(feature_views[VIEWS[1]]["dropped_features"]) == set(feature_views[VIEWS[2]]["dropped_features"])-{"ICULOS", "HospAdmTime"}
    old_lgb = previous.candidate_by_id("lgb_bce_lag_observed_leaf63")
    old_xgb = previous.candidate_by_id("xgb_lag_observed_depth5")
    candidates = {}
    for identifier, view in zip(("lgb_789", "lgb_736", "lgb_734"), VIEWS):
        candidates[identifier] = {"id": identifier, "model": "lightgbm", "target": "binary", "feature_view": view,
                                  "parameters": old_lgb["parameters"], "selection_metric": "auroc", "max_rounds": 1000}
    candidates["xgb_789"] = {"id": "xgb_789", "model": "xgboost", "target": "binary", "feature_view": VIEWS[0],
                              "parameters": old_xgb["parameters"], "selection_metric": "auroc", "max_rounds": 1000}
    candidates["lr_789"] = {"id": "lr_789", "model": "lr", "target": "binary", "feature_view": VIEWS[0],
        "parameters": {"C": 1., "penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "tol": 1e-4,
                       "random_state": 42, "class_weight": None, "n_jobs": 4}, "selection_metric": "auroc"}
    ebm = {"interactions": 0, "max_bins": 16, "min_samples_leaf": 400, "reg_lambda": 50.,
           "max_leaves": 2, "learning_rate": .03, "greedy_ratio": 0., "smoothing_rounds": 25,
           "validation_size": 0, "outer_bags": 1, "inner_bags": 0, "early_stopping_rounds": 0,
           "random_state": 42, "n_jobs": 4}
    for rounds in (150, 300):
        identifier = f"ebm{rounds}_789"
        candidates[identifier] = {"id": identifier, "model": "ebm", "target": "binary", "feature_view": VIEWS[0],
                                  "parameters": {**ebm, "max_rounds": rounds}, "selection_metric": "auroc"}
    protocol = {"standard_id": "SEPSIS_FULL_RECORD_V2", "ensemble_enabled": False,
        "feature_views": feature_views, "feature_names": {key: [names[i] for i in value["kept_column_indices"]] for key,value in feature_views.items()},
        "candidates": candidates, "maximum_rounds": 1000, "auc_early_stopping_rounds": 50, "min_delta": 1e-5,
        "split_policy": "Unchanged A+B hospital/event patient70/15/15; unchanged three training folds and inner15% calibration roles",
        "roles_sha256": content_hash(previous.DATA / "cv_patient_roles.csv"),
        "patient_splits_sha256": content_hash(baseline.DATA / "patient_splits.csv"),
        "primary_feature_view": VIEWS[0], "feature_selection": "No feature selection on test or ablation results: primary model comparison prespecified common789 columns.736/734 answer the independent clock ablation question.",
        "primary_models": ["lr", "ebm", "lightgbm"], "supplementary_models": ["xgboost"],
        "ebm_selection": "Choose150/300 fixed budgets by mean outer-fold original-label AUROC; these are development estimates, not nested unbiased scores.",
        "preprocessing": {"lr": "Fit-only per-column median(all-missing0), then fit-only standardization; no clipping or resampling",
                          "ebm": "Fit-only binning; nativeNaN; additive interactions0; no random-hour validation, no calibration rows in binning",
                          "trees": "NativeNaN, unchanged model parameters; only column views differ for clock ablations"},
        "thresholds": "Each CV model calibrates exact globalUS threshold on inner calibration only, then evaluates held patients. Final models use originalvalidation threshold; no cooldown.",
        "rounds": "LGB/XGB median inner-selected CV rounds; EBM fixed candidate budgets; LR convergence cap2000",
        "scope": "Current A+B full-record benchmark, including available post-onset hours; distinct from proposal pre-onset A-only/B transfer task",
        "test_history": "Historically exposed exploratory benchmark; no new candidates or commonfeature choices using test outcomes",
        "reuse": "Verified original789 LGB/XGB CV and final artifacts, preserving their historical signature and provenance",
        "ablation_reference": "lgb_789", "plan_sha256": content_hash(ROOT / "configs/us_next_experiments.json")}
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "Protocol changed; create a new study directory"
    else:
        write_json(path, protocol)
    manifest = {"models": {}}
    for identifier, historical in (("lgb_789", old_lgb["id"]), ("xgb_789", old_xgb["id"])):
        model_suffix = "model.txt" if identifier.startswith("lgb") else "model.ubj"
        entry = {"historical_candidate": historical, "cv": {}}
        for fold in (0,1,2,-1):
            directory = previous.OUT / "final" / historical if fold == -1 else previous.OUT / "cv" / f"fold{fold}" / historical
            files = ([model_suffix, "results.json", "validation_score.npy", "threshold.json"] if fold == -1 else
                     [model_suffix, "results.json", "cal_score.npy", "cal_patients.csv", "held_score.npy", "held_patients.csv"])
            record = {"directory": str(directory.relative_to(ROOT)),
                      "files_sha256": {name: content_hash(available(directory/name)) for name in files}}
            if fold == -1:
                entry["final"] = record
                target = OUT / "final" / identifier
            else:
                entry["cv"][str(fold)] = record
                target = OUT / "cv" / f"fold{fold}" / identifier
            target.mkdir(parents=True, exist_ok=True)
            write_json(target / "reuse.json", {"historical_candidate": historical, **record})
        manifest["models"][identifier] = entry
    path = OUT / "reuse_manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest
    else:
        write_json(path, manifest)
    write_json(REPORT / "protocol.json", protocol)
    write_json(REPORT / "reuse_manifest.json", manifest)
    pd.read_csv(previous.DATA / "cv_patient_roles.csv").to_csv(REPORT / "cv_patient_roles.csv", index=False)
    return protocol
