"""Run predeclared, training-only stages before a frozen exploratory test replay."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from sepsis_warning import challenge_improvement as study
from sepsis_warning import friend_563 as baseline
from sepsis_warning.ab_expansion import content_hash, write_json


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen selection changed: {path}"
    else:
        write_json(path, value)


def jobs(items):
    pending, active = list(items), []
    logs = study.OUT / "logs"
    logs.mkdir(exist_ok=True)
    last = time.monotonic()
    while pending or active:
        while pending and len(active) < 2:
            identifier, fold = pending.pop(0)
            path = logs / f"{identifier}_fold{fold}.log"
            stream = path.open("w")
            environment = baseline.runtime_environment()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen([sys.executable, "-B", "-u", str(ROOT / "scripts/fit_challenge_improvement.py"),
                                         "--candidate", identifier, "--fold", str(fold)],
                                        cwd=ROOT, env=environment,
                                        stdout=stream, stderr=subprocess.STDOUT)
            active.append((process, stream, identifier, fold, path))
            print(f"Started {identifier}, fold={fold}", flush=True)
        for item in list(active):
            process, stream, identifier, fold, path = item
            status = process.poll()
            if status is None:
                continue
            stream.close()
            active.remove(item)
            if status:
                for other, handle, *_ in active:
                    other.terminate()
                    handle.close()
                print(path.read_text()[-10000:], flush=True)
                raise RuntimeError(f"Failed {identifier}/{fold}: {status}")
            print(path.read_text()[-1200:], flush=True)
        if time.monotonic()-last > 35:
            print("Active: " + ", ".join(f"{identifier}/fold{fold}" for _,_,identifier,fold,_ in active)
                  + f"; queued={len(pending)}", flush=True)
            last = time.monotonic()
        if pending or active:
            time.sleep(.5)


def summarize():
    patients = pd.read_csv(baseline.DATA / "train_patients.csv")
    labels = np.load(baseline.DATA / "train_labels.npy")
    delta = np.load(study.DATA / "train_delta.npy")
    scaled = np.rint(delta * 180).astype(np.int64)
    destination = study.OUT / "cv_predictions"
    destination.mkdir(exist_ok=True)
    summaries, fold_records = [], []
    candidates = json.loads((study.OUT / "candidates.json").read_text())
    for candidate in candidates:
        identifier = candidate["id"]
        directories = [study.OUT / "cv" / f"fold{fold}" / identifier for fold in range(3)]
        if not all((directory / "results.json").exists() for directory in directories):
            continue
        scores = np.full(len(labels), np.nan)
        alarms = np.zeros(len(labels), np.int8)
        rounds, aucs, aps = [], [], []
        for fold, directory in enumerate(directories):
            result = json.loads((directory / "results.json").read_text())
            held = pd.read_csv(directory / "held_patients.csv")
            original_held = patients.loc[patients.cv_fold.eq(fold)]
            np.testing.assert_array_equal(held.patient_id, original_held.patient_id)
            index = study.row_indices(original_held)
            prediction = np.load(directory / "held_score.npy")
            assert len(prediction) == len(index) and np.isnan(scores[index]).all()
            scores[index] = prediction
            alarms[index] = prediction >= result["calibration"]["threshold"]
            rounds.append(result["selected_rounds"])
            aucs.append(result["evaluation"]["auroc"])
            aps.append(result["evaluation"]["auprc"])
            fold_records.append({"candidate": identifier, "fold": fold, "model": candidate["model"],
                                 "target": candidate["target"], "feature_set": candidate["feature_set"],
                                 "selected_rounds": rounds[-1], "fit_seconds": result["fit_seconds"],
                                 **result["evaluation"]})
        assert np.isfinite(scores).all()
        np.save(destination / f"{identifier}_score.npy", scores)
        np.save(destination / f"{identifier}_alarm.npy", alarms)
        summaries.append({"candidate": identifier, "model": candidate["model"], "target": candidate["target"],
                          "feature_set": candidate["feature_set"], "feature_count": len(study.load_protocol()["feature_names"][candidate["feature_set"]]),
                          **study.ranking_scores(labels, scores), "mean_fold_auroc": float(np.mean(aucs)),
                          "mean_fold_auprc": float(np.mean(aps)),
                          "utility_score": float(scaled[alarms.astype(bool)].sum()/np.maximum(scaled, 0).sum()),
                          "final_rounds": int(np.median(rounds))})
    frame = pd.DataFrame(summaries)
    frame.to_csv(study.REPORT / "cv_metrics.csv", index=False)
    pd.DataFrame(fold_records).to_csv(study.REPORT / "cv_fold_metrics.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    return frame


def leader(frame, metric, identifiers=None):
    if identifiers is not None:
        frame = frame.loc[frame.candidate.isin(identifiers)]
    assert len(frame)
    return frame.sort_values([metric, "candidate"], ascending=[False, True]).iloc[0].candidate


def clone(identifier, parent, feature_set=None, overrides=None):
    old = study.candidate_by_id(parent)
    return study.candidate(identifier, old["model"], old["target"], feature_set or old["feature_set"],
                           {**old["parameters"], **(overrides or {})})


def development():
    study.prepare()
    initial = [study.candidate("lgb_bce_base", "lightgbm"), study.candidate("xgb_bce_base", "xgboost"),
               study.candidate("lgb_delta_base", "lightgbm", "utility_regression"),
               study.candidate("lgb_utility_class_base", "lightgbm", "utility_classification")]
    study.save_candidates(initial)
    jobs([(entry["id"], fold) for fold in range(3) for entry in initial])
    frame = summarize()
    utility_parent = leader(frame, "utility_score", [entry["id"] for entry in initial if entry["model"] == "lightgbm"])
    freeze(study.OUT / "stage1_selection.json", {"utility_parent": utility_parent})
    # Each feature block has a paired original-label branch and a utility branch.
    parents = list(dict.fromkeys(["lgb_bce_base", utility_parent]))
    additions = [clone(f"{parent.removesuffix('_base')}_{profile}", parent, profile)
                 for parent in parents for profile in ("lag", "lag_observed", "all")]
    study.save_candidates(additions)
    jobs([(entry["id"], fold) for fold in range(3) for entry in additions])
    frame = summarize()
    lgb = frame.loc[frame.model.eq("lightgbm")]
    auc_parent = leader(lgb, "mean_fold_auroc")
    utility_parent = leader(lgb, "utility_score")
    binary_parent = leader(lgb.loc[lgb.target.eq("binary")], "mean_fold_auroc")
    freeze(study.OUT / "stage2_selection.json", {"auc_parent": auc_parent, "utility_parent": utility_parent,
                                                 "xgb_feature_parent": binary_parent})
    more = []
    lgb_overrides = {
        "leaf31": {"max_depth": 5, "num_leaves": 31, "min_data_in_leaf": 500, "lambda_l2": 10.,
                   "feature_fraction": .8, "bagging_fraction": .8, "bagging_freq": 1},
        "leaf63": {"max_depth": 6, "num_leaves": 63, "min_data_in_leaf": 1000, "lambda_l2": 20.,
                   "feature_fraction": .9, "bagging_fraction": .9, "bagging_freq": 1}}
    for parent in dict.fromkeys([auc_parent, utility_parent]):
        for suffix, override in lgb_overrides.items():
            more.append(clone(f"{parent}_{suffix}", parent, overrides=override))
    chosen_feature = study.candidate_by_id(binary_parent)["feature_set"]
    for suffix, override in {
        "depth3": {"max_depth": 3, "max_leaves": 8, "min_child_weight": 10., "lambda": 20.,
                   "subsample": .9, "colsample_bytree": .9},
        "depth5": {"max_depth": 5, "max_leaves": 32, "min_child_weight": 10., "lambda": 10.,
                   "subsample": .8, "colsample_bytree": .8}}.items():
        more.append(study.candidate(f"xgb_{chosen_feature}_{suffix}", "xgboost", "binary", chosen_feature, override))
    study.save_candidates(more)
    jobs([(entry["id"], fold) for fold in range(3) for entry in more])
    frame = summarize()
    write_json(study.OUT / "rounds.json", {row.candidate: int(row.final_rounds) for row in frame.itertuples(index=False)})
    selection = {"auc_leader": leader(frame, "mean_fold_auroc"), "utility_leader": leader(frame, "utility_score"),
                 "binary_lgb_leader": leader(frame.loc[frame.model.eq("lightgbm") & frame.target.eq("binary")], "mean_fold_auroc"),
                 "binary_xgb_leader": leader(frame.loc[frame.model.eq("xgboost")], "mean_fold_auroc"),
                 "selection_data": "Training patients only; no validation/test candidate selection",
                 "candidate_count": len(frame)}
    freeze(study.OUT / "model_selection.json", selection)
    print("FROZEN TRAINING SELECTION: " + json.dumps(selection), flush=True)


def rank_transform(reference, values):
    reference = np.sort(reference)
    return np.searchsorted(reference, values, side="right") / len(reference)


def blend_scores(definition, component_scores, reference_scores=None):
    identifiers = definition["components"]
    values = [component_scores[name] for name in identifiers]
    if definition["method"] == "percentile_rank":
        values = [rank_transform(reference_scores[name], component_scores[name]) for name in identifiers]
    return definition["weight_first"] * values[0] + (1-definition["weight_first"]) * values[1]


def ensemble_development(selection):
    from fit_challenge_improvement import exact_utility_threshold, utility_at_threshold
    patients = pd.read_csv(baseline.DATA / "train_patients.csv")
    labels = np.load(baseline.DATA / "train_labels.npy")
    delta = np.load(study.DATA / "train_delta.npy")
    scaled = np.rint(delta * 180).astype(np.int64)
    definitions = [{"id": "blend_binary_half", "method": "probability_mean",
                    "components": [selection["binary_lgb_leader"], selection["binary_xgb_leader"]],
                    "weight_first": .5}]
    if selection["auc_leader"] != selection["utility_leader"]:
        for weight in (.25, .5, .75):
            definitions.append({"id": f"blend_rank_w{int(weight*100)}", "method": "percentile_rank",
                                "components": [selection["auc_leader"], selection["utility_leader"]],
                                "weight_first": weight})
    freeze(study.OUT / "ensemble_definitions.json", definitions)
    summaries, fold_rows = [], []
    for definition in definitions:
        scores = np.full(len(labels), np.nan)
        alarms = np.zeros(len(labels), np.int8)
        aucs, aps = [], []
        for fold in range(3):
            _, cal, held = study.cv_roles(fold)
            cal_index, held_index = study.row_indices(cal), study.row_indices(held)
            component_cal, component_held = {}, {}
            for identifier in definition["components"]:
                directory = study.OUT / "cv" / f"fold{fold}" / identifier
                np.testing.assert_array_equal(pd.read_csv(directory / "cal_patients.csv").patient_id, cal.patient_id)
                component_cal[identifier] = np.load(directory / "cal_score.npy")
                component_held[identifier] = np.load(directory / "held_score.npy")
            cal_blend = blend_scores(definition, component_cal, component_cal)
            held_blend = blend_scores(definition, component_held, component_cal)
            calibration = exact_utility_threshold(cal_blend, delta[cal_index])
            ranking = study.ranking_scores(labels[held_index], held_blend)
            utility = utility_at_threshold(held_blend, delta[held_index], calibration["threshold"])
            scores[held_index] = held_blend
            alarms[held_index] = held_blend >= calibration["threshold"]
            aucs.append(ranking["auroc"])
            aps.append(ranking["auprc"])
            out = study.OUT / "ensembles/cv" / f"fold{fold}" / definition["id"]
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / "cal_score.npy", cal_blend)
            np.save(out / "held_score.npy", held_blend)
            write_json(out / "results.json", {"definition": definition, "fold": fold, "calibration": calibration,
                                               "evaluation": {**ranking, "utility_score": utility},
                                               "rank_reference": "Same fold's inner calibration predictions only",
                                               "outer_holdout_used_for_threshold": False})
            fold_rows.append({"candidate": definition["id"], "fold": fold, **ranking, "utility_score": utility})
        assert np.isfinite(scores).all()
        np.save(study.OUT / "cv_predictions" / f"{definition['id']}_score.npy", scores)
        np.save(study.OUT / "cv_predictions" / f"{definition['id']}_alarm.npy", alarms)
        summaries.append({"candidate": definition["id"], **study.ranking_scores(labels, scores),
                          "mean_fold_auroc": float(np.mean(aucs)), "mean_fold_auprc": float(np.mean(aps)),
                          "utility_score": float(scaled[alarms.astype(bool)].sum()/np.maximum(scaled, 0).sum())})
    frame = pd.DataFrame(summaries)
    frame.to_csv(study.REPORT / "cv_ensemble_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(study.REPORT / "cv_ensemble_fold_metrics.csv", index=False)
    selected = {"auc_ensemble": leader(frame, "mean_fold_auroc"), "utility_ensemble": leader(frame, "utility_score")}
    freeze(study.OUT / "ensemble_selection.json", selected)
    print("Training-only ensemble results:\n" + frame.to_string(index=False), flush=True)
    return definitions, selected


def persist_evaluation(split, identifier, scores, threshold):
    from sepsis_warning.challenge_evaluation import evaluate_scores
    from sepsis_warning.utility import patient_utility
    labels = np.load(baseline.DATA / f"{split}_labels.npy")
    patients = pd.read_csv(baseline.DATA / f"{split}_patients.csv")
    result = evaluate_scores(labels, scores, patients, threshold)
    out = study.OUT / "evaluation" / split / identifier
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "score.npy", scores)
    contributions = [{"patient_id": p.patient_id,
                      **patient_utility(labels[int(p.start):int(p.stop)], scores[int(p.start):int(p.stop)] >= threshold)}
                     for p in patients.itertuples(index=False)]
    pd.DataFrame(contributions).to_csv(out / "patient_utilities.csv", index=False)
    write_json(out / "metrics.json", result)
    return {"split": split, "candidate": identifier, **{key: result[key] for key in
            ("auroc", "auprc", "utility_score", "threshold", "hour_precision", "hour_recall", "hour_fpr",
             "hour_f1", "hour_accuracy", "patients", "hours", "positive_hours")}}


def verify_input_hashes(splits):
    records = {}
    for split in splits:
        case = json.loads(study.available(baseline.DATA / f"{split}_case.json").read_text())
        for suffix, key in (("features.npy", "feature_sha256"), ("labels.npy", "labels_sha256"),
                            ("patients.csv", "patients_sha256")):
            path = study.available(baseline.DATA / f"{split}_{suffix}")
            actual = content_hash(path)
            assert actual == case[key], path
            records[str(path.relative_to(ROOT))] = actual
        marker = json.loads((study.DATA / f"{split}_extra_case.json").read_text())
        path = study.available(study.DATA / f"{split}_extra.npy")
        actual = content_hash(path)
        assert actual == marker["feature_sha256"]
        assert marker["source_sha256"] == content_hash(ROOT / "src/sepsis_warning/challenge_features.py")
        records[str(path.relative_to(ROOT))] = actual
    from sepsis_warning.challenge_utility import utility_targets
    patients = pd.read_csv(baseline.DATA / "train_patients.csv")
    labels = np.load(baseline.DATA / "train_labels.npy")
    np.testing.assert_array_equal(utility_targets(labels, patients)["delta"], np.load(study.DATA / "train_delta.npy"))
    records["processed_data/challenge_improvement/train_delta.npy"] = content_hash(study.DATA / "train_delta.npy")
    return records


def audit_additions():
    """Describe the training additions without fitting any transformation."""
    from sepsis_warning.challenge_features import feature_names
    matrix = np.load(study.DATA / "train_extra.npy", mmap_mode="r")
    counts = np.zeros(matrix.shape[1], np.int64)
    zeros = counts.copy()
    sums = np.zeros(matrix.shape[1], np.float64)
    squares = sums.copy()
    minimum = np.full(matrix.shape[1], np.inf)
    maximum = np.full(matrix.shape[1], -np.inf)
    for start in range(0, len(matrix), 10000):
        values = matrix[start:start+10000].astype(np.float64)
        observed = np.isfinite(values)
        clean = np.where(observed, values, 0)
        counts += observed.sum(axis=0)
        zeros += (observed & (values == 0)).sum(axis=0)
        sums += clean.sum(axis=0)
        squares += np.square(clean).sum(axis=0)
        minimum = np.minimum(minimum, np.where(observed, values, np.inf).min(axis=0))
        maximum = np.maximum(maximum, np.where(observed, values, -np.inf).max(axis=0))
    mean = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    variance = np.divide(squares, counts, out=np.full_like(sums, np.nan), where=counts > 0) - np.square(mean)
    minimum[counts == 0] = maximum[counts == 0] = np.nan
    pd.DataFrame({"feature": feature_names(), "training_missing_fraction": 1-counts/len(matrix),
                  "training_zero_fraction": zeros/len(matrix), "observed_mean": mean,
                  "observed_sd": np.sqrt(np.maximum(variance, 0)), "observed_min": minimum,
                  "observed_max": maximum}).to_csv(study.REPORT / "training_feature_audit.csv", index=False)
    delta = np.load(study.DATA / "train_delta.npy")
    labels = np.load(baseline.DATA / "train_labels.npy")
    write_json(study.REPORT / "training_target_audit.json", {
        "hours": len(labels), "original_positive_hours": int(labels.sum()),
        "positive_utility_hours": int((delta > 0).sum()), "zero_utility_hours": int((delta == 0).sum()),
        "negative_utility_hours": int((delta < 0).sum()), "minimum_delta": float(delta.min()),
        "maximum_delta": float(delta.max()), "training_only": True})


def finish():
    from fit_challenge_improvement import exact_utility_threshold, predict
    from sepsis_warning.challenge_utility import utility_targets
    from sepsis_warning.challenge_evaluation import bootstrap_differences
    audit_additions()
    selection = json.loads(study.available(study.OUT / "model_selection.json").read_text())
    definitions, ensemble_selection = ensemble_development(selection)
    final_ids = list(dict.fromkeys(["lgb_bce_base", "xgb_bce_base", selection["auc_leader"], selection["utility_leader"],
                                    selection["binary_lgb_leader"], selection["binary_xgb_leader"]]))
    ensemble_ids = list(dict.fromkeys(ensemble_selection.values()))
    chosen_definitions = [d for d in definitions if d["id"] in ensemble_ids]
    freeze(study.OUT / "final_selection.json", {"models": final_ids, "ensembles": chosen_definitions,
                                                "source": "All choices from training patient CV"})
    study.build_extra("validation")
    input_hashes = verify_input_hashes(("train", "validation"))
    jobs([(identifier, -1) for identifier in final_ids])
    validation_y = np.load(baseline.DATA / "validation_labels.npy")
    validation_patients = pd.read_csv(baseline.DATA / "validation_patients.csv")
    validation_delta = utility_targets(validation_y, validation_patients)["delta"]
    thresholds, validation_scores, model_hashes, threshold_hashes = {}, {}, {}, {}
    results = []
    for identifier in final_ids:
        directory = study.OUT / "final" / identifier
        candidate = study.candidate_by_id(identifier)
        scores = np.load(directory / "validation_score.npy")
        calibration = exact_utility_threshold(scores, validation_delta)
        write_json(directory / "threshold.json", calibration)
        thresholds[identifier] = calibration["threshold"]
        validation_scores[identifier] = scores
        model_hashes[identifier] = content_hash(directory / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt"))
        threshold_hashes[identifier] = content_hash(directory / "threshold.json")
        results.append(persist_evaluation("validation", identifier, scores, thresholds[identifier]))
    cdf_hashes = {}
    references = {}
    for definition in chosen_definitions:
        for identifier in definition["components"]:
            path = study.OUT / "cv_predictions" / f"{identifier}_score.npy"
            references[identifier] = np.load(path)
            cdf_hashes[identifier] = content_hash(path)
        identifier = definition["id"]
        scores = blend_scores(definition, validation_scores, references)
        calibration = exact_utility_threshold(scores, validation_delta)
        directory = study.OUT / "final" / identifier
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "threshold.json", calibration)
        thresholds[identifier] = calibration["threshold"]
        threshold_hashes[identifier] = content_hash(directory / "threshold.json")
        results.append(persist_evaluation("validation", identifier, scores, thresholds[identifier]))
    frozen = {"model_sha256": model_hashes, "threshold_sha256": threshold_hashes,
              "rank_reference_sha256": cdf_hashes, "rounds_sha256": content_hash(study.OUT / "rounds.json"),
              "selection_sha256": content_hash(study.OUT / "final_selection.json"),
              "model_selection_sha256": content_hash(study.OUT / "model_selection.json"),
              "ensemble_definitions_sha256": content_hash(study.OUT / "ensemble_definitions.json"),
              "input_sha256": input_hashes,
              "source_sha256": {name: content_hash(ROOT / name) for name in (
                  "src/sepsis_warning/challenge_improvement.py", "src/sepsis_warning/challenge_features.py",
                  "src/sepsis_warning/challenge_utility.py", "src/sepsis_warning/challenge_evaluation.py",
                  "scripts/fit_challenge_improvement.py", "scripts/run_challenge_improvement.py")},
              "final_candidate_ids": final_ids, "ensembles": chosen_definitions,
              "test_used_for_selection": False, "historical_test_exposure": True}
    freeze(study.OUT / "frozen_before_test.json", frozen)
    study.build_extra("test")
    write_json(study.REPORT / "test_input_sha256.json", verify_input_hashes(("test",)))
    test_scores = {}
    for identifier in final_ids:
        candidate = study.candidate_by_id(identifier)
        directory = study.OUT / "final" / identifier
        model = baseline.load_model(candidate["model"], directory)
        rounds = json.loads((directory / "results.json").read_text())["selected_rounds"]
        x = study.load_features("test", candidate["feature_set"])
        scores = predict(model, candidate["model"], x, rounds)
        test_scores[identifier] = scores
        results.append(persist_evaluation("test", identifier, scores, thresholds[identifier]))
        assert content_hash(directory / ("model.ubj" if candidate["model"] == "xgboost" else "model.txt")) == model_hashes[identifier]
        del model, x
        gc.collect()
        print("Frozen test: " + json.dumps(results[-1]), flush=True)
    for definition in chosen_definitions:
        identifier = definition["id"]
        scores = blend_scores(definition, test_scores, references)
        test_scores[identifier] = scores
        results.append(persist_evaluation("test", identifier, scores, thresholds[identifier]))
        print("Frozen test ensemble: " + json.dumps(results[-1]), flush=True)
    for identifier, sha in threshold_hashes.items():
        assert content_hash(study.OUT / "final" / identifier / "threshold.json") == sha
    frame = pd.DataFrame(results)
    frame.to_csv(study.REPORT / "metrics.csv", index=False)
    test_y = np.load(baseline.DATA / "test_labels.npy")
    test_patients = pd.read_csv(baseline.DATA / "test_patients.csv")
    intervals = {}
    reference = "lgb_bce_base"
    # Candidates are selected before test. Intervals retain their frozen thresholds.
    for identifier in list(dict.fromkeys([selection["auc_leader"], selection["utility_leader"], *ensemble_ids])):
        if identifier == reference:
            continue
        intervals[identifier] = bootstrap_differences(test_y, test_patients, test_scores[reference], test_scores[identifier],
                                                     thresholds[reference], thresholds[identifier], repetitions=500, seed=2026)
    write_json(study.REPORT / "paired_test_intervals.json", {"reference": reference, "comparisons": intervals})
    for name in ("protocol.json", "candidates.json", "model_selection.json", "ensemble_selection.json",
                 "ensemble_definitions.json", "final_selection.json", "rounds.json", "frozen_before_test.json"):
        write_json(study.REPORT / name, json.loads((study.OUT / name).read_text()))
    write_report(frame, selection, ensemble_selection, intervals)
    write_json(study.OUT / "completed.json", {"candidate_count": selection["candidate_count"],
                                                "final_models": len(final_ids), "evaluations": len(results),
                                                "test_used_for_selection": False})
    print(frame.to_string(index=False), flush=True)


def write_report(frame, selection, ensemble_selection, intervals):
    lines = ["# 竞赛方案启发的效用目标、时序特征与树模型优化", "",
             "在已冻结的朋友563维基础上开展逐项对照，沿用A+B各院患者70/15/15名单及医院×事件分层三折。"
             "每个外折训练池再留15%患者作内部早停和阈值校准；其余85%拟合，外折不参与选择轮数与阈值。"
             "跨候选选择仍使用这些外折开发成绩，因此最佳候选的CV不是完全嵌套的无偏泛化估计。", "",
             "## 方法与选择", "",
             "比较原标签BCE、报警边际效用ΔU回归、效用标签1[ΔU>0]及权重|ΔU|的二分类。"
             "加权分类权重仅按拟合患者子集的均值缩放；回归分数不是概率。全部AUROC/AP仍针对原始SepsisLabel。", "",
             "新增228维分成lag56维、真实观测历史170维、临床组合2维，对应619/789/791维逐步消融。"
             "所有新增输入只依赖当前及过去，不读标签；化验差值基于最近两次真实测量。"
             "临床组合为近期HR/SBP和SBP−DBP，不声称完整SOFA。大矩阵、模型留在本地processed_data/outputs。", "",
             "US目标每10轮在内部校准患者上计算精确最大效用，连续8次无改善早停；普通分类按内部AUROC早停50轮。"
             "最终轮数为三折内层所选轮数中位数，再用全部27,911位训练患者重训。"
             "验证集仅在所有模型与集成选择冻结后选择最终单阈值。", "",
             "本轮BCE基线的最终轮数重新由上述内层流程确定，与前一轮固定276/216轮的已发布模型可能不同；"
             "主要差异比较采用本轮同流程的BCE基线，不能把旧→新基线变化归因于新增特征或训练目标。", "",
             f"训练CV选择的AUROC模型：`{selection['auc_leader']}`；US模型：`{selection['utility_leader']}`。",
             f"训练CV选择的集成：AUROC `{ensemble_selection['auc_ensemble']}`；US `{ensemble_selection['utility_ensemble']}`。", "",
             "## 完整记录逐小时结果", "", "| 集合 | 候选 | AUROC | AUPRC/AP | US |",
             "|---|---|---:|---:|---:|"]
    for row in frame.itertuples(index=False):
        lines.append(f"| {row.split} | {row.candidate} | {row.auroc:.6f} | {row.auprc:.6f} | {row.utility_score:.6f} |")
    lines += ["", "验证集US直接用于选阈值，具有校准乐观性。测试集采用冻结阈值，无报警冷却。"
              "已有测试患者在历史工作中被查看过；这是两家已知医院内的探索性固定留出比较，不是首次独立测试或外院验证。"
              "完整记录AUROC包含发病后识别；发病前六小时检测率单独保存在每个metrics.json中。", "",
              "## 相对本轮563维LightGBM BCE基线的测试差异", "",
              "| 训练CV选定候选 | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔUS [95% CI] |",
              "|---|---|---|---|"]
    for identifier, record in intervals.items():
        fields = [record["difference"][key] for key in ("auroc", "auprc", "utility_score")]
        text = [f"{value['point']:+.6f} [{value['lower']:+.6f}, {value['upper']:+.6f}]" for value in fields]
        lines.append(f"| {identifier} | " + " | ".join(text) + " |")
    lines += ["", "500次患者配对bootstrap保留模型和阈值；区间反映评价样本变化，不包括训练及候选选择变化。"
              "若区间跨0，不能据此宣称稳定改进。分医院成绩、误报/召回、效用时段分解见outputs/.../evaluation。", "",
              "## 复现", "", "```bash", ".venv/bin/python -u scripts/run_challenge_improvement.py --stage all",
              ".venv/bin/python scripts/verify_challenge_improvement.py --final", "```", "",
              "脚本用原有隔离原生运行时加载XGBoost/LightGBM。详情见protocol.json、candidates.json和cv_fold_metrics.csv。"
              "本轮不重分患者，不引入外部预训练模型。Signature变换留待本轮目标、特征、容量与集成消融完成后再评估。", "",
              "参考：[冠军效用回归与signature](https://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf)、"
              "[SBU效用加权](https://www.cinc.org/archives/2019/pdf/CinC2019-459.pdf)、"
              "[Sepsyd历史快照](https://www.cinc.org/archives/2019/pdf/CinC2019-423.pdf)、"
              "[SailOcean集成](https://www.cinc.org/archives/2019/pdf/CinC2019-020.pdf)。", ""]
    (study.REPORT / "report.md").write_text("\n".join(lines))


def main():
    # The coordinator also reloads XGBoost for final inference. Give it the
    # same pinned package/OpenMP environment as its fitting subprocesses.
    if os.environ.get("INFO5600_CHALLENGE_RUNTIME_READY") != "1":
        environment = baseline.runtime_environment()
        environment["INFO5600_CHALLENGE_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", "-u", __file__, *sys.argv[1:]], environment)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["development", "finish", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("development", "all"):
        development()
    if args.stage in ("finish", "all"):
        finish()


if __name__ == "__main__":
    main()
