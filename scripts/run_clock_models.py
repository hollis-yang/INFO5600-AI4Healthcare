"""Run frozen-patient clock ablations and a matched interpretable-model study."""
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
from sepsis_warning import clock_models as study
from sepsis_warning import friend_563 as baseline
from sepsis_warning import challenge_improvement as previous
from sepsis_warning.ab_expansion import content_hash, write_json


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        write_json(path, value)


def jobs(items, concurrency=2):
    pending, active = list(items), []
    logs = study.OUT / "logs"
    logs.mkdir(exist_ok=True)
    last = time.monotonic()
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                identifier, fold = pending.pop(0)
                model = study.candidate_by_id(identifier)["model"]
                script = "fit_interpretable_clock.py" if model in ("lr", "ebm") else "fit_clock_models.py"
                path = logs / f"{identifier}_fold{fold}.log"
                stream = path.open("w")
                environment = baseline.runtime_environment()
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                if model == "lr":
                    # A binary lbfgs model has one class task. Avoid a loky
                    # float64 memmap copy; preserve the estimator's parameters.
                    environment["JOBLIB_MULTIPROCESSING"] = "0"
                    environment["VECLIB_MAXIMUM_THREADS"] = "4"
                    destination = study.OUT / "final" / identifier if fold == -1 else study.OUT / "cv" / f"fold{fold}" / identifier
                    if not (destination / "results.json").exists():
                        write_json(destination / "execution_environment.json", {
                            "JOBLIB_MULTIPROCESSING": "0", "VECLIB_MAXIMUM_THREADS": "4",
                            "reason": "One binary class task; sequential joblib backend avoids process data copies",
                            "estimator_parameters_changed": False,
                            "historical_cv_retrained_for_runtime_change": False})
                process = subprocess.Popen([sys.executable, "-B", "-u", str(ROOT / "scripts" / script),
                                             "--candidate", identifier, "--fold", str(fold)],
                                            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                active.append((process, stream, identifier, fold, path))
                print(f"Started {identifier}, fold={fold}", flush=True)
            for item in list(active):
                process, stream, identifier, fold, path = item
                if process.poll() is None:
                    continue
                stream.close()
                active.remove(item)
                print(path.read_text()[-1500:], flush=True)
                if process.returncode:
                    raise RuntimeError(f"Fit failed {identifier}/{fold}: {process.returncode}")
            if time.monotonic()-last > 35:
                print("Active: " + ", ".join(f"{name}/fold{fold}" for _,_,name,fold,_ in active)
                      + f"; queued={len(pending)}", flush=True)
                last = time.monotonic()
            if pending or active:
                time.sleep(.5)
    finally:
        for process, stream, *_ in active:
            if process.poll() is None:
                process.terminate()
        for process, stream, *_ in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()


def summarize(identifiers):
    people = pd.read_csv(baseline.DATA / "train_patients.csv")
    labels = np.load(baseline.DATA / "train_labels.npy")
    delta = np.rint(np.load(previous.DATA / "train_delta.npy")*180).astype(np.int64)
    destination = study.OUT / "cv_predictions"
    destination.mkdir(exist_ok=True)
    rows, folds = [], []
    for identifier in identifiers:
        candidate = study.candidate_by_id(identifier)
        directories = [study.resolve_directory(identifier, fold) for fold in range(3)]
        if not all((directory / "results.json").exists() for directory in directories):
            continue
        scores = np.full(len(labels), np.nan)
        alarm = np.zeros(len(labels), np.int8)
        aucs, aps, rounds = [], [], []
        for fold, directory in enumerate(directories):
            result = json.loads((directory / "results.json").read_text())
            original_held = people.loc[people.cv_fold.eq(fold)]
            held = pd.read_csv(directory / "held_patients.csv")
            np.testing.assert_array_equal(held.patient_id, original_held.patient_id)
            index = study.row_indices(original_held)
            p = np.load(directory / "held_score.npy")
            assert len(p) == len(index) and np.isnan(scores[index]).all()
            scores[index] = p
            alarm[index] = p >= result["calibration"]["threshold"]
            ranking = previous.ranking_scores(labels[index], p)
            aucs.append(ranking["auroc"]); aps.append(ranking["auprc"])
            rounds.append(int(result["selected_rounds"]))
            folds.append({"candidate": identifier, "model": candidate["model"], "feature_view": candidate["feature_view"],
                          "fold": fold, **ranking, "utility_score": float(delta[index][alarm[index].astype(bool)].sum()/np.maximum(delta[index],0).sum()),
                          "selected_rounds": rounds[-1], "fit_seconds": result["fit_seconds"]})
        assert np.isfinite(scores).all()
        np.save(destination / f"{identifier}_score.npy", scores)
        np.save(destination / f"{identifier}_alarm.npy", alarm)
        rows.append({"candidate": identifier, "model": candidate["model"], "feature_view": candidate["feature_view"],
                     "feature_count": len(study.load_protocol()["feature_names"][candidate["feature_view"]]),
                     **previous.ranking_scores(labels, scores), "mean_fold_auroc": float(np.mean(aucs)),
                     "mean_fold_auprc": float(np.mean(aps)),
                     "utility_score": float(delta[alarm.astype(bool)].sum()/np.maximum(delta,0).sum()),
                     "median_cv_iterations": int(np.median(rounds)),
                     "final_rounds": int(candidate["parameters"]["max_iter"]) if candidate["model"] == "lr" else int(np.median(rounds))})
    frame = pd.DataFrame(rows)
    frame.to_csv(study.REPORT / "cv_metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(study.REPORT / "cv_fold_metrics.csv", index=False)
    print(frame.to_string(index=False), flush=True)
    return frame


def ablation():
    study.prepare()
    jobs([(identifier, fold) for fold in range(3) for identifier in ("lgb_736", "lgb_734")])
    frame = summarize(("lgb_789", "lgb_736", "lgb_734"))
    frame.to_csv(study.REPORT / "clock_ablation_cv.csv", index=False)


def models():
    study.prepare()
    # A single LR at a time avoids simultaneous float64 solver copies.
    jobs([("lr_789", fold) for fold in range(3)], concurrency=1)
    # One EBM outer bag trains mostly serially. Three patient-fold jobs fit
    # within memory while using more than one core overall.
    jobs([(identifier, fold) for fold in range(3) for identifier in ("ebm150_789", "ebm300_789")], concurrency=3)
    frame = summarize(list(study.load_protocol()["candidates"]))
    ebms = frame.loc[frame.model.eq("ebm")].sort_values(["mean_fold_auroc", "final_rounds"], ascending=[False, True])
    assert len(ebms) == 2
    selected = ebms.iloc[0].candidate
    freeze(study.OUT / "model_selection.json", {
        "primary_feature_view": "baseline_789", "primary_candidates": ["lr_789", selected, "lgb_789"],
        "supplementary_candidates": ["xgb_789"], "ablation_candidates": ["lgb_789", "lgb_736", "lgb_734"],
        "selected_ebm": selected, "selection_data": "Training patient CV only; no test-driven feature or model choice",
        "criterion": "Fixed LR/LGB/XGB settings; EBM150/300 selected by mean-fold AUROC, then lower budget on ties",
        "ensemble_enabled": False})
    write_json(study.OUT / "rounds.json", {row.candidate: int(row.final_rounds) for row in frame.itertuples(index=False)})
    print("FROZEN MATCHED MODEL SELECTION: " + selected, flush=True)


def input_hashes(splits):
    records = {}
    for split in splits:
        case = json.loads(study.available(baseline.DATA / f"{split}_case.json").read_text())
        for suffix, key in (("features.npy", "feature_sha256"), ("labels.npy", "labels_sha256"),
                            ("patients.csv", "patients_sha256")):
            path = study.available(baseline.DATA / f"{split}_{suffix}")
            actual = content_hash(path)
            assert actual == case[key], path
            records[str(path.relative_to(ROOT))] = actual
        path = study.available(previous.DATA / f"{split}_extra.npy")
        marker = json.loads(study.available(previous.DATA / f"{split}_extra_case.json").read_text())
        actual = content_hash(path)
        assert actual == marker["feature_sha256"]
        records[str(path.relative_to(ROOT))] = actual
    for path in (previous.DATA / "cv_patient_roles.csv", baseline.DATA / "patient_splits.csv",
                 previous.DATA / "train_delta.npy", ROOT / "configs/us_next_experiments.json"):
        records[str(path.relative_to(ROOT))] = content_hash(study.available(path))
    return records


def persist_evaluation(split, identifier, scores, threshold):
    from sepsis_warning.challenge_evaluation import evaluate_scores
    from sepsis_warning.utility import patient_utility
    labels = np.load(study.available(baseline.DATA / f"{split}_labels.npy"))
    patients = pd.read_csv(study.available(baseline.DATA / f"{split}_patients.csv"))
    result = evaluate_scores(labels, scores, patients, threshold)
    out = study.OUT / "evaluation" / split / identifier
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "score.npy", scores)
    values = [{"patient_id": p.patient_id,
               **patient_utility(labels[int(p.start):int(p.stop)], scores[int(p.start):int(p.stop)] >= threshold)}
              for p in patients.itertuples(index=False)]
    pd.DataFrame(values).to_csv(out / "patient_utilities.csv", index=False)
    write_json(out / "metrics.json", result)
    return {"split": split, "candidate": identifier,
            "model": study.candidate_by_id(identifier)["model"],
            "feature_count": len(study.load_protocol()["feature_names"][study.candidate_by_id(identifier)["feature_view"]]),
            **{key: result[key] for key in ("auroc", "auprc", "utility_score", "threshold", "hour_precision", "hour_recall",
                                            "hour_fpr", "hour_f1", "hour_accuracy", "patients", "hours", "positive_hours")}}


def time_strata(labels, patients, alarms, identifier, split):
    """Descriptive elapsed-ICU strata, never final record length as a predictor."""
    groups = ((1,12), (13,24), (25,48), (49,72), (73,120), (121,1000000))
    counts = np.zeros((len(groups), 4), dtype=np.int64)
    for p in patients.itertuples(index=False):
        a,b = int(p.start), int(p.stop)
        y, alarm = labels[a:b], np.asarray(alarms[a:b], dtype=bool)
        hour = int(p.first_hour)+np.arange(b-a)
        positive = np.flatnonzero(y)
        pre6 = np.zeros(b-a, bool)
        if len(positive):
            first = int(positive[0]); pre6[first:first+6] = True
        for index,(lo,hi) in enumerate(groups):
            mask = (hour >= lo) & (hour <= hi)
            if len(positive):
                counts[index, 2] += (mask & pre6).sum()
                counts[index, 3] += (mask & pre6 & alarm).sum()
            else:
                counts[index, 0] += mask.sum()
                counts[index, 1] += (mask & alarm).sum()
    rows = []
    for (lo,hi), values in zip(groups, counts):
        nh,fp,ph,tp = map(int, values)
        rows.append({"split": split, "candidate": identifier, "current_iculos": f"{lo}-{hi}" if hi<1000000 else ">120",
                     "nonseptic_hours": nh, "nonseptic_alarm_hours": fp, "pre6_hours": ph, "pre6_alarm_hours": tp,
                     "nonseptic_fp_hour_fraction": fp/nh if nh else None,
                     "pre6_hour_sensitivity": tp/ph if ph else None})
    return rows


def finish():
    import joblib
    from sepsis_warning.challenge_utility import utility_targets
    from sepsis_warning.alarm_policy_analysis import exact_threshold
    from sepsis_warning.clock_interpretable import predict_bundle
    from sepsis_warning.challenge_evaluation import bootstrap_differences
    selection = json.loads(study.available(study.OUT / "model_selection.json").read_text())
    cv = summarize(list(study.load_protocol()["candidates"]))
    ebms = cv.loc[cv.model.eq("ebm")].sort_values(["mean_fold_auroc", "final_rounds"], ascending=[False, True])
    assert ebms.iloc[0].candidate == selection["selected_ebm"]
    write_json(study.OUT / "rounds.json", {row.candidate: int(row.final_rounds) for row in cv.itertuples(index=False)})
    identifiers = list(dict.fromkeys(selection["ablation_candidates"]+selection["primary_candidates"]+selection["supplementary_candidates"]))
    freeze(study.OUT / "final_selection.json", {"models": identifiers, "primary_candidates": selection["primary_candidates"],
                                                "ablation_candidates": selection["ablation_candidates"],
                                                "supplementary_candidates": selection["supplementary_candidates"],
                                                "ensemble_enabled": False, "selection_data": "Training CV only"})
    hashes = input_hashes(("train", "validation"))
    # EBM's one bag uses roughly one native core. Its matrix can coexist with
    # two tree fits. The full LR solver then runs alone to bound float64 memory.
    jobs([("lgb_736", -1), ("lgb_734", -1), (selection["selected_ebm"], -1)], concurrency=3)
    jobs([("lr_789", -1)], concurrency=1)
    validation_y = np.load(baseline.DATA / "validation_labels.npy")
    validation_patients = pd.read_csv(baseline.DATA / "validation_patients.csv")
    validation_delta = utility_targets(validation_y, validation_patients)["delta"]
    thresholds, model_hashes, threshold_hashes, model_paths = {}, {}, {}, {}
    results = []
    for identifier in identifiers:
        candidate = study.candidate_by_id(identifier)
        directory = study.resolve_directory(identifier, -1)
        scores = np.load(study.available(directory / "validation_score.npy"))
        calibration = exact_threshold(scores, validation_delta)
        target = study.OUT / "final" / identifier
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "threshold.json", {"threshold": calibration["threshold"], "max_validation_utility": calibration["utility"],
                                               "source": "Validation only, after training-CV model/round choices"})
        thresholds[identifier] = calibration["threshold"]
        if identifier in ("lgb_789", "xgb_789"):
            assert thresholds[identifier] == json.loads((directory / "threshold.json").read_text())["threshold"]
        suffix = "model.joblib" if candidate["model"] in ("lr", "ebm") else "model.txt" if candidate["model"] == "lightgbm" else "model.ubj"
        model_paths[identifier] = str((directory / suffix).relative_to(ROOT))
        model_hashes[identifier] = content_hash(directory / suffix)
        threshold_hashes[identifier] = content_hash(target / "threshold.json")
        results.append(persist_evaluation("validation", identifier, scores, thresholds[identifier]))
    source_files = ("src/sepsis_warning/clock_models.py", "src/sepsis_warning/clock_interpretable.py",
                    "scripts/fit_clock_models.py", "scripts/fit_interpretable_clock.py", "scripts/run_clock_models.py",
                    "scripts/fit_challenge_improvement.py")
    frozen = {"model_sha256": model_hashes, "model_paths": model_paths, "threshold_sha256": threshold_hashes,
              "source_sha256": {name: content_hash(ROOT/name) for name in source_files}, "input_sha256": hashes,
              "rounds_sha256": content_hash(study.OUT / "rounds.json"), "selection_sha256": content_hash(study.OUT / "final_selection.json"),
              "model_selection_sha256": content_hash(study.OUT / "model_selection.json"),
              "protocol_sha256": content_hash(study.OUT / "protocol.json"), "reuse_manifest_sha256": content_hash(study.OUT / "reuse_manifest.json"),
              "final_candidate_ids": identifiers, "test_used_for_selection": False, "historical_test_exposure": True,
              "ensemble_enabled": False}
    if (study.OUT / "scheduling_events.json").exists():
        frozen["scheduling_events_sha256"] = content_hash(study.OUT / "scheduling_events.json")
    frozen["lr_execution_environment_sha256"] = content_hash(study.OUT / "final/lr_789/execution_environment.json")
    freeze(study.OUT / "frozen_before_test.json", frozen)
    write_json(study.REPORT / "test_input_sha256.json", input_hashes(("test",)))
    test_scores = {}
    strata = []
    test_y = np.load(baseline.DATA / "test_labels.npy")
    test_patients = pd.read_csv(baseline.DATA / "test_patients.csv")
    for identifier in identifiers:
        candidate = study.candidate_by_id(identifier)
        directory = study.resolve_directory(identifier, -1)
        x = study.load_features("test", candidate["feature_view"])
        if candidate["model"] in ("lr", "ebm"):
            model = joblib.load(directory / "model.joblib")
            scores = predict_bundle(model, x)
        else:
            model = baseline.load_model(candidate["model"], directory)
            if candidate["model"] == "lightgbm":
                scores = model.predict(x, num_threads=4)
            else:
                scores = model.inplace_predict(x)
        assert np.isfinite(scores).all()
        test_scores[identifier] = scores
        results.append(persist_evaluation("test", identifier, scores, thresholds[identifier]))
        strata += time_strata(test_y, test_patients, scores >= thresholds[identifier], identifier, "test")
        assert content_hash(ROOT / model_paths[identifier]) == model_hashes[identifier]
        print("Frozen test: " + json.dumps(results[-1]), flush=True)
        del model,x
        gc.collect()
    train_y = np.load(baseline.DATA / "train_labels.npy")
    train_patients = pd.read_csv(baseline.DATA / "train_patients.csv")
    for identifier in identifiers:
        alarms = np.load(study.OUT / "cv_predictions" / f"{identifier}_alarm.npy")
        strata += time_strata(train_y, train_patients, alarms, identifier, "train_OOF")
    pd.DataFrame(strata).to_csv(study.REPORT / "current_iculos_metrics.csv", index=False)
    for identifier,expected in threshold_hashes.items():
        assert content_hash(study.OUT / "final" / identifier / "threshold.json") == expected
    frame = pd.DataFrame(results)
    frame.to_csv(study.REPORT / "metrics.csv", index=False)
    frame.loc[frame.candidate.isin(selection["ablation_candidates"])].to_csv(study.REPORT / "clock_ablation_metrics.csv", index=False)
    frame.loc[frame.candidate.isin(selection["primary_candidates"]+selection["supplementary_candidates"])].to_csv(study.REPORT / "model_comparison_metrics.csv", index=False)
    intervals = {"clock_ablation": {}, "primary_vs_lr": {}}
    for identifier in ("lgb_736", "lgb_734"):
        intervals["clock_ablation"][identifier] = bootstrap_differences(test_y, test_patients, test_scores["lgb_789"], test_scores[identifier],
            thresholds["lgb_789"], thresholds[identifier], repetitions=500, seed=2026)
    for identifier in (selection["selected_ebm"], "lgb_789"):
        intervals["primary_vs_lr"][identifier] = bootstrap_differences(test_y, test_patients, test_scores["lr_789"], test_scores[identifier],
            thresholds["lr_789"], thresholds[identifier], repetitions=500, seed=2026)
    write_json(study.REPORT / "paired_test_intervals.json", intervals)
    for name in ("protocol.json", "reuse_manifest.json", "model_selection.json", "final_selection.json", "rounds.json", "frozen_before_test.json"):
        write_json(study.REPORT / name, json.loads((study.OUT / name).read_text()))
    if (study.OUT / "scheduling_events.json").exists():
        write_json(study.REPORT / "scheduling_events.json", json.loads((study.OUT / "scheduling_events.json").read_text()))
    write_json(study.REPORT / "lr_execution_environment.json", json.loads((study.OUT / "final/lr_789/execution_environment.json").read_text()))
    write_report(frame, selection, intervals)
    write_json(study.OUT / "completed.json", {"ablation_views": 3, "primary_models": 3, "supplementary_models": 1,
                                               "new_CV_fits": 15, "reused_CV_fits": 6, "new_final_fits": 4, "reused_final_fits": 2,
                                               "final_fit_count_scope": "Completed new models entering evaluation; canceled attempts are separate in scheduling_events.json",
                                               "completed_new_final_candidates": [name for name in study.load_protocol()["candidates"]
                                                   if name not in ("lgb_789", "xgb_789") and (study.OUT/"final"/name/"results.json").exists()],
                                               "evaluations": len(frame), "ensemble_enabled": False, "test_used_for_selection": False})
    print(frame.to_string(index=False), flush=True)


def write_report(frame, selection, intervals):
    cv = pd.read_csv(study.REPORT / "cv_metrics.csv")
    lines = ["# 时钟特征消融与统一单模型对比", "",
             "沿用当前A+B完整记录研究：27,911训练患者、5,981验证、5,982测试；相同医院×事件分层患者名单、三折及内部15%校准患者。"
             "原始SepsisLabel不再次前移，完整记录包含可用发病后0–3小时。模型集成关闭。", "",
             "## 时钟消融", "",
             "789为原主模型；736仅去53个时钟派生列、保留原ICULOS/HospAdmTime；734去全部55显式时钟列。"
             "测量年龄/间隔/次数与生理历史保留，累计生理统计仍可能代理时长。全部LGB超参保持原789模型设置，早停规则、患者角色相同；"
             "各自从内部校准集选轮数及阈值。最终轮数为三折所选轮数中位数。", "",
             "| 训练CV方案 | 维数 | 平均折AUROC | 平均折AP | 合并外折US |", "|---|---:|---:|---:|---:|"]
    for identifier in selection["ablation_candidates"]:
        r = cv.loc[cv.candidate.eq(identifier)].iloc[0]
        lines.append(f"| {identifier} | {r.feature_count} | {r.mean_fold_auroc:.6f} | {r.mean_fold_auprc:.6f} | {r.utility_score:.6f} |")
    lines += ["", "## 固定测试集结果", "", "| 方案 | 角色 | 维数 | AUROC | AUPRC/AP | US |", "|---|---|---:|---:|---:|---:|"]
    primary = set(selection["primary_candidates"])
    for r in frame.loc[frame.split.eq("test")].itertuples(index=False):
        role = "主三模型" if r.candidate in primary else "XGB补充" if r.model == "xgboost" else "时钟消融"
        lines.append(f"| {r.candidate} | {role} | {r.feature_count} | {r.auroc:.6f} | {r.auprc:.6f} | {r.utility_score:.6f} |")
    lines += ["", "## 三模型的共同口径", "",
             "主表预先指定共同789列信息，LR/加性EBM/LightGBM使用完全相同患者、小时、标签与评分方法；XGBoost为补充。"
             "不根据消融测试成绩替换公共特征。LR固定L2/C=1，无类别权重，仅拟合患者学习中位数及标准化，保留全缺失列为0。"
             "EBM原生NaN，interactions=0，单袋且validation_size=0，分箱只见拟合患者，禁用按小时随机验证；150/300轮固定候选按训练外折平均AUROC选择。"
             "原LGB/XGB789的3折和最终模型均核对参数/名单/列名后显式复用，其历史signature和SHA保留，并未伪造新拟合。", "",
             f"选中的EBM为`{selection['selected_ebm']}`。LR的迭代数是求解收敛信息，不作为调参轮数；最终仍使用C=1、max_iter=2000。"
             "树模型沿用此前训练内选定设置，本轮LR是一组预设基线、EBM是两组预算，并非对各算法进行对称的穷尽调参。", "",
             "## 配对测试不确定性", "", "| 对照组 | 候选 | ΔAUROC [95% CI] | ΔAP [95% CI] | ΔUS [95% CI] |",
             "|---|---|---|---|---|"]
    for group, comparisons in intervals.items():
        for identifier,record in comparisons.items():
            values = [record["difference"][key] for key in ("auroc", "auprc", "utility_score")]
            text = [f"{x['point']:+.6f} [{x['lower']:+.6f}, {x['upper']:+.6f}]" for x in values]
            lines.append(f"| {group} | {identifier} | " + " | ".join(text) + " |")
    lines += ["", "500次患者配对bootstrap固定模型和验证阈值，区间不包含训练及候选选择变化。US采用固定官方评分器、每小时输出、无报警冷却。"
              "验证集US用于阈值校准而非独立估计；CV参与轮数/EBM预算选择，是开发估计。现有测试患者已有历史暴露，属于已知两院的探索性固定留出比较。", "",
              "## 与proposal的关系", "",
              "本轮补齐原proposal的LR/EBM/LightGBM架构对比，但使用当前A+B完整记录统一基准。原proposal的发病前六小时AUPRC、A开发/B迁移和6小时报警抑制是另一个研究终点；"
              "不能用此完整记录表替代原终点，或把旧312维LR/EBM成绩拼入新表。", "",
              "## 复现与产物", "", "```bash", ".venv/bin/python -B -u scripts/run_clock_models.py --stage all",
              ".venv/bin/python -B scripts/verify_clock_models.py --final", "```", "",
              "完整指标见metrics.csv，消融与主表分别为clock_ablation_metrics.csv/model_comparison_metrics.csv；"
              "current_iculos_metrics.csv另报当前已知ICU时段的非septic误报与pre6小时命中。"
              "新模型与预测留在outputs/clock_models，列掩码/角色引用在processed_data/clock_models，无重复存储大特征矩阵。", ""]
    if (study.OUT / "scheduling_events.json").exists():
        lines += ["拟合调度及取消的未完成预拟合记录见scheduling_events.json；这些未完成尝试没有参与候选选择或评分。", ""]
    lines += ["LR最终拟合只调整执行后端：JOBLIB_MULTIPROCESSING=0避免二分类单一求解任务的loky进程副本，VECLIB_MAXIMUM_THREADS=4。"
              "估计器仍保留n_jobs=4及相同C/L2/lbfgs/容差；历史CV模型不因执行后端调整而重训。运行配置见lr_execution_environment.json。", ""]
    (study.REPORT / "report.md").write_text("\n".join(lines))


def main():
    if os.environ.get("INFO5600_CLOCK_COORDINATOR_RUNTIME_READY") != "1":
        environment = baseline.runtime_environment()
        environment["INFO5600_CLOCK_COORDINATOR_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", "-u", __file__, *sys.argv[1:]], environment)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ablation", "models", "finish", "prefit-ebm150", "prefit-clocks", "all"], default="all")
    args = parser.parse_args()
    if args.stage == "prefit-ebm150":
        study.prepare()
        jobs([("ebm150_789", -1)], concurrency=1)
    if args.stage == "prefit-clocks":
        study.prepare()
        frame = summarize(["lgb_789", "lgb_736", "lgb_734"])
        write_json(study.OUT / "rounds.json", {row.candidate: int(row.final_rounds) for row in frame.itertuples(index=False)})
        jobs([("lgb_736", -1), ("lgb_734", -1)], concurrency=1)
    if args.stage in ("ablation", "all"):
        ablation()
    if args.stage in ("models", "all"):
        models()
    if args.stage in ("finish", "all"):
        finish()


if __name__ == "__main__":
    main()
