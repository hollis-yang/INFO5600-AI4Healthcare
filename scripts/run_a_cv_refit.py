"""Run full-A patient CV, freeze OOF thresholds, refit all A, then evaluate B."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from sepsis_warning.a_cv_refit import DATA, OUT, REPORT, MODELS, prepare, predict_full_patients
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.ab_regularization import runtime_environment
from sepsis_warning.data import config
from sepsis_warning.evaluation import ranking_metrics, select_alert_threshold
from sepsis_warning.utility import normalized_utility, patient_utility


def run_jobs(jobs):
    pending, active = list(jobs), []
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    last_update = time.monotonic()
    while pending or active:
        while pending and len(active) < 2:
            fold, model = pending.pop(0)
            path = logs / f"fold{fold}_{model}.log"
            stream = path.open("w")
            process = subprocess.Popen([sys.executable, str(ROOT / "scripts/fit_a_cv_refit.py"),
                                        "--fold", str(fold), "--model", model],
                                       cwd=ROOT, env=runtime_environment(), stdout=stream, stderr=subprocess.STDOUT)
            active.append((process, stream, fold, model, path))
            print(f"Started fold={fold} model={model}", flush=True)
        for item in list(active):
            process, stream, fold, model, path = item
            code = process.poll()
            if code is None:
                continue
            stream.close()
            active.remove(item)
            if code:
                for other, handle, *_ in active:
                    other.terminate(); handle.close()
                print(path.read_text()[-10000:], flush=True)
                raise RuntimeError(f"Worker failed: fold={fold} model={model}, code={code}")
            print(path.read_text()[-1200:], flush=True)
        if time.monotonic()-last_update >= 40:
            print("Active: " + ", ".join(f"fold={f}/{m}" for _, _, f, m, _ in active)
                  + f"; queued={len(pending)}", flush=True)
            last_update = time.monotonic()
        if pending or active:
            time.sleep(.5)


def calibrate():
    rows = pd.read_csv(DATA / "A_rows.csv.gz")
    patients = pd.read_csv(DATA / "A_patients.csv")
    row_folds = rows.patient_id.map(patients.set_index("patient_id").fold).to_numpy()
    thresholds, summary, fold_records = {}, [], []
    for model in MODELS:
        directory = OUT / "oof" / model
        directory.mkdir(parents=True, exist_ok=True)
        oof = np.full(len(rows), np.nan)
        components, full_tables, full_probabilities, full_labels = [], [], [], []
        for fold in range(3):
            source = OUT / "cv" / f"fold{fold}" / model
            result = json.loads((source / "results.json").read_text())
            p = np.load(source / "probability.npy", mmap_mode="r")
            labels = np.load(source / "labels.npy")
            mask = np.load(source / "task_eligible.npy")
            table = pd.read_csv(source / "patients.csv").set_index("patient_id")
            held = rows.loc[row_folds == fold]
            assert set(held.patient_id) == set(table.index)
            location = (held.patient_id.map(table.start).to_numpy()+held.hour.to_numpy()
                        -held.patient_id.map(table.first_hour).to_numpy()).astype(int)
            assert mask[location].all() and len(location) == int(mask.sum())
            np.testing.assert_array_equal(labels[location], held.label)
            oof[row_folds == fold] = p[location]
            contribution = pd.read_csv(source / "patient_utilities.csv")
            contribution["fold"] = fold
            components.append(contribution)
            full_tables.append(table.reset_index())
            full_probabilities.append(p)
            full_labels.append(labels)
            evaluation = result["evaluation"]
            fold_records.append({"model": model, "fold": fold,
                                 "training_patients": result["training_patients"],
                                 "evaluation_patients": evaluation["patients"],
                                 "auroc": evaluation["ranking"]["auroc"],
                                 "auprc": evaluation["ranking"]["auprc_average_precision"],
                                 "utility_score": evaluation["utility"]["normalized_utility"],
                                 "internally_calibrated_threshold": result["threshold"],
                                 "fitted_trees": result["details"].get("inner_selected_trees"),
                                 "fit_seconds": result["fit_seconds"]})
        assert np.isfinite(oof).all()
        np.save(directory / "eligible_probability.npy", oof)
        ranking = ranking_metrics(rows.label.to_numpy(), oof)
        threshold, search = select_alert_threshold(rows, oof, config())
        thresholds[model] = threshold
        search.to_csv(directory / "threshold_search.csv", index=False)
        pd.concat(components, ignore_index=True).to_csv(directory / "honest_cv_patient_utilities.csv", index=False)
        honest = normalized_utility(pd.concat(components, ignore_index=True))
        # This additional score uses the same OOF patients that selected the final threshold;
        # label it as calibration rather than independent outer-CV performance.
        calibrated = []
        for table, probability, labels in zip(full_tables, full_probabilities, full_labels):
            for patient in table.itertuples(index=False):
                start, stop = int(patient.start), int(patient.stop)
                calibrated.append({"patient_id": patient.patient_id,
                                   **patient_utility(labels[start:stop], probability[start:stop] >= threshold)})
        frame = pd.DataFrame(calibrated)
        frame.to_csv(directory / "pooled_threshold_calibration_patient_utilities.csv", index=False)
        calibration_utility = normalized_utility(frame)
        summary.append({"model": model, "evaluation": "A_patient_3CV_OOF", "patients": 20131,
                        "event_patients": 1585, "eligible_hours": len(rows), "full_hours": 788042,
                        "auroc": ranking["auroc"], "auprc": ranking["auprc_average_precision"],
                        "utility_score": honest["normalized_utility"],
                        "utility_threshold_policy": "Independent inner A calibration in each outer fold",
                        "final_OOF_threshold": threshold,
                        "OOF_threshold_calibration_utility": calibration_utility["normalized_utility"]})
        print(f"A OOF {model}: AUROC={ranking['auroc']:.6f}, AP={ranking['auprc_average_precision']:.6f}, "
              f"independent-fold utility={honest['normalized_utility']:.6f}, final threshold={threshold:.9f}", flush=True)
    write_json(OUT / "thresholds.json", thresholds)
    write_json(OUT / "A_oof_metrics.json", summary)
    pd.DataFrame(fold_records).to_csv(OUT / "cv_fold_metrics.csv", index=False)
    write_json(OUT / "selection.json", {"threshold_fit_on": "A OOF only", "B_used": False,
                                       "model_settings_changed": False, "thresholds": thresholds})
    return summary


def evaluate_b():
    # Freeze all models and thresholds on disk before opening B evaluation trajectories.
    frozen = {m: content_hash(OUT / "final" / m / "model.joblib") for m in MODELS}
    freeze = {"model_sha256": frozen, "thresholds_sha256": content_hash(OUT / "thresholds.json"),
              "selection_sha256": content_hash(OUT / "selection.json"), "B_used_for_selection": False}
    write_json(OUT / "frozen_before_B.json", freeze)
    thresholds = json.loads((OUT / "thresholds.json").read_text())
    split = pd.read_csv(ROOT / "processed_data/splits/patient_splits.csv")
    identifiers = split.loc[split.hospital.eq("B"), "patient_id"].tolist()
    assert len(identifiers) == 19743
    results = []
    for model in MODELS:
        model_path = OUT / "final" / model / "model.joblib"
        bundle = joblib.load(model_path)
        assert bundle["training_patients"] == 20131 and bundle["fit_hospitals"] == ["A"]
        assert bundle["threshold"] == thresholds[model]
        assert bundle["preprocessor"].fit_rows_ == 670920
        result = predict_full_patients(bundle, identifiers, OUT / "B" / model)
        assert result["patients"] == 19743 and result["event_patients"] == 905
        assert result["eligible_hours"] == 641470 and result["full_hours"] == 757924
        results.append({"model": model, "evaluation": "B_after_full_A_refit", "patients": result["patients"],
                        "event_patients": result["event_patients"], "eligible_hours": result["eligible_hours"],
                        "full_hours": result["full_hours"], "auroc": result["ranking"]["auroc"],
                        "auprc": result["ranking"]["auprc_average_precision"],
                        "utility_score": result["utility"]["normalized_utility"],
                        "utility_threshold_policy": "Frozen pooled A OOF threshold", "final_OOF_threshold": result["threshold"]})
        assert content_hash(model_path) == frozen[model]
        print("B refit " + json.dumps(results[-1]), flush=True)
    assert content_hash(OUT / "thresholds.json") == freeze["thresholds_sha256"]
    assert content_hash(OUT / "selection.json") == freeze["selection_sha256"]
    write_json(OUT / "B_metrics.json", results)
    return results


def report(a_metrics, b_metrics, protocol):
    for name, expected in protocol["input_fingerprints"].items():
        assert content_hash(ROOT / name) == expected, name
    case = json.loads((DATA / "case.json").read_text())
    for name, key in [("A_features.npy", "features_sha256"), ("A_rows.csv.gz", "rows_sha256"),
                      ("A_patients.csv", "patients_sha256")]:
        assert content_hash(DATA / name) == case[key], name
    folds = pd.read_csv(OUT / "cv_fold_metrics.csv")
    means = folds.groupby("model")[["auroc", "auprc", "utility_score"]].agg(["mean", "std"])
    means.columns = ["_".join(c) for c in means.columns]
    means.reset_index().to_csv(REPORT / "cv_mean_sd.csv", index=False)
    folds.to_csv(REPORT / "cv_fold_metrics.csv", index=False)
    metrics = pd.DataFrame(a_metrics+b_metrics)
    metrics.to_csv(REPORT / "metrics.csv", index=False)
    final_records = {m: json.loads((OUT / "final" / m / "results.json").read_text()) for m in MODELS}
    write_json(REPORT / "final_fit_metadata.json", final_records)
    for name in ["protocol.json", "thresholds.json", "selection.json", "frozen_before_B.json",
                 "A_oof_metrics.json", "B_metrics.json"]:
        write_json(REPORT / name, json.loads((OUT / name).read_text()))
    pd.read_csv(DATA / "A_patients.csv").to_csv(REPORT / "A_patient_cv_folds.csv", index=False)
    verification = {"input_fingerprints_unchanged": True, "original_split_files_unchanged": True,
                    "A_original_test_included_in_this_new_development": True,
                    "outer_cv_patient_disjoint": True, "B_used_for_selection": False,
                    "final_model_training_patients": 20131, "final_model_training_hours": 670920,
                    "A_OOF_patients": 20131, "B_patients": 19743,
                    "source_fingerprints": {str(p.relative_to(ROOT)): content_hash(p) for p in [
                        ROOT / "scripts/run_a_cv_refit.py", ROOT / "scripts/fit_a_cv_refit.py",
                        ROOT / "src/sepsis_warning/a_cv_refit.py"]}}
    write_json(REPORT / "verification.json", verification)
    lines = ["# 全 A 患者交叉验证与重训后 B 迁移评价", "",
             "沿用固定的 312 个因果特征及 LR、EBM、LightGBM 参数，没有开展新的特征或超参数搜索。"
             "本轮将全部符合条件的 A 患者纳入开发，原 A 的 70/15/15 划分文件保留，但原 A 测试患者已进入此次交叉验证及最终训练。", "",
             "A：20,131 位患者，1,585 位发病患者，670,920 个符合条件的发病前小时；B：19,743 位患者，905 位发病患者，641,470 个符合条件小时。", "",
             "## 主要结果", "", "| 评价 | 模型 | AUROC | AUPRC | US |", "|---|---|---:|---:|---:|"]
    for row in a_metrics+b_metrics:
        label = "A 三折患者 OOF" if row["evaluation"].startswith("A_") else "全 A 重训 → 全 B"
        lines.append(f"| {label} | {row['model']} | {row['auroc']:.6f} | {row['auprc']:.6f} | {row['utility_score']:.6f} |")
    lines += ["", "AUROC/AUPRC 是严格发病前未来六小时任务的 pooled 指标；US 使用同批患者的完整原始轨迹，"
              "保留原 SepsisLabel，不再平移，采用固定版本官方评分器，每小时阈值输出、不施加报警冷却。", "",
              "A 的主 US：每个外层折阈值仅在该折训练部分的独立内部校准患者上确定，然后用于外层未见患者；"
              "将全部外层评价患者的原始效用相加后统一标准化，不能平均各折标准化分数。", "",
              "最终 B 阈值：先从全部 A 的折外预测按既定误报预算选出，再冻结并用于全 A 重训模型。"
              "A OOF 在该最终阈值下的 US 属于阈值校准成绩，另存 OOF_threshold_calibration_utility，不作为独立 CV 主 US。", "",
              "## 训练与阈值流程", "",
              "三折按 A 患者的发病状态分层，随机种子 2026。同一患者的全部小时始终在同一折。"
              "每折训练患者内部再按患者分层留出 20% 校准阈值；内层插补、裁剪和标准化仅在内层拟合患者上学习。"
              "随后在完整外层训练部分重新拟合预处理和模型，再评价该折未见患者。", "",
              "LightGBM 在内部 A 校准患者上按 binary_logloss 早停选树数，之后按该树数重训。"
              "全 A 最终模型也在全 A 内部重新选择树数，不依赖 B。EBM 保持三组患者级早停 bag；"
              "内部 bags 共享当前拟合分区的预处理和分箱，外层评价患者及 B 不进入其中。", "",
              "阈值继续使用原先的临床规则：六小时重复报警抑制后，每 100 个符合条件患者小时误报不超过一次，"
              "优先最大化事件检出率。该规则不等于直接最大化官方 US。", "",
              "## 限制", "",
              "B 在此前实验中已多次查看，本轮属于探索性外院迁移评价，不能称为新取得的完全盲法外部测试。"
              "A 的配置沿用历史开发选择，本轮 CV 评价的是这些已冻结配置，不是独立于全部历史选模过程的全新估计。"
              "A OOF 阈值由较小训练模型的预测确定，转移到全 A 模型后概率分布可能变化。"
              "指标是本次固定模型的点估计，未把交叉验证折间标准差当成置信区间。", "",
              "cv_fold_metrics.csv 保存每折结果；cv_mean_sd.csv 保存三折均值和标准差，它们不同于 pooled OOF。"
              "所有模型、预测和校准记录位于本地 outputs/a_only_cv_refit，处理数据单独存放于 processed_data/a_only_cv_refit。"]
    (REPORT / "report.md").write_text("\n".join(lines)+"\n")
    write_json(OUT / "completed.json", {"A_models": 9, "final_models": 3, "B_evaluations": 3,
                                       "B_used_for_selection": False, "metrics": a_metrics+b_metrics})
    print(metrics[["evaluation", "model", "auroc", "auprc", "utility_score"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    protocol = prepare()
    if args.prepare_only:
        return
    run_jobs([(fold, model) for fold in range(3) for model in ["lr", "lightgbm", "ebm"]])
    a_metrics = calibrate()
    run_jobs([(-1, model) for model in ["lr", "lightgbm", "ebm"]])
    b_metrics = evaluate_b()
    report(a_metrics, b_metrics, protocol)


if __name__ == "__main__":
    main()
