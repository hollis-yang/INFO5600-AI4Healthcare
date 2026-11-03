"""Consolidate existing results and derive hourly classification metrics without refitting."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sepsis_warning.ab_expansion import content_hash
from sepsis_warning.ab_regularization import OUTPUT, PROCESSED, REPORT, output_directory
from sepsis_warning.evaluation import ranking_metrics

MODELS = ["lr", "ebm", "lightgbm"]
SPLITS = ["A_validation", "B_validation", "pooled_validation"]
NAMES = {"lr": "LR", "ebm": "EBM", "lightgbm": "LightGBM"}


def markdown(frame, digits=6):
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "\\|")
    lines = ["| " + " | ".join(frame.columns) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |"
              for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def build():
    inputs = {}

    def record(path):
        path = Path(path)
        inputs[str(path.relative_to(ROOT))] = content_hash(path)
        return path

    def csv(name):
        return pd.read_csv(record(REPORT / name))

    archive = json.loads(record(REPORT / "model_metadata_archive.json").read_text())["results"]
    jobs = json.loads(record(REPORT / "validation_manifest.json").read_text())
    selection = json.loads(record(REPORT / "selection.json").read_text())
    completion = json.loads(record(REPORT / "completed.json").read_text())
    cv = csv("cv_summary.csv")
    validation = csv("selected_validation_metrics.csv")
    alerts = csv("alert_comparison.csv")
    utility = csv("utility/all_scores.csv")
    effects = csv("utility/paired_effects.csv")
    missing = csv("missingness/cross_hospital.csv")
    missing_summary = csv("missingness/patient_composition.csv")
    hospital_folds = csv("missingness/hospital_classifier_folds.csv")
    hospital_diagnostic = json.loads(record(REPORT / "missingness/hospital_classifier.json").read_text())
    rows = {s: pd.read_csv(record(PROCESSED / f"{s}_rows.csv.gz")) for s in SPLITS[:2]}
    rows["pooled_validation"] = pd.concat(list(rows.values()), ignore_index=True)
    train_rows = pd.read_csv(record(PROCESSED / "training_rows.csv.gz"))

    cohorts = []
    tables = [("A_training", train_rows.loc[train_rows.hospital.eq("A")]),
              ("B_training", train_rows.loc[train_rows.hospital.eq("B")]),
              ("pooled_training", train_rows), *rows.items()]
    for name, table in tables:
        full = utility.loc[utility.evaluation.eq(name), "full_hours"].unique()
        cohorts.append({"cohort": name, "patients": int(table.patient_id.nunique()),
                        "event_patients": int(table.groupby("patient_id").label.max().sum()),
                        "eligible_hours": len(table), "positive_hours": int(table.label.sum()),
                        "positive_hour_prevalence": float(table.label.mean()),
                        "full_utility_hours": int(full[0]) if len(full) else None})
    cohorts = pd.DataFrame(cohorts)

    training, hourly, combined = [], [], []
    for job in jobs:
        model = job["model"]
        directory = output_directory(**job)
        result_key = str((directory / "results.json").relative_to(ROOT))
        result = archive[result_key]
        assert result["candidate"] == selection["selected_candidates"][model]
        threshold = result["threshold"]
        probabilities = {s: np.load(record(directory / f"{s}_probability.npy")) for s in SPLITS[:2]}
        probabilities["pooled_validation"] = np.concatenate(list(probabilities.values()))
        for hospital, label in [("A", "A_training"), ("B", "B_training"), ("pooled", "pooled_training")]:
            training.append({"model": model, "candidate": job["candidate"], "evaluation": label,
                             **result["training"][hospital],
                             "macro_hospital_auprc": result["training"]["macro_hospital_auprc"]})
        for split in SPLITS:
            y = rows[split].label.to_numpy()
            probability = probabilities[split]
            assert len(y) == len(probability) and np.isfinite(probability).all()
            ranked = ranking_metrics(y, probability)
            source = validation.loc[validation.model.eq(model) & validation.evaluation.eq(split)].iloc[0]
            for metric, value in ranked.items():
                np.testing.assert_allclose(value, source[metric], atol=1e-12, rtol=0)
            predicted = probability >= threshold
            tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
            assert tn + fp + fn + tp == len(y) and tp + fn == y.sum()
            hour = {"model": model, "candidate": job["candidate"], "evaluation": split,
                    "threshold": threshold, "hour_accuracy": float(accuracy_score(y, predicted)),
                    "hour_precision": float(precision_score(y, predicted, zero_division=0)),
                    "hour_recall": float(recall_score(y, predicted)),
                    "hour_specificity": float(tn / (tn + fp)),
                    "hour_f1": float(f1_score(y, predicted)),
                    "hour_balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                    "hour_tp": int(tp), "hour_fp": int(fp), "hour_fn": int(fn), "hour_tn": int(tn)}
            hourly.append(hour)
            alarm = alerts.loc[alerts.model.eq(model) & alerts.method.eq("selected") & alerts.evaluation.eq(split)].iloc[0].to_dict()
            np.testing.assert_allclose(threshold, alarm["threshold"], atol=1e-12, rtol=0)
            for key in ["model", "method", "evaluation", "threshold"]:
                alarm.pop(key)
            u = utility.loc[utility.stage.eq("A_B_cv") & utility.model.eq(model) & utility.evaluation.eq(split)]
            budget = u.loc[u.policy.eq("existing_validation_budget")].iloc[0]
            np.testing.assert_allclose(threshold, budget.threshold, atol=1e-12, rtol=0)
            combined.append({**hour, **ranked, **alarm,
                             "missed_event_patients": int(alarm["event_patients"] - alarm["detected_event_patients"]),
                             "utility_existing_budget": float(budget.normalized_utility),
                             "utility_fixed_0p5": float(u.loc[u.policy.eq("fixed_0p5"), "normalized_utility"].iloc[0]),
                             "utility_full_hours": int(budget.full_hours)})
    training, hourly, combined = map(pd.DataFrame, [training, hourly, combined])
    assert len(training) == len(hourly) == len(combined) == 9

    primary = utility.loc[utility.policy.eq("existing_validation_budget")]
    utility_table = primary.pivot(index=["model", "evaluation"], columns="stage", values="normalized_utility").reset_index()
    contrast = effects.loc[effects.policy.eq("existing_validation_budget") & effects.reference.eq("A_only"),
                           ["model", "evaluation", "difference", "lower", "upper"]]
    utility_table = utility_table.merge(contrast, on=["model", "evaluation"])
    default_utility = utility.loc[utility.policy.eq("fixed_0p5")].pivot(
        index=["model", "evaluation"], columns="stage", values="normalized_utility").reset_index()
    chosen = cv.loc[cv.selected, ["model", "mean_training_macro_auprc", "mean_macro_auprc",
                                  "fold_sd_macro_auprc", "oof_A_auprc", "oof_B_auprc", "oof_macro_auprc"]]
    missing_columns = ["variable", "A_raw_hour_missing_rate", "B_raw_hour_missing_rate",
                       "A_latest_hour_missing_rate", "B_latest_hour_missing_rate"]
    selected_alerts = alerts.loc[alerts.method.eq("selected")]
    text = f"""# 全部指标汇总

本文件汇总聊天中报告的指标，并保存可复算的小时级 Accuracy、Precision、Recall、Specificity、F1、Balanced Accuracy 与混淆矩阵。所选模型来自已冻结的医院分层 CV；本次只读取已保存预测，未重新训练、选择参数或改变阈值。

三个模型均保留原 A+B 设置和权重，因此“原 A+B”与“本轮所选”的验证指标、阈值和 utility 相同。本轮未重新评分测试集。所有 CSV 保留未四舍五入的数值；下列 Markdown 浮点值显示六位小数，比例使用 0–1。

## 队列与评价单位

{markdown(cohorts)}

发病前任务仅使用 eligible_hours；官方 utility 对同一批验证患者使用完整 raw 轨迹，共 233,738 小时、5,981 位患者。阳性比例很低，合并验证集始终预测阴性的 Accuracy 为 {1 - rows['pooled_validation'].label.mean():.6f}，因此不能凭 Accuracy 判断预警价值。

## 全部正则化候选

CV 指标先对 A/B 医院 AUPRC 取平均，再对三折取平均。候选必须平均提升至少 0.001 且至少改善两折，才替换原设置；否则保留 baseline。折间 SD 是描述性波动，不是置信区间。EBM 的 stronger/weaker 联合改变叶子下限及 L2：800/100 与 200/10，原设置为 400/50；LightGBM 对应 400/50 与 100/1，原设置为 200/10。LR 原 C=1，较小 C 表示更强 L2。

{markdown(cv)}

所选模型训练折与 OOF 分数（macro 均指两院平均；OOF 是拼接各折未见患者预测后再评分）：

{markdown(chosen)}

本轮 CV 共 {completion['cv_models']} 次拟合，主实验墙钟耗时 {completion['elapsed_seconds'] / 60:.2f} 分钟；各候选 mean_fit_seconds 为单次拟合平均耗时，不能直接当作并行运行总时长。

## 完整训练集拟合指标

{markdown(training)}

训练分数用于观察拟合程度，不代表未见患者上的泛化性能。完整小型模型结果和逐折分数分别见 [model_metadata_archive.json](model_metadata_archive.json) 与 [cv_fold_results.csv](cv_fold_results.csv)。

## 固定验证集概率指标

{markdown(validation)}

AUROC/AUPRC 越大越好，Brier 越小越好。AUPRC 使用 Average Precision。各队列阳性比例不同，应在同一评价队列比较模型。概率校准目前报告 Brier，没有计算校准斜率或 ECE。

## 小时级分类指标与混淆矩阵

每个符合条件小时直接判断 probability >= threshold，不应用 6 小时报警冷却。hour_tp/fp/fn/tn 的单位均为小时；hour_recall 与患者级 event_sensitivity 的统计单位不同。

{markdown(hourly)}

本表保存为 [hourly_classification_metrics.csv](hourly_classification_metrics.csv)。概率指标、小时分类、患者报警与两种 utility 阈值已合并为 [all_model_metrics.csv](all_model_metrics.csv)，每个模型/验证队列一行。

## 患者级报警指标

以下采用 6 小时冷却；真报警位于发病前六小时，早于该窗口或非发病患者上的报警计为误报。阈值在同一个 A+B 混合验证集选择，目标为每 100 个符合条件的患者小时不超过 1 次误报；不保证 A/B 分别达标。漏检患者数及所有原始报警计数也保存在 all_model_metrics.csv。

{markdown(selected_alerts)}

## 官方完整轨迹 Utility Score

采用固定版本 [PhysioNet/CinC 2019 官方评分代码](../../third_party/physionet2019/README.md)，保留原始 SepsisLabel，不再平移标签。每小时直接二值判断，不套用报警冷却。0 对应始终不报警，1 对应本队列理想预测，负分允许。合并分数先累加患者效用再标准化，不是两院分数的算术平均。

原有预算阈值下，A-only、原 A+B 和本轮 CV 所选的结果及 A+B 减 A-only 差值区间：

{markdown(utility_table)}

以下为固定 0.5 阈值的完整对照；其接近零的效用不能与上表既定预算阈值混用：

{markdown(default_utility, digits=9)}

Utility 采用完整小时轨迹，与仅发病前任务是不同评价口径；当前成人新发队列也不是官方隐藏测试队列。原始 observed/best/inaction/worst 效用及标准化分母保存在 [utility/all_scores.csv](utility/all_scores.csv)，全部 36 组配对差值区间保存在 [utility/paired_effects.csv](utility/paired_effects.csv)。Utility 未用于本轮模型或阈值选择。

## 两院缺失分布

全部分析只使用 A+B 训练患者。原始缺失表示该小时没有观测；最近值缺失对生理指标采用六小时有效期，背景变量仍取当前值。

{markdown(missing_summary)}

完整 40 变量的原始与最近值缺失率：

{markdown(missing[missing_columns])}

患者等权、统一发病比例及前 24 小时对照的全部数值在 [missingness/cross_hospital.csv](missingness/cross_hospital.csv)；按发病状态和 ICU 小时段的表分别为 [by_event.csv](missingness/by_event.csv) 与 [by_hour_band.csv](missingness/by_hour_band.csv)。这些描述性对照不构成因果调整，不能单凭差异推断缺失导致预测失败。

## 缺失模式辨别医院的诊断

只用训练患者完整符合条件轨迹的 40 项缺失比例识别医院，整体 OOF 医院 AUROC 为 {hospital_diagnostic['pooled_oof_hospital_auroc']:.6f}；这些全段诊断特征没有进入逐小时 sepsis 模型。

{markdown(hospital_folds)}

## 不确定性与复现

外层验证集此前已经查看过，结果属于探索性评价。已有 bootstrap 针对患者配对差值，采用 300 次重采样，条件于已训练模型和固定阈值，未作多重比较校正；没有计算各 AUROC/AUPRC 的绝对置信区间。所选模型直接复用原权重时，差值及区间恒为 0，不能解释为一次重新训练的不确定性。

运行 `python scripts/report_all_metrics.py` 可从既有结果和本地预测重建本汇总；`python scripts/report_all_metrics.py --verify` 只核对重建结果与快照。输入预测和处理后数据继续保留在本地忽略目录，输入摘要和本脚本摘要见 [all_metrics_manifest.json](all_metrics_manifest.json)。生成过程中会重算验证 AUROC/AUPRC/Brier 并与既有结果核对，检查分类混淆矩阵总数和阳性小时数。

原研究报告与图见 [report.md](report.md)，utility 口径说明见 [utility/report.md](utility/report.md)。
"""
    products = {"cohort_metrics.csv": cohorts.to_csv(index=False),
                "training_metrics.csv": training.to_csv(index=False),
                "hourly_classification_metrics.csv": hourly.to_csv(index=False),
                "all_model_metrics.csv": combined.to_csv(index=False),
                "all_metrics.md": text}
    return products, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    products, inputs = build()
    source = {str(Path(__file__).resolve().relative_to(ROOT)): content_hash(__file__)}
    if args.verify:
        metadata = json.loads((REPORT / "all_metrics_manifest.json").read_text())
        assert metadata["input_fingerprints"] == inputs and metadata["source_fingerprints"] == source
        for name, body in products.items():
            assert (REPORT / name).read_text() == body, name
            assert content_hash(REPORT / name) == metadata["output_fingerprints"][name], name
        print("All consolidated metric tables and source/input hashes verified.")
    else:
        REPORT.mkdir(parents=True, exist_ok=True)
        for name, body in products.items():
            (REPORT / name).write_text(body)
        metadata = {"hourly_metric_rows": 9, "combined_validation_rows": 9, "training_metric_rows": 9,
                    "hourly_metrics_use_suppression": False, "alert_suppression_hours": 6,
                    "models_refitted": False, "thresholds_changed": False, "test_sets_evaluated": [],
                    "input_fingerprints": inputs, "source_fingerprints": source,
                    "output_fingerprints": {name: content_hash(REPORT / name) for name in products}}
        (REPORT / "all_metrics_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("Wrote complete metrics report and four CSV tables.")


if __name__ == "__main__":
    main()
