"""Create the regularization/class-weight comparison and curate a small Git snapshot."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/"outputs/.cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from sepsis_warning.data import save_json
from sepsis_warning.evaluation import paired_patient_bootstrap


def main():
    directory = ROOT/"outputs/ebm_regularized"
    experiment = json.loads((directory/"experiment_results.json").read_text())
    lr = json.loads((ROOT/"outputs/lr_results.json").read_text())
    old = json.loads((ROOT/"outputs/ebm_results.json").read_text())
    cases = experiment["validation_cases"]
    selection = experiment["selection"]
    rows = []
    def row(name, validation, training, alarms, oob=None, weight=1, raw_brier=None):
        return {"model": name, "positive_weight": weight, "validation_auroc": validation["auroc"],
                "validation_auprc": validation["auprc_average_precision"], "validation_brier": validation["brier_score"],
                "raw_weighted_brier": validation["brier_score"] if raw_brier is None else raw_brier,
                "training_auprc": training["auprc_average_precision"],
                "internal_oob_auprc": None if oob is None else oob["auprc_average_precision"],
                "event_sensitivity": alarms["event_sensitivity"], "alert_ppv": alarms["alert_ppv"],
                "false_alerts_per_100_patient_hours": alarms["false_alerts_per_100_patient_hours"],
                "detected_events": alarms["detected_event_patients"], "false_alert_episodes": alarms["false_alert_episodes"],
                "median_lead_hours": alarms["median_true_alert_lead_hours"]}
    rows.append(row("LR baseline", lr["validation"], lr["training"], lr["validation_selected_threshold_alert_metrics"]))
    rows.append(row("EBM initial", old["validation"], old["training"], old["validation_selected_threshold_alert_metrics"]))
    for case in cases:
        rows.append(row(case["name"], case["validation"], case["training"], case["alerts"], case["oob"],
                        case["positive_weight"], case["validation_raw"]["brier_score"]))
    table = pd.DataFrame(rows)
    table["selected_internally"] = table.model == selection["selected_case"]
    table.to_csv(directory/"comparison.csv", index=False)
    print(table.to_string(index=False), flush=True)
    reference = pd.read_csv(ROOT/"outputs/lr_validation_predictions.csv.gz")
    selected = next(case for case in cases if case["selected_internally"])
    unweighted = next(case for case in cases if case["name"] == selection["selected_regularization"])
    interval_results = {}
    pairs = [("selected_vs_lr", "LR", selection["selected_case"]),
             ("regularized_vs_initial", "initial", selection["selected_regularization"])]
    for case in cases:
        if case["positive_weight"] > 1:
            pairs.append((f"weight{case['positive_weight']:g}_vs_weight1", selection["selected_regularization"], case["name"]))
    def predictions(name):
        if name == "LR":
            return reference
        if name == "initial":
            return pd.read_csv(ROOT/"outputs/ebm_validation_predictions.csv.gz")
        return pd.read_csv(directory/name/"validation_predictions.csv.gz")
    for label, left, right in pairs:
        a, b = predictions(left), predictions(right)
        metadata = ["patient_id", "hospital", "hour", "label", "onset_hour"]
        pd.testing.assert_frame_equal(a[metadata], b[metadata])
        print(f"Paired patient bootstrap: {label}", flush=True)
        paired = paired_patient_bootstrap(a, a.probability.to_numpy(), b.probability.to_numpy(), 300, 42)
        interval_results[label] = {"reference_model": left, "candidate_model": right,
                                  "reference": paired["lr"], "candidate": paired["ebm"],
                                  "candidate_minus_reference": paired["ebm_minus_lr"],
                                  "valid_resamples": paired["valid_resamples"]}
    save_json(directory/"paired_intervals.json", interval_results)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.calibration import calibration_curve
    colors = ["#24788c", "#aaa6a1", "#bf6546"]
    plotted = [("LR", reference), ("Initial EBM", predictions("initial")),
               (f"Selected EBM (weight {selected['positive_weight']:g})", predictions(selected["name"]))]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    largest = 0.0
    for color, (name, frame) in zip(colors, plotted):
        probability, labels = frame.probability.to_numpy(), frame.label.to_numpy()
        precision, recall, _ = precision_recall_curve(labels, probability)
        axes[0].plot(recall, precision, color=color, label=name)
        fpr, tpr, _ = roc_curve(labels, probability)
        axes[1].plot(fpr, tpr, color=color, label=name)
        observed, predicted = calibration_curve(labels, probability, n_bins=10, strategy="quantile")
        axes[2].plot(predicted, observed, color=color, marker="o", label=name)
        largest = max(largest, observed.max(), predicted.max())
    axes[0].axhline(reference.label.mean(), color="gray", linestyle="--", label="Prevalence")
    axes[0].set(title="Precision-recall", xlabel="Recall", ylabel="Precision")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[2].plot([0, largest*1.1], [0, largest*1.1], color="gray", linestyle="--")
    axes[2].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    for ax in axes: ax.legend(fontsize=8)
    fig.suptitle("Hospital A validation - selection made inside A training only")
    fig.tight_layout()
    fig.savefig(directory/"selected_model_comparison.png", dpi=180)
    plt.close(fig)

    weight_rows = [case for case in cases if case["name"] == selection["selected_regularization"] or case["positive_weight"] > 1]
    weight_rows.sort(key=lambda case: case["positive_weight"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ticks = np.arange(len(weight_rows))
    names = [f"Weight {case['positive_weight']:g}" for case in weight_rows]
    axes[0].bar(ticks, [case["validation"]["auprc_average_precision"] for case in weight_rows], color="#24788c")
    axes[0].set(title="Validation AUPRC", xticks=ticks, xticklabels=names)
    axes[1].bar(ticks-.16, [case["validation_raw"]["brier_score"] for case in weight_rows], width=.32, color="#aaa6a1", label="Raw weighted probability")
    axes[1].bar(ticks+.16, [case["validation"]["brier_score"] for case in weight_rows], width=.32, color="#bf6546", label="Analytic prior correction")
    axes[1].set(title="Brier score (lower is better)", xticks=ticks, xticklabels=names)
    axes[1].legend(fontsize=8)
    axes[2].bar(ticks-.16, [100*case["alerts"]["event_sensitivity"] for case in weight_rows], width=.32, color="#24788c", label="Event sensitivity")
    axes[2].bar(ticks+.16, [100*case["alerts"]["alert_ppv"] for case in weight_rows], width=.32, color="#bf6546", label="Alert PPV")
    axes[2].set(title="At the same false-alert ceiling", ylabel="Percent", xticks=ticks, xticklabels=names)
    axes[2].legend(fontsize=8)
    fig.suptitle("Class-weight ablation - same regularization, patient bags and bin cuts")
    fig.tight_layout()
    fig.savefig(directory/"class_weight_comparison.png", dpi=180)
    plt.close(fig)

    selected_ci = interval_results["selected_vs_lr"]["candidate_minus_reference"]["auprc_average_precision"]
    outer_best = max(cases, key=lambda case: case["validation"]["auprc_average_precision"])
    report = f'''EBM 过拟合控制与类别权重实验

外层患者划分保持不变，原始数据及缓存输入校验一致；未评价医院 A 测试集或医院 B。
首轮基线已保留，所有新模型继续使用同一份 312 个输入变量和 A 训练拟合的预处理器。

主要发现：初版 EBM 的训练 AUPRC {old['training']['auprc_average_precision']:.4f}、验证 {old['validation']['auprc_average_precision']:.4f}，
内部选定改进模型的训练 AUPRC {selected['training']['auprc_average_precision']:.4f}、验证 {selected['validation']['auprc_average_precision']:.4f}，训练—验证差距明显缩小。
内部选择保留阳性权重 {selection['selected_positive_weight']:g}；本轮权重 2、5 的收益应结合配对差值区间判断，不能仅凭点估计认定改善。
改进模型与 LR 的 AUPRC 差值 95% CI 为 [{selected_ci['lower']:+.4f}, {selected_ci['upper']:+.4f}]，仍不能据此确认优于 LR。

一、设计
每个模型有 3 个按患者分组的 bag，每个 bag 使用 80% 的 A 训练患者学习、20% 的患者早停。
同一患者所有小时在一个 bag 内保持相同身份；这些内部留出标签用于早停和设置选择，不作为独立测试。
共享无监督预处理及分箱来自 A 训练范围，未触及外层验证或测试集。
先比较 32 bins / min leaf 200 / lambda 10 与 16 bins / min leaf 400 / lambda 50。
两档均最多 150 轮、叶节点最多 2 个、25 轮平滑、15 轮等待早停，保存 Native 最优损失检查点。
再固定内部选定的设置、患者 bags 和无权重的分箱边界，比较阳性权重 1、2、5。
所有数据保留真实比例，未做过采样或欠采样。

二、内部选择（先于读取外层验证预测）
选定复杂度：{selection['selected_regularization']}。
选定模型：{selection['selected_case']}；阳性权重：{selection['selected_positive_weight']:g}。
选择指标为内部患者 OOB AUPRC，选择不使用外层验证集。
OOB/留出结果参与早停和选择，可能偏乐观；只将外层验证作为本轮性能对照。
外层 AUPRC 最高的候选为 {outer_best['name']}（{outer_best['validation']['auprc_average_precision']:.4f}）；仍按事先记录的内部选择保留 {selected['name']}，没有事后按外层结果改选。

三、医院 A 验证集结果
{table.to_string(index=False)}

内部选定模型相对 LR 的 AUPRC 差值为 {selected['validation']['auprc_average_precision']-lr['validation']['auprc_average_precision']:+.4f}。
按患者配对 bootstrap 95% CI：[{selected_ci['lower']:+.4f}, {selected_ci['upper']:+.4f}]。
其他差值置信区间在 paired_intervals.json 中。

四、概率与报警口径
类别加权会改变学习概率。权重 >1 时预先规定从原始 log-odds 减去 log(weight)，做解析先验修正。
这不是在验证集上拟合的校准，也不保证校准正确；表中保留修正前和修正后的 Brier 分数。
AUPRC/AUROC 不受这一单调概率修正影响。
各模型报警阈值仍在外层验证集按每 100 患者小时最多 1 次误报选定，6 小时内抑制重复报警。
报警结果在同一验证集选择和报告，因此是探索性结果；实际误报率未必完全相同。

五、限制与文件
初次 EBM 与改进模型同时改变了多项复杂度控制，不能把改进归因于其中一个单独参数。
权重实验固定了复杂度、分箱和患者 bags，可以更直接检验类别权重的影响。
外层验证集已在前几轮开发中查看过，最终泛化结论仍需保留的测试集。
模型：outputs/ebm_regularized/selected_model.joblib；输入仍为原始因果特征经过现有 preprocessor 的矩阵。
逐项指标：comparison.csv；完整实验：experiment_results.json；配对区间：paired_intervals.json。
'''
    (directory/"experiment_report.txt").write_text(report)
    snapshot = ROOT/"reports/ebm_regularized"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ["comparison.csv", "experiment_results.json", "internal_selection.json", "paired_intervals.json",
                 "patient_bags.json", "input_fingerprints.json", "experiment_report.txt",
                 "selected_model_comparison.png", "class_weight_comparison.png"]:
        shutil.copy2(directory/name, snapshot/name)
    shutil.copy2(ROOT/"processed_data/splits/ebm_inner_patient_bags.csv", snapshot/"inner_patient_bags.csv")
    print("Saved local experiment report and curated Git snapshot.", flush=True)


if __name__ == "__main__":
    main()
