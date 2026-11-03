"""Compare frozen LR and additive EBM validation predictions with paired patient CIs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "outputs/.cache"))
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sepsis_warning.data import config, save_json
from sepsis_warning.evaluation import paired_patient_bootstrap, ranking_metrics


def main():
    cfg = config()
    output = ROOT / cfg["data"]["output_directory"]
    lr_result = json.loads((output / "lr_results.json").read_text())
    ebm_result = json.loads((output / "ebm_results.json").read_text())
    lr = pd.read_csv(output / "lr_validation_predictions.csv.gz")
    ebm = pd.read_csv(output / "ebm_validation_predictions.csv.gz")
    metadata = ["patient_id", "hospital", "hour", "label", "onset_hour"]
    pd.testing.assert_frame_equal(lr[metadata], ebm[metadata])
    assert lr_result["test_sets_evaluated"] == ebm_result["test_sets_evaluated"] == []
    scores = {"LR": lr.probability.to_numpy(), "EBM": ebm.probability.to_numpy()}
    labels = lr.label.to_numpy()
    print("Bootstrapping paired LR/EBM differences using identical patient resamples...", flush=True)
    intervals = paired_patient_bootstrap(lr, scores["LR"], scores["EBM"],
                                         cfg["lr"]["bootstrap_samples"], cfg["split"]["random_seed"])
    rows = []
    for name, result in [("LR", lr_result), ("EBM", ebm_result)]:
        observed = ranking_metrics(labels, scores[name])
        for key, value in observed.items():
            np.testing.assert_allclose(value, result["validation"][key], atol=1e-12)
        alarms = result["validation_selected_threshold_alert_metrics"]
        rows.append({"model": name, **observed,
                     **{key: alarms[key] for key in ["event_sensitivity", "alert_ppv",
                         "false_alerts_per_100_patient_hours", "median_true_alert_lead_hours",
                         "detected_event_patients", "event_patients", "total_alert_episodes", "threshold"]},
                     "fit_seconds": result["fit_seconds"]})
    table = pd.DataFrame(rows).set_index("model")
    table.to_csv(output / "lr_ebm_comparison.csv")
    differences = {key: float(table.loc["EBM", key]-table.loc["LR", key])
                   for key in ["auroc", "auprc_average_precision", "brier_score", "event_sensitivity", "alert_ppv"]}
    combined = {"evaluation_split": "hospital_A_validation", "patients": int(lr.patient_id.nunique()),
                "hours": len(lr), "positive_hours": int(labels.sum()),
                "same_patient_split": True, "same_preprocessed_inputs": True,
                "no_test_sets_evaluated": True, "model_metrics": table.reset_index().to_dict("records"),
                "training_metrics": {"LR": lr_result["training"], "EBM": ebm_result["training"]},
                "ebm_minus_lr": differences, "paired_patient_bootstrap_95_ci": intervals,
                "bootstrap_samples": cfg["lr"]["bootstrap_samples"],
                "alert_comparison": "Same validation false-alert ceiling and suppression rule; realized false-alert rates can differ.",
                "limitations": ["Only validation performance; test and external generalization remain unmeasured.",
                                "Each model's alarm threshold was selected and reported on the same validation cohort.",
                                "One fixed configuration per model, no hyperparameter search or CV.",
                                "EBM has no early stopping or bagging; its train-validation gap must be considered.",
                                "Recorded fit times use LR with 4 threads and EBM with 1 job; not a controlled speed benchmark."]}
    save_json(output / "lr_ebm_comparison.json", combined)
    print(table.to_string(), flush=True)
    print("Paired EBM-LR metric intervals:", intervals["ebm_minus_lr"], flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"LR": "#24788c", "EBM": "#bf6546"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    calibration_rows = []
    for name, probability in scores.items():
        precision, recall, _ = precision_recall_curve(labels, probability)
        axes[0].plot(recall, precision, color=colors[name], label=f"{name} AP = {table.loc[name, 'auprc_average_precision']:.3f}")
        fpr, tpr, _ = roc_curve(labels, probability)
        axes[1].plot(fpr, tpr, color=colors[name], label=f"{name} AUROC = {table.loc[name, 'auroc']:.3f}")
        observed, predicted = calibration_curve(labels, probability, n_bins=10, strategy="quantile")
        axes[2].plot(predicted, observed, marker="o", color=colors[name], label=name)
        calibration_rows.extend({"model": name, "predicted": float(a), "observed": float(b)} for a, b in zip(predicted, observed))
    axes[0].axhline(labels.mean(), color="gray", linestyle="--", label=f"Prevalence = {labels.mean():.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    lim = max(max(r["predicted"], r["observed"]) for r in calibration_rows)*1.1
    axes[2].plot([0, lim], [0, lim], color="gray", linestyle="--")
    axes[2].set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Calibration (quantile bins)", xlim=(0, lim), ylim=(0, lim))
    for ax in axes:
        ax.legend()
    fig.suptitle("Hospital A validation - identical patients, features and preprocessing")
    fig.tight_layout()
    fig.savefig(output / "lr_ebm_validation_comparison.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(calibration_rows).to_csv(output / "lr_ebm_calibration_bins.csv", index=False)

    bundle = joblib.load(output / "ebm_baseline.joblib")
    model, processor = bundle["model"], bundle["preprocessor"]
    importance = pd.read_csv(output / "ebm_feature_importance.csv")
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.feature, top.mean_absolute_log_odds_contribution, color=colors["EBM"])
    ax.set(xlabel="Training-weighted mean absolute log-odds contribution", title="Additive EBM: 20 most influential terms")
    fig.tight_layout()
    fig.savefig(output / "ebm_feature_importance.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    shape_rows = []
    for ax, name in zip(axes.flat, importance.head(6).feature):
        term_index = model.term_names_.index(name)
        feature_index = model.term_features_[term_index][0]
        cuts = model.bins_[feature_index][0]
        bounds = model.feature_bounds_[feature_index]
        edges = np.r_[bounds[0], cuts, bounds[1]]
        contribution = model.term_scores_[term_index][1:-1]
        assert len(edges) == len(contribution)+1
        physical_edges = edges*processor.scaler_.scale_[feature_index]+processor.scaler_.mean_[feature_index]
        ax.stairs(contribution, physical_edges, baseline=None, color=colors["EBM"], linewidth=1.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set(title=name, xlabel="Feature value (before standardization)", ylabel="Log-odds contribution")
        shape_rows.extend({"feature": name, "lower": float(a), "upper": float(b), "contribution": float(c)}
                          for a, b, c in zip(physical_edges[:-1], physical_edges[1:], contribution))
    fig.suptitle("EBM shape functions - model associations among correlated features")
    fig.tight_layout()
    fig.savefig(output / "ebm_shape_functions.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(shape_rows).to_csv(output / "ebm_shape_functions.csv", index=False)

    delta_ap = differences["auprc_average_precision"]
    delta_ci = intervals["ebm_minus_lr"]["auprc_average_precision"]
    report = f'''EBM 与 LR：医院 A 验证集对比

数据一致性：训练 469,344 小时、验证 101,888 小时（3,020 位患者），同一份 312 维预处理矩阵。
未重新划分患者、未重新拟合预处理器、未抽样或添加类别权重；LR 已保存的预测已用其模型重新核对。
两类测试集均未评价。训练输入、划分清单和 LR 参考文件的 SHA-256 在 EBM 训练前后完全一致。

EBM：纯加性模型，interactions=0；64 个分箱、学习率 0.03、1,000 轮、叶节点最少 100 个样本。
训练使用所有 A 训练行，关闭随机小时内部验证和早停；只有 1 个固定配置，未调参或做交叉验证。

指标                              LR              EBM
AUROC                         {table.loc['LR','auroc']:.4f}          {table.loc['EBM','auroc']:.4f}
AUPRC (average precision)     {table.loc['LR','auprc_average_precision']:.4f}          {table.loc['EBM','auprc_average_precision']:.4f}
Brier score (越低越好)         {table.loc['LR','brier_score']:.5f}         {table.loc['EBM','brier_score']:.5f}
事件敏感度                     {100*table.loc['LR','event_sensitivity']:.1f}%            {100*table.loc['EBM','event_sensitivity']:.1f}%
报警 PPV                       {100*table.loc['LR','alert_ppv']:.1f}%             {100*table.loc['EBM','alert_ppv']:.1f}%
每 100 患者小时误报             {table.loc['LR','false_alerts_per_100_patient_hours']:.3f}           {table.loc['EBM','false_alerts_per_100_patient_hours']:.3f}
检出事件数 / 238                {int(table.loc['LR','detected_event_patients'])}              {int(table.loc['EBM','detected_event_patients'])}
成功检出事件的中位提前量         {table.loc['LR','median_true_alert_lead_hours']:.1f} 小时          {table.loc['EBM','median_true_alert_lead_hours']:.1f} 小时
训练时间（秒）                 {table.loc['LR','fit_seconds']:.1f}             {table.loc['EBM','fit_seconds']:.1f}

训练集 AUPRC：LR {lr_result['training']['auprc_average_precision']:.4f}，EBM {ebm_result['training']['auprc_average_precision']:.4f}。
验证集 AUPRC：LR {lr_result['validation']['auprc_average_precision']:.4f}，EBM {ebm_result['validation']['auprc_average_precision']:.4f}。
需要结合训练与验证表现判断过拟合；当前 EBM 配置关闭了早停和 bagging，没有经过调参。
训练时间为实际运行记录，LR 使用 4 个线程、EBM 使用 1 个任务，不属于严格控制并行度的速度基准测试。

AUPRC 绝对差值（EBM-LR）：{delta_ap:+.4f}。
按患者配对 bootstrap 95% CI：[{delta_ci['lower']:+.4f}, {delta_ci['upper']:+.4f}]，共 {intervals['valid_resamples']} 次有效重采样。
阳性小时比例 / 无区分能力参考 AUPRC 为 {labels.mean():.4f}。

每个模型均在同一验证集按最多 1 次误报 / 100 患者小时选择阈值，并采用 6 小时重复报警抑制。
两者实际误报率可以不同；提前量只针对成功检出的事件。阈值选择和报告使用同一验证集，属于探索性效果。
纯加性 EBM 的形状曲线提供非线性关联解释；特征间相关性和临床测量流程会影响解释，不能作因果推断。
本轮只比较验证集，不能据此确认跨医院收益或 final 的医院 B 研究目标。
当前结果只代表这个固定配置，不能推广为所有 EBM 都优于或劣于 LR。
下一轮可在 A 训练集内部按患者做早停或交叉验证，并检验更少轮数、更强正则化和 bagging；保持现有外层划分不变。

完整数值：outputs/lr_ebm_comparison.json
比较图：outputs/lr_ebm_validation_comparison.png
EBM 解释图：outputs/ebm_feature_importance.png、outputs/ebm_shape_functions.png
模型：outputs/ebm_baseline.joblib
'''
    (output / "ebm_vs_lr_report.txt").write_text(report)
    print("Saved comparison metrics, paired intervals, report and EBM explanations.", flush=True)


if __name__ == "__main__":
    main()
