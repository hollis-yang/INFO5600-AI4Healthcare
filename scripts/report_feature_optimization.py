"""Summarize internal feature screening and the three frozen validation selections."""
from __future__ import annotations

import json
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/"outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT/"outputs/.cache"))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sepsis_warning.data import COLUMNS, FEATURE_NAMES, save_json, environment
from sepsis_warning.feature_engineering import LABS, feature_names
from sepsis_warning.history_study import bootstrap_ap_contrast
from sepsis_warning.evaluation import ranking_metrics
from sepsis_warning.optimization import OUTPUT, PROCESSED, REPORT, settings
from sepsis_warning.augmentation import ADDED_OUTPUT, ADDED_REPORT, ADDED_PROFILES


def main():
    include_added = "--include-augmentation" in sys.argv
    target, snapshot = (ADDED_OUTPUT, ADDED_REPORT) if include_added else (OUTPUT, REPORT)
    completion = json.loads((target/"completed.json").read_text())
    selection = json.loads((target/"selection.json").read_text())
    study = settings()
    jobs = json.loads((OUTPUT/"cv_manifest.json").read_text())
    if include_added:
        jobs += json.loads((ADDED_OUTPUT/"cv_manifest.json").read_text())
        study["study"]["profiles"] += list(ADDED_PROFILES)
        if any(j["profile"] == "baseline_context_splines" for j in jobs):
            study["study"]["profiles"].append("baseline_context_splines")
    train_rows = pd.read_csv(ROOT/"processed_data/train_rows.csv.gz")
    fold_records, pooled, full_scores = [], {}, {}
    for job in jobs:
        result = json.loads((OUTPUT/"cv"/job["case"]/f"{job['model']}_results.json").read_text())
        metrics, meta = result["evaluation"], result["case"]
        fold_records.append({"profile": job["profile"], "model": job["model"], "fold": job["fold"],
                             "auprc": metrics["auprc_average_precision"], "auroc": metrics["auroc"],
                             "brier": metrics["brier_score"], "training_auprc": result["training"]["auprc_average_precision"],
                             "features": meta["features"], "training_patients": meta["training_patients"],
                             "training_event_patients": meta["training_event_patients"], "fit_seconds": result["fit_seconds"],
                             "inner_selected_trees": result["details"].get("inner_selected_trees"),
                             "warning_count": len(result["warnings"]), "case": job["case"], "signature": result["signature"]})
        key = (job["model"], job["profile"])
        if key not in pooled:
            pooled[key] = np.full(len(train_rows), np.nan)
        indices = np.load(PROCESSED/"cases"/job["case"]/"evaluation_indices.npy")
        assert np.isnan(pooled[key][indices]).all()
        pooled[key][indices] = np.load(OUTPUT/"cv"/job["case"]/f"{job['model']}_probability.npy")
    folds = pd.DataFrame(fold_records)
    summary = folds.groupby(["model", "profile"], sort=False).agg(
        mean_fold_auprc=("auprc", "mean"), fold_sd_auprc=("auprc", "std"),
        mean_fold_auroc=("auroc", "mean"), mean_training_auprc=("training_auprc", "mean"),
        mean_fold_brier=("brier", "mean"), features_min=("features", "min"), features_max=("features", "max"),
        mean_fit_seconds=("fit_seconds", "mean")).reset_index()
    scores = []
    for (model, profile), probability in pooled.items():
        assert np.isfinite(probability).all()
        np.save(target/f"{model}_{profile}_oof_probability.npy", probability)
        metrics = ranking_metrics(train_rows.label.to_numpy(), probability)
        scores.append({"model": model, "profile": profile, "pooled_oof_auprc": metrics["auprc_average_precision"],
                       "pooled_oof_auroc": metrics["auroc"]})
    summary = summary.merge(pd.DataFrame(scores), on=["model", "profile"]).merge(
        pd.DataFrame(selection["cv_comparisons"]).drop(columns="mean_fold_auprc"), on=["model", "profile"])
    assert len(summary) == len(pooled) == len(selection["cv_comparisons"])
    summary["selected"] = [selection["selected_profiles"][m] == p for m, p in zip(summary.model, summary.profile)]
    folds.to_csv(target/"cv_fold_results.csv", index=False)
    summary.to_csv(target/"cv_summary.csv", index=False)
    availability = []
    original = np.load(ROOT/"processed_data/train_features.npy", mmap_mode="r")
    modified = np.load(PROCESSED/"train_combined_features.npy", mmap_mode="r")
    names = feature_names("combined")
    for variable in LABS:
        gap = modified[:, names.index(f"{variable}__last2_interval_24h")]
        observed = gap[np.isfinite(gap)]
        availability.append({"variable": variable,
            "original_delta6h_missing_rate": float(np.isnan(original[:, FEATURE_NAMES.index(f"{variable}__delta_6h")]).mean()),
            "new_last2_24h_missing_rate": float(np.isnan(gap).mean()),
            "median_last2_interval_when_available": float(np.median(observed)) if len(observed) else None})
    pd.DataFrame(availability).to_csv(target/"lab_trend_availability.csv", index=False)
    preprocessing = pd.read_csv(ROOT/"outputs/preprocessing_parameters.csv").set_index("feature")
    outliers = []
    for variable in COLUMNS[:34]:
        name = variable+"__latest"
        if name not in preprocessing.index:
            continue
        values = original[:, FEATURE_NAMES.index(name)]
        available = np.isfinite(values)
        low,high = preprocessing.loc[name,["clip_lower","clip_upper"]]
        clipped = available & ((values < low)|(values > high))
        def rate_for(label):
            mask = train_rows.label.eq(label)
            denominator = available[mask].sum()
            return float(clipped[mask].sum()/denominator) if denominator else None
        outliers.append({"variable":variable,"minimum":float(np.min(values[available])),"maximum":float(np.max(values[available])),
                         "clip_lower":low,"clip_upper":high,"observed_hours":int(available.sum()),
                         "clipped_fraction_observed":float(clipped.sum()/available.sum()),
                         "clipped_fraction_positive_observed":rate_for(1),
                         "clipped_fraction_negative_observed":rate_for(0)})
    pd.DataFrame(outliers).to_csv(target/"outlier_audit.csv",index=False)
    runtime = environment()
    runtime["packages"].update({"interpret-core":version("interpret-core"),"lightgbm":version("lightgbm")})
    save_json(target/"runtime_environment.json",runtime)
    validation = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    labels = validation.label.to_numpy()
    final_records, effects, model_effects = [], {}, {}
    previous_best = pd.read_csv(ROOT/"reports/history_study/full_training_comparison.csv").groupby("model").validation_auprc.max()
    for job in json.loads((target/"validation_manifest.json").read_text()):
        model = job["model"]
        directory = OUTPUT/"validation"/job["case"]
        result = json.loads((directory/f"{model}_results.json").read_text())
        base = json.loads((ROOT/f"outputs/history_study/history6h_full/{model}_results.json").read_text())
        baseline_probability = np.load(ROOT/f"outputs/history_study/history6h_full/{model}_validation_probability.npy")
        probability = np.load(directory/f"{model}_probability.npy")
        full_scores[model] = (baseline_probability, probability)
        print("Patient-paired validation contrast:", model, flush=True)
        effects[model] = bootstrap_ap_contrast(validation, [baseline_probability], [probability],
                                             repetitions=study["study"]["bootstrap_samples"])
        final_records.append({"model": model, "selected_profile": job["profile"], "features": result["case"]["features"],
            "baseline_6h_auprc": base["validation"]["auprc_average_precision"],
            "previous_history_best_auprc": float(previous_best[model]),
            "optimized_auprc": result["evaluation"]["auprc_average_precision"],
            "ap_difference": effects[model]["difference"], "difference_lower": effects[model]["lower"], "difference_upper": effects[model]["upper"],
            "baseline_auroc": base["validation"]["auroc"], "optimized_auroc": result["evaluation"]["auroc"],
            "baseline_brier": base["validation"]["brier_score"], "optimized_brier": result["evaluation"]["brier_score"],
            "training_auprc": result["training"]["auprc_average_precision"],
            "baseline_event_sensitivity": base["alerts"]["event_sensitivity"], "event_sensitivity": result["alerts"]["event_sensitivity"],
            "baseline_alert_ppv": base["alerts"]["alert_ppv"], "alert_ppv": result["alerts"]["alert_ppv"],
            "false_alerts_per_100_patient_hours": result["alerts"]["false_alerts_per_100_patient_hours"],
            "median_lead_hours": result["alerts"]["median_true_alert_lead_hours"],
            "selected_threshold": result["alerts"]["threshold"], "case": job["case"]})
    for left, right in [("ebm", "lr"), ("lightgbm", "lr"), ("lightgbm", "ebm")]:
        model_effects[f"{left}_minus_{right}"] = bootstrap_ap_contrast(validation,
            [full_scores[right][1]], [full_scores[left][1]], repetitions=study["study"]["bootstrap_samples"])
    final = pd.DataFrame(final_records).sort_values("model")
    final.to_csv(target/"validation_comparison.csv", index=False)
    save_json(target/"validation_effects.json", effects)
    save_json(target/"selected_model_effects.json", model_effects)
    archive = {str(path.relative_to(ROOT)):json.loads(path.read_text()) for path in
               sorted(OUTPUT.glob("*/*/*_results.json"))}
    save_json(target/"model_metadata_archive.json",{"available_model_results":len(archive),"results":archive})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model_names = {"lr": "Logistic regression", "ebm": "Additive EBM", "lightgbm": "LightGBM"}
    short = {"baseline": "Baseline", "native_missing": "Native missing", "lab_trends": "Lab trends",
             "vital_dynamics": "Vital dynamics", "clinical_combinations": "Combinations",
             "without_context_time": "No time / unit", "without_clipping": "No clipping", "lr_splines": "LR splines",
             "preserve_baseline":"Add dynamics", "preserve_baseline_splines":"Add + splines",
             "baseline_context_splines":"Context splines"}
    colors = {"lr": "#24788c", "ebm": "#bf6546", "lightgbm": "#598653"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, model in zip(axes, study["study"]["models"]):
        part = summary.loc[summary.model == model].set_index("profile")
        order = [p for p in study["study"]["profiles"] if p in part.index]
        for position, profile in enumerate(order):
            values = folds.loc[(folds.model == model) & (folds.profile == profile)].sort_values("fold").auprc
            ax.scatter(position+np.linspace(-.12,.12,3), values, s=18, alpha=.55, color=colors[model])
            ax.scatter(position, part.loc[profile,"mean_fold_auprc"], marker="D", s=45,
                       edgecolor="black" if part.loc[profile,"selected"] else "none", color=colors[model])
        ax.axhline(part.loc["baseline","mean_fold_auprc"], linestyle="--", color="gray", linewidth=1)
        ax.set(title=model_names[model], xticks=np.arange(len(order)), xticklabels=[short[p] for p in order])
        ax.tick_params(axis="x", rotation=50)
    axes[0].set_ylabel("AUPRC on held-out training patients")
    fig.suptitle("Feature screening inside A training - three patient folds")
    fig.text(.5,.01,"Dots: individual folds. Diamonds: fold mean; black outline: prespecified CV selection. Outer validation is not used for selection.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(target/"cv_feature_comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,4.8))
    ordered = final.set_index("model").loc[study["study"]["models"]]
    positions = np.arange(3)
    for ax, left, right, title in [(axes[0], "baseline_6h_auprc", "optimized_auprc", "Validation AUPRC"),
                                  (axes[1], "baseline_event_sensitivity", "event_sensitivity", "Event sensitivity under false-alert ceiling")]:
        for offset, column, label, color in [(-.18,left,"Original 6h baseline","#8aa4ad"),(.18,right,"CV-selected features","#bf6546")]:
            bars = ax.bar(positions+offset, ordered[column], width=.34, label=label, color=color)
            ax.bar_label(bars, labels=[f"{v:.4f}" if column.endswith("auprc") else f"{v:.1%}" for v in ordered[column]], padding=3,fontsize=9)
        ax.set(xticks=positions,xticklabels=[model_names[m] for m in ordered.index],title=title,ylim=(0,max(ordered[[left,right]].max())*1.22))
        ax.legend(fontsize=8)
    fig.suptitle("Original A validation patients - all classifiers retain their settings")
    fig.tight_layout()
    fig.savefig(target/"validation_model_comparison.png",dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    for model, (base, selected) in full_scores.items():
        for probability, style, suffix in [(base,"--","original"),(selected,"-","selected")]:
            fpr,tpr,_ = roc_curve(labels,probability)
            precision,recall,_ = precision_recall_curve(labels,probability)
            axes[0].plot(fpr,tpr,color=colors[model],linestyle=style,label=f"{model}: {suffix}")
            axes[1].plot(recall,precision,color=colors[model],linestyle=style,label=f"{model}: {suffix}")
    axes[0].plot([0,1],[0,1],color="gray",linestyle=":")
    axes[1].axhline(labels.mean(),color="gray",linestyle=":")
    axes[0].set(xlabel="False positive rate",ylabel="True positive rate",title="ROC")
    axes[1].set(xlabel="Recall",ylabel="Precision",title="Precision-recall",ylim=(0,1))
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target/"validation_ranking_curves.png",dpi=180)
    plt.close(fig)

    selection_note = "选择文件在新外层验证模型评估前保存。每个模型只评估这一份 CV 选中的方案，没有根据外层验证分数重新换方案。"
    if include_added:
        added_design = json.loads((ADDED_OUTPUT/"design.json").read_text())
        timing = "补充对照在初始 A 验证模型评估前固定。" if not added_design["initial_outer_validation_manifest_present_at_design"] else "固定补充对照时已有第一阶段验证文件。"
        selection_note = timing+"第一阶段和最终阶段均保留评估结果；最终选择仅汇总两阶段训练 CV 分数，初始外层验证分数没有参与选择。"
    addition_note = ("preserve_baseline：保留全部 318 项原始特征，追加 141 项新化验趋势、生命体征动态和组合特征；树模型保留原生缺失。\npreserve_baseline_splines：在保留原始特征的增量方案上追加同样的 LR 样条。" if include_added else "")
    if include_added and (ADDED_OUTPUT/"context_spline_design.json").exists():
        addition_note += "\nbaseline_context_splines：仅 LR，在完整原始特征上，对 HR/Resp/Temp/SBP/ICULOS/HospAdmTime 增加折内拟合的样条；分位数节点重复时使用训练范围内等距节点。该补充在第一阶段验证已完成后根据训练 CV 消融结果提出，其选择仍只使用训练 CV。"
    observations = []
    for row in final.itertuples():
        if row.difference_lower > 0:
            finding = "患者配对区间支持相对 6h 基线的改善"
        elif row.difference_upper < 0:
            finding = "患者配对区间支持相对 6h 基线的下降"
        else:
            finding = "患者配对区间包含 0，尚无明确改善证据"
        observations.append(f"{row.model} 选择 {row.selected_profile}：A 验证 AUPRC {row.baseline_6h_auprc:.4f} -> {row.optimized_auprc:.4f}，差值 {row.ap_difference:+.4f}，95% CI [{row.difference_lower:+.4f}, {row.difference_upper:+.4f}]；{finding}。")
        if row.optimized_auprc < row.previous_history_best_auprc-1e-12:
            observations.append(f"{row.model} 的本轮点估计尚未超过上轮保留期限实验的最高点估计 {row.previous_history_best_auprc:.4f}；上轮最高值也属于已查看验证集上的探索性结果。")
    text = f'''第一轮数据处理与特征优化：三模型、患者交叉验证

主要观察
{chr(10).join(observations)}

实验协议
原始队列、患者划分、未来 6h 预测目标和最近值 6h 保留期限不变。
LR、EBM、LightGBM 参数与此前 6h 基线完全一致，没有随结果调整模型参数。
在原有 A 训练患者中按发病状态分层做三折患者交叉验证，每折重新拟合插补、截尾、标准化和样条。
模型内部停止仍按患者分组；内部停止患者共享本折训练分区的无监督预处理，外层 CV 患者不参与拟合。
先完成 {completion['distinct_cv_models']} 个独立 CV 模型，产生 {completion['cv_comparisons']} 行比较；LR 的原生缺失对照与基线相同，复用结果。
选择规则预先固定：平均折 AUPRC 比基线高至少 0.001，且至少两折改善；符合者中取平均折 AUPRC 最高者，否则保留基线。
{selection_note}
A 测试集和医院 B 未使用。

特征方案
baseline：原始特征和中位数/众数插补。
native_missing：仅 EBM/LightGBM 保留 NaN，有限值的截尾和缩放保持一致；LR 与基线相同。
lab_trends：在 native_missing 上，将 26 种化验的 6h 均值/最小/最大/变化量，替换为过去 24h 内最后两次真实测量的差值、每小时变化率、间隔和 24h 测量次数。
vital_dynamics：在 lab_trends 上，加入七种生命体征的 3/12h 斜率、12h 波动、相对过去 12h 本人中位数的偏离和 12h 测量次数。
clinical_combinations：在 vital_dynamics 上，加入 HR/SBP 和 SBP-DBP；测量不超过 3h 旧、两项时差不超过 2h。
without_context_time：从 clinical_combinations 去除 ICULOS/HospAdmTime/Unit1/Unit2 及其缺失标记，保留其余采样特征。
without_clipping：从 clinical_combinations 去除分位数截尾，其他流程不变；这是敏感性分析，不代表极端值都正确。
lr_splines：仅 LR；在 clinical_combinations 上，对 HR/Resp/Temp/SBP 最近值增加训练折拟合的少量样条。
{addition_note}
所有趋势使用原始观测值计算；不对填补后的常数计算变化量，不回填未来数值。

交叉验证结果
折间标准差反映三折差异，不是置信区间；OOF 汇总值不参与筛选。
{summary.to_string(index=False)}

冻结选择
{json.dumps(selection['selected_profiles'],ensure_ascii=False,indent=2)}

A 验证集结果
baseline_6h 是受控基线；previous_history_best 是上轮 6/12/24h 中最高的验证点估计，仅供透明对照，不参与本轮选择。
AP 差值区间使用相同验证患者的 300 次配对 bootstrap，以已训练模型为条件。
{final.to_string(index=False)}

化验趋势可用性
{pd.DataFrame(availability).to_string(index=False)}

解释边界
这些候选方案检验了不同工程假设，不能预先保证三个模型同时改善。
三折训练集合相互重叠，折分数不是独立重复试验；多项 CV 筛选可能产生选择偏差。
此前 A 验证集已被多次查看，本轮结果仍为开发性验证；最终泛化和跨医院结论需要保留测试集。
报警阈值仍在 A 验证集按每 100 合格患者小时最多 1 次误报、6h 抑制选择，同集合报警点估计可能偏乐观。
这一轮没有增加患者，也不隔离阳性病例数量的作用，不能据此宣称样本量充足或已达上限。
模型和大矩阵保留在本地 outputs/feature_optimization/ 与 processed_data/feature_optimization/。
'''
    (target/"experiment_report.txt").write_text("\n".join(line.rstrip() for line in text.splitlines())+"\n")
    snapshot.mkdir(parents=True,exist_ok=True)
    for name in ["protocol.json","input_fingerprints.json","patient_cv_folds.csv","selection.json","completed.json",
                 "cv_fold_results.csv","cv_summary.csv","validation_comparison.csv","validation_effects.json",
                 "selected_model_effects.json","lab_trend_availability.csv","experiment_report.txt",
                 "outlier_audit.csv","runtime_environment.json","model_metadata_archive.json","cv_feature_comparison.png","validation_model_comparison.png","validation_ranking_curves.png"]:
        source = OUTPUT/name if name in ["protocol.json","input_fingerprints.json","patient_cv_folds.csv"] else target/name
        shutil.copy2(source,snapshot/name)
    if include_added:
        shutil.copy2(ADDED_OUTPUT/"design.json",snapshot/"added_design.json")
        shutil.copy2(OUTPUT/"selection.json",snapshot/"initial_selection.json")
        shutil.copy2(OUTPUT/"completed.json",snapshot/"initial_completed.json")
        for name in ["context_spline_design.json","selection_before_context_splines.json","completed_before_context_splines.json"]:
            if (ADDED_OUTPUT/name).exists():shutil.copy2(ADDED_OUTPUT/name,snapshot/name)
    if (OUTPUT/"metadata_recovery.json").exists():shutil.copy2(OUTPUT/"metadata_recovery.json",snapshot/"metadata_recovery.json")
    print(final.to_string(index=False),flush=True)
    print("Feature optimization report complete.",flush=True)


if __name__ == "__main__":
    main()
