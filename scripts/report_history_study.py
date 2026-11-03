"""Report history retention and patient learning curves for all three models."""
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

import numpy as np
import pandas as pd
from sepsis_warning.data import save_json
from sepsis_warning.history_study import settings, bootstrap_ap_contrast


def main():
    directory = ROOT/"outputs/history_study"
    completion = json.loads((directory/"completed.json").read_text())
    study = settings()["study"]
    records = [json.loads(path.read_text()) for path in sorted(directory.glob("*/*_results.json"))]
    assert len(records) == completion["model_results"] == 45
    rows, model_settings = [], {}
    by_key = {}
    for record in records:
        trial, alarms = record["trial"], record["alerts"]
        model_settings[record["model"]] = record["parameters"]
        by_key[(trial["retention_hours"], trial["fraction"], trial["sample_seed"], record["model"])] = record
        rows.append({"retention_hours": trial["retention_hours"], "model": record["model"],
                     "fraction": trial["fraction"], "sample_seed": trial["sample_seed"],
                     "patients": trial["patients"], "event_patients": trial["event_patients"],
                     "training_hours": trial["hours"], "features": trial["features"],
                     "training_auprc": record["training"]["auprc_average_precision"],
                     "validation_auprc": record["validation"]["auprc_average_precision"],
                     "validation_auroc": record["validation"]["auroc"],
                     "validation_brier": record["validation"]["brier_score"],
                     "event_sensitivity": alarms["event_sensitivity"], "alert_ppv": alarms["alert_ppv"],
                     "false_alerts_per_100_patient_hours": alarms["false_alerts_per_100_patient_hours"],
                     "median_lead_hours": alarms["median_true_alert_lead_hours"],
                     "selected_threshold": alarms["threshold"],
                     "inner_selected_trees": record["details"].get("inner_selected_trees"),
                     "fit_seconds": record["fit_seconds"], "cached_reference_model": record["cached_reference_model"],
                     "trial": trial["name"]})
    frame = pd.DataFrame(rows).sort_values(["model", "retention_hours", "fraction", "sample_seed"])
    save_json(directory/"model_settings.json", {"models": model_settings,
              "lightgbm_stopping": settings()["lightgbm_stopping"],
              "note": "Hyperparameters and stopping rules are fixed; fitted stopping iterations may differ."})
    frame.to_csv(directory/"all_results.csv", index=False)
    full = frame.loc[frame.fraction == 1].copy()
    full.to_csv(directory/"full_training_comparison.csv", index=False)
    summary = frame.groupby(["model", "retention_hours", "fraction"]).agg(
        patients=("patients", "first"), event_patients=("event_patients", "first"),
        mean_auprc=("validation_auprc", "mean"), min_auprc=("validation_auprc", "min"),
        max_auprc=("validation_auprc", "max"), mean_auroc=("validation_auroc", "mean"),
        mean_training_auprc=("training_auprc", "mean"), min_training_auprc=("training_auprc", "min"),
        max_training_auprc=("training_auprc", "max"),
        mean_brier=("validation_brier", "mean"), runs=("model", "size")).reset_index()
    summary.to_csv(directory/"learning_curve_summary.csv", index=False)
    validation = pd.read_csv(ROOT/"processed_data/validation_rows.csv.gz")
    probability_cache = {}
    def scores(record):
        key = (record["trial"]["name"], record["model"])
        if key not in probability_cache:
            probability_cache[key] = np.load(directory/key[0]/f"{key[1]}_validation_probability.npy")
        return probability_cache[key]
    def contrast(references, candidates):
        return bootstrap_ap_contrast(validation, [scores(r) for r in references],
                                     [scores(r) for r in candidates],
                                     repetitions=study["bootstrap_samples"])
    effects, sample_effects, model_effects = {}, {}, {}
    for model in study["models"]:
        base = by_key[(6, 1.0, None, model)]
        for retention in [12, 24]:
            candidate = by_key[(retention, 1.0, None, model)]
            key = f"{model}_{retention}h_minus_6h"
            print("Paired history contrast:", key, flush=True)
            effects[key] = contrast([base], [candidate])
        for retention in study["lab_retention_hours"]:
            complete = by_key[(retention, 1.0, None, model)]
            for fraction in [.25, .5]:
                smaller = [by_key[(retention, fraction, seed, model)] for seed in study["sample_seeds"]]
                key = f"{model}_{retention}h_full_minus_{int(fraction*100)}pct"
                print("Patient learning contrast:", key, flush=True)
                sample_effects[key] = contrast(smaller, [complete])
            quarter = [by_key[(retention, .25, seed, model)] for seed in study["sample_seeds"]]
            half = [by_key[(retention, .5, seed, model)] for seed in study["sample_seeds"]]
            key = f"{model}_{retention}h_50pct_minus_25pct"
            sample_effects[key] = contrast(quarter, half)
    for retention in study["lab_retention_hours"]:
        reference = by_key[(retention, 1.0, None, "lr")]
        for model in ["ebm", "lightgbm"]:
            candidate = by_key[(retention, 1.0, None, model)]
            key = f"{model}_minus_lr_{retention}h"
            model_effects[key] = contrast([reference], [candidate])
        model_effects[f"lightgbm_minus_ebm_{retention}h"] = contrast(
            [by_key[(retention, 1.0, None, "ebm")]], [by_key[(retention, 1.0, None, "lightgbm")]])
    save_json(directory/"history_effects.json", effects)
    save_json(directory/"sample_size_effects.json", sample_effects)
    save_json(directory/"model_effects.json", model_effects)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"lr": "Logistic regression", "ebm": "Additive EBM", "lightgbm": "LightGBM"}
    colors = {6: "#24788c", 12: "#bf6546", 24: "#598653"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, model in zip(axes, study["models"]):
        for retention in study["lab_retention_hours"]:
            part = summary.loc[(summary.model == model) & (summary.retention_hours == retention)].sort_values("fraction")
            ax.plot(part.fraction*100, part.mean_auprc, marker="o", label=f"Labs retained {retention}h", color=colors[retention])
            ax.fill_between(part.fraction*100, part.min_auprc, part.max_auprc, color=colors[retention], alpha=.13)
        ax.set(title=names[model], xlabel="Fraction of fixed A-training patients (%)", xticks=[25, 50, 100])
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Validation AUPRC (average precision)")
    fig.suptitle("Patient learning curves - all model settings fixed")
    fig.text(.5, .01, "Shading: range of two nested patient subsamples; full-data point is shared. It is not a confidence interval.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(directory/"patient_learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, model in zip(axes, study["models"]):
        part = summary.loc[(summary.model == model) & (summary.retention_hours == 6)].sort_values("fraction")
        ax.plot(part.fraction*100, part.mean_training_auprc, marker="o", color="#bf6546", label="Training")
        ax.fill_between(part.fraction*100, part.min_training_auprc, part.max_training_auprc, color="#bf6546", alpha=.13)
        ax.plot(part.fraction*100, part.mean_auprc, marker="o", color="#24788c", label="Fixed validation")
        ax.fill_between(part.fraction*100, part.min_auprc, part.max_auprc, color="#24788c", alpha=.13)
        ax.set(title=names[model], xlabel="Fraction of fixed A-training patients (%)", ylabel="AUPRC", xticks=[25,50,100])
        ax.legend()
    fig.suptitle("Training versus validation learning curves - original 6h lab retention")
    fig.text(.5,.01,"Shading: range of two training subsamples, not a confidence interval. Full training cohort is shared.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(directory/"training_validation_learning_curves.png",dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    centers = np.arange(3)
    for offset, retention in zip([-.25, 0, .25], study["lab_retention_hours"]):
        values = [float(full.loc[(full.model == model) & (full.retention_hours == retention), "validation_auprc"].iloc[0]) for model in study["models"]]
        bars = ax.bar(centers+offset, values, width=.24, color=colors[retention], label=f"{retention}h")
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=9)
    ax.axhline(validation.label.mean(), linestyle="--", color="gray", label="Positive-hour prevalence")
    ax.set(xticks=centers, xticklabels=[names[model] for model in study["models"]],
           ylabel="Validation AUPRC", title="Lab-history retention: full training data", ylim=(0, full.validation_auprc.max()*1.25))
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory/"history_model_comparison.png", dpi=180)
    plt.close(fig)

    members = []
    for trial_dir in sorted((ROOT/"processed_data/history_study").glob("history6h_*")):
        if not (trial_dir/"trial.json").exists(): continue
        trial = json.loads((trial_dir/"trial.json").read_text())
        member = pd.read_csv(trial_dir/"patients.csv")
        member["fraction"], member["sample_seed"] = trial["fraction"], trial["sample_seed"]
        members.append(member)
    pd.concat(members, ignore_index=True).to_csv(directory/"patient_samples.csv", index=False)
    protocol = {"retention_intervention": "Only laboratory latest-value expiry changes; vital expiry, rolling statistics and target horizon remain 6h.",
                "sample_design": "Two event-stratified nested patient samples at 25% and 50%; a single shared 100% cohort.",
                "model_settings": "LR and EBM fixed at their previous settings; one prespecified LightGBM baseline.",
                "preprocessing": "Re-fitted using only each selected training subset; all three models share the same transformed matrices per trial.",
                "validation": "Identical 3,020 patients and 101,888 hours in every run; real class proportions preserved.",
                "test_sets_evaluated": [], "uncertainty": f"{study['bootstrap_samples']} paired patient bootstraps, conditional on the two training subsamples.",
                "caveats": ["Only two subsample repetitions; curves cannot establish an ultimate data-size ceiling.",
                            "Event and non-event patient counts vary together; positive-case scarcity is not isolated.",
                            "Fixed hyperparameters may have different effective regularization at different sample sizes.",
                            "Only latest-value retention changes; this does not test longer trend windows or all preprocessing choices.",
                            "Multiple exploratory comparisons on a previously viewed development validation set; no final generalization claim."]}
    save_json(directory/"protocol.json", protocol)
    effect_table = pd.DataFrame([{"comparison": key, **value} for key, value in effects.items()])
    learning_table = pd.DataFrame([{"comparison": key, **value} for key, value in sample_effects.items()])
    model_table = pd.DataFrame([{"comparison": key, **value} for key, value in model_effects.items()])
    availability = pd.read_csv(directory/"history_availability.csv")
    availability = availability.loc[availability.feature.isin(["Creatinine", "WBC", "Platelets", "Lactate"])].pivot(
        index="feature", columns="retention_hours", values="latest_missing_rate")
    best = full.loc[full.validation_auprc.idxmax()]
    observations = [f"全量训练的最高 AUPRC 点估计是 {best['model']} / {int(best['retention_hours'])}h：{best['validation_auprc']:.4f}；这不是对最佳配置的独立测试确认。"]
    if all(effect["lower"] <= 0 <= effect["upper"] for effect in effects.values()):
        observations.append("延长化验保留期限的六项全量比较，其配对 95% 区间全部包含 0，尚无明确改善证据；这也不能证明不同保留期限等效。")
    for model in study["models"]:
        part = summary.loc[(summary.model == model) & (summary.retention_hours == 6)].sort_values("fraction")
        values = " / ".join(f"{v:.4f}" for v in part.mean_auprc)
        gap = part.mean_training_auprc-part.mean_auprc
        observations.append(f"{model} 在原始 6h 策略下，25% / 50% / 100% 患者的验证 AUPRC 为 {values}；训练减验证的 AUPRC 差距由 25% 时的 {gap.iloc[0]:.4f} 变为全量时的 {gap.iloc[-1]:.4f}。")
    observations.append("原始 6h 策略下，LR/EBM 的点估计随患者数上升，但后半段增益较小；LightGBM 的 50% 平均分数高于全量，曲线并非单调。小样本的训练—验证差距更大，但现有结果不足以将整体表现单独归咎于样本量；训练患者组成和停止规则也可能影响曲线。")
    observations.append("患者 bootstrap 区间以已训练模型为条件，不覆盖所有可能训练样本的波动；两次训练抽样不足以确认总体学习曲线形状。")
    report = f'''历史化验保留策略、患者学习曲线与三模型比较

一、实验设计
仅改变化验最近值的保留期限：6、12、24 小时。生命体征保留期限、趋势窗口和预测目标均维持 6 小时。
原始数据、队列和外层患者划分不变；不使用医院 A 测试集或医院 B。
三种模型每次使用相同的训练患者、特征定义、插补/缩放和验证患者。
LR 和 EBM 参数不变；LightGBM 使用预先固定的一套基线参数。
LightGBM 仅在训练患者内部按患者选择树数，再用所有本轮训练患者重拟合。
学习曲线：25%、50%、100% 患者；25%/50% 分别使用两组分层嵌套抽样。
插补、截尾和标准化只从本轮被选中的训练患者拟合；不借用全训练集的统计量。
共 15 组数据条件、45 份模型结果。6h 全量 LR/EBM 经核对后复用原模型。

主要观察
{chr(10).join(observations)}

二、全训练集下的三模型比较
{full[['retention_hours','model','patients','event_patients','validation_auroc','validation_auprc','validation_brier','event_sensitivity','alert_ppv','false_alerts_per_100_patient_hours']].to_string(index=False)}

三、延长保留期限相对 6h 的 AUPRC 差值
正值表示延长保留期限更好；区间为按患者配对 bootstrap 的 95% CI。
{effect_table[['comparison','difference','lower','upper']].to_string(index=False)}

四、患者学习曲线
{summary.to_string(index=False)}

五、不同患者样本量之间的 AUPRC 差值
较小样本取两次抽样模型的 AUPRC 均值，不是把预测平均成一个集成模型。
{learning_table[['comparison','difference','lower','upper']].to_string(index=False)}

六、全训练集三模型之间的 AUPRC 差值
比较名中的前者减去后者；区间为相同验证患者上的配对 bootstrap 95% CI。
{model_table[['comparison','difference','lower','upper']].to_string(index=False)}

七、训练集中最近化验值的缺失率
行是化验，列是保留小时数；0.5 表示 50% 的合格小时没有保留期限内的最近化验值。
{availability.to_string()}

八、解释边界
缺失率降低不等于预测一定改善，旧值可能有信息，也可能过时；应按各模型的配对差值判断。
学习曲线只反映当前固定设置和现有样本范围；两组抽样不能证明更多数据必然有效或无效。
曲线同时改变阳性与阴性患者数量，不能单独判定阳性病例稀少或阴性病例数量的作用。
本轮仅测试最近化验值保留期限，未测试 12/24h 趋势等其他特征设计。
外层验证集已在此前开发中查看；多项比较是探索性分析，最终结论仍需保留的测试集。
报警阈值仍在本验证集按每 100 患者小时最多 1 次误报选择，并有 6h 抑制期，报警点估计可能偏乐观。

文件：all_results.csv、full_training_comparison.csv、learning_curve_summary.csv。
区间：history_effects.json、sample_size_effects.json、model_effects.json。
图：history_model_comparison.png、patient_learning_curves.png、training_validation_learning_curves.png。
模型和处理后的数据保留在本地 outputs/history_study/ 与 processed_data/history_study/。
'''
    (directory/"experiment_report.txt").write_text("\n".join(line.rstrip() for line in report.splitlines())+"\n")
    snapshot = ROOT/"reports/history_study"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ["all_results.csv", "full_training_comparison.csv", "learning_curve_summary.csv", "history_availability.csv",
                 "history_effects.json", "sample_size_effects.json", "model_effects.json", "patient_samples.csv",
                 "protocol.json", "model_settings.json", "input_fingerprints.json", "completed.json", "experiment_report.txt",
                 "history_model_comparison.png", "patient_learning_curves.png", "training_validation_learning_curves.png",
                 "ap_implementation_check.json"]:
        shutil.copy2(directory/name, snapshot/name)
    print(full[["retention_hours", "model", "validation_auprc", "validation_auroc"]].to_string(index=False), flush=True)
    print("Report and curated snapshot complete.", flush=True)


if __name__ == "__main__":
    main()
