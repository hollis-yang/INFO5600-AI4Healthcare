"""Report the frozen A+B experiment, including matched-size and hospital controls."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME",str(ROOT/"outputs/.cache"))

import numpy as np
import pandas as pd
from sepsis_warning.ab_expansion import OUTPUT,PROCESSED,REPORT,EVALUATION_SPLITS,result_directory,content_hash,write_json,settings
from sepsis_warning.data import FEATURE_NAMES,COLUMNS
from sepsis_warning.history_study import bootstrap_ap_contrast
from threadpoolctl import threadpool_limits

MODEL_NAMES = {"lr":"LR","ebm":"EBM","lightgbm":"LightGBM"}
SPLIT_NAMES = {"A_validation":"A 验证集","B_validation":"B 验证集","B_test":"B 留出测试集"}


def probability(cohort,profile,model,split):
    return np.load(result_directory(cohort+"__"+profile)/f"{model}_{split}_probability.npy")


def markdown(frame):
    # Avoid an optional tabulate dependency in the reproducible reporting environment.
    def cell(value):
        if isinstance(value,(float,np.floating)):return f"{value:.4f}"
        return str(value)
    lines = ["| "+" | ".join(frame.columns)+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]
    lines += ["| "+" | ".join(cell(value) for value in row)+" |" for row in frame.itertuples(index=False,name=None)]
    return "\n".join(lines)


def distribution_audit():
    a = np.load(ROOT/"processed_data/train_features.npy",mmap_mode="r")
    b = np.load(ROOT/"processed_data/external_test_features.npy",mmap_mode="r")
    split = pd.read_csv(PROCESSED/"B_patient_splits.csv")
    rows = pd.read_csv(ROOT/"processed_data/external_test_rows.csv.gz")
    b_indices = np.flatnonzero(rows.patient_id.isin(split.loc[split.expansion_split.eq("B_train"),"patient_id"]))
    records = []
    for name in COLUMNS[:34]:
        index = FEATURE_NAMES.index(name+"__latest")
        av,bv = a[:,index],b[b_indices,index]
        afin,bfin = av[np.isfinite(av)],bv[np.isfinite(bv)]
        record = {"variable":name,"A_train_latest_missing_rate":float(np.isnan(av).mean()),
                  "B_train_latest_missing_rate":float(np.isnan(bv).mean())}
        record["B_minus_A_missing_rate"] = record["B_train_latest_missing_rate"]-record["A_train_latest_missing_rate"]
        for prefix,values in [("A",afin),("B",bfin)]:
            for label,q in [("q05",.05),("median",.5),("q95",.95)]:
                record[f"{prefix}_observed_{label}"] = float(np.quantile(values,q)) if len(values) else None
        records.append(record)
    pd.DataFrame(records).to_csv(OUTPUT/"training_hospital_shift.csv",index=False)
    train = pd.read_csv(PROCESSED/"cases/A_B_full__common/train_rows.csv.gz")
    patients = train.groupby("patient_id").agg(hospital=("hospital","first"),has_event=("label","max"),eligible_hours=("hour","size"))
    patient_records = []
    for hospital,part in patients.groupby("hospital"):
        patient_records.append({"hospital":hospital,"patients":len(part),"event_patients":int(part.has_event.sum()),
                                "event_patient_prevalence":float(part.has_event.mean()),
                                "median_eligible_hours":float(part.eligible_hours.median()),
                                "p90_eligible_hours":float(part.eligible_hours.quantile(.9))})
    pd.DataFrame(patient_records).to_csv(OUTPUT/"training_patient_composition.csv",index=False)


def compute_effects(only_available=False):
    effects = []
    repetitions = settings()["study"]["bootstrap_samples"]
    cache = OUTPUT/"effect_cache";cache.mkdir(exist_ok=True)
    for split in EVALUATION_SPLITS:
        rows = pd.read_csv(PROCESSED/f"{split}_rows.csv.gz")
        row_hash = content_hash(PROCESSED/f"{split}_rows.csv.gz")
        for profile,models in [("common",["lr","ebm","lightgbm"]),("enhanced_lightgbm",["lightgbm"])]:
            for model in models:
                candidate = result_directory("A_B_full__"+profile)/f"{model}_{split}_probability.npy"
                references = {"full_minus_A_only":["A_only"]}
                if profile == "common":references["full_minus_mean_matched"] = ["A_B_matched_seed2026","A_B_matched_seed2027"]
                for name,cohorts in references.items():
                    paths = [result_directory(cohort+"__"+profile)/f"{model}_{split}_probability.npy" for cohort in cohorts]
                    if only_available and not all(p.exists() for p in [candidate,*paths]):continue
                    signature = {"rows":row_hash,"probabilities":[content_hash(p) for p in [candidate,*paths]],
                                 "repetitions":repetitions,"seed":42,"analysis_source":content_hash(ROOT/"src/sepsis_warning/history_study.py")}
                    target = cache/f"{split}_{profile}_{model}_{name}.json"
                    if target.exists() and json.loads(target.read_text())["signature"] == signature:
                        result = json.loads(target.read_text())["result"]
                    else:
                        print(f"Patient-paired AP contrast {split}/{profile}/{model}/{name}",flush=True)
                        with threadpool_limits(limits=1):
                            result = bootstrap_ap_contrast(rows,[np.load(p) for p in paths],[np.load(candidate)],repetitions=repetitions)
                        write_json(target,{"signature":signature,"result":result})
                    effects.append({"evaluation":split,"profile":profile,"model":model,"contrast":name,**result})
    return pd.DataFrame(effects)


def make_plots(metrics,effects,training):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    models = ["lr","ebm","lightgbm"]
    colors = {"A_only":"#367a95","A_B_full":"#c27543"}
    fig,axes = plt.subplots(1,2,figsize=(10,4.2),sharey=False)
    for ax,split,title in zip(axes,["A_validation","B_test"],["Original A validation","Held-out B patients"]):
        data = metrics.loc[metrics.profile.eq("common")&metrics.evaluation.eq(split)]
        x = np.arange(3)
        for offset,cohort,label in [(-.18,"A_only","Train A only"),(.18,"A_B_full","Train A+B full")]:
            values = data.loc[data.cohort.eq(cohort)].set_index("model").loc[models,"auprc_average_precision"]
            ax.bar(x+offset,values,width=.34,label=label,color=colors[cohort])
        prevalence = data.positive_hour_prevalence.iloc[0]
        ax.axhline(prevalence,color="gray",linestyle="--",linewidth=1,label="Positive-hour prevalence")
        ax.set(xticks=x,xticklabels=[MODEL_NAMES[m] for m in models],ylabel="AUPRC (average precision)",title=title,
               ylim=(0,float(metrics.auprc_average_precision.max())*1.4))
        ax.legend(fontsize=8,loc="upper center")
    fig.suptitle("Same 312 features and classifier settings; fixed evaluation patients")
    fig.tight_layout()
    fig.savefig(OUTPUT/"auprc_comparison.png",dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(10,4.2))
    for ax,split,title in zip(axes,["A_validation","B_test"],["Original A validation","Held-out B patients"]):
        data = metrics.loc[metrics.profile.eq("common")&metrics.evaluation.eq(split)]
        for model,color in zip(models,["#367a95","#c27543","#648a55"]):
            matched = data.loc[data.model.eq(model)&data.cohort.str.contains("matched"),"auprc_average_precision"]
            full = data.loc[data.model.eq(model)&data.cohort.eq("A_B_full"),"auprc_average_precision"].iloc[0]
            ns = [int(training.loc[training.cohort.eq(c),"patients"].iloc[0]) for c in ["A_only","A_B_full"]]
            ax.plot(ns,[matched.mean(),full],marker="o",label=MODEL_NAMES[model],color=color)
            ax.vlines(ns[0],matched.min(),matched.max(),color=color,linewidth=3)
        ax.set(title=title,xlabel="Mixed A+B training patients",ylabel="AUPRC",xticks=ns)
    fig.suptitle("Mixed-source size control: two matched draws versus the full pool")
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.045),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.005,"Short vertical lines are the range of two training draws, not confidence intervals.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1));fig.savefig(OUTPUT/"matched_size_comparison.png",dpi=180);plt.close(fig)
    part = effects.loc[effects.profile.eq("common")&effects.contrast.eq("full_minus_A_only")&effects.evaluation.isin(["A_validation","B_test"])]
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,split,title in zip(axes,["A_validation","B_test"],["A validation","B held-out test"]):
        view = part.loc[part.evaluation.eq(split)].set_index("model").loc[models]
        ax.errorbar(view.difference,np.arange(3),xerr=np.vstack([view.difference-view.lower,view.upper-view.difference]),fmt="o",capsize=4,color="#367a95")
        ax.axvline(0,color="gray",linestyle="--")
        ax.set(yticks=np.arange(3),yticklabels=[MODEL_NAMES[m] for m in models],title=title,xlabel="AUPRC: full A+B minus A only")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x",labelsize=9)
    fig.suptitle("Patient-paired bootstrap 95% intervals (300 resamples)")
    fig.tight_layout();fig.savefig(OUTPUT/"expansion_effects.png",dpi=180);plt.close(fig)


def main():
    completed = json.loads((OUTPUT/"completed.json").read_text())
    jobs = json.loads((OUTPUT/"model_manifest.json").read_text())
    protocol = json.loads((OUTPUT/"protocol.json").read_text())
    records,archive = [],{}
    for job in jobs:
        path = result_directory(job["case"])/f"{job['model']}_results.json"
        result = json.loads(path.read_text());archive[str(path.relative_to(ROOT))] = result
        for split,metrics in result["evaluation"].items():
            records.append({**job,"evaluation":split,**metrics,**result["alerts"][split],
                            "training_patients":result["case"]["training_patients"],
                            "training_event_patients":result["case"]["training_event_patients"],
                            "features":result["case"]["features"],"fit_seconds":result["fit_seconds"],
                            "reused_A_only_model":result["reused_A_only_model"],
                            "training_auprc":result["training"]["auprc_average_precision"],
                            "inner_selected_trees":result["details"].get("inner_selected_trees"),
                            "warning_count":len(result["warnings"])})
    metrics = pd.DataFrame(records)
    metrics.to_csv(OUTPUT/"all_model_metrics.csv",index=False)
    write_json(OUTPUT/"model_metadata_archive.json",{"available_model_results":len(archive),"results":archive})
    effects = compute_effects()
    effects.to_csv(OUTPUT/"paired_ap_effects.csv",index=False)
    training = pd.read_csv(OUTPUT/"training_cohorts.csv")
    evaluation = pd.read_csv(OUTPUT/"evaluation_cohorts.csv")
    distribution_audit()
    make_plots(metrics,effects,training)
    main = metrics.loc[metrics.profile.eq("common")&metrics.cohort.isin(["A_only","A_B_full"])&metrics.evaluation.isin(["A_validation","B_test"])]
    comparison = main.pivot(index=["model","evaluation"],columns="cohort",values="auprc_average_precision").reset_index()
    comparison["change"] = comparison.A_B_full-comparison.A_only
    comparison.to_csv(OUTPUT/"primary_comparison.csv",index=False)
    alarms = main[["model","cohort","evaluation","event_sensitivity","alert_ppv","false_alerts_per_100_patient_hours","median_true_alert_lead_hours"]]
    alarms.to_csv(OUTPUT/"primary_alert_comparison.csv",index=False)
    secondary = metrics.loc[metrics.profile.eq("enhanced_lightgbm")&metrics.evaluation.isin(["A_validation","B_test"]),
                            ["model","cohort","evaluation","auprc_average_precision","auroc","brier_score"]]
    secondary.to_csv(OUTPUT/"secondary_lightgbm.csv",index=False)
    compact_effects = effects.loc[effects.evaluation.isin(["A_validation","B_test"]),
                                 ["profile","model","evaluation","contrast","difference","lower","upper"]]
    size_control = metrics.loc[metrics.profile.eq("common")&metrics.evaluation.isin(["A_validation","B_test"])&metrics.cohort.ne("A_only")].pivot(
        index=["model","evaluation"],columns="cohort",values="auprc_average_precision").reset_index()
    size_control["matched_mean"] = size_control[["A_B_matched_seed2026","A_B_matched_seed2027"]].mean(axis=1)
    size_control["full_minus_matched_mean"] = size_control.A_B_full-size_control.matched_mean
    size_control.to_csv(OUTPUT/"size_control_comparison.csv",index=False)
    n = training.set_index("cohort")
    shift = pd.read_csv(OUTPUT/"training_hospital_shift.csv").sort_values("B_minus_A_missing_rate",key=abs,ascending=False).head(8)
    primary_effects = effects.loc[effects.profile.eq("common")&effects.contrast.eq("full_minus_A_only")]
    supported_b = primary_effects.loc[primary_effects.evaluation.eq("B_test")&primary_effects.lower.gt(0),"model"].tolist()
    supported_a = primary_effects.loc[primary_effects.evaluation.eq("A_validation")&primary_effects.lower.gt(0),"model"].tolist()
    b_description = "、".join(MODEL_NAMES[m] for m in supported_b) or "无模型"
    a_description = "、".join(MODEL_NAMES[m] for m in supported_a) or "无模型"
    text = f"""# A+B 训练扩充实验

本实验已经完成 14 组模型评价：复用 4 个 A-only 模型，新训练 10 个模型。原 A 训练/验证/测试名单及原始数据保持不变；A 测试集未评价。

## 本轮判断

主实验在 B 留出集上，{b_description} 的扩充增益区间完全高于 0；在原 A 验证集上，{a_description} 的扩充增益区间完全高于 0。这些是未作多重校正的探索性、患者配对区间。扩充对不同医院和模型的影响不同，不能仅凭总小时记录增加就认定问题已解决。

总训练人数接近翻倍，发病患者增加约 57%。等规模 A+B 对照进一步显示：LR 的规模增益缺乏稳定支持；EBM 与原特征 LightGBM 在 B 测试上存在正向规模信号，但 A 验证未见明确收益，且 B 验证的规模差异区间仍跨 0。这些信号条件于两个训练抽样，仍需更多训练内重采样验证。两院测量缺失率存在很大差异，应同时处理数据来源分布、发病患者数量和模型复杂度。

## 实验回答什么

- A-only 与全量 A+B：评价加入医院 B 训练患者后的整体效果，同时改变了样本量、医院来源和病例比例，不能将差异全部归因于样本量。
- 两个等规模 A+B 对照与全量 A+B：按医院×发病状态比例抽样，两个种子均为 2026/2027；患者数与原 A 训练集一致，混合比例约相同，主要观察混合数据池内的规模效应。
- 主比较三个模型统一使用原 A 定义的 312 个特征。额外复核上一轮 349 特征、不裁剪、原生缺失的 LightGBM；两种特征方案分别比较，不混为模型架构效果。

## 队列与固定设置

{markdown(training)}

训练人数从 {int(n.loc['A_only','patients']):,} 增加至 {int(n.loc['A_B_full','patients']):,}，发病患者从 {int(n.loc['A_only','event_patients']):,} 增加至 {int(n.loc['A_B_full','event_patients']):,}，小时数从 {int(n.loc['A_only','hours']):,} 增至 {int(n.loc['A_B_full','hours']):,}。等规模混合对照只有 880 位发病患者，低于 A-only 的 1,109 位，这是保持全量混合池自然病例比例的结果。

B 在患者层面按发病状态分层，以种子 42 划为 70%/15%/15%。原 A 划分保留。全部评价队列如下：

{markdown(evaluation)}

沿用六小时预测目标、六小时历史有效期、原固定超参数和内部患者级早停规则；没有类别重加权、过采样或重新调参。每个扩充队列单独拟合训练集插补、裁剪和标准化参数。原 A 全缺失而删除的特征不会因 B 数据可用而加入主实验。

每个模型只在同一个 A+B 混合验证集选择一个报警阈值，目标为每 100 个符合条件的患者小时不超过 1 次误报，报警后抑制重复报警 6 小时。阈值原样用于 A 验证、B 验证和 B 测试。A-only 仅指风险模型在 A 训练，阈值校准也使用 B 验证；因此它并非完全不接触 B 数据的部署方案。混合验证的误报限制不保证各医院分别达标，须读实际误报率。

## 主结果：小时级 AUPRC

{markdown(comparison)}

![AUPRC comparison](auprc_comparison.png)

## 患者配对置信区间与等规模对照

下表为候选减参考的 AUPRC 差异；full_minus_mean_matched 比较全量与两个等规模对照的平均 AP，并非先平均预测概率。按评价患者重采样 300 次，报告 95% 百分位区间。

{markdown(size_control)}

{markdown(compact_effects)}

![Expansion effects](expansion_effects.png)

![Matched size control](matched_size_comparison.png)

区间仅反映给定已训练模型及两个训练子集条件下的评价患者不确定性，不包含重新训练的波动；两个训练抽样不足以给出完整的训练样本不确定性。多组探索性比较未作多重校正；不应以单个正向点估计宣称扩充稳定有效。

## 报警表现

{markdown(alarms)}

## 上一轮 LightGBM 特征方案的扩充复核

{markdown(secondary)}

## 两院训练数据差异

下表按最近值缺失率的绝对差异列出最大的八个变量，仅使用 A/B 各自训练患者。数值为小时加权，不能代表患者层面的独立样本数量。完整的缺失率及观测分位数见 training_hospital_shift.csv，患者病例比例和可用时长见 training_patient_composition.csv。

{markdown(shift[['variable','A_train_latest_missing_rate','B_train_latest_missing_rate','B_minus_A_missing_rate']])}

## 解释边界与下一步

1. B 测试患者没有参与拟合或阈值选择，但其医院已经进入 A+B 训练。它属于同院未见患者检验，不能再称为独立外院验证；原来的跨院研究结果仍保留为历史实验。
2. B 患者此前整体被保留，当前按预先固定的划分对多个固定候选一次性评价，没有根据 B 测试选择特征或参数。原 A 验证已多次查看，本研究仍属于探索性分析。
3. 相同数值超参数在不同样本量下可能产生不同的有效正则化强度。本轮刻意固定参数，结论针对当前模型设置，不能证明增加数据在最优调参后一定无效或一定有效。
4. AUPRC 必须在同一评价队列内比较；A/B 患病率不同，不能直接用跨队列 AP 大小判断难度。完整 AUROC、Brier、训练 AP、早停树数、耗时与报警统计见 all_model_metrics.csv。
5. 下一轮先结合等规模对照判断规模效应，再在训练集内部交叉验证处理医院分布与正则化。继续保留 A 测试；若需要真正的外院验证，必须另找未参与训练的医院。

运行及验证方法见仓库 README 的 A+B expansion experiment。protocol.json、模型元数据、固定患者名单和校验记录与本报告一起保存；大型特征和模型仍分别存放在 processed_data/ab_expansion 与 outputs/ab_expansion。
"""
    (OUTPUT/"report.md").write_text(text)
    write_json(OUTPUT/"report_manifest.json",{"model_results":len(jobs),"metric_rows":len(metrics),"paired_contrasts":len(effects),
                "original_A_test_untouched":True,"B_test_unseen_hospital":False,
                "analysis_source_fingerprints":{f"scripts/{name}":content_hash(ROOT/"scripts"/name) for name in
                    ["report_ab_expansion.py","verify_ab_expansion.py","resume_ab_expansion.py","recover_ab_inputs.py"]},
                "cohort_membership_sha256":content_hash(OUTPUT/"cohort_membership.csv"),
                "protocol_sha256":content_hash(OUTPUT/"protocol.json")})
    REPORT.mkdir(parents=True,exist_ok=True)
    names = ["protocol.json","completed.json","training_cohorts.csv","evaluation_cohorts.csv","model_manifest.json",
             "input_fingerprints.json","all_model_metrics.csv","model_metadata_archive.json","paired_ap_effects.csv",
             "training_hospital_shift.csv","training_patient_composition.csv","primary_comparison.csv","primary_alert_comparison.csv",
             "secondary_lightgbm.csv","size_control_comparison.csv","auprc_comparison.png","matched_size_comparison.png","expansion_effects.png","report.md","report_manifest.json"]
    for name in names:shutil.copyfile(OUTPUT/name,REPORT/name)
    for name in ["input_recovery.json","patient_summary_recovery.json","row_cache_recovery.json","preprocessor_recovery.json","local_runtime.json","post_run_inputs.json","secondary_input_fingerprints.json"]:
        if (OUTPUT/name).exists():shutil.copyfile(OUTPUT/name,REPORT/name)
    shutil.copyfile(PROCESSED/"B_patient_splits.csv",REPORT/"B_patient_splits.csv")
    print(comparison.to_string(index=False),flush=True)
    print("A+B report and versioned snapshots written.",flush=True)


if __name__ == "__main__":main()
