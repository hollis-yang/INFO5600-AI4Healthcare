"""Report hospital-balanced CV, frozen validation choices and training-only missingness."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import OUTPUT,PROCESSED,REPORT,AB_CASE,AB_MODELS,VALIDATION_SPLITS,output_directory,model_metrics
from sepsis_warning.ab_expansion import content_hash,write_json
from sepsis_warning.evaluation import ranking_metrics,alert_replay
from sepsis_warning.history_study import bootstrap_ap_contrast
from sepsis_warning.data import config

NAMES={"lr":"LR","ebm":"EBM","lightgbm":"LightGBM"}


def markdown(frame):
    def cell(v):return f"{v:.4f}" if isinstance(v,(float,np.floating)) else str(v)
    return "\n".join(["| "+" | ".join(frame.columns)+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]+
        ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])


def baseline_validation(protocol):
    out=OUTPUT/"baseline_validation";out.mkdir(exist_ok=True)
    rows={s:pd.read_csv(PROCESSED/f"{s}_rows.csv.gz") for s in VALIDATION_SPLITS}
    records={}
    for model in protocol["settings"]["study"]["models"]:
        path=AB_MODELS/f"{model}_model.joblib"
        assert content_hash(path)==protocol["input_fingerprints"][str(path.relative_to(ROOT))]
        bundle=joblib.load(path);scores={};metrics={};alerts={}
        for split in VALIDATION_SPLITS:
            x=np.load(AB_CASE/f"{split}_X.npy",mmap_mode="r")
            with threadpool_limits(limits=2):
                p=bundle["model"].booster_.predict(x) if model=="lightgbm" else bundle["model"].predict_proba(x)[:,1]
            np.save(out/f"{model}_{split}_probability.npy",p);scores[split]=p
            metrics[split]=ranking_metrics(rows[split].label.to_numpy(),p)
            alerts[split],_,_=alert_replay(rows[split],p,bundle["threshold"],config()["alerts"]["suppression_hours"])
        pooled=pd.concat([rows[s] for s in VALIDATION_SPLITS],ignore_index=True)
        combined=np.concatenate([scores[s] for s in VALIDATION_SPLITS])
        metrics["pooled_validation"]=ranking_metrics(pooled.label.to_numpy(),combined)
        metrics["macro_hospital_auprc"]=float(np.mean([metrics[s]["auprc_average_precision"] for s in VALIDATION_SPLITS]))
        alerts["pooled_validation"],_,_=alert_replay(pooled,combined,bundle["threshold"],config()["alerts"]["suppression_hours"])
        records[model]={"evaluation":metrics,"alerts":alerts,"threshold":bundle["threshold"],"source_model_sha256":content_hash(path),"test_sets_evaluated":[]}
    write_json(out/"results.json",records)
    return records


def main():
    completion=json.loads((OUTPUT/"completed.json").read_text())
    selection=json.loads((OUTPUT/"selection.json").read_text())
    protocol=json.loads((OUTPUT/"protocol.json").read_text())
    jobs=json.loads((OUTPUT/"cv_manifest.json").read_text())
    train_rows=pd.read_csv(PROCESSED/"training_rows.csv.gz")
    folds,scores,archive=[],{},{}
    for job in jobs:
        path=output_directory(**job)/"results.json";r=json.loads(path.read_text());archive[str(path.relative_to(ROOT))]=r
        metrics=r["evaluation"]
        folds.append({**job,"macro_hospital_auprc":metrics["macro_hospital_auprc"],"A_auprc":metrics["A"]["auprc_average_precision"],
            "B_auprc":metrics["B"]["auprc_average_precision"],"pooled_auprc":metrics["pooled"]["auprc_average_precision"],
            "A_auroc":metrics["A"]["auroc"],"B_auroc":metrics["B"]["auroc"],"training_macro_auprc":r["training"]["macro_hospital_auprc"],
            "fit_seconds":r["fit_seconds"],"inner_selected_trees":r["details"].get("inner_selected_trees"),"warnings":len(r["warnings"])})
        cycles=r["details"].get("executed_cycles_including_patience",[])
        folds[-1]["max_executed_ebm_cycles"]=float(max(cycles)) if cycles else None
        folds[-1]["ebm_bags_at_round_cap"]=int(sum(value>=r["parameters"].get("max_rounds",np.inf)-1e-10 for value in cycles)) if cycles else None
        key=(job["model"],job["candidate"])
        if key not in scores:scores[key]=np.full(len(train_rows),np.nan)
        indices=np.load(PROCESSED/f"fold{job['fold']}"/"evaluation_indices.npy")
        assert np.isnan(scores[key][indices]).all()
        scores[key][indices]=np.load(path.parent/"probability.npy")
    folds=pd.DataFrame(folds);folds.to_csv(OUTPUT/"cv_fold_results.csv",index=False)
    summary=folds.groupby(["model","candidate"],sort=False).agg(mean_macro_auprc=("macro_hospital_auprc","mean"),
        fold_sd_macro_auprc=("macro_hospital_auprc","std"),mean_A_auprc=("A_auprc","mean"),mean_B_auprc=("B_auprc","mean"),
        mean_training_macro_auprc=("training_macro_auprc","mean"),mean_fit_seconds=("fit_seconds","mean")).reset_index()
    oof=[]
    for (model,candidate),p in scores.items():
        assert np.isfinite(p).all();np.save(OUTPUT/f"{model}_{candidate}_oof_probability.npy",p)
        metrics=model_metrics(train_rows,p)
        oof.append({"model":model,"candidate":candidate,"oof_macro_auprc":metrics["macro_hospital_auprc"],
            "oof_A_auprc":metrics["A"]["auprc_average_precision"],"oof_B_auprc":metrics["B"]["auprc_average_precision"]})
    summary=summary.merge(pd.DataFrame(selection["cv_comparisons"]).drop(columns="mean_macro_hospital_auprc"),on=["model","candidate"]).merge(pd.DataFrame(oof),on=["model","candidate"])
    summary["selected"]=[selection["selected_candidates"][m]==c for m,c in zip(summary.model,summary.candidate)]
    summary.to_csv(OUTPUT/"cv_summary.csv",index=False)
    baseline=baseline_validation(protocol)
    comparison,alarms,effects,full_records=[],[],[],[]
    for job in json.loads((OUTPUT/"validation_manifest.json").read_text()):
        path=output_directory(**job)/"results.json";r=json.loads(path.read_text());archive[str(path.relative_to(ROOT))]=r
        model=job["model"];bundle=joblib.load(path.parent/"model.joblib")
        for split in VALIDATION_SPLITS:
            table=pd.read_csv(PROCESSED/f"{split}_rows.csv.gz")
            a=np.load(OUTPUT/"baseline_validation"/f"{model}_{split}_probability.npy")
            b=np.load(path.parent/f"{split}_probability.npy")
            with threadpool_limits(limits=1):effect=bootstrap_ap_contrast(table,[a],[b],repetitions=protocol["settings"]["study"]["bootstrap_samples"])
            effects.append({"model":model,"candidate":job["candidate"],"evaluation":split,**effect})
            comparison.append({"model":model,"selected_candidate":job["candidate"],"evaluation":split,
                "baseline_auprc":baseline[model]["evaluation"][split]["auprc_average_precision"],"selected_auprc":r["evaluation"][split]["auprc_average_precision"],
                "ap_difference":effect["difference"],"difference_lower":effect["lower"],"difference_upper":effect["upper"],
                "baseline_auroc":baseline[model]["evaluation"][split]["auroc"],"selected_auroc":r["evaluation"][split]["auroc"],
                "baseline_brier":baseline[model]["evaluation"][split]["brier_score"],"selected_brier":r["evaluation"][split]["brier_score"]})
        for method,record in [("baseline",baseline[model]),("selected",r)]:
            for split in [*VALIDATION_SPLITS,"pooled_validation"]:alarms.append({"model":model,"method":method,"evaluation":split,**record["alerts"][split]})
        for split in [*VALIDATION_SPLITS,"pooled_validation"]:full_records.append({"model":model,"candidate":job["candidate"],"evaluation":split,**r["evaluation"][split]})
        m=bundle["model"];names=bundle["preprocessor"].names_
        importance=np.abs(m.coef_[0]) if model=="lr" else m.term_importances() if model=="ebm" else m.booster_.feature_importance("gain")
        pd.DataFrame({"feature":names,"importance":importance,"measurement_availability":[name.endswith(("__missing_now","__never_measured","__hours_since","__count_6h")) for name in names]}).sort_values("importance",ascending=False).to_csv(OUTPUT/f"{model}_feature_importance.csv",index=False)
    comparison=pd.DataFrame(comparison);alarms=pd.DataFrame(alarms)
    comparison.to_csv(OUTPUT/"validation_comparison.csv",index=False);alarms.to_csv(OUTPUT/"alert_comparison.csv",index=False)
    pd.DataFrame(full_records).to_csv(OUTPUT/"selected_validation_metrics.csv",index=False)
    write_json(OUTPUT/"validation_effects.json",effects)
    write_json(OUTPUT/"model_metadata_archive.json",{"available_model_results":len(archive),"results":archive})
    make_plots(folds,summary,comparison,selection,protocol)
    missing=pd.read_csv(OUTPUT/"missingness/cross_hospital.csv")
    missing_top=missing.sort_values("B_minus_A_raw_hour_missing_rate",key=abs,ascending=False).head(10)[
        ["variable","A_raw_hour_missing_rate","B_raw_hour_missing_rate","A_latest_hour_missing_rate","B_latest_hour_missing_rate"]]
    controls=missing.sort_values("B_minus_A_raw_patient_mean_missing_rate",key=abs,ascending=False).head(8)[
        ["variable","B_minus_A_raw_patient_mean_missing_rate","B_minus_A_raw_event_standardized_patient_missing_rate","B_minus_A_raw_first24_patient_mean_missing_rate"]]
    diagnostic=json.loads((OUTPUT/"missingness/hospital_classifier.json").read_text())
    chosen=summary.loc[summary.selected,["model","candidate","mean_macro_auprc","mean_A_auprc","mean_B_auprc","mean_gain_vs_baseline","improved_folds"]]
    changed=[f"{NAMES[m]} → {c}" for m,c in selection["selected_candidates"].items() if c!="baseline"]
    change_description="；".join(changed) if changed else "没有候选达到预先规定的替换门槛，三个模型均保留原设置"
    reused_count=sum(record["case"]["stage"]=="validation" and record["reused_full_baseline"] for record in archive.values())
    positive=[]
    for split in VALIDATION_SPLITS:
        models=comparison.loc[comparison.evaluation.eq(split)&comparison.difference_lower.gt(0),"model"].tolist()
        positive.append(split+"："+("、".join(NAMES[m] for m in models) if models else "无"))
    text=f"""# A+B 医院分层交叉验证与正则化

在原有 A+B 训练池（27,911 位患者、1,742 位发病患者、918,004 条符合条件的小时记录）上完成 33 次 CV 拟合，固定 312 个特征，只改变正则化参数。现有 A/B 训练、验证和测试名单均未改变；本轮未评价任何测试集。

本轮训练内选择：**{change_description}**。固定验证集上，增益区间完全高于 0 的模型为：{'；'.join(positive)}。这些是未作多重校正的探索性比较，不能将训练 CV 的改进直接等同于验证集收益。

三条所选模型评价记录中，{reused_count} 条直接复用了原 A+B 模型权重。保留原模型时，验证差值及其配对区间为 0，表示相同权重的比较，不是重新训练后波动的估计。

两院缺失差异在患者等权、统一发病比例和前 24 小时对照后仍然明显；仅凭训练患者的缺失比例识别医院，OOF AUROC 为 {diagnostic['pooled_oof_hospital_auroc']:.4f}。这支持记录分布存在强烈的医院来源差异；其对 sepsis 预测误差的因果影响还需训练内消融检验。

## 训练内选择结果

{markdown(chosen)}

完整参数见 protocol.json 的 candidates。LR 比较 C=0.001/0.01/0.1/1/10；EBM 联合比较 (min_samples_leaf, reg_lambda)=(200,10)/(400,50)/(800,100)；LightGBM 联合比较 (min_child_samples, reg_lambda)=(100,1)/(200,10)/(400,50)。树的叶子数量、分箱、学习率和迭代上限等保持原设置；联合设置的效果不能拆解为单个参数的因果效果。

三折按患者分组，并对医院×发病状态分层，种子 2026。同一患者全部小时属于同一折。每折单独拟合插补、裁剪和标准化；只使用该折训练患者，外折患者不进入预处理拟合。内部早停沿用原先按发病状态分层的患者袋，全部来自该外折训练分区。

选择指标先在 A、B 各自计算 AUPRC，再对两院取平均，最后对三折取平均。候选必须比原设置提升至少 0.001，且至少改善两折，才有资格替换原设置；否则保留 baseline。selection.json 在外层验证评价之前保存，外层验证指标不参与选择。折间标准差不是置信区间，OOF 指标与用于选择的平均折指标分别保存。

![CV](cv_regularization.png)

## 固定 A/B 验证集结果

{markdown(comparison)}

这些外层验证集此前已经查看过，当前结果仍属于探索性分析。表中差值为所选模型减原 A+B 模型，按评价患者配对重采样 300 次；区间条件于已选定模型，不包含再次调参或重新训练的波动，多组比较未做多重校正。

![Validation](validation_comparison.png)

![Effects](validation_effects.png)

本轮没有重新评分 B 测试患者，也没有使用仍保留的 A 测试。B 在上一轮已经进入训练，因此后续 B 留出评价仍属于同院未见患者检验，不能恢复为独立外院验证。

## 报警评价

每个所选模型在相同的 A+B 混合验证集校准一个阈值，目标为每 100 个符合条件的患者小时不超过 1 次误报，报警后抑制 6 小时。该混合预算不保证每个医院分别达标。原模型参考阈值来自上一轮同一混合验证队列。

{markdown(alarms[['model','method','evaluation','event_sensitivity','alert_ppv','false_alerts_per_100_patient_hours','median_true_alert_lead_hours']])}

验证集上选择阈值并在同一集合报告的报警表现可能乐观；当前结果用于研究比较。训练模型与预处理、早停分组、阈值和预测都保存在 outputs/ab_regularization 下。

## 两院训练数据的缺失差异

原始小时缺失指该小时没有观测；最近值缺失指六小时有效期后仍没有可用生理指标。背景变量保持当前值，不做该生理指标有效期处理。以下比例均为 0–1，0.10 表示 10%。全部分析仅使用训练患者及其符合条件的小时，不包含外层验证或测试患者。

{markdown(missing_top)}

![Missingness](missingness/raw_and_latest.png)

## 记录长度与病例比例对照

下表为 B 减 A 的原始缺失率差：先让每位患者等权，再将两院发病患者比例统一到训练池的 {1742/27911:.4%}，并另看每位患者前 24 小时。病例比例标准化只处理发病状态构成，不是年龄、严重度或其他临床因素的因果调整；小时分组也不能排除记录结束和队列选择造成的偏差。

{markdown(controls)}

![Patient controls](missingness/patient_weighted_controls.png)

![Hour bands](missingness/hour_band_gaps.png)

缺失模式辨别医院的诊断实验，使用每位训练患者全段符合条件记录的 40 项原始缺失比例，采用相同三折、折内标准化和固定 LR，医院 B 为正类；OOF AUROC 为 **{diagnostic['pooled_oof_hospital_auroc']:.4f}**。这量化了两院记录模式的可区分程度。全段聚合仅用于该诊断，未进入逐小时 sepsis 模型，医院编号也未作为 sepsis 模型输入。

明显的缺失分布差异支持医院来源偏移确实存在；单凭该分析不能断定缺失导致了预测失败，也不能判断记录错误。主模型的缺失标记、观测次数与测量间隔可能携带医院流程信息，同时也可能携带有效风险信息。三种所选模型的特征重要性表保存了这些特征的标记；重要性受特征相关性影响，不等于因果贡献，三个模型的重要性尺度不能直接比较。

## 后续建议与复现

按用户补充要求，另计算 [完整轨迹官方 Utility Score](utility/report.md)，比较 A-only、A+B 原参数与本轮 CV 所选模型。该项使用未再平移的原始标签、完整小时轨迹及既定阈值，采用固定版本的 [官方评分代码](https://github.com/physionetchallenges/evaluation-2019)，不参与正则化或阈值选择；与本报告的发病前任务分开解释。

先据固定验证结果判断正则化是否带来可重复的改进。若增益有限，下一轮可在训练内预先设计缺失信息消融，以及医院分层的概率校准/报警预算比较；不能从目前缺失率差异直接推导哪一种处理必然改善效果。保留现有测试患者，外院泛化仍需要未参与训练的新医院。

运行命令、缓存目录及校验方式见仓库 README。protocol.json 和 selection.json 分别记录训练前设计与训练内选择，patient_cv_folds.csv 保存固定折名单；model_metadata_archive.json 归档 36 个小型模型结果记录。大型模型、预测和矩阵保存在本地的 outputs/ab_regularization、processed_data/ab_regularization，未加入 Git。
"""
    (OUTPUT/"report.md").write_text(text)
    write_json(OUTPUT/"report_manifest.json",{"cv_models":len(jobs),"validation_models":3,"test_sets_evaluated":[],"selection_sha256":content_hash(OUTPUT/"selection.json"),
        "analysis_source_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in [Path(__file__),ROOT/"scripts/verify_ab_regularization.py"]}})
    REPORT.mkdir(parents=True,exist_ok=True)
    names=["protocol.json","selection.json","completed.json","patient_cv_folds.csv","fold_strata.csv","cv_manifest.json","validation_manifest.json",
        "cv_fold_results.csv","cv_summary.csv","validation_comparison.csv","selected_validation_metrics.csv","validation_effects.json","alert_comparison.csv",
        "model_metadata_archive.json","cv_regularization.png","validation_comparison.png","validation_effects.png","report.md","report_manifest.json"]
    names += [f"{m}_feature_importance.csv" for m in ["lr","ebm","lightgbm"]]
    for name in names:shutil.copyfile(OUTPUT/name,REPORT/name)
    (REPORT/"missingness").mkdir(exist_ok=True)
    for path in (OUTPUT/"missingness").iterdir():
        if path.name!="hospital_classifier_oof.csv":shutil.copyfile(path,REPORT/"missingness"/path.name)
    print(chosen.to_string(index=False),flush=True);print(comparison.to_string(index=False),flush=True)
    print("Regularization and missingness report written.",flush=True)


def make_plots(folds,summary,comparison,selection,protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator,FormatStrFormatter
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True)
    for ax,model in zip(axes,["lr","ebm","lightgbm"]):
        order=[c["candidate"] for c in protocol["candidates"] if c["model"]==model]
        part=summary.loc[summary.model.eq(model)].set_index("candidate")
        for j,candidate in enumerate(order):
            values=folds.loc[folds.model.eq(model)&folds.candidate.eq(candidate)].sort_values("fold").macro_hospital_auprc
            ax.scatter(j+np.linspace(-.12,.12,3),values,color="#367a95",s=20,alpha=.5)
            ax.scatter(j,part.loc[candidate,"mean_macro_auprc"],marker="D",s=45,color="#c27543",
                edgecolor="black" if candidate==selection["selected_candidates"][model] else "none")
        ax.axhline(part.loc["baseline","mean_macro_auprc"],linestyle="--",color="gray",linewidth=1)
        labels=["C="+str(c["parameters"]["C"]) if model=="lr" else c["candidate"] for c in protocol["candidates"] if c["model"]==model]
        ax.set(title=NAMES[model],xticks=np.arange(len(order)),xticklabels=labels);ax.tick_params(axis="x",rotation=30,labelsize=8)
    axes[0].set_ylabel("Mean hospital-specific AUPRC on held-out patients")
    fig.suptitle("Hospital/event-stratified patient CV: regularization only")
    fig.text(.5,.015,"Dots: three folds. Diamonds: mean; black outline: frozen selection. Dashed line: original setting.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1));fig.savefig(OUTPUT/"cv_regularization.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    models=["lr","ebm","lightgbm"];x=np.arange(3)
    maximum=float(comparison[["baseline_auprc","selected_auprc"]].to_numpy().max())*1.15
    for ax,split in zip(axes,VALIDATION_SPLITS):
        part=comparison.loc[comparison.evaluation.eq(split)].set_index("model").loc[models]
        ax.bar(x-.18,part.baseline_auprc,width=.34,label="Original A+B",color="#367a95")
        ax.bar(x+.18,part.selected_auprc,width=.34,label="CV selected",color="#c27543")
        ax.set(xticks=x,xticklabels=[NAMES[m] for m in models],ylim=(0,maximum),ylabel="AUPRC",title=split.replace("_"," "))
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.005),ncol=2,frameon=False)
    fig.suptitle("Frozen training-CV choices on unchanged validation patients")
    fig.tight_layout(rect=(0,.1,1,1));fig.savefig(OUTPUT/"validation_comparison.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,VALIDATION_SPLITS):
        part=comparison.loc[comparison.evaluation.eq(split)].set_index("model").loc[models]
        ax.hlines(np.arange(3),part.difference_lower,part.difference_upper,color="#367a95")
        ax.scatter(part.ap_difference,np.arange(3),color="#367a95")
        ax.axvline(0,color="gray",linestyle="--");ax.set(yticks=np.arange(3),yticklabels=[NAMES[m] for m in models],title=split.replace("_"," "),xlabel="AUPRC: selected minus original A+B")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5));ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"));ax.tick_params(axis="x",labelsize=9)
    fig.suptitle("Patient-paired bootstrap 95% intervals; no test evaluation")
    fig.tight_layout();fig.savefig(OUTPUT/"validation_effects.png",dpi=180);plt.close(fig)


if __name__=="__main__":main()
