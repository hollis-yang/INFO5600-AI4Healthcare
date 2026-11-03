"""Official utility on complete trajectories of the unchanged validation patients."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/"outputs/.matplotlib"))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sepsis_warning.ab_regularization import OUTPUT,PROCESSED,REPORT,AB_CASE,AB_MODELS,VALIDATION_SPLITS,output_directory
from sepsis_warning.ab_expansion import content_hash,write_json
from sepsis_warning.data import causal_features,assess_patient,config,FEATURE_NAMES
from sepsis_warning.history_study import digest
from sepsis_warning.utility import patient_utility,normalized_utility,paired_utility_contrast

OUT=OUTPUT/"utility"
DATA=PROCESSED/"utility"


def settings():
    with (ROOT/"configs/utility_comparison.toml").open("rb") as stream:return tomllib.load(stream)


def references():
    return [{"stage":stage,"model":model,"path":ROOT/"outputs/ab_expansion/models"/case/f"{model}_model.joblib"}
        for stage,case in [("A_only","A_only__common"),("A_B_baseline","A_B_full__common")] for model in ["lr","ebm","lightgbm"]]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    study=settings();refs=references()
    paths=[ROOT/"processed_data/raw_records.npy",ROOT/"outputs/patient_summary.csv",ROOT/"processed_data/splits/patient_splits.csv",
        ROOT/"processed_data/ab_expansion/B_patient_splits.csv"]
    paths += [ROOT/"processed_data/ab_expansion"/f"{s}_rows.csv.gz" for s in VALIDATION_SPLITS]
    paths += [r["path"] for r in refs]
    sources=[Path(__file__),ROOT/"configs/utility_comparison.toml",ROOT/"configs/baseline.toml",ROOT/"src/sepsis_warning/utility.py",ROOT/"src/sepsis_warning/data.py",
        ROOT/"third_party/physionet2019/evaluate_sepsis_score.py",ROOT/"third_party/physionet2019/LICENSE.txt",ROOT/"third_party/physionet2019/UPSTREAM.json"]
    for p in paths:
        if getattr(p.stat(),"st_flags",0)&0x40000000:raise OSError(f"Required utility input unavailable: {p}")
    protocol={"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":study,"input_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in paths},
        "source_fingerprints":{str(p.relative_to(ROOT)):content_hash(p) for p in sources},"upstream":json.loads((ROOT/"third_party/physionet2019/UPSTREAM.json").read_text()),
        "regularization_selection_existed_at_design":(OUTPUT/"selection.json").exists(),"added_during_regularization_study_at_user_request":True,
        "model_choice_uses_utility":False,"test_sets_evaluated":[]}
    if (OUT/"protocol.json").exists():
        previous=json.loads((OUT/"protocol.json").read_text());a,b=dict(previous),dict(protocol)
        for key in ["created_at_utc","regularization_selection_existed_at_design"]:a.pop(key);b.pop(key)
        assert a==b,"Utility design changed; preserve earlier results separately"
        protocol=previous
    write_json(OUT/"protocol.json",protocol)
    raw=np.load(ROOT/"processed_data/raw_records.npy",mmap_mode="r")
    audit=pd.read_csv(ROOT/"outputs/patient_summary.csv").set_index("patient_id")
    for split in VALIDATION_SPLITS:
        eligible_rows=pd.read_csv(ROOT/"processed_data/ab_expansion"/f"{split}_rows.csv.gz")
        identifiers=eligible_rows.patient_id.drop_duplicates().tolist()
        signature=digest({"protocol":protocol,"split":split,"patient_ids":identifiers})
        directory=DATA/split;directory.mkdir(exist_ok=True)
        if (directory/"case.json").exists() and json.loads((directory/"case.json").read_text())["signature"]==signature:continue
        n=int(audit.loc[identifiers,"n_hours"].sum())
        matrix=np.lib.format.open_memmap(directory/"features.npy",mode="w+",dtype=np.float32,shape=(n,318))
        records,patients,position=[],[],0
        for identifier in identifiers:
            patient=audit.loc[identifier];values=raw[int(patient.raw_start):int(patient.raw_stop)]
            assert bool(patient.included)
            reason,eligible,onset=assess_patient(values,config());assert reason=="included"
            features=causal_features(values,config());matrix[position:position+len(values)]=features
            for hour,label,is_eligible in zip(values[:,39],values[:,40],eligible):
                records.append((identifier,patient.hospital,int(hour),int(label),onset,bool(is_eligible)))
            patients.append({"patient_id":identifier,"hospital":patient.hospital,"start":position,"stop":position+len(values),"hours":len(values),
                "has_event":bool(values[:,40].any()),"raw_start":int(patient.raw_start),"raw_stop":int(patient.raw_stop)})
            position+=len(values)
        assert position==n;matrix.flush();del matrix
        table=pd.DataFrame(records,columns=["patient_id","hospital","hour","label","onset_hour","task_eligible"])
        pd.testing.assert_frame_equal(table.loc[table.task_eligible,eligible_rows.columns].reset_index(drop=True),eligible_rows)
        table.to_csv(directory/"rows.csv.gz",index=False,compression="gzip")
        pd.DataFrame(patients).to_csv(directory/"patients.csv",index=False)
        np.save(directory/"labels.npy",table.label.to_numpy(np.int8))
        write_json(directory/"case.json",{"signature":signature,"evaluation":split,"patients":len(patients),"full_hours":n,"eligible_task_hours":len(eligible_rows),
            "all_raw_hours_retained":True,"labels_shifted":False,"features_sha256":content_hash(directory/"features.npy"),"labels_sha256":content_hash(directory/"labels.npy")})
        print(f"Prepared utility {split}: {len(patients)} patients, {n:,} full hours; {len(eligible_rows):,} original task hours",flush=True)
    return protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--prepare-only",action="store_true");args=parser.parse_args()
    protocol=prepare()
    if args.prepare_only:return
    completion=json.loads((OUTPUT/"completed.json").read_text());selection=json.loads((OUTPUT/"selection.json").read_text())
    assert completion["selected_candidates"]==selection["selected_candidates"] and not selection["outer_validation_used_for_selection"]
    models=references()+[{"stage":"A_B_cv","model":m,"path":output_directory("validation",-1,m,c)/"model.joblib"} for m,c in selection["selected_candidates"].items()]
    metadata,all_components,results=[],{},[]
    for item in models:
        stage,model,path=item["stage"],item["model"],item["path"]
        bundle=joblib.load(path);processor=bundle["preprocessor"];classifier=bundle["model"]
        assert len(processor.names_)==312 and bundle["threshold"] is not None
        output=OUT/stage/model;output.mkdir(parents=True,exist_ok=True)
        thresholds={"existing_validation_budget":float(bundle["threshold"]),"fixed_0p5":.5}
        checks=[]
        for split in VALIDATION_SPLITS:
            directory=DATA/split;features=np.load(directory/"features.npy",mmap_mode="r")
            table=pd.read_csv(directory/"rows.csv.gz");patients=pd.read_csv(directory/"patients.csv")
            labels=np.load(directory/"labels.npy");np.testing.assert_array_equal(labels,table.label)
            probability=np.empty(len(labels))
            for start in range(0,len(features),10000):
                matrix=processor.transform(features[start:start+10000])
                with threadpool_limits(limits=protocol["settings"]["study"]["prediction_threads"]):
                    probability[start:start+len(matrix)]=classifier.booster_.predict(matrix,num_threads=2) if model=="lightgbm" else classifier.predict_proba(matrix)[:,1]
            assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
            np.save(output/f"{split}_probability.npy",probability)
            # Full-trajectory generation agrees with the established pre-onset pipeline.
            eligible_positions=np.flatnonzero(table.task_eligible)
            probes=np.unique(np.linspace(0,len(eligible_positions)-1,64,dtype=int))
            source=ROOT/"processed_data/ab_expansion/cases"/("A_only__common" if stage=="A_only" else "A_B_full__common")
            old=np.load(source/f"{split}_X.npy",mmap_mode="r")[probes]
            new=processor.transform(features[eligible_positions[probes]])
            np.testing.assert_allclose(new,old,rtol=0,atol=0,equal_nan=True)
            with threadpool_limits(limits=2):expected=classifier.booster_.predict(old,num_threads=2) if model=="lightgbm" else classifier.predict_proba(old)[:,1]
            np.testing.assert_allclose(expected,probability[eligible_positions[probes]],rtol=0,atol=1e-12)
            checks.append(split)
            for policy,threshold in thresholds.items():
                records=[]
                for patient in patients.itertuples(index=False):
                    begin,end=int(patient.start),int(patient.stop)
                    components=patient_utility(labels[begin:end],probability[begin:end]>=threshold)
                    records.append({"patient_id":patient.patient_id,"hospital":patient.hospital,"hours":patient.hours,"has_event":patient.has_event,**components})
                frame=pd.DataFrame(records);frame.to_csv(output/f"{split}_{policy}_patient_utilities.csv",index=False)
                all_components[stage,model,policy,split]=frame
                results.append({"stage":stage,"model":model,"policy":policy,"evaluation":split,"threshold":threshold,
                    "patients":len(patients),"event_patients":int(patients.has_event.sum()),"full_hours":len(labels),**normalized_utility(frame)})
        for policy,threshold in thresholds.items():
            pooled=pd.concat([all_components[stage,model,policy,s] for s in VALIDATION_SPLITS],ignore_index=True)
            all_components[stage,model,policy,"pooled_validation"]=pooled
            results.append({"stage":stage,"model":model,"policy":policy,"evaluation":"pooled_validation","threshold":threshold,
                "patients":len(pooled),"event_patients":int(pooled.has_event.sum()),"full_hours":int(pooled.hours.sum()),**normalized_utility(pooled)})
        metadata.append({"stage":stage,"model":model,"source_model":str(path.relative_to(ROOT)),"source_model_sha256":content_hash(path),
            "thresholds":thresholds,"eligible_pipeline_probes_verified":checks,"parameters":bundle["parameters"]})
        print(f"Scored full-trajectory utility: {stage}/{model}",flush=True)
    frame=pd.DataFrame(results);frame.to_csv(OUT/"all_scores.csv",index=False)
    effects=[]
    with threadpool_limits(limits=1):
        for policy in protocol["settings"]["study"]["threshold_policies"]:
            for split in [*VALIDATION_SPLITS,"pooled_validation"]:
                for model in ["lr","ebm","lightgbm"]:
                    for before,after in [("A_only","A_B_baseline"),("A_B_baseline","A_B_cv")]:
                        contrast=paired_utility_contrast(all_components[before,model,policy,split],all_components[after,model,policy,split],
                            repetitions=protocol["settings"]["study"]["bootstrap_samples"],seed=protocol["settings"]["study"]["bootstrap_seed"])
                        effects.append({"reference":before,"candidate":after,"model":model,"policy":policy,"evaluation":split,**contrast})
    effects=pd.DataFrame(effects);effects.to_csv(OUT/"paired_effects.csv",index=False)
    write_json(OUT/"model_manifest.json",metadata)
    for mapping in [protocol["input_fingerprints"],protocol["source_fingerprints"]]:
        for name,expected in mapping.items():assert content_hash(ROOT/name)==expected
    write_json(OUT/"completed.json",{"models":len(metadata),"metric_rows":len(frame),"paired_contrasts":len(effects),"test_sets_evaluated":[],
        "regularization_selection_sha256":content_hash(OUTPUT/"selection.json"),"utility_used_for_model_or_threshold_selection":False})
    write_report(frame,effects,protocol)


def write_report(scores,effects,protocol):
    from report_ab_regularization import markdown
    primary=scores.loc[scores.policy.eq("existing_validation_budget")]
    comparison=primary.pivot(index=["model","evaluation"],columns="stage",values="normalized_utility").reset_index()
    comparison["CV_minus_AB_baseline"]=comparison.A_B_cv-comparison.A_B_baseline
    comparison.to_csv(OUT/"primary_comparison.csv",index=False)
    paired=effects.loc[effects.policy.eq("existing_validation_budget"),["reference","candidate","model","evaluation","difference","lower","upper"]]
    text=f"""# 完整轨迹官方 Utility Score 补充评价

使用 [PhysioNet/CinC 2019 官方评分代码](https://github.com/physionetchallenges/evaluation-2019/tree/467c49b514542be7a4a0bafe40fa2c3b064dda2e)，保留原始 BSD 2-Clause 许可及固定上游代码摘要。此项是在正则化 CV 运行期间按用户要求增加的补充评价，不参与已有的选参规则。

三个阶段均为共同 312 特征：A-only（上一轮 A+B 实验中的 A-only 风险模型）、A+B 原参数、当前训练 CV 选出的 A+B 版本。评价使用同一批 A 验证患者和 B 验证患者；患者划分、模型权重和既定阈值保持不变，测试集未评分。

## 原始阈值下的官方标准化效用

{markdown(comparison)}

0 对应始终不报警，1 对应本队列理想预测；负分表示比始终不报警更差，分数不是概率。合并分数先累加患者的原始效用，再统一标准化，不是两院分数的算术平均。

## 评分口径

- 每位患者恢复完整原始小时轨迹，包括发病后及原自定义任务删除的阴性末尾小时。保持原 SepsisLabel，不再移动标签。
- 官方函数从首次正标签位置加 6 小时重建基准发病时间，使用 -12/-6/+3 小时的效用窗口及官方误报/漏报代价。
- 每小时直接按 probability >= 原模型阈值产生二值预测，不套用自定义报警的 6 小时冷却期。所有历史阶段的原阈值均来自上轮同一 A+B 混合验证的误报预算校准；当前版本使用本轮同一混合验证的校准阈值。
- 本轮没有为 utility 调整模型或阈值。另列固定 0.5 阈值作为敏感性对照，详见 all_scores.csv。原误报预算只约束自定义任务的报警事件，不代表这些完整轨迹逐小时预测也符合相同报警负担。
- 模型训练仍是发病前六小时风险任务，完整轨迹 utility 是另外的评分视角，不替换原 AUPRC 或报警评价。该成人新发队列也是官方数据的子集，不能将分数直接当作挑战赛隐藏测试成绩。

## 患者配对差值区间

{markdown(paired)}

按患者配对重采样 300 次，区间条件于已训练模型及既定阈值，多组比较未做多重校正；验证集此前已查看过，结果仍属探索性。完整患者效用分解和概率保存在本地 outputs/ab_regularization/utility，规范、摘要和对比表保存在本报告目录。
"""
    (OUT/"report.md").write_text(text)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names={"lr":"LR","ebm":"EBM","lightgbm":"LightGBM"}
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),sharey=True)
    models=["lr","ebm","lightgbm"];positions=np.arange(3)
    for ax,split in zip(axes,VALIDATION_SPLITS):
        part=primary.loc[primary.evaluation.eq(split)]
        for offset,stage,label,color in [(-.25,"A_only","A only","#8d9ba7"),(0,"A_B_baseline","A+B original","#367a95"),(.25,"A_B_cv","A+B CV selected","#c27543")]:
            values=part.loc[part.stage.eq(stage)].set_index("model").loc[models,"normalized_utility"]
            ax.bar(positions+offset,values,width=.23,label=label,color=color)
        ax.axhline(0,color="gray",linewidth=.8);ax.set(xticks=positions,xticklabels=[names[m] for m in models],title=split.replace("_"," "),ylabel="Official normalized utility")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,.005),ncol=3,frameon=False,fontsize=9)
    fig.suptitle("Official utility on full validation trajectories; frozen thresholds")
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(OUT/"utility_comparison.png",dpi=180);plt.close(fig)
    target=REPORT/"utility";target.mkdir(parents=True,exist_ok=True)
    for name in ["protocol.json","all_scores.csv","primary_comparison.csv","paired_effects.csv","model_manifest.json","completed.json","report.md","utility_comparison.png"]:
        import shutil
        shutil.copyfile(OUT/name,target/name)
    print(comparison.to_string(index=False),flush=True)


if __name__=="__main__":main()
