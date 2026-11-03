"""Matched full-record comparison using the friend's unchanged 563 features."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from sepsis_warning.friend_563 import (DATA, OUT, REPORT, MODELS, prepare, build_split,
    runtime_environment, load_model, predict, score, select_utility_threshold)
from sepsis_warning.ab_expansion import content_hash, write_json
from sepsis_warning.evaluation import ranking_metrics


def jobs(items):
    pending, active = list(items), []
    logs = OUT / "logs"; logs.mkdir(exist_ok=True)
    last = time.monotonic()
    while pending or active:
        while pending and len(active) < 2:
            fold, model = pending.pop(0)
            path = logs / f"fold{fold}_{model}.log"; stream = path.open("w")
            process = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/fit_friend_563.py"),
                                       "--model", model, "--fold", str(fold)],
                                      cwd=ROOT, env=runtime_environment(), stdout=stream, stderr=subprocess.STDOUT)
            active.append((process, stream, fold, model, path))
            print(f"Started {model} fold={fold}", flush=True)
        for item in list(active):
            process, stream, fold, model, path = item
            status = process.poll()
            if status is None: continue
            stream.close(); active.remove(item)
            if status:
                for other, handle, *_ in active: other.terminate(); handle.close()
                print(path.read_text()[-10000:], flush=True)
                raise RuntimeError(f"Fit failed {model}/{fold}, status={status}")
            print(path.read_text()[-1800:], flush=True)
        if time.monotonic()-last > 40:
            print("Active: " + ", ".join(f"{m}/fold{f}" for _,_,f,m,_ in active)+f"; queued={len(pending)}", flush=True)
            last = time.monotonic()
        if pending or active: time.sleep(.5)


def finish_cv():
    patients = pd.read_csv(DATA / "train_patients.csv")
    labels = np.load(DATA / "train_labels.npy")
    original = patients.set_index("patient_id")
    rounds, fold_records, pooled = {}, [], []
    for model in MODELS:
        oof = np.full(len(labels), np.nan)
        for fold in range(3):
            directory = OUT / "cv" / f"fold{fold}" / model
            result = json.loads((directory / "results.json").read_text())
            held = pd.read_csv(directory / "held_patients.csv")
            probability = np.load(directory / "probability.npy")
            y = np.load(directory / "labels.npy")
            assert held.cv_fold.eq(fold).all()
            for person in held.itertuples(index=False):
                target = original.loc[person.patient_id]
                a,b = int(target.start),int(target.stop)
                start,stop = int(person.start),int(person.stop)
                np.testing.assert_array_equal(labels[a:b],y[start:stop])
                assert np.isnan(oof[a:b]).all()
                oof[a:b] = probability[start:stop]
            fold_records.append({"model":model,"fold":fold,"selected_rounds":result["selected_rounds"],
                                 "auroc":result["evaluation"]["auroc"],
                                 "auprc":result["evaluation"]["auprc_average_precision"],
                                 "fit_seconds":result["fit_seconds"]})
        assert np.isfinite(oof).all()
        np.save(OUT / f"{model}_train_OOF_probability.npy",oof)
        ranking=ranking_metrics(labels,oof)
        selected=[r["selected_rounds"] for r in fold_records if r["model"]==model]
        rounds[model]=int(np.median(selected))
        pooled.append({"split":"train_OOF_development","model":model,"auroc":ranking["auroc"],
                       "auprc":ranking["auprc_average_precision"],"selected_final_rounds":rounds[model]})
        print(f"CV {model}: AUROC={ranking['auroc']:.6f}, AP={ranking['auprc_average_precision']:.6f}, final rounds={rounds[model]}",flush=True)
    write_json(OUT / "rounds.json", rounds)
    pd.DataFrame(fold_records).to_csv(REPORT / "cv_fold_metrics.csv",index=False)
    pd.DataFrame(pooled).to_csv(REPORT / "cv_oof_metrics.csv",index=False)


def evaluate():
    results=[]; model_hashes={}; threshold_hashes={}
    y=np.load(DATA / "validation_labels.npy"); patients=pd.read_csv(DATA / "validation_patients.csv")
    for model in MODELS:
        directory=OUT / "final" / model
        p=np.load(directory / "validation_probability.npy")
        threshold=select_utility_threshold(y,p,patients)
        write_json(directory / "threshold.json",threshold)
        scored=score("validation",model,p,threshold["threshold"])
        np.testing.assert_allclose(scored["utility"]["normalized_utility"],threshold["max_validation_utility"],rtol=0,atol=1e-12)
        results.append(flatten("validation",model,scored))
        model_path=directory / ("model.ubj" if model=="xgboost" else "model.txt")
        model_hashes[model]=content_hash(model_path); threshold_hashes[model]=content_hash(directory / "threshold.json")
    write_json(OUT / "frozen_before_test.json",{"model_sha256":model_hashes,"threshold_sha256":threshold_hashes,
                                             "rounds_sha256":content_hash(OUT / "rounds.json"),
                                             "test_used_for_selection":False})
    # Features are deterministic, but defer their test construction until choices are frozen.
    build_split("test")
    x=np.load(DATA / "test_features.npy",mmap_mode="r")
    for model in MODELS:
        directory=OUT / "final" / model
        threshold=json.loads((directory / "threshold.json").read_text())["threshold"]
        booster=load_model(model,directory)
        probability=predict(model,booster,x)
        scored=score("test",model,probability,threshold)
        results.append(flatten("test",model,scored))
        path=directory / ("model.ubj" if model=="xgboost" else "model.txt")
        assert content_hash(path)==model_hashes[model]
        assert content_hash(directory / "threshold.json")==threshold_hashes[model]
        print("Frozen test: "+json.dumps(results[-1]),flush=True)
    return results


def flatten(split,model,result):
    return {"split":split,"model":model,"auroc":result["ranking"]["auroc"],
            "auprc":result["ranking"]["auprc_average_precision"],
            "utility_score":result["utility"]["normalized_utility"],"threshold":result["threshold"],
            "patients":result["patients"],"hours":result["hours"],"positive_hours":result["positive_hours"]}


def report(results):
    frame=pd.DataFrame(results); frame.to_csv(REPORT / "metrics.csv",index=False)
    for name in ["protocol.json","rounds.json","frozen_before_test.json"]:
        write_json(REPORT / name,json.loads((OUT / name).read_text()))
    records={m:json.loads((OUT / "final" / m / "results.json").read_text()) for m in MODELS}
    write_json(REPORT / "final_model_metadata.json",records)
    write_json(REPORT / "raw_provenance.json",json.loads((DATA / "raw_case.json").read_text()))
    pd.read_csv(DATA / "patient_splits.csv")[["patient_id","hospital","split","cv_fold","has_event","hours"]].to_csv(REPORT / "patient_splits.csv",index=False)
    cv=pd.read_csv(REPORT / "cv_fold_metrics.csv")
    means=cv.groupby("model")[["auroc","auprc"]].agg(["mean","std"])
    means.columns=["_".join(c) for c in means.columns]
    means.reset_index().to_csv(REPORT / "cv_mean_sd.csv",index=False)
    lines=["# 朋友的563维特征：XGBoost与LightGBM完整记录对比","",
           "使用main提交32ee355b9373e81a9d3bdb1f1b4a22883e04678e原样发布的augmented特征，保留全部入选患者的完整记录及原始标签。"
           "从Git中的原始PSV逐一验证SHA-256并按float64解析，再用原特征代码产生float32的563列矩阵；未复用旧float32原始数值缓存。","",
           "A+B各院按患者分层70/15/15，种子42；训练集三折按医院×事件分层，种子2026。名单与既有A+B实验逐人一致。"
           "训练27,911人/1,081,518小时；验证5,981人/233,738小时；测试5,982人/230,710小时。","",
           "## 完整记录逐小时结果","","| 集合 | 模型 | AUROC | AUPRC/AP | US |","|---|---|---:|---:|---:|"]
    for row in results: lines.append(f"| {row['split']} | {row['model']} | {row['auroc']:.6f} | {row['auprc']:.6f} | {row['utility_score']:.6f} |")
    lines += ["","验证集US是在该集合上直接最大化Utility得到的校准成绩，不能当独立泛化估计；测试集使用已冻结模型及验证阈值。"
              "US采用固定版本官方评分器，完整原标签不再次前移，每小时输出、不施加报警冷却。","",
              "## 公平性与训练设置","",
              "两者共用563列、患者集合及折号，原生处理NaN；不插补、不标准化、不裁剪，不做类别加权或重采样。"
              "固定深度4、约16叶、学习率0.05、最多1000轮、255 bins和L2=10；XGBoost min_child_weight=5，LightGBM min_data_in_leaf=200，"
              "两者参数含义不同，不能将它们理解为数学等价模型。","",
              "训练三折按AUROC早停（50轮，min_delta=1e-5）；外折同时用于选择轮数，所以CV仅为开发估计，并非完全嵌套的独立估计。"
              "每个模型取三折最佳轮数中位数，再用全部训练患者重训；验证集不参与拟合或早停，只选择Utility阈值。"
              "测试特征在模型/阈值冻结后才构建，未使用测试结果调整任何方案。","",
              "main没有发布朋友训练参数，因此本轮是预先固定配置的受控基线，不是截图模型的精确复现，也不是穷尽调参后的算法排名。"
              "源数据在历史实验中已被查看；本轮属于两家已知医院内的探索性患者留出评价，不是独立外院测试。"
              "完整记录指标包含发病后识别，不代表发病前六小时预警效果。","",
              "完整参数、软件版本、实际轮数、原始数据及特征源码校验值见同目录JSON；cv_fold_metrics.csv和cv_oof_metrics.csv保留训练CV结果。"
              "大型矩阵在processed_data/friend_563_comparison，模型与预测在outputs/friend_563_comparison，未加入Git。"]
    (REPORT / "report.md").write_text("\n".join(lines)+"\n")
    write_json(OUT / "completed.json",{"CV_models":6,"final_models":2,"evaluations":4,"test_used_for_selection":False})
    print(frame.to_string(index=False),flush=True)


def main():
    prepare()
    jobs([(fold,model) for fold in range(3) for model in MODELS])
    finish_cv()
    jobs([(-1,model) for model in MODELS])
    report(evaluate())


if __name__=="__main__": main()
