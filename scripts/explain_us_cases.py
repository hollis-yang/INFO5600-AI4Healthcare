"""Training-OOF case attribution for utility errors; no fitting or held-test reads.

Explain the first false alarm in the 20 nonseptic patients with most false-alarm
hours, onset-3h in 20 randomly selected pre6 misses, and the first pre6 alarm in
20 randomly selected pre6 detections. Add the earliest alarm at ICULOS>=73h
in those same false-alarm patients, without replacing ineligible patients.
Explanations use each patient's own
outer-fold model and LightGBM's native TreeSHAP contribution output.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BASE = ROOT / "processed_data/friend_563_comparison"
OUT = ROOT / "outputs/challenge_improvement"
REPORT = ROOT / "reports/us_loss_analysis"
PRIMARY = "lgb_bce_lag_observed_leaf63"
FEATURE_SET = "lag_observed"
SEED = 2026
N_PER_GROUP = 20


def available(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Cloud-evicted input; restore its original checksum first: {path}")
    return path


def read_json(path):
    return json.loads(available(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with available(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def feature_group(name):
    # Only explicit admission clocks and their transformations belong here.
    # Measurement age/interval/frequency deliberately remain separate.
    if "ICULOS" in name or "HospAdmTime" in name or name == "sep_prefix_length":
        return "admission_clock"
    if name in {"Age", "Gender", "sep_Age_last", "sep_Gender_last"}:
        return "demographics"
    if name.startswith("challenge__") and "__last2_observed_delta" in name:
        return "last_two_real_measurement_change"
    if (name.endswith(("_age6", "_missing_now", "_observed"))
            or "_run_" in name or "_observed_" in name or name.startswith("sep_coverage")
            or "__last_observed_age_h" in name or "__last2_observed_interval_h" in name):
        return "measurement_pattern"
    if name.startswith("challenge__") and "__lag" in name:
        return "recent_vital_snapshot"
    return "physiology_value_and_summary"


def load_training():
    import numpy as np
    import pandas as pd

    patients = pd.read_csv(available(BASE / "train_patients.csv"))
    labels = np.load(available(BASE / "train_labels.npy"), mmap_mode="r")
    scores = np.full(len(labels), np.nan)
    threshold = np.full(len(labels), np.nan)
    fold_records, provenance = {}, {}
    assert int(patients.hours.sum()) == len(labels)
    for fold in range(3):
        directory = OUT / "cv" / f"fold{fold}" / PRIMARY
        record = read_json(directory / "results.json")
        held = pd.read_csv(available(directory / "held_patients.csv"))
        original = patients.loc[patients.cv_fold.eq(fold)]
        np.testing.assert_array_equal(held.patient_id, original.patient_id)
        held_scores = np.load(available(directory / "held_score.npy"))
        assert sha256(directory / "held_score.npy") == record["score_sha256"]
        assert sha256(directory / "model.txt") == record["model_sha256"]
        assert len(held_scores) == int(held.hours.sum())
        for local, global_patient in zip(held.itertuples(index=False), original.itertuples(index=False)):
            a, b = int(global_patient.start), int(global_patient.stop)
            values = held_scores[int(local.start):int(local.stop)]
            assert len(values) == b-a
            scores[a:b] = values
            threshold[a:b] = float(record["calibration"]["threshold"])
        fold_records[fold] = record
        for name in ("results.json", "held_score.npy", "held_patients.csv", "model.txt"):
            path = directory / name
            provenance[str(path.relative_to(ROOT))] = sha256(path)
    assert np.isfinite(scores).all() and np.isfinite(threshold).all()
    assert patients.patient_id.is_unique
    return patients, labels, scores, threshold, fold_records, provenance


def select_cases(patients, labels, scores, thresholds):
    import numpy as np
    import pandas as pd

    nonseptic, misses, detected = [], [], []
    for patient in patients.itertuples(index=False):
        a, b = int(patient.start), int(patient.stop)
        y, p = labels[a:b], scores[a:b]
        cutoff = float(thresholds[a])
        assert np.all(thresholds[a:b] == cutoff)
        alarms = p >= cutoff
        common = {"patient_id": patient.patient_id, "hospital": patient.hospital,
                  "fold": int(patient.cv_fold), "patient_hours": b-a,
                  "first_hour": int(patient.first_hour),
                  "alarm_hours": int(alarms.sum()), "threshold": cutoff,
                  "patient_start": a, "patient_stop": b}
        if not y.any():
            if alarms.any():
                point = int(np.flatnonzero(alarms)[0])
                nonseptic.append({**common, "case_group": "top_nonseptic_false_alarm",
                                  "has_event": 0, "onset_row": None,
                                  "relative_to_onset_h": None, "patient_row": point,
                                  "explanation_point": "first_alarm"})
            continue
        onset = int(np.flatnonzero(y)[0]) + 6
        relative = np.arange(len(y)) - onset
        pre6 = (relative >= -6) & (relative < 0)
        hit = np.flatnonzero(alarms & pre6)
        if len(hit):
            point = int(hit[0])
            detected.append({**common, "case_group": "sampled_pre6_detected",
                             "has_event": 1, "onset_row": onset,
                             "relative_to_onset_h": int(relative[point]), "patient_row": point,
                             "explanation_point": "first_alarm_within_pre6"})
        else:
            point = onset-3
            assert 0 <= point < len(y) and not alarms[point]
            misses.append({**common, "case_group": "sampled_pre6_missed",
                           "has_event": 1, "onset_row": onset,
                           "relative_to_onset_h": -3, "patient_row": point,
                           "explanation_point": "onset_minus3"})
    nonseptic_pool_size = len(nonseptic)
    nonseptic = sorted(nonseptic, key=lambda p: (-p["alarm_hours"], p["patient_id"]))[:N_PER_GROUP]

    def random_sample(records, seed):
        records = sorted(records, key=lambda p: p["patient_id"])
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(records), size=min(N_PER_GROUP, len(records)), replace=False))
        return [records[int(i)] for i in indices]

    selected = nonseptic + random_sample(misses, SEED) + random_sample(detected, SEED+1)
    assert len(selected) == 3*N_PER_GROUP
    late_points = []
    for first in nonseptic:
        a, b = first["patient_start"], first["patient_stop"]
        current_hours = first["first_hour"] + np.arange(b-a)
        eligible = np.flatnonzero((scores[a:b] >= thresholds[a:b]) & (current_hours >= 73))
        if len(eligible):
            late_points.append({**first, "case_group": "same_patients_late_nonseptic_false_alarm",
                                "patient_row": int(eligible[0]),
                                "explanation_point": "first_alarm_at_ICULOS_at_least73"})
    selected += late_points
    cases = pd.DataFrame(selected)
    cases["global_row"] = cases.patient_start + cases.patient_row
    assert not cases.duplicated(["case_group", "patient_id"]).any()
    assert 3*N_PER_GROUP <= len(cases) <= 4*N_PER_GROUP
    return cases, {"false_alarm_eligible_patients": nonseptic_pool_size,
                   "pre6_missed_pool": len(misses), "pre6_detected_pool": len(detected),
                   "late_FP_same_patients_eligible": len(late_points)}


def run():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    import sepsis_warning.challenge_improvement as study

    patients, labels, scores, thresholds, fold_records, provenance = load_training()
    cases, pools = select_cases(patients, labels, scores, thresholds)
    previous_cases = REPORT / "attribution_cases.csv"
    if previous_cases.exists():
        previous = pd.read_csv(available(previous_cases)).head(3*N_PER_GROUP)
        pd.testing.assert_frame_equal(previous[["case_group", "patient_id", "global_row"]].reset_index(drop=True),
                                      cases.head(3*N_PER_GROUP)[["case_group", "patient_id", "global_row"]].reset_index(drop=True))
    names = read_json(OUT / "protocol.json")["feature_names"][FEATURE_SET]
    assert len(names) == len(set(names)) == 789
    groups = np.array([feature_group(name) for name in names], dtype=object)
    clocks = np.flatnonzero(groups == "admission_clock")
    assert len(clocks) == 55
    assert not any("age6" in names[i] or "last_observed_age" in names[i] for i in clocks)
    raw_features = study.load_features("train", FEATURE_SET, cases.global_row.to_numpy(dtype=int))
    assert raw_features.shape == (len(cases), len(names))
    attribution = np.empty((len(cases), len(names)+1), np.float64)
    predicted = np.empty(len(cases), np.float64)
    raw_score = np.empty(len(cases), np.float64)
    model_probes = []
    for fold in range(3):
        rows = np.flatnonzero(cases.fold.to_numpy() == fold)
        if not len(rows):
            continue
        fit, cal, held = study.cv_roles(fold)
        selected_ids = set(cases.iloc[rows].patient_id)
        assert selected_ids <= set(held.patient_id)
        assert not selected_ids & set(fit.patient_id)
        assert not selected_ids & set(cal.patient_id)
        model = lgb.Booster(model_file=str(available(OUT / "cv" / f"fold{fold}" / PRIMARY / "model.txt")))
        assert model.feature_name() == names
        selected_rounds = int(fold_records[fold]["selected_rounds"])
        attribution[rows] = model.predict(raw_features[rows], pred_contrib=True,
                                          num_iteration=selected_rounds, num_threads=4)
        predicted[rows] = model.predict(raw_features[rows], num_iteration=selected_rounds, num_threads=4)
        raw_score[rows] = model.predict(raw_features[rows], raw_score=True,
                                        num_iteration=selected_rounds, num_threads=4)
        model_probes.append({"fold": fold, "points": len(rows), "unique_patients": len(selected_ids),
                             "rounds": selected_rounds})
    np.testing.assert_allclose(attribution.sum(axis=1), raw_score, rtol=0, atol=1e-9)
    sigmoid = 1/(1+np.exp(-raw_score))
    np.testing.assert_allclose(sigmoid, predicted, rtol=0, atol=1e-12)
    expected = scores[cases.global_row.to_numpy(dtype=int)]
    np.testing.assert_allclose(predicted, expected, rtol=0, atol=1e-12)
    assert np.isfinite(attribution).all()
    cases["saved_score"] = expected
    cases["replayed_score"] = predicted
    cases["raw_log_odds"] = raw_score
    cases["expected_base_log_odds"] = attribution[:, -1]
    cases["prediction_absolute_error"] = np.abs(predicted-expected)
    cases["raw_score_reconstruction_error"] = np.abs(attribution.sum(axis=1)-raw_score)
    cases["current_ICULOS"] = raw_features[:, names.index("ICULOS")]
    cases["current_HospAdmTime"] = raw_features[:, names.index("HospAdmTime")]
    late_mask = cases.case_group.eq("same_patients_late_nonseptic_false_alarm")
    assert (cases.loc[late_mask, "current_ICULOS"] >= 73).all()
    assert (cases.loc[late_mask, "saved_score"] >= cases.loc[late_mask, "threshold"]).all()
    feature_shap = attribution[:, :-1]
    absolute_total = np.abs(feature_shap).sum(axis=1)
    positive_total = np.maximum(feature_shap, 0).sum(axis=1)
    group_rows, top_rows, all_feature_rows = [], [], []
    for i, case in enumerate(cases.itertuples(index=False)):
        clock = feature_shap[i, clocks]
        cases.loc[i, "clock_signed_log_odds"] = clock.sum()
        cases.loc[i, "clock_absolute_log_odds"] = np.abs(clock).sum()
        cases.loc[i, "clock_absolute_share"] = np.abs(clock).sum()/absolute_total[i]
        cases.loc[i, "clock_positive_support_share"] = np.maximum(clock, 0).sum()/positive_total[i] if positive_total[i] else 0.
        for group in sorted(set(groups)):
            index = np.flatnonzero(groups == group)
            contribution = feature_shap[i, index]
            group_rows.append({"case_group": case.case_group, "patient_id": case.patient_id,
                               "feature_group": group, "feature_count": len(index),
                               "signed_log_odds": float(contribution.sum()),
                               "absolute_log_odds": float(np.abs(contribution).sum()),
                               "absolute_share": float(np.abs(contribution).sum()/absolute_total[i]),
                               "positive_support_share": float(np.maximum(contribution, 0).sum()/positive_total[i]) if positive_total[i] else 0.})
        top = np.argsort(-np.abs(feature_shap[i]), kind="stable")[:8]
        for rank, index in enumerate(top, 1):
            top_rows.append({"case_group": case.case_group, "patient_id": case.patient_id,
                             "rank_absolute": rank, "feature": names[index],
                             "feature_group": groups[index], "feature_value": float(raw_features[i, index]),
                             "shap_log_odds": float(feature_shap[i, index])})
    group_frame = pd.DataFrame(group_rows)
    group_summary = group_frame.groupby(["case_group", "feature_group"], sort=True).agg(
        selected_points=("patient_id", "size"), mean_signed_log_odds=("signed_log_odds", "mean"),
        median_signed_log_odds=("signed_log_odds", "median"), mean_absolute_share=("absolute_share", "mean"),
        median_absolute_share=("absolute_share", "median"),
        mean_positive_support_share=("positive_support_share", "mean"),
        fraction_positive_signed=("signed_log_odds", lambda v: float((v > 0).mean()))).reset_index()
    for case_group in cases.case_group.unique():
        selected = cases.case_group.eq(case_group).to_numpy()
        for j, name in enumerate(names):
            values = feature_shap[selected, j]
            all_feature_rows.append({"case_group": case_group, "feature": name,
                                     "feature_group": groups[j], "selected_points": int(selected.sum()),
                                     "mean_absolute_log_odds": float(np.abs(values).mean()),
                                     "mean_signed_log_odds": float(values.mean())})
    feature_summary = pd.DataFrame(all_feature_rows).sort_values(
        ["case_group", "mean_absolute_log_odds", "feature"], ascending=[True, False, True])
    clock_summary = feature_summary.loc[feature_summary.feature_group.eq("admission_clock")].copy()
    clock_summary["share_of_clock_mean_absolute"] = clock_summary.mean_absolute_log_odds / clock_summary.groupby(
        "case_group").mean_absolute_log_odds.transform("sum")
    first_points = cases.loc[cases.case_group.eq("top_nonseptic_false_alarm")].set_index("patient_id")
    late_points = cases.loc[cases.case_group.eq("same_patients_late_nonseptic_false_alarm")].set_index("patient_id")
    paired_rows = []
    for patient_id, late in late_points.iterrows():
        first = first_points.loc[patient_id]
        paired_rows.append({"patient_id": patient_id, "first_ICULOS": first.current_ICULOS,
                            "late_ICULOS": late.current_ICULOS, "first_score": first.saved_score,
                            "late_score": late.saved_score, "same_explanation_row": first.global_row == late.global_row,
                            "first_clock_signed": first.clock_signed_log_odds,
                            "late_clock_signed": late.clock_signed_log_odds,
                            "clock_signed_difference": late.clock_signed_log_odds-first.clock_signed_log_odds,
                            "first_clock_absolute_share": first.clock_absolute_share,
                            "late_clock_absolute_share": late.clock_absolute_share,
                            "clock_share_difference": late.clock_absolute_share-first.clock_absolute_share})
    REPORT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(REPORT / "attribution_cases.csv", index=False)
    pd.DataFrame(top_rows).to_csv(REPORT / "attribution_top_features.csv", index=False)
    group_frame.to_csv(REPORT / "attribution_groups.csv", index=False)
    group_summary.to_csv(REPORT / "attribution_group_summary.csv", index=False)
    feature_summary.to_csv(REPORT / "attribution_feature_summary.csv", index=False)
    clock_summary.to_csv(REPORT / "attribution_clock_feature_summary.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(REPORT / "attribution_paired_fp_clock.csv", index=False)
    protocol = {"model": PRIMARY, "feature_set": FEATURE_SET, "features": len(names),
                "explained_points": len(cases), "unique_patients": int(cases.patient_id.nunique()),
                "training_only": True, "validation_read": False, "test_read": False, "model_fitted": False,
                "nonseptic_selection": "20 patients with most OOF false-alarm hours; patient_id breaks ties",
                "FN_selection": "Patient-ID-sorted pre6-miss pool, sample 20 without replacement, seed 2026",
                "TP_selection": "Patient-ID-sorted pre6-detected pool, sample 20 without replacement, seed 2027",
                "late_FP_selection": "Same top20 FP patients; first OOF alarm at observed ICULOS>=73h; skip if none, never replace",
                "points": {"FP": "first OOF alarm", "FN": "onset-3h", "TP": "first alarm within onset-6..onset-1",
                           "late_FP": "first OOF alarm at ICULOS>=73h"},
                "clock_feature_count": len(clocks), "clock_features": [names[i] for i in clocks],
                "feature_groups": {group: [name for name, assigned in zip(names, groups) if assigned == group]
                                   for group in sorted(set(groups))},
                "sample_pools": pools, "native_replay": model_probes,
                "maximum_probability_error": float(np.max(np.abs(predicted-expected))),
                "maximum_logit_reconstruction_error": float(np.max(np.abs(attribution.sum(axis=1)-raw_score))),
                "original_60_points_preserved": True,
                "same_row_in_first_and_late_FP": int(sum(p["same_explanation_row"] for p in paired_rows)),
                "provenance_sha256": provenance,
                "script_sha256": sha256(__file__),
                "limitations": ["Selected diagnostic cases are not representative global SHAP importance.",
                                "Native TreeSHAP log-odds association does not prove causal error mechanisms.",
                                "Correlated physiological and sampling variables can share or proxy admission-time information.",
                                "Final record length is used only to describe selected cases, never as a prediction feature."]}
    write_json(REPORT / "attribution_protocol.json", protocol)
    lines = ["# 训练外折病例的特征归因", "",
             "主模型为789维、63叶LightGBM。仅使用训练患者的对应外折模型与原内部校准阈值，无拟合、无验证或测试读取。"
             f"LightGBM内置TreeSHAP解释log-odds；{len(cases)}点的贡献和/逆sigmoid/保存预测已逐点核验。", "",
             "选择20位误报小时最多的非脓毒症患者，解释第一次报警；在发病前6小时全漏及成功检出池中分别固定种子随机抽20人，"
             "解释发病前3小时及前6小时内第一次报警。原60点保留，并在同20位误报患者中补充当前已知ICULOS≥73h后的第一次报警，"
             f"有{len(late_points)}位符合；无该点者跳过，不换患者。首次与晚期点可能重合，均不是总体重要性。", "",
             "显式入院时钟组共55列，包含ICULOS/HospAdmTime、其窗口统计/累计能量/末值/缺失统计以及prefix_length；"
             "测量年龄、测量间隔和采样次数属于测量模式，保留在时钟组之外。", "",
             "| 病例组 | 时钟绝对贡献占比均值 | 中位数 | 时钟净贡献均值(log-odds) | 时钟净贡献为正比例 |",
             "|---|---:|---:|---:|---:|"]
    for group in cases.case_group.unique():
        part = group_summary.loc[group_summary.case_group.eq(group) & group_summary.feature_group.eq("admission_clock")].iloc[0]
        lines.append(f"| {group} | {part.mean_absolute_share:.1%} | {part.median_absolute_share:.1%} | {part.mean_signed_log_odds:+.3f} | {part.fraction_positive_signed:.1%} |")
    lines += ["", "绝对占比的分母为全部789列SHAP绝对值之和，不含模型基准截距；正负时钟贡献可相互抵消。"
              "病例错误不等于某个特征造成错误，相关特征间的归因分配也不是唯一机制证明。", "",
              "解释时点的ICULOS中位数：" + "；".join(f"{group} {part.current_ICULOS.median():.1f}h"
                                                      for group, part in cases.groupby("case_group")) + "。"
              "这些病例组没有按入院时长或病情匹配，组间归因差异不能当因果对照。", "",
              "各组贡献最大的5项（按该组20点平均绝对贡献）：", ""]
    for group in cases.case_group.unique():
        top = feature_summary.loc[feature_summary.case_group.eq(group)].head(5)
        lines.append(group + ": " + "；".join(f"{p.feature} ({p.mean_absolute_log_odds:.3f})" for p in top.itertuples(index=False)))
    lines += ["", "显式时钟组内最大的三项（该组病例平均绝对贡献占时钟组平均绝对贡献的比例）：", ""]
    for group in cases.case_group.unique():
        top = clock_summary.loc[clock_summary.case_group.eq(group)].head(3)
        lines.append(group + ": " + "；".join(f"{p.feature} ({p.share_of_clock_mean_absolute:.1%})" for p in top.itertuples(index=False)))
    lines += ["", "可将显式时钟组消融列为训练患者内部的候选实验；本分析不保证去掉时钟能提高US，"
              "也不能用这些病例替代整体患者评价。未移除测量age/interval/count特征。"
              "O2Sat等生理变量的累计能量仍归入生理统计；它们也可能代理病程长度，因此移除55列不等于完全消除时间代理。", "",
              "输出：[病例](attribution_cases.csv)、[每点top8](attribution_top_features.csv)、"
              "[按组汇总](attribution_group_summary.csv)、[全部特征均值](attribution_feature_summary.csv)、"
              "[同患者首报/晚期时钟对照](attribution_paired_fp_clock.csv)、[时钟组内特征](attribution_clock_feature_summary.csv)、"
              "[协议与重放校验](attribution_protocol.json)。", ""]
    (REPORT / "attribution_report.md").write_text("\n".join(lines))
    print(group_summary.to_string(index=False))
    print(json.dumps({"points": len(cases), "unique_patients": int(cases.patient_id.nunique()),
                      "probability_error": protocol["maximum_probability_error"],
                      "logit_reconstruction_error": protocol["maximum_logit_reconstruction_error"]}))


def main():
    if os.environ.get("INFO5600_ATTRIBUTION_RUNTIME_READY") != "1":
        from sepsis_warning.friend_563 import runtime_environment
        environment = runtime_environment()
        environment["INFO5600_ATTRIBUTION_RUNTIME_READY"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        os.execve(sys.executable, [sys.executable, "-B", "-u", __file__, *sys.argv[1:]], environment)
    run()


if __name__ == "__main__":
    main()
