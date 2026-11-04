"""Export a verified single-model review without training or combining scores.

The original competition experiment remains frozen for provenance. This entry
point reads its small metric artifacts, checks the training-selected model IDs
and frozen thresholds, and writes the current single-model result table.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/single_model_review.json"
REPORT = ROOT / "reports/single_model_review"


def available(path: Path) -> Path:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError(path)
    if getattr(path.stat(), "st_flags", 0) & 0x40000000:
        raise OSError(f"Cloud-evicted input needs restoration before review: {path}")
    return path


def read_json(path: Path):
    return json.loads(available(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with available(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=".single-model-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
    os.replace(temporary, path)


def write_json(path: Path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def checked_inputs(config):
    assert config["schema_version"] == 1
    assert config["ensemble_enabled"] is False
    planned = config["planned_project_comparison"]
    assert planned["ensemble_enabled"] is False
    assert planned["primary_project_models"] == ["lr", "ebm", "lightgbm"]
    assert planned["status"] == "planning_only_no_new_fits"

    source_paths = {key: ROOT / config[key] for key in (
        "source_model_metrics_csv", "source_model_selection_json", "source_freeze_json",
        "source_final_selection_json", "source_independent_verification_json")}
    frozen = read_json(source_paths["source_freeze_json"])
    selected = read_json(source_paths["source_model_selection_json"])
    final = read_json(source_paths["source_final_selection_json"])
    assert frozen["test_used_for_selection"] is False
    assert sha256(source_paths["source_model_selection_json"]) == frozen["model_selection_sha256"]
    assert sha256(source_paths["source_final_selection_json"]) == frozen["selection_sha256"]
    assert read_json(source_paths["source_independent_verification_json"])["passed"] is True
    assert selected["auc_leader"] == config["primary_single_model"]
    assert selected["utility_leader"] == config["utility_alternative"]
    assert selected["binary_xgb_leader"] == config["supplementary_optimized_xgboost"]

    candidates = config["review_candidates"]
    identifiers = [candidate["id"] for candidate in candidates]
    assert len(identifiers) == len(set(identifiers)) == 5
    assert set(identifiers) == set(final["models"]) == set(frozen["final_candidate_ids"])
    assert not any("blend" in name.lower() or "ensemble" in name.lower() for name in identifiers)
    model_evidence = {}
    for candidate in candidates:
        identifier = candidate["id"]
        path = ROOT / "outputs/challenge_improvement/final" / identifier / "results.json"
        metadata = read_json(path)
        assert metadata["candidate"] == identifier and metadata["fold"] == -1
        assert metadata["feature_count"] == candidate["feature_count"]
        assert metadata["model_sha256"] == frozen["model_sha256"][identifier]
        assert metadata["test_read"] is False and metadata["validation_used_for_fitting"] is False
        model_evidence[str(path.relative_to(ROOT))] = sha256(path)
    source = source_paths["source_model_metrics_csv"]
    with available(source).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    assert fields and len(rows) == config["source_expected_rows"]
    assert len({(row["split"], row["candidate"]) for row in rows}) == len(rows)

    source_index = {(row["split"], row["candidate"]): row for row in rows}
    reviewed, evidence = [], {}
    float_fields = ("auroc", "auprc", "utility_score", "threshold", "hour_precision",
                    "hour_recall", "hour_fpr", "hour_f1", "hour_accuracy")
    integer_fields = ("patients", "hours", "positive_hours")
    for split, counts in config["expected_splits"].items():
        for candidate in candidates:
            identifier = candidate["id"]
            row = source_index[(split, identifier)]
            evaluated_path = ROOT / "outputs/challenge_improvement/evaluation" / split / identifier / "metrics.json"
            evaluated = read_json(evaluated_path)
            for name in float_fields:
                actual = float(row[name])
                assert math.isfinite(actual)
                assert math.isclose(actual, float(evaluated[name]), rel_tol=0, abs_tol=1e-12), (split, identifier, name)
            for name in integer_fields:
                assert int(row[name]) == int(evaluated[name]) == counts[name], (split, identifier, name)
            threshold_path = ROOT / "outputs/challenge_improvement/final" / identifier / "threshold.json"
            assert sha256(threshold_path) == frozen["threshold_sha256"][identifier]
            threshold = read_json(threshold_path)["threshold"]
            assert float(row["threshold"]) == threshold
            reviewed.append({**row, "review_role": candidate["role"],
                             "display_name": candidate["display_name"], "feature_count": candidate["feature_count"]})
            evidence[str(evaluated_path.relative_to(ROOT))] = sha256(evaluated_path)
    assert len(reviewed) == len(candidates) * len(config["expected_splits"]) == 10
    assert not any("blend" in row["candidate"].lower() or "ensemble" in row["candidate"].lower() for row in reviewed)
    provenance = {"config_sha256": sha256(CONFIG),
                  "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths.values()},
                  "source_evaluation_sha256": evidence,
                  "source_model_metadata_sha256": model_evidence,
                  "source_model_metrics_csv": str(source.relative_to(ROOT)),
                  "source_rows": len(rows), "exported_rows": len(reviewed), "single_models": len(candidates),
                  "rows_per_split": {split: sum(row["split"] == split for row in reviewed) for split in config["expected_splits"]},
                  "primary_single_model": config["primary_single_model"], "utility_alternative": config["utility_alternative"],
                  "selection_origin": "Previously frozen training-CV model_selection.json",
                  "new_training_performed": False, "new_threshold_selection_performed": False,
                  "score_combination_performed": False, "ensemble_enabled": False,
                  "historical_test_exposure": bool(frozen["historical_test_exposure"]),
                  "checks_passed": True}
    return reviewed, fields + ["review_role", "display_name", "feature_count"], provenance


def write_report(config, rows, loss_audit_run):
    lines = ["# 当前单模型分析入口", "",
             "当前主模型为训练交叉验证预先选出的789维、63叶LightGBM，US备选为791维LightGBM。"
             "优化XGBoost及两项563维基线作为附加比较；本入口直接复用已冻结模型的指标，不重新拟合或选择阈值。", "",
             "当前主流程关闭集成，结果表只包含单模型。此前包含集成的实验保留在历史冻结产物中，用于复核。", "",
             "## 固定单模型结果", "", "| 集合 | 角色 | 单模型 | 维度 | AUROC | AUPRC/AP | US |",
             "|---|---|---|---:|---:|---:|---:|"]
    roles = {"baseline": "基线", "supplementary_baseline": "附加基线", "primary_single_model": "主模型",
             "utility_alternative": "US备选", "supplementary_optimized": "附加对照"}
    for row in rows:
        lines.append(f"| {row['split']} | {roles[row['review_role']]} | {row['display_name']} | {row['feature_count']} | "
                     f"{float(row['auroc']):.6f} | {float(row['auprc']):.6f} | {float(row['utility_score']):.6f} |")
    lines += ["", "验证集US是阈值校准成绩。测试集沿用原先冻结阈值；本次整理不按测试成绩重新选择主模型。"
              "当前实验使用A+B各院固定70/15/15患者名单、完整原标签和逐小时报警，无冷却。"
              "两院均进入训练，且测试患者已有历史暴露，结果属于已知医院内的探索性患者留出比较。", "",
              "## 提案主线与第三模型", "",
              f"[原提案第5页]({config['proposal_model_slide']})的主三模型为L2逻辑回归、加性EBM与LightGBM，XGBoost作为附加树模型对照。"
              "如果在现有两棵树的对比中补充一种模型，EBM能补充结构和解释性；完整主表仍应保留LR简单基线。", "",
              "| 提案主模型 | 作用 | 新789维完整记录协议状态 |", "|---|---|---|",
              "| LR | 简单、可解释的基线；插补及标准化只在拟合患者内学习 | 尚未训练 |",
              "| 加性EBM | 提案的可解释性研究模型；interactions=0，原生NaN，保持患者隔离 | 尚未训练 |",
              "| LightGBM | 非线性性能对照；原生NaN | 已有上述冻结结果 |", "",
              "配置中的三模型字段仅是下一轮比较计划，本入口不会启动LR或EBM训练。"
              "新比较应先固定共同患者名单、小时纳入范围、特征信息、标签和报警规则，再在拟合患者内学习预处理。", "",
              "## 两种研究口径", "",
              "原提案研究未发病时未来六小时风险：A院开发、B院迁移，主要看发病前AUPRC、误报预算和预警时机；"
              "官方完整轨迹US是补充实验。近期789/791维结果则是A+B完整记录的竞赛补充分析。", "",
              "[已有A-only迁移结果](../a_only_cv_refit/report.md)采用312维、严格发病前AUROC/AP及误报预算确定的阈值。"
              "这些LR/EBM成绩不能直接拼入当前完整记录表；如选择当前新协议，需在同一口径下重新拟合它们。"
              "已有B院结果多次查看过，仍需保留探索性评价说明。", "",
              "## 复现", "", "```sh", ".venv/bin/python -B scripts/run_single_model_review.py", "```", "",
              "可选追加US损失审计：", "", "```sh", ".venv/bin/python -B scripts/run_single_model_review.py --with-loss-audit", "```", "",
              "输出[single_model_metrics.csv](single_model_metrics.csv)仅含5个单模型的验证/测试共10行；"
              "[provenance.json](provenance.json)记录来源SHA、数量和逐项一致性校验。"
              "历史完整结果见[原实验报告](../challenge_improvement/report.md)，"
              "效用目标对照见[原目标诊断](../challenge_improvement/objective_diagnostics.md)。", ""]
    if loss_audit_run:
        lines += ["本次命令已执行独立US损失审计，其结果位于`reports/us_loss_analysis/`。", ""]
    write_text(REPORT / "report.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-loss-audit", action="store_true",
                        help="Also run the separate US-loss audit without fitting models")
    args = parser.parse_args()
    config = read_json(CONFIG)
    if args.with_loss_audit:
        audit = available(ROOT / "scripts/analyze_us_losses.py")
        subprocess.run([sys.executable, "-B", str(audit)], cwd=ROOT, check=True)
    rows, fields, provenance = checked_inputs(config)
    REPORT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=REPORT,
                                     prefix=".single-model-", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, REPORT / "single_model_metrics.csv")
    provenance["loss_audit_run"] = args.with_loss_audit
    write_json(REPORT / "provenance.json", provenance)
    write_report(config, rows, args.with_loss_audit)
    print(f"Verified and exported {len(rows)} rows for {provenance['single_models']} single models.")
    print(f"Primary: {config['primary_single_model']}; US alternative: {config['utility_alternative']}")
    print(f"Report: {REPORT.relative_to(ROOT) / 'report.md'}")


if __name__ == "__main__":
    main()
