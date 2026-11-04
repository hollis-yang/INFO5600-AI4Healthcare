# 当前单模型分析入口

当前主模型为训练交叉验证预先选出的789维、63叶LightGBM，US备选为791维LightGBM。优化XGBoost及两项563维基线作为附加比较；本入口直接复用已冻结模型的指标，不重新拟合或选择阈值。

当前主流程关闭集成，结果表只包含单模型。此前包含集成的实验保留在历史冻结产物中，用于复核。

## 固定单模型结果

| 集合 | 角色 | 单模型 | 维度 | AUROC | AUPRC/AP | US |
|---|---|---|---:|---:|---:|---:|
| validation | 基线 | LightGBM 基线 | 563 | 0.858210 | 0.143361 | 0.430379 |
| validation | 附加基线 | XGBoost 基线 | 563 | 0.858201 | 0.143791 | 0.433508 |
| validation | 主模型 | LightGBM 主模型（63叶） | 789 | 0.859889 | 0.148187 | 0.438971 |
| validation | US备选 | LightGBM US 备选 | 791 | 0.857820 | 0.146381 | 0.437841 |
| validation | 附加对照 | XGBoost 优化对照（深度5） | 789 | 0.858966 | 0.148186 | 0.442523 |
| test | 基线 | LightGBM 基线 | 563 | 0.847685 | 0.145551 | 0.403780 |
| test | 附加基线 | XGBoost 基线 | 563 | 0.845487 | 0.142896 | 0.402554 |
| test | 主模型 | LightGBM 主模型（63叶） | 789 | 0.850076 | 0.150794 | 0.413198 |
| test | US备选 | LightGBM US 备选 | 791 | 0.848840 | 0.150004 | 0.413894 |
| test | 附加对照 | XGBoost 优化对照（深度5） | 789 | 0.850820 | 0.145455 | 0.405514 |

验证集US是阈值校准成绩。测试集沿用原先冻结阈值；本次整理不按测试成绩重新选择主模型。当前实验使用A+B各院固定70/15/15患者名单、完整原标签和逐小时报警，无冷却。两院均进入训练，且测试患者已有历史暴露，结果属于已知医院内的探索性患者留出比较。

## 提案主线与第三模型

[原提案第5页](https://docs.google.com/presentation/d/1qQN6P4p-iK33ZLMlUx7Z8wyTcubnhfJMR9BgzA7NdmU/edit#slide=id.h45f296164c6b7e81_7_84)的主三模型为L2逻辑回归、加性EBM与LightGBM，XGBoost作为附加树模型对照。如果在现有两棵树的对比中补充一种模型，EBM能补充结构和解释性；完整主表仍应保留LR简单基线。

| 提案主模型 | 作用 | 新789维完整记录协议状态 |
|---|---|---|
| LR | 简单、可解释的基线；插补及标准化只在拟合患者内学习 | 尚未训练 |
| 加性EBM | 提案的可解释性研究模型；interactions=0，原生NaN，保持患者隔离 | 尚未训练 |
| LightGBM | 非线性性能对照；原生NaN | 已有上述冻结结果 |

配置中的三模型字段仅是下一轮比较计划，本入口不会启动LR或EBM训练。新比较应先固定共同患者名单、小时纳入范围、特征信息、标签和报警规则，再在拟合患者内学习预处理。

## 两种研究口径

原提案研究未发病时未来六小时风险：A院开发、B院迁移，主要看发病前AUPRC、误报预算和预警时机；官方完整轨迹US是补充实验。近期789/791维结果则是A+B完整记录的竞赛补充分析。

[已有A-only迁移结果](../a_only_cv_refit/report.md)采用312维、严格发病前AUROC/AP及误报预算确定的阈值。这些LR/EBM成绩不能直接拼入当前完整记录表；如选择当前新协议，需在同一口径下重新拟合它们。已有B院结果多次查看过，仍需保留探索性评价说明。

## 复现

```sh
.venv/bin/python -B scripts/run_single_model_review.py
```

可选追加US损失审计：

```sh
.venv/bin/python -B scripts/run_single_model_review.py --with-loss-audit
```

输出[single_model_metrics.csv](single_model_metrics.csv)仅含5个单模型的验证/测试共10行；[provenance.json](provenance.json)记录来源SHA、数量和逐项一致性校验。历史完整结果见[原实验报告](../challenge_improvement/report.md)，效用目标对照见[原目标诊断](../challenge_improvement/objective_diagnostics.md)。
