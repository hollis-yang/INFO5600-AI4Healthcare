# 竞赛方案启发的效用目标、时序特征与树模型优化

在已冻结的朋友563维基础上开展逐项对照，沿用A+B各院患者70/15/15名单及医院×事件分层三折。每个外折训练池再留15%患者作内部早停和阈值校准；其余85%拟合，外折不参与选择轮数与阈值。跨候选选择仍使用这些外折开发成绩，因此最佳候选的CV不是完全嵌套的无偏泛化估计。

## 方法与选择

比较原标签BCE、报警边际效用ΔU回归、效用标签1[ΔU>0]及权重|ΔU|的二分类。加权分类权重仅按拟合患者子集的均值缩放；回归分数不是概率。全部AUROC/AP仍针对原始SepsisLabel。

新增228维分成lag56维、真实观测历史170维、临床组合2维，对应619/789/791维逐步消融。所有新增输入只依赖当前及过去，不读标签；化验差值基于最近两次真实测量。临床组合为近期HR/SBP和SBP−DBP，不声称完整SOFA。大矩阵、模型留在本地processed_data/outputs。

US目标每10轮在内部校准患者上计算精确最大效用，连续8次无改善早停；普通分类按内部AUROC早停50轮。最终轮数为三折内层所选轮数中位数，再用全部27,911位训练患者重训。验证集仅在所有模型与集成选择冻结后选择最终单阈值。

本轮BCE基线的最终轮数重新由上述内层流程确定，与前一轮固定276/216轮的已发布模型可能不同；主要差异比较采用本轮同流程的BCE基线，不能把旧→新基线变化归因于新增特征或训练目标。

训练CV选择的AUROC模型：`lgb_bce_lag_observed_leaf63`；US模型：`lgb_bce_all`。
训练CV选择的集成：AUROC `blend_binary_half`；US `blend_binary_half`。

## 完整记录逐小时结果

| 集合 | 候选 | AUROC | AUPRC/AP | US |
|---|---|---:|---:|---:|
| validation | lgb_bce_base | 0.858210 | 0.143361 | 0.430379 |
| validation | xgb_bce_base | 0.858201 | 0.143791 | 0.433508 |
| validation | lgb_bce_lag_observed_leaf63 | 0.859889 | 0.148187 | 0.438971 |
| validation | lgb_bce_all | 0.857820 | 0.146381 | 0.437841 |
| validation | xgb_lag_observed_depth5 | 0.858966 | 0.148186 | 0.442523 |
| validation | blend_binary_half | 0.860295 | 0.150718 | 0.442497 |
| test | lgb_bce_base | 0.847685 | 0.145551 | 0.403780 |
| test | xgb_bce_base | 0.845487 | 0.142896 | 0.402554 |
| test | lgb_bce_lag_observed_leaf63 | 0.850076 | 0.150794 | 0.413198 |
| test | lgb_bce_all | 0.848840 | 0.150004 | 0.413894 |
| test | xgb_lag_observed_depth5 | 0.850820 | 0.145455 | 0.405514 |
| test | blend_binary_half | 0.851394 | 0.150718 | 0.414016 |

验证集US直接用于选阈值，具有校准乐观性。测试集采用冻结阈值，无报警冷却。已有测试患者在历史工作中被查看过；这是两家已知医院内的探索性固定留出比较，不是首次独立测试或外院验证。完整记录AUROC包含发病后识别；发病前六小时检测率单独保存在每个metrics.json中。

## 相对本轮563维LightGBM BCE基线的测试差异

| 训练CV选定候选 | ΔAUROC [95% CI] | ΔAUPRC [95% CI] | ΔUS [95% CI] |
|---|---|---|---|
| lgb_bce_lag_observed_leaf63 | +0.002391 [-0.000811, +0.005754] | +0.005244 [-0.004704, +0.015717] | +0.009418 [-0.003505, +0.024776] |
| lgb_bce_all | +0.001155 [-0.001561, +0.003684] | +0.004453 [-0.004458, +0.013256] | +0.010114 [-0.002037, +0.021808] |
| blend_binary_half | +0.003709 [+0.000898, +0.006771] | +0.005167 [-0.002874, +0.014140] | +0.010236 [-0.003571, +0.024734] |

500次患者配对bootstrap保留模型和阈值；区间反映评价样本变化，不包括训练及候选选择变化。若区间跨0，不能据此宣称稳定改进。分医院成绩、误报/召回、效用时段分解见outputs/.../evaluation。

## 复现

```bash
.venv/bin/python -u scripts/run_challenge_improvement.py --stage all
.venv/bin/python scripts/verify_challenge_improvement.py --final
```

脚本用原有隔离原生运行时加载XGBoost/LightGBM。详情见protocol.json、candidates.json和cv_fold_metrics.csv。本轮不重分患者，不引入外部预训练模型。Signature变换留待本轮目标、特征、容量与集成消融完成后再评估。

参考：[冠军效用回归与signature](https://www.cinc.org/archives/2019/pdf/CinC2019-014.pdf)、[SBU效用加权](https://www.cinc.org/archives/2019/pdf/CinC2019-459.pdf)、[Sepsyd历史快照](https://www.cinc.org/archives/2019/pdf/CinC2019-423.pdf)、[SailOcean集成](https://www.cinc.org/archives/2019/pdf/CinC2019-020.pdf)。
