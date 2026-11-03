# 完整轨迹官方 Utility Score 补充评价

使用 [PhysioNet/CinC 2019 官方评分代码](https://github.com/physionetchallenges/evaluation-2019/tree/467c49b514542be7a4a0bafe40fa2c3b064dda2e)，保留原始 BSD 2-Clause 许可及固定上游代码摘要。此项是在正则化 CV 运行期间按用户要求增加的补充评价，不参与已有的选参规则。

三个阶段均为共同 312 特征：A-only（上一轮 A+B 实验中的 A-only 风险模型）、A+B 原参数、当前训练 CV 选出的 A+B 版本。评价使用同一批 A 验证患者和 B 验证患者；患者划分、模型权重和既定阈值保持不变，测试集未评分。

## 原始阈值下的官方标准化效用

| model | evaluation | A_B_baseline | A_B_cv | A_only | CV_minus_AB_baseline |
| --- | --- | --- | --- | --- | --- |
| ebm | A_validation | 0.2966 | 0.2966 | 0.2958 | 0.0000 |
| ebm | B_validation | 0.2689 | 0.2689 | 0.1894 | 0.0000 |
| ebm | pooled_validation | 0.2865 | 0.2865 | 0.2569 | 0.0000 |
| lightgbm | A_validation | 0.3375 | 0.3375 | 0.3071 | 0.0000 |
| lightgbm | B_validation | 0.3519 | 0.3519 | 0.2580 | 0.0000 |
| lightgbm | pooled_validation | 0.3428 | 0.3428 | 0.2891 | 0.0000 |
| lr | A_validation | 0.2630 | 0.2630 | 0.2793 | 0.0000 |
| lr | B_validation | 0.2334 | 0.2334 | 0.1022 | 0.0000 |
| lr | pooled_validation | 0.2522 | 0.2522 | 0.2145 | 0.0000 |

0 对应始终不报警，1 对应本队列理想预测；负分表示比始终不报警更差，分数不是概率。合并分数先累加患者的原始效用，再统一标准化，不是两院分数的算术平均。

## 评分口径

- 每位患者恢复完整原始小时轨迹，包括发病后及原自定义任务删除的阴性末尾小时。保持原 SepsisLabel，不再移动标签。
- 官方函数从首次正标签位置加 6 小时重建基准发病时间，使用 -12/-6/+3 小时的效用窗口及官方误报/漏报代价。
- 每小时直接按 probability >= 原模型阈值产生二值预测，不套用自定义报警的 6 小时冷却期。所有历史阶段的原阈值均来自上轮同一 A+B 混合验证的误报预算校准；当前版本使用本轮同一混合验证的校准阈值。
- 本轮没有为 utility 调整模型或阈值。另列固定 0.5 阈值作为敏感性对照，详见 all_scores.csv。原误报预算只约束自定义任务的报警事件，不代表这些完整轨迹逐小时预测也符合相同报警负担。
- 模型训练仍是发病前六小时风险任务，完整轨迹 utility 是另外的评分视角，不替换原 AUPRC 或报警评价。该成人新发队列也是官方数据的子集，不能将分数直接当作挑战赛隐藏测试成绩。

## 患者配对差值区间

| reference | candidate | model | evaluation | difference | lower | upper |
| --- | --- | --- | --- | --- | --- | --- |
| A_only | A_B_baseline | lr | A_validation | -0.0163 | -0.0394 | 0.0061 |
| A_B_baseline | A_B_cv | lr | A_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | ebm | A_validation | 0.0007 | -0.0177 | 0.0169 |
| A_B_baseline | A_B_cv | ebm | A_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | lightgbm | A_validation | 0.0304 | 0.0139 | 0.0496 |
| A_B_baseline | A_B_cv | lightgbm | A_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | lr | B_validation | 0.1312 | 0.0944 | 0.1707 |
| A_B_baseline | A_B_cv | lr | B_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | ebm | B_validation | 0.0795 | 0.0450 | 0.1112 |
| A_B_baseline | A_B_cv | ebm | B_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | lightgbm | B_validation | 0.0940 | 0.0488 | 0.1380 |
| A_B_baseline | A_B_cv | lightgbm | B_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | lr | pooled_validation | 0.0377 | 0.0168 | 0.0629 |
| A_B_baseline | A_B_cv | lr | pooled_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | ebm | pooled_validation | 0.0296 | 0.0116 | 0.0480 |
| A_B_baseline | A_B_cv | ebm | pooled_validation | 0.0000 | 0.0000 | 0.0000 |
| A_only | A_B_baseline | lightgbm | pooled_validation | 0.0537 | 0.0317 | 0.0734 |
| A_B_baseline | A_B_cv | lightgbm | pooled_validation | 0.0000 | 0.0000 | 0.0000 |

按患者配对重采样 300 次，区间条件于已训练模型及既定阈值，多组比较未做多重校正；验证集此前已查看过，结果仍属探索性。完整患者效用分解和概率保存在本地 outputs/ab_regularization/utility，规范、摘要和对比表保存在本报告目录。
