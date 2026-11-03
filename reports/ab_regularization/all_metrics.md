# 全部指标汇总

本文件汇总聊天中报告的指标，并保存可复算的小时级 Accuracy、Precision、Recall、Specificity、F1、Balanced Accuracy 与混淆矩阵。所选模型来自已冻结的医院分层 CV；本次只读取已保存预测，未重新训练、选择参数或改变阈值。

三个模型均保留原 A+B 设置和权重，因此“原 A+B”与“本轮所选”的验证指标、阈值和 utility 相同。本轮未重新评分测试集。所有 CSV 保留未四舍五入的数值；下列 Markdown 浮点值显示六位小数，比例使用 0–1。

## 队列与评价单位

| cohort | patients | event_patients | eligible_hours | positive_hours | positive_hour_prevalence | full_utility_hours |
| --- | --- | --- | --- | --- | --- | --- |
| A_training | 14091 | 1109 | 469344 | 6654 | 0.014177 | — |
| B_training | 13820 | 633 | 448660 | 3798 | 0.008465 | — |
| pooled_training | 27911 | 1742 | 918004 | 10452 | 0.011386 | — |
| A_validation | 3020 | 238 | 101888 | 1428 | 0.014015 | 119455.000000 |
| B_validation | 2961 | 136 | 96822 | 816 | 0.008428 | 114283.000000 |
| pooled_validation | 5981 | 374 | 198710 | 2244 | 0.011293 | 233738.000000 |

发病前任务仅使用 eligible_hours；官方 utility 对同一批验证患者使用完整 raw 轨迹，共 233,738 小时、5,981 位患者。阳性比例很低，合并验证集始终预测阴性的 Accuracy 为 0.988707，因此不能凭 Accuracy 判断预警价值。

## 全部正则化候选

CV 指标先对 A/B 医院 AUPRC 取平均，再对三折取平均。候选必须平均提升至少 0.001 且至少改善两折，才替换原设置；否则保留 baseline。折间 SD 是描述性波动，不是置信区间。EBM 的 stronger/weaker 联合改变叶子下限及 L2：800/100 与 200/10，原设置为 400/50；LightGBM 对应 400/50 与 100/1，原设置为 200/10。LR 原 C=1，较小 C 表示更强 L2。

| model | candidate | mean_macro_auprc | fold_sd_macro_auprc | mean_A_auprc | mean_B_auprc | mean_training_macro_auprc | mean_fit_seconds | mean_gain_vs_baseline | improved_folds | eligible | oof_macro_auprc | oof_A_auprc | oof_B_auprc | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | baseline | 0.068094 | 0.004817 | 0.067849 | 0.068338 | 0.096287 | 415.036667 | 0.000000 | 0 | False | 0.066448 | 0.067258 | 0.065638 | True |
| ebm | stronger | 0.068517 | 0.004832 | 0.068177 | 0.068856 | 0.095521 | 464.423333 | 0.000423 | 3 | False | 0.066862 | 0.067596 | 0.066128 | False |
| ebm | weaker | 0.067442 | 0.004803 | 0.067349 | 0.067535 | 0.097432 | 403.986667 | -0.000651 | 0 | False | 0.065845 | 0.066738 | 0.064952 | False |
| lightgbm | baseline | 0.074536 | 0.002792 | 0.077503 | 0.071569 | 0.181587 | 90.310000 | 0.000000 | 0 | False | 0.072997 | 0.076554 | 0.069441 | True |
| lightgbm | stronger | 0.073269 | 0.004924 | 0.075399 | 0.071140 | 0.163565 | 110.336667 | -0.001266 | 1 | False | 0.071312 | 0.074024 | 0.068601 | False |
| lightgbm | weaker | 0.072514 | 0.005326 | 0.078283 | 0.066746 | 0.253114 | 87.996667 | -0.002021 | 1 | False | 0.070610 | 0.077176 | 0.064044 | False |
| lr | C_0p001 | 0.058031 | 0.003819 | 0.062212 | 0.053850 | 0.074214 | 2.653333 | 0.000732 | 2 | False | 0.056459 | 0.061565 | 0.051354 | False |
| lr | C_0p01 | 0.057365 | 0.003454 | 0.061440 | 0.053289 | 0.076460 | 4.330000 | 0.000066 | 3 | False | 0.055681 | 0.060704 | 0.050659 | False |
| lr | C_0p1 | 0.057348 | 0.003440 | 0.061579 | 0.053117 | 0.076765 | 6.070000 | 0.000049 | 3 | False | 0.055535 | 0.060588 | 0.050482 | False |
| lr | C_10p0 | 0.057287 | 0.003464 | 0.061272 | 0.053302 | 0.077040 | 6.520000 | -0.000012 | 0 | False | 0.055479 | 0.060286 | 0.050673 | False |
| lr | baseline | 0.057299 | 0.003458 | 0.061291 | 0.053307 | 0.077009 | 7.316667 | 0.000000 | 0 | False | 0.055498 | 0.060315 | 0.050681 | True |

所选模型训练折与 OOF 分数（macro 均指两院平均；OOF 是拼接各折未见患者预测后再评分）：

| model | mean_training_macro_auprc | mean_macro_auprc | fold_sd_macro_auprc | oof_A_auprc | oof_B_auprc | oof_macro_auprc |
| --- | --- | --- | --- | --- | --- | --- |
| ebm | 0.096287 | 0.068094 | 0.004817 | 0.067258 | 0.065638 | 0.066448 |
| lightgbm | 0.181587 | 0.074536 | 0.002792 | 0.076554 | 0.069441 | 0.072997 |
| lr | 0.077009 | 0.057299 | 0.003458 | 0.060315 | 0.050681 | 0.055498 |

本轮 CV 共 33 次拟合，主实验墙钟耗时 41.94 分钟；各候选 mean_fit_seconds 为单次拟合平均耗时，不能直接当作并行运行总时长。

## 完整训练集拟合指标

| model | candidate | evaluation | auroc | auprc_average_precision | brier_score | positive_hour_prevalence | macro_hospital_auprc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lr | baseline | A_training | 0.800191 | 0.071222 | 0.013609 | 0.014177 | 0.071113 |
| lr | baseline | B_training | 0.833703 | 0.071004 | 0.008124 | 0.008465 | 0.071113 |
| lr | baseline | pooled_training | 0.819497 | 0.070742 | 0.010928 | 0.011386 | 0.071113 |
| ebm | baseline | A_training | 0.835106 | 0.088142 | 0.013400 | 0.014177 | 0.090905 |
| ebm | baseline | B_training | 0.865828 | 0.093669 | 0.008005 | 0.008465 | 0.090905 |
| ebm | baseline | pooled_training | 0.852601 | 0.089990 | 0.010763 | 0.011386 | 0.090905 |
| lightgbm | baseline | A_training | 0.858507 | 0.130712 | 0.013142 | 0.014177 | 0.137048 |
| lightgbm | baseline | B_training | 0.886449 | 0.143383 | 0.007847 | 0.008465 | 0.137048 |
| lightgbm | baseline | pooled_training | 0.874007 | 0.135397 | 0.010554 | 0.011386 | 0.137048 |

训练分数用于观察拟合程度，不代表未见患者上的泛化性能。完整小型模型结果和逐折分数分别见 [model_metadata_archive.json](model_metadata_archive.json) 与 [cv_fold_results.csv](cv_fold_results.csv)。

## 固定验证集概率指标

| model | candidate | evaluation | auroc | auprc_average_precision | brier_score | positive_hour_prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| lr | baseline | A_validation | 0.785906 | 0.069482 | 0.013474 | 0.014015 |
| lr | baseline | B_validation | 0.815318 | 0.051222 | 0.008291 | 0.008428 |
| lr | baseline | pooled_validation | 0.804744 | 0.061108 | 0.010949 | 0.011293 |
| ebm | baseline | A_validation | 0.807117 | 0.072516 | 0.013405 | 0.014015 |
| ebm | baseline | B_validation | 0.835969 | 0.065572 | 0.008128 | 0.008428 |
| ebm | baseline | pooled_validation | 0.825288 | 0.069808 | 0.010833 | 0.011293 |
| lightgbm | baseline | A_validation | 0.816779 | 0.081466 | 0.013305 | 0.014015 |
| lightgbm | baseline | B_validation | 0.848486 | 0.079877 | 0.008013 | 0.008428 |
| lightgbm | baseline | pooled_validation | 0.837154 | 0.080251 | 0.010726 | 0.011293 |

AUROC/AUPRC 越大越好，Brier 越小越好。AUPRC 使用 Average Precision。各队列阳性比例不同，应在同一评价队列比较模型。概率校准目前报告 Brier，没有计算校准斜率或 ECE。

## 小时级分类指标与混淆矩阵

每个符合条件小时直接判断 probability >= threshold，不应用 6 小时报警冷却。hour_tp/fp/fn/tn 的单位均为小时；hour_recall 与患者级 event_sensitivity 的统计单位不同。

| model | candidate | evaluation | threshold | hour_accuracy | hour_precision | hour_recall | hour_specificity | hour_f1 | hour_balanced_accuracy | hour_tp | hour_fp | hour_fn | hour_tn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | baseline | A_validation | 0.045988 | 0.943487 | 0.085408 | 0.312325 | 0.952459 | 0.134135 | 0.632392 | 446 | 4776 | 982 | 95684 |
| lr | baseline | B_validation | 0.045988 | 0.967580 | 0.074074 | 0.247549 | 0.973700 | 0.114028 | 0.610624 | 202 | 2525 | 614 | 93481 |
| lr | baseline | pooled_validation | 0.045988 | 0.955226 | 0.081520 | 0.288770 | 0.962838 | 0.127146 | 0.625804 | 648 | 7301 | 1596 | 189165 |
| ebm | baseline | A_validation | 0.051064 | 0.944537 | 0.094176 | 0.343137 | 0.953086 | 0.147791 | 0.648112 | 490 | 4713 | 938 | 95747 |
| ebm | baseline | B_validation | 0.051064 | 0.968148 | 0.087036 | 0.292892 | 0.973887 | 0.134194 | 0.633390 | 239 | 2507 | 577 | 93499 |
| ebm | baseline | pooled_validation | 0.051064 | 0.956041 | 0.091710 | 0.324866 | 0.963251 | 0.143039 | 0.644058 | 729 | 7220 | 1515 | 189246 |
| lightgbm | baseline | A_validation | 0.051759 | 0.933839 | 0.086408 | 0.388655 | 0.941589 | 0.141383 | 0.665122 | 555 | 5868 | 873 | 94592 |
| lightgbm | baseline | B_validation | 0.051759 | 0.961920 | 0.091375 | 0.393382 | 0.966752 | 0.148302 | 0.680067 | 321 | 3192 | 495 | 92814 |
| lightgbm | baseline | pooled_validation | 0.051759 | 0.947522 | 0.088164 | 0.390374 | 0.953885 | 0.143842 | 0.672130 | 876 | 9060 | 1368 | 187406 |

本表保存为 [hourly_classification_metrics.csv](hourly_classification_metrics.csv)。概率指标、小时分类、患者报警与两种 utility 阈值已合并为 [all_model_metrics.csv](all_model_metrics.csv)，每个模型/验证队列一行。

## 患者级报警指标

以下采用 6 小时冷却；真报警位于发病前六小时，早于该窗口或非发病患者上的报警计为误报。阈值在同一个 A+B 混合验证集选择，目标为每 100 个符合条件的患者小时不超过 1 次误报；不保证 A/B 分别达标。漏检患者数及所有原始报警计数也保存在 all_model_metrics.csv。

| model | method | evaluation | threshold | suppression_hours | eligible_patient_hours | event_patients | detected_event_patients | event_sensitivity | false_alert_episodes | false_alerts_per_100_patient_hours | total_alert_episodes | true_alert_episodes | alert_ppv | median_true_alert_lead_hours |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | selected | A_validation | 0.045988 | 6 | 101888 | 238 | 99 | 0.415966 | 1219 | 1.196412 | 1318 | 99 | 0.075114 | 3.000000 |
| lr | selected | B_validation | 0.045988 | 6 | 96822 | 136 | 54 | 0.397059 | 620 | 0.640350 | 674 | 54 | 0.080119 | 3.000000 |
| lr | selected | pooled_validation | 0.045988 | 6 | 198710 | 374 | 153 | 0.409091 | 1839 | 0.925469 | 1992 | 153 | 0.076807 | 3.000000 |
| ebm | selected | A_validation | 0.051064 | 6 | 101888 | 238 | 106 | 0.445378 | 1139 | 1.117894 | 1245 | 106 | 0.085141 | 3.000000 |
| ebm | selected | B_validation | 0.051064 | 6 | 96822 | 136 | 63 | 0.463235 | 637 | 0.657908 | 700 | 63 | 0.090000 | 3.000000 |
| ebm | selected | pooled_validation | 0.051064 | 6 | 198710 | 374 | 169 | 0.451872 | 1776 | 0.893765 | 1945 | 169 | 0.086889 | 3.000000 |
| lightgbm | selected | A_validation | 0.051759 | 6 | 101888 | 238 | 107 | 0.449580 | 1173 | 1.151264 | 1280 | 107 | 0.083594 | 4.000000 |
| lightgbm | selected | B_validation | 0.051759 | 6 | 96822 | 136 | 68 | 0.500000 | 692 | 0.714714 | 760 | 68 | 0.089474 | 3.000000 |
| lightgbm | selected | pooled_validation | 0.051759 | 6 | 198710 | 374 | 175 | 0.467914 | 1865 | 0.938554 | 2040 | 175 | 0.085784 | 3.000000 |

## 官方完整轨迹 Utility Score

采用固定版本 [PhysioNet/CinC 2019 官方评分代码](../../third_party/physionet2019/README.md)，保留原始 SepsisLabel，不再平移标签。每小时直接二值判断，不套用报警冷却。0 对应始终不报警，1 对应本队列理想预测，负分允许。合并分数先累加患者效用再标准化，不是两院分数的算术平均。

原有预算阈值下，A-only、原 A+B 和本轮 CV 所选的结果及 A+B 减 A-only 差值区间：

| model | evaluation | A_B_baseline | A_B_cv | A_only | difference | lower | upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ebm | A_validation | 0.296569 | 0.296569 | 0.295837 | 0.000732 | -0.017746 | 0.016885 |
| ebm | B_validation | 0.268946 | 0.268946 | 0.189405 | 0.079541 | 0.045049 | 0.111222 |
| ebm | pooled_validation | 0.286452 | 0.286452 | 0.256857 | 0.029595 | 0.011642 | 0.047995 |
| lightgbm | A_validation | 0.337462 | 0.337462 | 0.307098 | 0.030363 | 0.013893 | 0.049587 |
| lightgbm | B_validation | 0.351943 | 0.351943 | 0.257977 | 0.093966 | 0.048791 | 0.138034 |
| lightgbm | pooled_validation | 0.342765 | 0.342765 | 0.289108 | 0.053657 | 0.031673 | 0.073368 |
| lr | A_validation | 0.263048 | 0.263048 | 0.279349 | -0.016300 | -0.039409 | 0.006145 |
| lr | B_validation | 0.233357 | 0.233357 | 0.102168 | 0.131189 | 0.094439 | 0.170715 |
| lr | pooled_validation | 0.252174 | 0.252174 | 0.214457 | 0.037717 | 0.016840 | 0.062910 |

以下为固定 0.5 阈值的完整对照；其接近零的效用不能与上表既定预算阈值混用：

| model | evaluation | A_B_baseline | A_B_cv | A_only |
| --- | --- | --- | --- | --- |
| ebm | A_validation | -0.000012544 | -0.000012544 | 0.000000000 |
| ebm | B_validation | 0.001186600 | 0.001186600 | 0.000000000 |
| ebm | pooled_validation | 0.000426637 | 0.000426637 | 0.000000000 |
| lightgbm | A_validation | 0.000000000 | 0.000000000 | 0.000000000 |
| lightgbm | B_validation | 0.000000000 | 0.000000000 | 0.000000000 |
| lightgbm | pooled_validation | 0.000000000 | 0.000000000 | 0.000000000 |
| lr | A_validation | 0.002217483 | 0.002217483 | 0.001254390 |
| lr | B_validation | 0.005231170 | 0.005231170 | 0.005585703 |
| lr | pooled_validation | 0.003321232 | 0.003321232 | 0.002840713 |

Utility 采用完整小时轨迹，与仅发病前任务是不同评价口径；当前成人新发队列也不是官方隐藏测试队列。原始 observed/best/inaction/worst 效用及标准化分母保存在 [utility/all_scores.csv](utility/all_scores.csv)，全部 36 组配对差值区间保存在 [utility/paired_effects.csv](utility/paired_effects.csv)。Utility 未用于本轮模型或阈值选择。

## 两院缺失分布

全部分析只使用 A+B 训练患者。原始缺失表示该小时没有观测；最近值缺失对生理指标采用六小时有效期，背景变量仍取当前值。

| hospital | patients | event_patients | event_patient_prevalence | median_eligible_hours | median_patient_raw_missing_fraction | mean_hour_raw_missing_fraction | mean_hour_latest_missing_fraction | hours_without_any_lab_measurement_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 14091 | 1109 | 0.078703 | 33.000000 | 0.700833 | 0.695049 | 0.509960 | 0.673438 |
| B | 13820 | 633 | 0.045803 | 33.000000 | 0.691792 | 0.699681 | 0.564053 | 0.743024 |

完整 40 变量的原始与最近值缺失率：

| variable | A_raw_hour_missing_rate | B_raw_hour_missing_rate | A_latest_hour_missing_rate | B_latest_hour_missing_rate |
| --- | --- | --- | --- | --- |
| HR | 0.072397 | 0.117022 | 0.024046 | 0.035174 |
| O2Sat | 0.109960 | 0.135751 | 0.030417 | 0.038595 |
| Temp | 0.653214 | 0.657161 | 0.102224 | 0.119913 |
| SBP | 0.138613 | 0.134028 | 0.071167 | 0.037884 |
| MAP | 0.094187 | 0.141207 | 0.026428 | 0.048415 |
| DBP | 0.460790 | 0.134271 | 0.418704 | 0.037933 |
| Resp | 0.092246 | 0.207863 | 0.030854 | 0.061505 |
| EtCO2 | 1.000000 | 0.918069 | 1.000000 | 0.891733 |
| BaseExcess | 0.885696 | 0.997653 | 0.616145 | 0.992625 |
| HCO3 | 0.915346 | 0.998088 | 0.562852 | 0.994143 |
| FiO2 | 0.845469 | 0.975311 | 0.546169 | 0.883299 |
| pH | 0.875473 | 0.975672 | 0.582475 | 0.887440 |
| PaCO2 | 0.903870 | 0.975578 | 0.625639 | 0.887133 |
| SaO2 | 0.945905 | 0.979307 | 0.807923 | 0.904438 |
| AST | 0.983882 | 0.980937 | 0.913731 | 0.880504 |
| BUN | 0.914248 | 0.941192 | 0.556485 | 0.648137 |
| Alkalinephos | 0.984301 | 0.980881 | 0.916049 | 0.880248 |
| Calcium | 0.947753 | 0.926773 | 0.681289 | 0.606595 |
| Chloride | 0.912354 | 0.993416 | 0.555015 | 0.970365 |
| Creatinine | 0.930354 | 0.941160 | 0.576880 | 0.647967 |
| Bilirubin_direct | 0.998413 | 0.997408 | 0.991855 | 0.983252 |
| Glucose | 0.870138 | 0.772313 | 0.446438 | 0.291747 |
| Lactate | 0.962801 | 0.979403 | 0.848546 | 0.907817 |
| Magnesium | 0.918652 | 0.949180 | 0.583133 | 0.693835 |
| Phosphate | 0.946913 | 0.968495 | 0.676489 | 0.808031 |
| Potassium | 0.885272 | 0.918067 | 0.468722 | 0.574718 |
| Bilirubin_total | 0.986890 | 0.980830 | 0.917977 | 0.879976 |
| TroponinI | 0.998634 | 0.980350 | 0.991499 | 0.882695 |
| Hct | 0.875296 | 0.938051 | 0.443310 | 0.642257 |
| Hgb | 0.906529 | 0.937552 | 0.548968 | 0.638292 |
| PTT | 0.948547 | 0.989270 | 0.695238 | 0.933908 |
| WBC | 0.920772 | 0.943717 | 0.594500 | 0.663306 |
| Fibrinogen | 0.991716 | 0.993884 | 0.959846 | 0.963850 |
| Platelets | 0.931195 | 0.942850 | 0.593245 | 0.659881 |
| Age | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Gender | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Unit1 | 0.497068 | 0.292259 | 0.497068 | 0.292259 |
| Unit2 | 0.497068 | 0.292259 | 0.497068 | 0.292259 |
| HospAdmTime | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| ICULOS | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

患者等权、统一发病比例及前 24 小时对照的全部数值在 [missingness/cross_hospital.csv](missingness/cross_hospital.csv)；按发病状态和 ICU 小时段的表分别为 [by_event.csv](missingness/by_event.csv) 与 [by_hour_band.csv](missingness/by_hour_band.csv)。这些描述性对照不构成因果调整，不能单凭差异推断缺失导致预测失败。

## 缺失模式辨别医院的诊断

只用训练患者完整符合条件轨迹的 40 项缺失比例识别医院，整体 OOF 医院 AUROC 为 0.999538；这些全段诊断特征没有进入逐小时 sepsis 模型。

| fold | patients | hospital_auroc | balanced_accuracy |
| --- | --- | --- | --- |
| 0 | 9304 | 0.999618 | 0.996468 |
| 1 | 9304 | 0.999577 | 0.997009 |
| 2 | 9303 | 0.999425 | 0.995819 |

## 不确定性与复现

外层验证集此前已经查看过，结果属于探索性评价。已有 bootstrap 针对患者配对差值，采用 300 次重采样，条件于已训练模型和固定阈值，未作多重比较校正；没有计算各 AUROC/AUPRC 的绝对置信区间。所选模型直接复用原权重时，差值及区间恒为 0，不能解释为一次重新训练的不确定性。

运行 `python scripts/report_all_metrics.py` 可从既有结果和本地预测重建本汇总；`python scripts/report_all_metrics.py --verify` 只核对重建结果与快照。输入预测和处理后数据继续保留在本地忽略目录，输入摘要和本脚本摘要见 [all_metrics_manifest.json](all_metrics_manifest.json)。生成过程中会重算验证 AUROC/AUPRC/Brier 并与既有结果核对，检查分类混淆矩阵总数和阳性小时数。

原研究报告与图见 [report.md](report.md)，utility 口径说明见 [utility/report.md](utility/report.md)。
