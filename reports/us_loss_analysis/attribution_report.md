# 训练外折病例的特征归因

主模型为789维、63叶LightGBM。仅使用训练患者的对应外折模型与原内部校准阈值，无拟合、无验证或测试读取。LightGBM内置TreeSHAP解释log-odds；80点的贡献和/逆sigmoid/保存预测已逐点核验。

选择20位误报小时最多的非脓毒症患者，解释第一次报警；在发病前6小时全漏及成功检出池中分别固定种子随机抽20人，解释发病前3小时及前6小时内第一次报警。原60点保留，并在同20位误报患者中补充当前已知ICULOS≥73h后的第一次报警，有20位符合；无该点者跳过，不换患者。首次与晚期点可能重合，均不是总体重要性。

显式入院时钟组共55列，包含ICULOS/HospAdmTime、其窗口统计/累计能量/末值/缺失统计以及prefix_length；测量年龄、测量间隔和采样次数属于测量模式，保留在时钟组之外。

| 病例组 | 时钟绝对贡献占比均值 | 中位数 | 时钟净贡献均值(log-odds) | 时钟净贡献为正比例 |
|---|---:|---:|---:|---:|
| top_nonseptic_false_alarm | 8.2% | 6.7% | +0.162 | 65.0% |
| sampled_pre6_missed | 6.5% | 6.5% | -0.138 | 5.0% |
| sampled_pre6_detected | 18.2% | 17.4% | +0.878 | 90.0% |
| same_patients_late_nonseptic_false_alarm | 23.2% | 22.2% | +1.323 | 100.0% |

绝对占比的分母为全部789列SHAP绝对值之和，不含模型基准截距；正负时钟贡献可相互抵消。病例错误不等于某个特征造成错误，相关特征间的归因分配也不是唯一机制证明。

解释时点的ICULOS中位数：same_patients_late_nonseptic_false_alarm 73.0h；sampled_pre6_detected 92.5h；sampled_pre6_missed 16.5h；top_nonseptic_false_alarm 22.5h。这些病例组没有按入院时长或病情匹配，组间归因差异不能当因果对照。

各组贡献最大的5项（按该组20点平均绝对贡献）：

top_nonseptic_false_alarm: sep_ICULOS_energy (0.110)；FiO2_current (0.091)；sep_Temp_last (0.083)；EtCO2_age6 (0.079)；FiO2_sd6 (0.076)
sampled_pre6_missed: FiO2_current (0.146)；sep_Temp_last (0.060)；FiO2_age6 (0.057)；FiO2_sd6 (0.056)；sep_SBP_run_mean (0.056)
sampled_pre6_detected: sep_ICULOS_energy (0.472)；sep_ICULOS_energy_entropy (0.228)；ICULOS (0.156)；sep_O2Sat_energy (0.146)；FiO2_current (0.114)
same_patients_late_nonseptic_false_alarm: sep_ICULOS_energy (0.683)；sep_ICULOS_energy_entropy (0.306)；ICULOS (0.231)；sep_O2Sat_energy (0.215)；sep_Resp_energy_entropy (0.134)

显式时钟组内最大的三项（该组病例平均绝对贡献占时钟组平均绝对贡献的比例）：

top_nonseptic_false_alarm: sep_ICULOS_energy (29.2%)；sep_HospAdmTime_energy (17.6%)；HospAdmTime (15.7%)
sampled_pre6_missed: sep_ICULOS_energy (25.2%)；sep_ICULOS_energy_entropy (18.4%)；sep_HospAdmTime_energy (17.5%)
sampled_pre6_detected: sep_ICULOS_energy (46.3%)；sep_ICULOS_energy_entropy (22.3%)；ICULOS (15.3%)
same_patients_late_nonseptic_false_alarm: sep_ICULOS_energy (50.1%)；sep_ICULOS_energy_entropy (22.4%)；ICULOS (17.0%)

可将显式时钟组消融列为训练患者内部的候选实验；本分析不保证去掉时钟能提高US，也不能用这些病例替代整体患者评价。未移除测量age/interval/count特征。O2Sat等生理变量的累计能量仍归入生理统计；它们也可能代理病程长度，因此移除55列不等于完全消除时间代理。

输出：[病例](attribution_cases.csv)、[每点top8](attribution_top_features.csv)、[按组汇总](attribution_group_summary.csv)、[全部特征均值](attribution_feature_summary.csv)、[同患者首报/晚期时钟对照](attribution_paired_fp_clock.csv)、[时钟组内特征](attribution_clock_feature_summary.csv)、[协议与重放校验](attribution_protocol.json)。
