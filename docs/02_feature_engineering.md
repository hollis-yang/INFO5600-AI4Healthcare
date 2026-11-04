# 02 · 因果特征工程

目标是在每个小时生成当时可计算的特征，给后续模型使用。特征计算按患者独立执行，保留全部入选患者小时和原始标签。入口为 `sepsis_pipeline/build_features.py`，核心接口为 `sepsis_pipeline.features.transform(raw, profile)`。

## 主模型：563 维 augmented

| 模块 | 维度 | 定义 |
| --- | ---: | --- |
| 基础变量 | 208 | 34 个生命体征/检验变量 × 6 个统计量，加 Age、Gender、HospAdmTime、ICULOS |
| 数值及历史统计 | 201 | 9 个变量的 5/11 小时统计、累计能量、能量熵、历史平均差分、记录前缀长度和末次值 |
| 缺失与观测模式 | 154 | 36 个变量的观测/缺失连续段统计和累计观测比例等 144 维，加整体覆盖度及方差的 0～4 小时滞后值 10 维 |
| 合计 | **563** | 用于当前 XGBoost 主模型和 LightGBM 对照 |

基础 6 小时窗口是当前小时及前 5 小时。每个动态变量包括：窗口内末次观测值、观测均值、总体标准差、末次减最早观测值、距末次观测的小时数、当前是否缺失。窗口没有观测时保留 NaN，距末次观测记为 6；差值至少需要两次观测。

增强统计的 9 个变量为 HR、O2Sat、Temp、SBP、MAP、DBP、Resp、HospAdmTime、ICULOS。仅向前填充，然后计算 5/11 小时窗口的均值、最小、最大、中位数、总体方差和 95/99/5/1 百分位数。部分历史窗口仅使用已有记录。

累计能量为历史值平方和；能量熵基于归一化平方能量；平均差分使用截至当前的历史相邻差分。前缀长度只统计已经出现的行。缺失连续段统计包含已完成和当前未完成的观测/缺失段，不读取未来段长度。

该主特征集含明确住院时长和无界累计历史信息，可能与长记录患者持续高分有关；本次发布保持既有实现以便复现。

## 对照：410 维 recent

| 模块 | 维度 | 定义 |
| --- | ---: | --- |
| 紧凑基础变量 | 206 | 从基础 208 维去除 ICULOS 和 HospAdmTime |
| 局部窗口统计 | 204 | 34 个动态变量 × 6 个统计量：6/12 小时观测次数、12 小时均值/最小/最大、6 小时首末观测斜率 |
| 合计 | **410** | 去除显式时长和无界累计历史的对照实验 |

本对照最多使用当前及此前 11 个小时的数据，同时保留年龄和性别。缺失测量不会因基础末次值而被误算为新观测。它是对照方案，不能仅凭维度更少或去掉时长就声称效果更好。

## 输入与输出约束

- `transform` 每次只接收一位患者的 `n × 40` 输入；传入第 41 列标签会报错。
- 输出为 `float32`，每输入一小时对应一行特征，特征名称和顺序保存在 `feature_schema.json`。
- Unit1、Unit2 不参与两种特征集；医院、患者 ID、集合、折号、标签仅为元数据。
- NaN 保留给后续训练处理；无后向填充，无使用全体数据拟合的均值/中位数填补。后续填补、标准化或特征选择必须仅在当前训练折拟合。
- 每次生成特征前核对原始文件 SHA-256，确保使用与预处理相同的输入。

## 运行与接入模型

```bash
python -m sepsis_pipeline.build_features --profile augmented
python -m sepsis_pipeline.build_features --profile recent
# 只构建测试集合时：
python -m sepsis_pipeline.build_features --profile augmented --split test --output outputs/features-test
```

默认分别输出 `outputs/features/augmented/` 和 `outputs/features/recent/`；每个目录有 train/validation/test Parquet 和特征说明。分批写出以限制内存使用，`--batch-patients` 默认为 128。

```python
import json
import pandas as pd

directory = 'outputs/features/augmented'
with open(f'{directory}/feature_schema.json') as stream:
    schema = json.load(stream)
frame = pd.read_parquet(f'{directory}/train.parquet')
X = frame[schema['features']]  # 仅此清单是模型输入。
y = frame['label']
fold = frame['cv_fold']       # 用于患者级 CV，禁止放入 X。
```

逐小时样本可以用于拟合，但不得随机按小时重新划分数据。对训练折生成的风险分数做模型比较，验证集确定唯一阈值；同一测试集仅执行固定评估。
