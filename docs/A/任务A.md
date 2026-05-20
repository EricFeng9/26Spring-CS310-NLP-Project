# 任务A要求整理

## 任务定位

任务A负责完成 `Chain-of-Thought (CoT)` 与 `Self-Consistency (SC)` baseline 的复现，产出可复现、可统计、可对照论文趋势的正式实验结果。任务A只做 baseline，不引入额外改进方法。

任务A的核心目标有三个：

- 复现 `Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency`
- 比较单条 CoT 与 SC 的效果差异
- 给出与参考论文方法趋势的一致性结论

## 固定实验范围

### 固定模型

任务A复现的是主论文中的 CoT 与 Self-Consistency 方法机制，不限制具体模型 checkpoint。为适配当前集群硬件条件并保证 24 小时量级内完成完整矩阵，任务A正式纳入的 3 个轻量模型是：

- `qwen2_5_0_5b_instruct` (`Qwen/Qwen2.5-0.5B-Instruct`)
- `qwen2_5_1_5b_instruct` (`Qwen/Qwen2.5-1.5B-Instruct`)
- `tinyllama_1_1b_chat` (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`)

早期尝试的 `mistral_7b_instruct_v0_3`、`olmo3_7b_instruct` 和 `llama2_7b_hf` 因完整 Task A 运行时间过长，不作为最终正式口径。

### 固定推理模式

任务A只跑以下 3 类 baseline：

- `Zero-shot CoT`
- `Few-shot CoT`
- `Self-Consistency`

其中：

- 单条 CoT 指单次生成，不做多路径投票
- SC 指同一题采样多条推理路径，再做多数投票

### 固定数据集

任务A固定使用以下 3 个数据集：

- `GSM8K`
- `CSQA`
- `MMLU`

正式实验 split 口径固定为：

- `GSM8K`: `test`
- `CSQA`: `validation`
- `MMLU`: `validation`

### 固定任务边界

任务A阶段只做 baseline 复现，不做以下改动：

- 不加入复杂度筛选
- 不加入答案归一化投票
- 不加入语义投票
- 不加入复杂度感知提示
- 不做多模态
- 不做全量微调

这些内容属于任务 B / C 的扩展方向，不属于任务A正式交付范围。

## 参考论文与方法口径

任务A的 baseline 参考核心是：

- `Wang et al. Self-Consistency Improves Chain of Thought Reasoning`, NeurIPS 2022

方法口径应保持为：

- 单路径 CoT baseline
- sample-and-vote 的 Self-Consistency baseline

任务A要求对照的是“方法趋势”而不是逐数字复刻论文结果。也就是说，不要求结果和论文数值完全一致，但要求说明本项目中是否也观察到：

- `Self-Consistency` 通常优于单条 `CoT`

## 必须回答的问题

任务A最终必须回答以下问题：

1. 单条 `CoT` 与 `Self-Consistency` 相比，准确率是否提升。
2. 在 3 个轻量模型上，`Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency` 分别得到什么结果。
3. 本项目观察到的趋势是否与参考论文一致。

## 必须完成的实验

### 正式实验组合

任务A正式实验应覆盖：

- `3 个模型`
- `3 种推理模式`
- `3 个数据集`

总计：

- `3 × 3 × 3 = 27` 个正式配置

配置执行规则固定为：

- 由 `scripts/run_formal_model_group.py` 按 `model × mode × dataset` 动态生成 lightweight-100 配置

### 运行顺序要求

建议正式实验按以下顺序执行：

1. `zero_shot`
2. `few_shot`
3. `self_consistency`

原因是先跑低成本模式，再跑高成本模式，更利于集群调度与问题定位。

## 必须保留的输出

每个正式实验都必须保留：

- 逐题 `JSONL`
- 汇总 `CSV`
- 运行日志

推荐目录约定：

- 原始结果：`output/raw/A/final_runs/`
- 运行日志：`output/log/A/final_runs/`
- 最终表格：`output/log/A/final_tables/`
- 最终说明：`output/log/A/final_reports/`

## 必须产出的最终结果表

### 表1：正式 baseline 主结果表

作用：

- 展示不同模型、不同推理模式、不同数据集上的正式准确率

要求：

- 行：3 个模型
- 列：`dataset × mode`
- 指标：`accuracy`

### 表2：单条 CoT vs Self-Consistency 对照表

作用：

- 直接回答 SC 是否优于单条 CoT

建议字段：

- 模型名
- 单条 CoT 准确率
- SC 准确率
- 绝对提升

### 表3：Self-Consistency 路径统计表

作用：

- 证明 SC 真实执行了多路径采样与投票

建议字段：

- 路径数 `N`
- 成功抽取答案的路径比例
- 唯一答案数平均值
- 多数票占比平均值

### 表4：论文趋势对照表

作用：

- 对照参考论文，总结方法趋势是否一致

建议字段：

- 方法
- 论文中的结论
- 本项目结果趋势
- 是否一致

## 完成标准

以下条件同时满足，任务A才算完成：

- 已跑完 3 个模型的正式 baseline
- 已覆盖 `Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency`
- 已生成正式主结果表
- 已生成 `CoT vs SC` 对照表
- 已生成 `SC` 路径统计表
- 已写出论文趋势对照说明
- 已明确记录最终纳入 baseline 的模型名单
- 已明确记录未纳入模型及原因

## 不算完成的情况

以下情况都不算完成任务A：

- 只有 smoke test，没有正式全量结果
- 只有逐题输出，没有最终准确率表
- 只跑了 CoT，没有跑 SC
- 只跑了 SC，没有与单条 CoT 做对照
- 跑完后没有与论文趋势做总结

## 一句话交付定义

当项目中已经具备“3 个模型在 3 个数据集上的正式 CoT / SC baseline 结果、可复现的准确率表、单条 CoT 与 SC 的对照结论、以及与参考论文的方法趋势对照说明”时，任务A才算真正完成。

## 当前已完成结果快照

统计时间：`2026-05-20 10:05 CST`

当前正式口径：

- 模型：`qwen2_5_0_5b_instruct`、`qwen2_5_1_5b_instruct`、`tinyllama_1_1b_chat`
- 数据集：`GSM8K`、`CSQA`、`MMLU`
- 方法：`zero_shot`、`few_shot`、`self_consistency`
- 每个配置样本数：`100`
- 随机种子：`42`
- Self-Consistency 路径数：`3`
- 结果目录：`output/raw/A/lightweight_100/`

当前完成度：

- `zero_shot`：`9 / 9` 完成
- `few_shot`：`9 / 9` 完成
- `self_consistency`：`5 / 9` 完成
- 总计：`23 / 27` 完成

仍在运行或尚未完成的配置：

- `qwen2_5_1_5b_instruct × self_consistency × mmlu`
- `tinyllama_1_1b_chat × self_consistency × gsm8k`
- `tinyllama_1_1b_chat × self_consistency × csqa`
- `tinyllama_1_1b_chat × self_consistency × mmlu`

### 已完成主结果表

表中数值为准确率，括号内为 `correct / sample_count`。

| model | dataset | zero_shot | few_shot | self_consistency |
| --- | --- | --- | --- | --- |
| `qwen2_5_0_5b_instruct` | GSM8K | 28.00% (28/100) | 10.00% (10/100) | 24.00% (24/100) |
| `qwen2_5_0_5b_instruct` | CSQA | 16.00% (16/100) | 42.00% (42/100) | 22.00% (22/100) |
| `qwen2_5_0_5b_instruct` | MMLU | 6.00% (6/100) | 6.00% (6/100) | 11.00% (11/100) |
| `qwen2_5_1_5b_instruct` | GSM8K | 56.00% (56/100) | 66.00% (66/100) | 53.00% (53/100) |
| `qwen2_5_1_5b_instruct` | CSQA | 30.00% (30/100) | 65.00% (65/100) | 37.00% (37/100) |
| `qwen2_5_1_5b_instruct` | MMLU | 15.00% (15/100) | 37.00% (37/100) | 运行中 |
| `tinyllama_1_1b_chat` | GSM8K | 3.00% (3/100) | 2.00% (2/100) | 运行中 |
| `tinyllama_1_1b_chat` | CSQA | 16.00% (16/100) | 19.00% (19/100) | 待运行 |
| `tinyllama_1_1b_chat` | MMLU | 9.00% (9/100) | 16.00% (16/100) | 待运行 |

### 已完成 Self-Consistency 路径统计

`answer_extraction_success_rate` 表示 SC 的每条采样路径是否成功抽取到非空答案；`mean_unique_answers` 表示每题 3 条路径中平均产生多少个不同答案；`mean_majority_share` 表示多数票平均占比。

| model | dataset | sample_count | total_paths | answer_extraction_success_rate | mean_unique_answers | mean_majority_share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | GSM8K | 100 | 300 | 100.00% | 2.81 | 39.67% |
| `qwen2_5_0_5b_instruct` | CSQA | 100 | 300 | 100.00% | 2.46 | 51.33% |
| `qwen2_5_0_5b_instruct` | MMLU | 100 | 300 | 100.00% | 2.70 | 43.33% |
| `qwen2_5_1_5b_instruct` | GSM8K | 100 | 300 | 100.00% | 2.20 | 60.00% |
| `qwen2_5_1_5b_instruct` | CSQA | 100 | 300 | 100.00% | 1.54 | 82.00% |

### 当前中间观察

截至当前快照，Self-Consistency 在轻量模型上的提升并不稳定：

- `qwen2_5_0_5b_instruct`：SC 相比 zero-shot 在 CSQA、MMLU 有提升，但相比 few-shot 只在 MMLU 持平/略优。
- `qwen2_5_1_5b_instruct`：已完成的 GSM8K、CSQA 上，few-shot 明显强于 SC。
- `tinyllama_1_1b_chat`：SC 尚未完成；从 zero-shot/few-shot 看，模型本身推理和格式遵循能力较弱，后续分析需要单独说明。

该快照只是已完成结果统计，不作为最终结论。最终结论需要等待 27 个配置全部完成后，重新生成完整结果表、CoT vs SC 对照表和论文趋势说明。
