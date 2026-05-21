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

## 最终结果

统计时间：`2026-05-21`

当前正式口径已全部完成：

- 模型：`qwen2_5_0_5b_instruct`、`qwen2_5_1_5b_instruct`、`tinyllama_1_1b_chat`
- 数据集：`GSM8K`、`CSQA`、`MMLU`
- 方法：`zero_shot`、`few_shot`、`self_consistency`
- 每个配置样本数：`100`
- 随机种子：`42`
- Self-Consistency 路径数：`3`
- 结果目录：`output/raw/A/lightweight_100/`

完成度：

- `zero_shot`：`9 / 9`
- `few_shot`：`9 / 9`
- `self_consistency`：`9 / 9`
- 总计：`27 / 27`

### 正式 baseline 主结果表

表中数值为 `accuracy`，括号内为 `correct / sample_count`。

| model | GSM8K zero_shot | GSM8K few_shot | GSM8K SC | CSQA zero_shot | CSQA few_shot | CSQA SC | MMLU zero_shot | MMLU few_shot | MMLU SC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_0_5b_instruct` | 28.00% (28/100) | 10.00% (10/100) | 20.00% (20/100) | 33.00% (33/100) | 47.00% (47/100) | 43.00% (43/100) | 31.00% (31/100) | 23.00% (23/100) | 33.00% (33/100) |
| `qwen2_5_1_5b_instruct` | 56.00% (56/100) | 66.00% (66/100) | 67.00% (67/100) | 60.00% (60/100) | 66.00% (66/100) | 67.00% (67/100) | 41.00% (41/100) | 54.00% (54/100) | 50.00% (50/100) |
| `tinyllama_1_1b_chat` | 3.00% (3/100) | 2.00% (2/100) | 1.00% (1/100) | 20.00% (20/100) | 24.00% (24/100) | 27.00% (27/100) | 11.00% (11/100) | 16.00% (16/100) | 9.00% (9/100) |

### CoT vs SC 对照表

| model | dataset | zero_shot | few_shot | zero_shot_sc | few_shot_sc |
| --- | --- | --- | --- | --- | --- |
| `qwen2_5_0_5b_instruct` | GSM8K | 28.00% | 10.00% | 20.00% | 20.00% |
| `qwen2_5_0_5b_instruct` | CSQA | 33.00% | 47.00% | 43.00% | 43.00% |
| `qwen2_5_0_5b_instruct` | MMLU | 31.00% | 23.00% | 33.00% | 33.00% |
| `qwen2_5_1_5b_instruct` | GSM8K | 56.00% | 66.00% | 67.00% | 67.00% |
| `qwen2_5_1_5b_instruct` | CSQA | 60.00% | 66.00% | 67.00% | 67.00% |
| `qwen2_5_1_5b_instruct` | MMLU | 41.00% | 54.00% | 50.00% | 50.00% |
| `tinyllama_1_1b_chat` | GSM8K | 3.00% | 2.00% | 1.00% | 1.00% |
| `tinyllama_1_1b_chat` | CSQA | 20.00% | 24.00% | 27.00% | 27.00% |
| `tinyllama_1_1b_chat` | MMLU | 11.00% | 16.00% | 9.00% | 9.00% |

### SC 路径统计表

| model | dataset | sample_count | total_paths | answer_extraction_success_rate | mean_unique_answers | mean_majority_share |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | GSM8K | 100 | 300 | 100.00% | 2.85 | 38.33% |
| `qwen2_5_0_5b_instruct` | CSQA | 100 | 300 | 100.00% | 1.73 | 75.67% |
| `qwen2_5_0_5b_instruct` | MMLU | 100 | 300 | 100.00% | 2.37 | 54.33% |
| `qwen2_5_1_5b_instruct` | GSM8K | 100 | 300 | 100.00% | 1.92 | 69.33% |
| `qwen2_5_1_5b_instruct` | CSQA | 100 | 300 | 100.00% | 1.18 | 94.00% |
| `qwen2_5_1_5b_instruct` | MMLU | 100 | 300 | 100.00% | 1.63 | 79.00% |
| `tinyllama_1_1b_chat` | GSM8K | 100 | 300 | 100.00% | 2.93 | 35.67% |
| `tinyllama_1_1b_chat` | CSQA | 100 | 300 | 100.00% | 2.15 | 61.67% |
| `tinyllama_1_1b_chat` | MMLU | 100 | 300 | 100.00% | 2.78 | 40.67% |

### 结论

- `SC` 并非在所有模型和数据集上都优于单条 `CoT`，但它确实在部分设置下带来了稳定增益。
- 本项目与参考论文的结论是 `部分一致`，而不是对论文数值或增益幅度的直接复刻。
- 从模型规模看，`qwen2_5_1_5b_instruct` 的结果最接近论文趋势，`tinyllama_1_1b_chat` 偏离最明显，这也说明在本项目的硬件约束下，模型规模会显著影响 SC 的收益。
- 从数据集看，`CSQA` 上 SC 更容易体现投票收益，`GSM8K` 更依赖模型本身的推理质量，`MMLU` 则介于两者之间。

更具体地说：

- `qwen2_5_0_5b_instruct`
  - `SC` 相比 `zero_shot` 在 `CSQA` 和 `MMLU` 上有提升，但在 `GSM8K` 上回落。
  - `SC` 相比 `few_shot` 只在 `GSM8K` 和 `MMLU` 上略有收益，在 `CSQA` 上略低。
  - 说明对 0.5B 模型而言，SC 可以修正一部分单条 CoT 的随机性，但还不足以稳定压过更强的 few-shot baseline。

- `qwen2_5_1_5b_instruct`
  - `SC` 相比 `zero_shot` 在三个数据集上都提升，说明模型能力提升后，SC 的多数投票开始更容易发挥作用。
  - `SC` 相比 `few_shot` 在 `GSM8K` 和 `CSQA` 上基本持平或略高，在 `MMLU` 上略低。
  - 这组结果最符合论文主趋势，但也显示出 few-shot 仍然可能与 SC 形成竞争。

- `tinyllama_1_1b_chat`
  - `SC` 只在 `CSQA` 上明显优于 `zero_shot`，在 `GSM8K` 和 `MMLU` 上反而回落。
  - `SC` 相比 `few_shot` 在三个数据集上都没有形成稳定优势。
  - 这说明在较弱模型上，SC 的多路径采样会放大路径噪声，投票未必能抵消模型本身的推理不足。

- 从 SC 路径统计看，`qwen2_5_1_5b_instruct` 在 `CSQA` 上的 `mean_unique_answers` 最低、`mean_majority_share` 最高，说明路径间更容易形成一致答案，这也是它更接近论文趋势的重要原因之一。
- 相比之下，`tinyllama_1_1b_chat` 的路径多样性更高、共识更弱，SC 的票选优势被稀释，因此结果更不稳定。

### 交付文件

- 主结果表：`output/log/A/final_tables/baseline_main_results.csv`
- CoT vs SC 对照表：`output/log/A/final_tables/cot_vs_sc_comparison.csv`
- SC 路径统计表：`output/log/A/final_tables/sc_path_stats.csv`
- 论文对照说明：`output/log/A/final_reports/paper_comparison.md`
