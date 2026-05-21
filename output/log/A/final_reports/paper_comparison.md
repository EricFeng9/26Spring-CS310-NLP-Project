# 任务A论文对照说明

## 1. 核心参考论文
- Wang et al. *Self-Consistency Improves Chain of Thought Reasoning*

## 2. 对照边界说明
- 本项目复现的是论文方法：单条 CoT 与 Self-Consistency 的对照。
- 本项目没有复刻原论文的同模型、同数据集、同超参数数值结果。
- 本项目当前正式实验模型为 `Qwen2.5-0.5B-Instruct`、`Qwen2.5-1.5B-Instruct`、`TinyLlama-1.1B-Chat`，与原论文大模型设置不同。

## 3. 方法趋势对照
- 论文核心结论：Self-Consistency 通常优于单条 CoT。
- 本项目在 18 组同模型同数据集对照中：
  - 提升：11
  - 持平：0
  - 下降：7
- 本项目结论：总体上是“部分一致”，SC 在部分模型和数据集上带来提升，但不是全局稳定优于单条 CoT。

结论判定规则：
- 如果提升组数占多数，则记为“趋势一致”。
- 如果提升和下降混合明显，则记为“部分一致”。
- 如果下降占多数，则记为“不一致”。

## 4. 正式 baseline 主结果表

| model | GSM8K zero_shot | GSM8K few_shot | GSM8K SC | CSQA zero_shot | CSQA few_shot | CSQA SC | MMLU zero_shot | MMLU few_shot | MMLU SC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `qwen2_5_0_5b_instruct` | 28.00% (28/100) | 10.00% (10/100) | 20.00% (20/100) | 33.00% (33/100) | 47.00% (47/100) | 43.00% (43/100) | 31.00% (31/100) | 23.00% (23/100) | 33.00% (33/100) |
| `qwen2_5_1_5b_instruct` | 56.00% (56/100) | 66.00% (66/100) | 67.00% (67/100) | 60.00% (60/100) | 66.00% (66/100) | 67.00% (67/100) | 41.00% (41/100) | 54.00% (54/100) | 50.00% (50/100) |
| `tinyllama_1_1b_chat` | 3.00% (3/100) | 2.00% (2/100) | 1.00% (1/100) | 20.00% (20/100) | 24.00% (24/100) | 27.00% (27/100) | 11.00% (11/100) | 16.00% (16/100) | 9.00% (9/100) |

## 5. CoT vs SC 对照表

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

## 6. SC 路径统计表

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

## 7. 可能原因
- 当前模型规模明显小于原论文主实验模型。
- 当前数据集口径包含 `MMLU`，与原论文主 benchmark 组合不同。
- 第三个模型使用的是 `TinyLlama`，推理能力弱于更大指令模型。
- 由于只采样 3 条路径，SC 的投票优势比论文常见设置更弱，容易出现“部分提升、部分回落”的情况。