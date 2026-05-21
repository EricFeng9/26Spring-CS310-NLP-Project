# 任务B结果分析

## 1. 实验口径

任务B复用任务A的 Self-Consistency 结果，不重新生成模型输出。

输入目录：

```text
output/raw/A/lightweight_100/self_consistency/
```

实验模型：

- `qwen2_5_0_5b_instruct`
- `qwen2_5_1_5b_instruct`
- `tinyllama_1_1b_chat`

数据集：

- `gsm8k`
- `csqa`
- `mmlu`

对比方法：

- `original_sc`
- `normalized_vote`
- `filtered_vote`
- `normalized_filtered_vote`

消融实验：

- 组件消融：只归一化、只过滤、归一化加过滤
- 路径数消融：只使用前 `1 / 2 / 3` 条 SC 路径
- 数据集平均消融：跨 3 个模型统计每类数据集的平均收益

诊断实验：

- Oracle 上界：统计多条路径中是否至少有一条包含正确答案
- 多数票置信度分桶：按 `1/3`、`2/3`、`3/3` 投票强度统计准确率
- 正确性转移矩阵：统计从原始 SC 到改进 SC 的错转对、对转错

## 2. 主结果

| model | dataset | original_sc | normalized_vote | filtered_vote | normalized_filtered_vote |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | 43% | 43% | 43% | 43% |
| `qwen2_5_0_5b_instruct` | GSM8K | 20% | 45% | 27% | 45% |
| `qwen2_5_0_5b_instruct` | MMLU | 33% | 35% | 36% | 35% |
| `qwen2_5_1_5b_instruct` | CSQA | 67% | 67% | 67% | 67% |
| `qwen2_5_1_5b_instruct` | GSM8K | 67% | 69% | 67% | 69% |
| `qwen2_5_1_5b_instruct` | MMLU | 50% | 50% | 50% | 50% |
| `tinyllama_1_1b_chat` | CSQA | 27% | 26% | 29% | 29% |
| `tinyllama_1_1b_chat` | GSM8K | 1% | 4% | 3% | 5% |
| `tinyllama_1_1b_chat` | MMLU | 9% | 14% | 22% | 23% |

## 3. 关键观察

- `qwen2_5_0_5b_instruct` 在 GSM8K 上收益最大，`normalized_filtered_vote` 从 `20%` 提升到 `45%`，说明原始 SC 中有大量可通过数字归一化合并的等价答案。
- `qwen2_5_1_5b_instruct` 的原始输出已经较稳定，任务B改进主要在 GSM8K 上有效，归一化后准确率从 `67%` 提升到 `69%`。
- `tinyllama_1_1b_chat` 的输出格式更不稳定，因此过滤和归一化收益更明显，MMLU 从 `9%` 提升到 `23%`。
- 路径过滤对弱模型更有价值，因为弱模型更容易输出非标准选项、长段解释或无法直接投票的答案。
- 对 CSQA 的强模型结果没有变化，原因是原始路径的答案抽取成功率已经是 `100%`，平均多数票占比达到 `94%`。

## 4. 路径有效性

`normalized_filtered_vote` 的有效路径率：

| model | dataset | valid_path_rate | failure_rate |
| --- | --- | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | 100.00% | 0.00% |
| `qwen2_5_0_5b_instruct` | GSM8K | 99.67% | 0.33% |
| `qwen2_5_0_5b_instruct` | MMLU | 89.67% | 10.33% |
| `qwen2_5_1_5b_instruct` | CSQA | 100.00% | 0.00% |
| `qwen2_5_1_5b_instruct` | GSM8K | 100.00% | 0.00% |
| `qwen2_5_1_5b_instruct` | MMLU | 97.33% | 2.67% |
| `tinyllama_1_1b_chat` | CSQA | 90.67% | 9.33% |
| `tinyllama_1_1b_chat` | GSM8K | 98.00% | 2.00% |
| `tinyllama_1_1b_chat` | MMLU | 65.67% | 34.33% |

TinyLlama 在 MMLU 上的无效路径比例最高，说明原始 SC 的多数投票被大量格式异常或不可投票答案干扰。Qwen0.5B 在 GSM8K 上归一化后有效路径率接近 `100%`，但投票结果变化率很高，说明主要问题不是答案缺失，而是同一数值答案被不同格式拆散。

## 5. 投票变化

`normalized_filtered_vote` 的投票变化率：

| model | dataset | vote_changed_rate | wrong_to_correct | correct_to_wrong |
| --- | --- | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | 0% | 0 | 0 |
| `qwen2_5_0_5b_instruct` | GSM8K | 64% | 26 | 1 |
| `qwen2_5_0_5b_instruct` | MMLU | 17% | 2 | 0 |
| `qwen2_5_1_5b_instruct` | CSQA | 0% | 0 | 0 |
| `qwen2_5_1_5b_instruct` | GSM8K | 3% | 2 | 0 |
| `qwen2_5_1_5b_instruct` | MMLU | 2% | 0 | 0 |
| `tinyllama_1_1b_chat` | CSQA | 8% | 2 | 0 |
| `tinyllama_1_1b_chat` | GSM8K | 32% | 4 | 0 |
| `tinyllama_1_1b_chat` | MMLU | 46% | 14 | 0 |

这说明任务B方法主要修正了原始 SC 中由格式问题导致的错误投票。`qwen2_5_0_5b_instruct` 在 GSM8K 上有 1 个样本从正确变错误，需要在报告中说明后处理不是无风险的；总体收益仍明显大于损失。

## 6. 结论

任务B的轻量改进有效，尤其适用于输出格式不稳定的弱模型。答案归一化可以合并等价答案，路径过滤可以减少无效答案对多数投票的干扰。对于已经稳定输出标准答案的模型和数据集，后处理收益有限。

任务B最终推荐报告主方法使用 `normalized_filtered_vote`，因为它在主实验范围内整体收益最大，同时完全符合不训练、不改模型参数、只做 SC 后处理的要求。

## 7. 组件消融

跨 3 个模型的平均结果如下：

| dataset | original_sc | normalized_vote | filtered_vote | normalized_filtered_vote |
| --- | ---: | ---: | ---: | ---: |
| CSQA | 45.67% | 45.33% | 46.33% | 46.33% |
| GSM8K | 29.33% | 39.33% | 32.33% | 39.67% |
| MMLU | 30.67% | 33.00% | 36.00% | 36.00% |

组件结论：

- GSM8K 的主要收益来自答案归一化，平均准确率从 `29.33%` 提升到 `39.33%`。
- MMLU 的主要收益来自路径过滤，平均准确率从 `30.67%` 提升到 `36.00%`。
- CSQA 原始答案已经相对规范，平均收益较小，过滤后只提升 `0.67%`。
- `normalized_filtered_vote` 在三个数据集平均上都不低于原始 SC，是最稳定的任务B主方法。

## 8. 路径数消融

路径数消融只使用任务A已经生成的 3 条路径，不重新采样。这里对比 `original_sc` 和推荐方法 `normalized_filtered_vote`。

| model | dataset | method | N=1 | N=2 | N=3 |
| --- | --- | --- | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | original_sc | 38% | 38% | 43% |
| `qwen2_5_0_5b_instruct` | CSQA | normalized_filtered_vote | 38% | 38% | 43% |
| `qwen2_5_0_5b_instruct` | GSM8K | original_sc | 17% | 17% | 20% |
| `qwen2_5_0_5b_instruct` | GSM8K | normalized_filtered_vote | 37% | 37% | 45% |
| `qwen2_5_0_5b_instruct` | MMLU | original_sc | 32% | 32% | 33% |
| `qwen2_5_0_5b_instruct` | MMLU | normalized_filtered_vote | 34% | 34% | 35% |
| `qwen2_5_1_5b_instruct` | CSQA | original_sc | 65% | 65% | 67% |
| `qwen2_5_1_5b_instruct` | CSQA | normalized_filtered_vote | 65% | 65% | 67% |
| `qwen2_5_1_5b_instruct` | GSM8K | original_sc | 63% | 63% | 67% |
| `qwen2_5_1_5b_instruct` | GSM8K | normalized_filtered_vote | 64% | 64% | 69% |
| `qwen2_5_1_5b_instruct` | MMLU | original_sc | 51% | 51% | 50% |
| `qwen2_5_1_5b_instruct` | MMLU | normalized_filtered_vote | 51% | 51% | 50% |
| `tinyllama_1_1b_chat` | CSQA | original_sc | 27% | 27% | 27% |
| `tinyllama_1_1b_chat` | CSQA | normalized_filtered_vote | 28% | 30% | 29% |
| `tinyllama_1_1b_chat` | GSM8K | original_sc | 1% | 1% | 1% |
| `tinyllama_1_1b_chat` | GSM8K | normalized_filtered_vote | 4% | 5% | 5% |
| `tinyllama_1_1b_chat` | MMLU | original_sc | 8% | 8% | 9% |
| `tinyllama_1_1b_chat` | MMLU | normalized_filtered_vote | 12% | 17% | 23% |

路径数结论：

- 多路径对原始 SC 有帮助，但在部分模型和数据集上收益有限，例如 TinyLlama 的 GSM8K 原始 SC 始终是 `1%`。
- 改进后处理能放大多路径收益，最明显的是 TinyLlama 的 MMLU，从 `N=1` 的 `12%` 提升到 `N=3` 的 `23%`。
- `N=2` 不一定优于 `N=1`，原因是两票平票时仍依赖最早出现路径，奇数路径更适合多数投票。
- 当前任务A只生成 3 条路径，因此任务B路径数消融只能研究 `1 / 2 / 3`，更大路径数属于任务C范畴。

## 9. 消融输出文件

新增消融输出：

```text
output/log/B/component_ablation.csv
output/log/B/component_dataset_average.csv
output/log/B/num_paths_ablation.csv
```

## 10. Oracle 上界诊断

Oracle 上界表示“如果投票器总能从已有路径中选出正确答案，最高能达到多少准确率”。它不作为真实方法结果，只用于判断错误来自路径生成，还是来自投票聚合。

| model | dataset | original_sc | normalized_filtered | normalized_oracle | remaining_gap |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | 43% | 43% | 63% | 20% |
| `qwen2_5_0_5b_instruct` | GSM8K | 20% | 45% | 61% | 16% |
| `qwen2_5_0_5b_instruct` | MMLU | 33% | 35% | 65% | 30% |
| `qwen2_5_1_5b_instruct` | CSQA | 67% | 67% | 71% | 4% |
| `qwen2_5_1_5b_instruct` | GSM8K | 67% | 69% | 85% | 16% |
| `qwen2_5_1_5b_instruct` | MMLU | 50% | 50% | 66% | 16% |
| `tinyllama_1_1b_chat` | CSQA | 27% | 29% | 44% | 15% |
| `tinyllama_1_1b_chat` | GSM8K | 1% | 5% | 8% | 3% |
| `tinyllama_1_1b_chat` | MMLU | 9% | 23% | 40% | 17% |

Oracle 诊断结论：

- Qwen0.5B 的 GSM8K 原始 SC 只有 `20%`，但归一化 oracle 是 `61%`，说明很多题的正确答案已经出现在路径中，原始字符串投票没有把等价答案合并。
- Qwen1.5B 的 CSQA 原始 SC 已经达到 `67%`，oracle 只有 `71%`，说明该设置下后处理空间很小。
- TinyLlama 的 GSM8K oracle 只有 `8%`，说明主要瓶颈是模型推理能力，后处理无法凭空制造正确路径。

## 11. 正确性转移诊断

这里比较原始 SC 和推荐方法 `normalized_filtered_vote`。

| model | dataset | wrong_to_correct | correct_to_wrong | net_gain | prediction_changed_rate |
| --- | --- | ---: | ---: | ---: | ---: |
| `qwen2_5_0_5b_instruct` | CSQA | 0 | 0 | 0 | 0% |
| `qwen2_5_0_5b_instruct` | GSM8K | 26 | 1 | 25 | 64% |
| `qwen2_5_0_5b_instruct` | MMLU | 2 | 0 | 2 | 17% |
| `qwen2_5_1_5b_instruct` | CSQA | 0 | 0 | 0 | 0% |
| `qwen2_5_1_5b_instruct` | GSM8K | 2 | 0 | 2 | 3% |
| `qwen2_5_1_5b_instruct` | MMLU | 0 | 0 | 0 | 2% |
| `tinyllama_1_1b_chat` | CSQA | 2 | 0 | 2 | 8% |
| `tinyllama_1_1b_chat` | GSM8K | 4 | 0 | 4 | 32% |
| `tinyllama_1_1b_chat` | MMLU | 14 | 0 | 14 | 46% |

转移诊断结论：

- 改进方法最明显的收益来自错转对，尤其是 Qwen0.5B 的 GSM8K 和 TinyLlama 的 MMLU。
- Qwen0.5B 的 GSM8K 出现 1 个对转错，说明后处理规则不是无风险的，报告中不能把它描述成单调改进。
- TinyLlama 的 MMLU 预测变化率达到 `46%`，且 14 个样本错转对，说明过滤和归一化确实改变了大量原始投票。

## 12. 多数票置信度诊断

多数票占比越高，通常表示路径间更一致。该指标可以解释 SC 投票是否可靠。

代表性结果：

- `qwen2_5_1_5b_instruct` 在 GSM8K 上，`normalized_filtered_vote` 的 unanimous 样本准确率为 `92.31%`，明显高于 one_third 样本的 `25.00%`。
- `qwen2_5_0_5b_instruct` 在 GSM8K 上，`normalized_filtered_vote` 的 unanimous 样本准确率为 `82.35%`，two_thirds 为 `63.64%`，one_third 为 `20.00%`。
- TinyLlama 的 GSM8K 即使投票一致也不可靠，说明弱模型会出现“多条路径一致地错”的情况。

置信度诊断结论：

- 对 Qwen 模型，投票一致性可以作为可靠性信号，多数票越集中，准确率通常越高。
- 对 TinyLlama，投票一致性不能稳定代表正确性，路径质量本身过低时，多数票只会放大共同错误。

## 13. 诊断输出文件

新增诊断输出：

```text
output/log/B/oracle_path_analysis.csv
output/log/B/majority_confidence_buckets.csv
output/log/B/original_to_improved_transition.csv
output/log/B/model_average_diagnostics.csv
```
