# 任务C：Self-Consistency 参数消融实验结果分析

## 1. 任务定位

根据项目书中“任务C：消融实验与参数分析”的要求，本部分关注 Self-Consistency 推理阶段的关键超参数，而不提出新的后处理方法或训练方法。核心问题是：

- `num_paths` 增加后，SC 是否稳定提升，是否存在边际收益递减。
- `temperature` 改变后，路径多样性、投票稳定性和准确率如何变化。
- SC 的收益是否依赖模型能力和任务类型。

本报告使用任务A的 `N=3, T=0.7` Self-Consistency 结果作为 baseline，并在同一 sampled subset 上补充任务C实验。

## 2. 实验设置

### 2.1 正式实验口径

任务C主分析模型选择 `qwen2_5_1_5b_instruct`。原因是任务A结果显示，1.5B 模型相比 0.5B 和 TinyLlama 更符合主论文中“SC 在推理任务上通常带来收益”的趋势。

主实验覆盖：

```text
model: qwen2_5_1_5b_instruct
datasets: GSM8K, CSQA, MMLU
sample_size: 100
sample_seed: 42
max_new_tokens: 384
top_p: 0.95
do_sample: true
strict_answer_parsing: false
```

任务A baseline：

```text
num_paths = 3
temperature = 0.7
```

任务C新增配置：

```text
num_paths ablation: N = 5, 8, fixed T = 0.7
temperature ablation: T = 0.3, 1.0, fixed N = 3
```

此外，为检验结论是否依赖模型能力，补充了 `qwen2_5_0_5b_instruct` 和 `tinyllama_1_1b_chat` 在 GSM8K / CSQA 上的对比实验：

```text
N = 5, T = 0.7
N = 3, T = 0.3
```

### 2.2 运行环境

正式结果来自 titan 分区 `rtx8000` 节点，使用 `fjm-cu124` 环境：

```text
conda env: fjm-cu124
torch: 2.6.0+cu124
CUDA runtime: 12.4
GPU: Quadro RTX 8000
```

所有正式任务C结果保存在：

```text
output/raw/C_titan_cu124/self_consistency/
output/log/C_titan_cu124/
```

截至本报告整理时，`output/raw/C_titan_cu124/self_consistency/` 下共有 20 个 Task C JSONL 结果文件，所有结果文件均为 100 条样本。

## 3. 1.5B 主实验结果

### 3.1 准确率总表

下表中 `baseline` 指任务A的 `N=3,T=0.7`。

| Dataset | Baseline N=3,T=0.7 | N=5,T=0.7 | N=8,T=0.7 | N=3,T=0.3 | N=3,T=1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 67% | 70% | 77% | 75% | 57% |
| CSQA | 67% | 68% | 66% | 66% | 58% |
| MMLU | 50% | 53% | 55% | 54% | 47% |

整体上，GSM8K 最符合主论文趋势：更多路径和更稳定的采样都能带来明显收益。MMLU 有温和提升。CSQA 对路径数和低温不敏感，高温则明显下降。

### 3.2 路径统计总表

| Dataset | N | T | Acc. | Majority Share | Unique Answers | Entropy | Tie Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 3 | 0.7 | 67% | 0.693 | 1.92 | 0.777 | 0.27 |
| GSM8K | 5 | 0.7 | 70% | 0.636 | 2.71 | 1.122 | 0.19 |
| GSM8K | 8 | 0.7 | 77% | 0.646 | 3.49 | 1.292 | 0.11 |
| GSM8K | 3 | 0.3 | 75% | 0.810 | 1.57 | 0.488 | 0.14 |
| GSM8K | 3 | 1.0 | 57% | 0.583 | 2.25 | 1.027 | 0.48 |
| CSQA | 3 | 0.7 | 67% | 0.940 | 1.18 | 0.160 | 0.02 |
| CSQA | 5 | 0.7 | 68% | 0.924 | 1.32 | 0.242 | 0.00 |
| CSQA | 8 | 0.7 | 66% | 0.923 | 1.35 | 0.245 | 0.02 |
| CSQA | 3 | 0.3 | 66% | 0.983 | 1.05 | 0.046 | 0.00 |
| CSQA | 3 | 1.0 | 58% | 0.853 | 1.44 | 0.381 | 0.09 |
| MMLU | 3 | 0.7 | 50% | 0.790 | 1.63 | 0.543 | 0.14 |
| MMLU | 5 | 0.7 | 53% | 0.754 | 2.07 | 0.766 | 0.08 |
| MMLU | 8 | 0.7 | 55% | 0.740 | 2.33 | 0.843 | 0.08 |
| MMLU | 3 | 0.3 | 54% | 0.900 | 1.30 | 0.263 | 0.05 |
| MMLU | 3 | 1.0 | 47% | 0.720 | 1.84 | 0.716 | 0.22 |

`Majority Share` 表示每题最高票数占路径数比例的平均值；`Unique Answers` 和 `Entropy` 反映答案分布多样性；`Tie Rate` 表示最高票并列比例。

## 4. 路径数消融

路径数消融固定 `temperature = 0.7`。

### 4.1 GSM8K

GSM8K 随路径数增加持续提升：

```text
N=3: 67%
N=5: 70%
N=8: 77%
```

这说明数学推理任务能从多路径采样中获益。虽然路径数增加后 `Unique Answers` 从 1.92 增至 3.49，答案分布更分散，但更多候选推理链也提高了覆盖正确答案的概率，最终多数投票仍能带来净收益。

从成本看，`N=5` 相对 `N=3` 约为 1.67 倍路径成本，`N=8` 约为 2.67 倍路径成本。GSM8K 中 `N=8` 的收益较明显，说明在数学推理上增加路径数是值得考虑的。

### 4.2 MMLU

MMLU 也随路径数增加上升，但幅度较小：

```text
N=3: 50%
N=5: 53%
N=8: 55%
```

这表明 MMLU 的多学科选择题仍能从额外路径中获得一些收益，但收益小于 GSM8K。可能原因是 MMLU 更依赖知识覆盖和选项判别，额外采样能修正一部分随机错误，但无法弥补模型知识不足。

### 4.3 CSQA

CSQA 不呈现单调提升：

```text
N=3: 67%
N=5: 68%
N=8: 66%
```

CSQA 的多数票占比一直很高，`N=3` 时已达到 0.940，说明模型在常识选择题上本来就容易形成一致答案。增加路径数并没有显著提供更多有效信息，反而可能引入少量错误路径，使 `N=8` 略低于 baseline。

### 4.4 小结

路径数不是越多越好。GSM8K 的收益最明显，MMLU 有小幅收益，CSQA 基本饱和甚至略降。因此，SC 的路径数选择应结合任务类型和推理成本。

## 5. 温度消融

温度消融固定 `num_paths = 3`。

### 5.1 低温 T=0.3

低温在 GSM8K 和 MMLU 上有帮助：

```text
GSM8K: 67% -> 75%
MMLU: 50% -> 54%
CSQA: 67% -> 66%
```

GSM8K 的提升最大。低温降低采样随机性，使模型更倾向于生成稳定、连贯的推理链；对应地，GSM8K 的 `Majority Share` 从 0.693 升到 0.810，`Tie Rate` 从 0.27 降到 0.14。

CSQA 中低温没有提升准确率。其 `Majority Share` 从 0.940 升到 0.983，但准确率略降，说明更一致并不等于更正确。模型可能只是更稳定地重复原本的偏好答案。

### 5.2 高温 T=1.0

高温对三个数据集均有负面影响：

```text
GSM8K: 67% -> 57%
CSQA: 67% -> 58%
MMLU: 50% -> 47%
```

高温增加了答案多样性，但这种多样性主要表现为噪声。以 GSM8K 为例，`Tie Rate` 从 0.27 升到 0.48，接近一半题目出现最高票并列；`Majority Share` 从 0.693 降到 0.583，说明投票共识明显变弱。

### 5.3 小结

温度消融的结论很清楚：高温会破坏 SC 的投票稳定性；低温对需要连贯推理的 GSM8K 更有利，对 CSQA 这类已经高度一致的选择题任务不一定有效。

## 6. 模型能力对 SC 收益的影响

为检验任务C结论是否能推广到其他模型，补充比较了 0.5B、1.5B 和 TinyLlama 在 GSM8K / CSQA 上的表现。

| Model | Dataset | Baseline N=3,T=0.7 | N=5,T=0.7 | N=3,T=0.3 |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-0.5B | GSM8K | 20% | 20% | 13% |
| Qwen2.5-0.5B | CSQA | 43% | 45% | 44% |
| Qwen2.5-1.5B | GSM8K | 67% | 70% | 75% |
| Qwen2.5-1.5B | CSQA | 67% | 68% | 66% |
| TinyLlama-1.1B | GSM8K | 1% | 1% | 3% |
| TinyLlama-1.1B | CSQA | 27% | 26% | 21% |

结果显示，SC 的收益强烈依赖基模型能力：

- 1.5B 在 GSM8K 上能明显受益，说明其多条路径中包含足够多可被投票机制利用的正确推理。
- 0.5B 在 GSM8K 上没有从 `N=5` 获益，低温反而下降，说明模型本身推理能力不足时，多采样不能稳定产生正确路径。
- TinyLlama 在 GSM8K 上 baseline 仅 1%，SC 基本无法发挥作用；在 CSQA 上增加路径和低温都没有收益。

这与任务A观察一致：SC 不是独立于模型能力的“免费提升”。多数投票只有在候选路径中存在足够比例的正确答案时才有意义；如果模型大多数路径本身错误，投票只会稳定错误或放大噪声。

## 7. 与主论文结论的关系

Wang et al. 的 Self-Consistency 方法强调，通过采样多条 CoT 推理路径并对最终答案多数投票，可以提升推理鲁棒性。本项目在轻量模型与 sampled-100 设定下复现到的是“条件成立的趋势”：

- 在 `qwen2_5_1_5b_instruct + GSM8K` 上，路径数从 3 增加到 8，准确率从 67% 提升到 77%，最符合主论文趋势。
- 在 MMLU 上，路径数增加也带来小幅提升，说明 SC 对部分知识问答任务仍有帮助。
- 在 CSQA、0.5B 和 TinyLlama 上，收益不稳定，说明 SC 依赖任务类型和基模型质量。
- 高温采样虽然增加路径多样性，但会降低投票共识，导致准确率下降。

因此，本项目不是复刻主论文的大模型绝对数值，而是在课程算力允许的轻量模型上复现 SC 的机制，并进一步说明其适用边界。

## 8. 结论

任务C的主要结论如下：

1. **路径数增加不保证稳定提升。**  
   GSM8K 从 67% 提升到 77%，MMLU 从 50% 提升到 55%，但 CSQA 从 67% 到 68% 后又降到 66%。路径数收益具有明显任务依赖性。

2. **高温采样普遍有害。**  
   `T=1.0` 在 GSM8K、CSQA、MMLU 上均低于 baseline，并伴随更高的答案熵和 tie rate。高温带来的多样性主要是噪声，而不是有效推理路径。

3. **低温更适合数学推理。**  
   `T=0.3` 将 GSM8K 从 67% 提升到 75%，MMLU 从 50% 提升到 54%，但 CSQA 略降。这说明稳定采样有助于连贯推理，但不一定改善选择题偏差。

4. **SC 依赖基模型能力。**  
   1.5B 是本项目中最能体现 SC 趋势的模型；0.5B 和 TinyLlama 的收益弱或不稳定。若候选路径整体质量较低，多数投票无法弥补模型能力不足。

5. **参数选择比盲目增加采样更重要。**  
   在有限算力下，GSM8K 可优先考虑 `N=8,T=0.7` 或 `N=3,T=0.3`；CSQA 上 `N=3,T=0.7` 已经接近合适配置；高温不建议作为正式设置。

最终，任务C支持如下观点：

> Self-Consistency 的有效性并不只由路径数量决定，还取决于路径质量、采样温度、任务类型和基模型能力。在轻量模型与有限算力条件下，合理选择 `num_paths` 与 `temperature` 比盲目增加采样规模更重要。

## 9. 结果文件

原始结果：

```text
output/raw/C_titan_cu124/self_consistency/
```

1.5B 含 MMLU 汇总：

```text
output/log/C_titan_cu124/summary_1p5b_with_mmlu/sc_parameter_summary.csv
output/log/C_titan_cu124/summary_1p5b_with_mmlu/num_paths_ablation.csv
output/log/C_titan_cu124/summary_1p5b_with_mmlu/temperature_ablation.csv
output/log/C_titan_cu124/summary_1p5b_with_mmlu/path_diversity_statistics.csv
output/log/C_titan_cu124/summary_1p5b_with_mmlu/case_study_examples.jsonl
```

模型对比汇总：

```text
output/log/C_titan_cu124/summary_model_compare/sc_parameter_summary.csv
output/log/C_titan_cu124/summary_model_compare/num_paths_ablation.csv
output/log/C_titan_cu124/summary_model_compare/temperature_ablation.csv
output/log/C_titan_cu124/summary_model_compare/path_diversity_statistics.csv
output/log/C_titan_cu124/summary_model_compare/case_study_examples.jsonl
```

主要运行日志：

```text
output/log/C_titan_cu124/run_taskC_minimal_ablation_titan_cu124.log
output/log/C_titan_cu124/run_taskC_ext_n8_t07_n3_t10_titan_cu124.log
output/log/C_titan_cu124/run_taskC_ext_mmlu_titan_cu124.log
output/log/C_titan_cu124/run_taskC_ext_mmlu_n8_t10_titan_cu124.log
output/log/C_titan_cu124/run_taskC_ext_model_compare_qwen05_titan_cu124.log
output/log/C_titan_cu124/run_taskC_ext_model_compare_tinyllama_titan_cu124.log
```
