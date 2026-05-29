# 任务C：Self-Consistency 消融实验与参数分析计划

## 1. 任务定位

任务C负责分析 Self-Consistency 的关键推理超参数对效果、稳定性和推理成本的影响。

任务C不提出新的后处理方法，不修改模型参数，也不重复任务B的答案归一化和无效路径过滤。任务C的重点是回答：在固定 sampled subset 上，SC 的收益是否依赖路径数、采样温度和路径多样性。

## 2. 论文依据

主论文 Wang et al. *Self-Consistency Improves Chain of Thought Reasoning* 的核心机制是：

1. 对同一问题采样多条 chain-of-thought reasoning paths。
2. 从每条路径抽取最终答案。
3. 对最终答案进行多数投票，选择最一致的答案。

因此，任务C围绕两个直接影响 SC 的因素展开：

- 路径数 `num_paths`：决定投票证据量和推理成本。
- 采样温度 `temperature`：决定路径随机性、多样性和潜在噪声。

报告中要明确说明：本项目复现的是 SC 方法机制，不复刻原论文的大模型规模与完整 benchmark 数值。

## 3. 与任务A/B的边界

任务A：

- 提供 `lightweight-100` baseline。
- 默认 SC 设置为 `num_paths = 3`、`temperature = 0.7`、`top_p = 0.95`。
- 结果位置：`output/raw/A/lightweight_100/self_consistency/`。

任务B：

- 基于任务A已有 SC 路径做后处理改进。
- 主要分析答案归一化、无效路径过滤、投票变化。
- 已有 `output/log/B/num_paths_ablation.csv` 是在任务A已有 3 条路径内部截断得到的 `N = 1/2/3` 后处理消融，只能作为辅助分析，不能替代任务C的新路径数实验。

任务C：

- 需要在同一 sampled subset 上重新生成不同 `num_paths` 和 `temperature` 的 SC 输出。
- 重点比较原始 SC 在不同采样设置下的表现。
- 不使用任务B的改进投票作为主结果，避免混淆“方法改进”和“参数敏感性”。

## 4. 实验口径

固定设置：

- `sample_size = 100`
- `sample_seed = 42`
- `split` 与任务A一致：
  - `GSM8K`: `test`
  - `CSQA`: `validation`
  - `MMLU`: `validation`
- `max_new_tokens = 384`
- `top_p = 0.95`
- `do_sample = true`
- `strict_answer_parsing = false`
- prompt 使用任务A的 SC prompt 和 fixed few-shot examples。

主模型：

- `qwen2_5_1_5b_instruct`

原因：

- 在任务A中整体表现最稳定。
- GSM8K 与 CSQA 上 SC 均达到 `67/100`，有分析边际收益和温度扰动的空间。
- 模型规模仍适合课程集群完成多组 SC 推理。

主数据集：

- `gsm8k`
- `csqa`

原因：

- 分别代表数学推理和常识推理。
- 任务A中两者 baseline 较高，能够观察 SC 参数变化带来的真实波动。
- MMLU 可作为可选补充，不作为最低完成要求。

## 5. 实验矩阵

### 5.1 路径数消融

研究问题：

- 路径数增加是否提升准确率。
- 路径数增加后多数票是否更稳定。
- 额外路径是否带来边际收益递减。

主实验：

| model | dataset | num_paths | temperature |
| --- | --- | ---: | ---: |
| qwen2_5_1_5b_instruct | gsm8k | 3 | 0.7 |
| qwen2_5_1_5b_instruct | gsm8k | 5 | 0.7 |
| qwen2_5_1_5b_instruct | csqa | 3 | 0.7 |
| qwen2_5_1_5b_instruct | csqa | 5 | 0.7 |

可选扩展：

- `num_paths = 8`
- `mmlu`
- `qwen2_5_0_5b_instruct`

注意：`N = 3` 可以直接引用任务A结果作为 baseline，也可以重跑以保证和 C 输出目录完全一致。若时间有限，建议直接引用任务A的 `N = 3, T = 0.7`。

### 5.2 温度消融

研究问题：

- 较低温度是否让路径更一致但多样性不足。
- 较高温度是否增加路径多样性，同时引入更多错误或无效答案。
- 温度变化对数学推理和选择题任务是否不同。

主实验：

| model | dataset | num_paths | temperature |
| --- | --- | ---: | ---: |
| qwen2_5_1_5b_instruct | gsm8k | 3 | 0.3 |
| qwen2_5_1_5b_instruct | gsm8k | 3 | 0.7 |
| qwen2_5_1_5b_instruct | csqa | 3 | 0.3 |
| qwen2_5_1_5b_instruct | csqa | 3 | 0.7 |

可选扩展：

- `temperature = 1.0`
- 固定 `num_paths = 5` 再做温度消融，但这会明显增加推理成本。

最低完成矩阵：

```text
2 datasets × (N=5 新跑 + T=0.3 新跑) = 4 个新配置
```

其中 `N=3, T=0.7` 直接复用任务A作为共同 baseline。

## 6. 输出目录

建议：

```text
output/raw/C/self_consistency/
output/log/C/
output/figures/C/
```

原始结果文件命名建议：

```text
output/raw/C/self_consistency/qwen2_5_1_5b_instruct_gsm8k_n5_t0.7.jsonl
output/raw/C/self_consistency/qwen2_5_1_5b_instruct_csqa_n5_t0.7.jsonl
output/raw/C/self_consistency/qwen2_5_1_5b_instruct_gsm8k_n3_t0.3.jsonl
output/raw/C/self_consistency/qwen2_5_1_5b_instruct_csqa_n3_t0.3.jsonl
```

汇总表：

```text
output/log/C/num_paths_ablation.csv
output/log/C/temperature_ablation.csv
output/log/C/sc_parameter_summary.csv
output/log/C/path_diversity_statistics.csv
output/log/C/case_study_examples.jsonl
```

图表：

```text
output/figures/C/num_paths_accuracy.png
output/figures/C/temperature_accuracy.png
output/figures/C/diversity_vs_accuracy.png
```

## 7. 指标设计

主指标：

- `accuracy`
- `correct_count`
- `sample_count`

路径稳定性指标：

- `mean_majority_share`：每题最高票数 / `num_paths` 的平均值。
- `mean_unique_answers`：每题不同答案个数的平均值。
- `mean_answer_entropy`：答案分布熵，越高表示投票越分散。
- `tie_rate`：最高票并列的题目比例。

路径有效性指标：

- `answer_extraction_success_rate`
- `invalid_answer_rate`

成本指标：

- `total_paths = sample_count × num_paths`
- `relative_path_cost = num_paths / 3`
- `accuracy_gain_vs_A_sc`
- `gain_per_extra_path = (accuracy - baseline_accuracy) / (num_paths - 3)`，仅用于 `num_paths > 3`。

建议不要只看准确率，因为 sampled-100 上 1 题就是 1%，准确率小波动可能并不稳定。报告中应同时结合多数票占比、唯一答案数和案例分析。

## 8. 实现建议

建议新增两个脚本：

```text
scripts/run_task_c_sc_ablation.py
scripts/summarize_task_c_results.py
```

`run_task_c_sc_ablation.py` 职责：

- 动态构造 SC run_config。
- 单次加载模型，串行运行多个 dataset/parameter 配置。
- 支持参数：
  - `--model qwen2_5_1_5b_instruct`
  - `--datasets gsm8k csqa`
  - `--num-paths 5`
  - `--temperature 0.7`
  - `--output-dir output/raw/C/self_consistency`
  - `--sample-size 100`
  - `--sample-seed 42`
  - `--skip-existing`

`summarize_task_c_results.py` 职责：

- 读取任务A baseline 和任务C新增 JSONL。
- 统一计算准确率、路径多样性、有效性、成本指标。
- 写出 `num_paths_ablation.csv`、`temperature_ablation.csv` 和 `path_diversity_statistics.csv`。
- 可选生成图表。

实现时可以复用：

- `src.pipeline.run_self_consistency`
- `src.models.runner.LocalModelRunner`
- `src.sc.answering.majority_vote`
- `src.io.files.write_csv`

## 9. 推荐执行顺序

1. 确认任务A baseline 文件完整：

```bash
find output/raw/A/lightweight_100/self_consistency -name 'qwen2_5_1_5b_instruct_*.jsonl' -print
```

2. 先跑最小 C 矩阵：

```text
qwen2_5_1_5b_instruct + gsm8k + N=5 + T=0.7
qwen2_5_1_5b_instruct + csqa  + N=5 + T=0.7
qwen2_5_1_5b_instruct + gsm8k + N=3 + T=0.3
qwen2_5_1_5b_instruct + csqa  + N=3 + T=0.3
```

3. 汇总任务A baseline 与任务C结果。

4. 如果 GPU 时间充足，再补：

```text
N=8, T=0.7
N=3, T=1.0
MMLU
```

5. 写报告第 7、8 节。

## 10. 报告写作结构

任务C负责报告中的“消融实验”和“分析与讨论”部分，建议结构如下：

### 10.1 路径数消融

写清楚：

- 实验设置：固定模型、数据集、温度，只改变 `num_paths`。
- 表格：`N=3` vs `N=5`，可选 `N=8`。
- 结论：是否有准确率提升，是否出现边际收益递减。

重点解释：

- 如果准确率提升：更多路径提供了更可靠的多数投票。
- 如果准确率不升反降：小模型生成的额外路径可能带来噪声，多数票会放大系统性错误。
- 如果多数票占比升高但准确率不升：模型更一致了，但可能一致地答错。

### 10.2 温度消融

写清楚：

- 实验设置：固定模型、数据集、路径数，只改变 `temperature`。
- 表格：`T=0.3` vs `T=0.7`，可选 `T=1.0`。
- 结论：温度对路径多样性与答案正确率的权衡。

重点解释：

- 低温可能提高一致性，但减少探索。
- 高温可能提高多样性，但也增加无效或错误路径。
- 数学题和选择题可能表现不同：GSM8K 更依赖完整推理链，CSQA 更容易因选项先验形成一致答案。

### 10.3 与主论文结论的关系

建议表述：

- 本项目观察到 SC 的收益并非在小模型和 sampled-100 设置下全局稳定。
- 参数选择会显著影响 SC 效果。
- 这与主论文“多路径采样 + 投票能够提升推理鲁棒性”的机制一致，但受模型能力、路径质量和采样规模限制。

## 11. 风险与应对

风险一：GPU 时间不足。

应对：

- 最低只新跑 4 个配置。
- `N=3, T=0.7` 复用任务A。
- 优先 GSM8K 和 CSQA，不强制 MMLU。

风险二：准确率波动小，结论不明显。

应对：

- 强化路径多样性、多数票占比、tie rate 和案例分析。
- 把结论写成“在 sampled subset 上观察到的趋势”，不做过强泛化。

风险三：温度太低导致采样路径重复。

应对：

- 将重复路径比例和唯一答案数作为分析点。
- 若 `T=0.3` 几乎无变化，可补 `T=1.0` 作为高温对照。

## 12. 完成标准

最低完成标准：

- 完成 `qwen2_5_1_5b_instruct` 在 `gsm8k`、`csqa` 上的路径数消融。
- 完成同一模型同一数据集上的温度消融。
- 输出 `num_paths_ablation.csv` 和 `temperature_ablation.csv`。
- 报告中能解释 SC 收益是否稳定、是否存在边际收益递减、温度如何影响路径多样性。

理想完成标准：

- 增加 `N=8` 或 `T=1.0`。
- 增加 `mmlu`。
- 生成图表和 3 到 5 个典型案例，说明参数变化如何改变投票结果。
