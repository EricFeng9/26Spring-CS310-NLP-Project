# 基于 Self-Consistency 的思维链推理复现、改进与消融分析项目书

## 一、项目题目

**基于 Self-Consistency 的大语言模型思维链推理复现、轻量改进与消融分析**

英文题目：

**Reproduction, Lightweight Improvement, and Ablation Study of Chain-of-Thought Reasoning with Self-Consistency**

## 二、项目背景与研究动机

大语言模型在数学推理、常识推理和多学科问答等任务中已经表现出较强的推理能力。`Chain-of-Thought (CoT)` 通过显式引导模型生成中间推理步骤，使模型能够将复杂问题拆解为多个较简单的推理环节，从而提升推理任务表现。

然而，单条 CoT 推理路径存在明显不稳定性：同一模型在同一问题上可能因为采样随机性、提示敏感性或局部推理错误而得到不同答案。`Self-Consistency (SC)` 方法通过对同一问题采样多条推理路径，再对最终答案进行多数投票，能够在不训练模型参数的情况下提升推理鲁棒性。该方法由 Wang et al. 在 *Self-Consistency Improves Chain of Thought Reasoning* 中系统提出，是当前 CoT 推理研究中的经典 baseline。

本项目围绕 CoT 与 SC 展开，目标不是进行大模型微调，而是在课程可承受算力范围内完成以下工作：

- 复现 Zero-shot CoT、Few-shot CoT 和原始 Self-Consistency baseline
- 在固定评测样本上比较单条 CoT 与 SC 的效果差异
- 设计低成本的 SC 改进方法，分析其对准确率与路径有效性的影响
- 进行小规模消融实验，研究路径数、采样温度等因素对 SC 的影响

## 三、项目总体目标

本项目希望回答以下核心问题：

1. 在多个 7B 级开源模型上，`Self-Consistency` 是否相比单条 `CoT` 带来更稳定或更高的准确率。
2. 原始 SC 的主要问题是什么，例如路径重复、答案格式不一致、无效推理路径等。
3. 简单的路径过滤或答案归一化方法是否能够在不增加训练成本的情况下提升 SC 的有效性。
4. SC 的性能是否会受到路径数、采样温度等推理超参数影响。

最终交付物包括：

- 可运行的 CoT / SC 推理代码
- 三个模型、三个数据集、三种推理方式的 baseline 结果
- 改进方法与原始 SC 的对比实验
- 消融实验结果表和图表
- 项目报告中的方法说明、实验分析和结论

## 四、参考工作

本项目主要参考以下工作：

- Wang et al. *Self-Consistency Improves Chain of Thought Reasoning*. NeurIPS 2022.
- Fu et al. *Complexity-Based Prompting for Multi-step Reasoning*. EMNLP 2024.
- Li et al. *On the Limitation of Self-Consistency*. ACL 2025.

其中，Wang et al. 的 Self-Consistency 是本项目任务 A 的核心复现对象；后续关于路径质量、路径冗余和采样参数敏感性的讨论，为任务 B 和任务 C 提供实验动机。

## 五、实验资源与评测口径

### 5.1 模型选择

本项目使用当前集群环境中可稳定运行的 7B 级模型：

- `mistral_7b_instruct_v0_3`
- `olmo3_7b_instruct`
- `llama2_7b_hf`

原计划中曾考虑的 `Qwen2.5-7B-Instruct` 和 `Phi-3.5-mini-instruct` 在当前环境下存在输出异常或乱码问题，因此不纳入最终正式实验。

### 5.2 数据集选择

本项目覆盖三类典型 NLP 推理任务：

- `GSM8K`：数学推理
- `CSQA`：常识推理
- `MMLU`：多学科知识与推理

对应 split 设置为：

- `GSM8K`: `test`
- `CSQA`: `validation`
- `MMLU`: `validation`

### 5.3 固定样本评测方案

由于课程计算集群 GPU 时间有限，而全量运行 `3 模型 × 3 数据集 × 3 方法` 的 27 个配置需要极高推理成本，本项目采用固定随机种子的 sampled evaluation 作为正式课程实验口径。

具体设置如下：

- 对每个数据集固定随机采样 `200` 条样本
- 随机种子固定为 `42`
- `GSM8K`、`CSQA`、`MMLU` 分别生成一份固定 sampled subset
- 任务 A、任务 B、任务 C 均使用同一批 sampled subset

这样做的原因是：

- 保证 A/B/C 实验之间公平可比
- 保证实验可复现
- 保证项目在课程集群资源内可完成
- 保留完整的模型、数据集和方法维度

报告中将明确说明：本项目结果不是全量 benchmark 数值，而是在固定 sampled subset 上进行的课程项目复现实验与方法分析。

## 六、总体技术路线

项目整体流程如下：

1. 准备三个数据集的固定 sampled subset。
2. 构造 Zero-shot CoT、Few-shot CoT 和 Self-Consistency prompt。
3. 使用三个 7B 模型分别生成推理路径和最终答案。
4. 对单条 CoT 输出进行答案抽取并计算准确率。
5. 对 SC 的多条路径进行多数投票，记录路径、投票结果和准确率。
6. 在同一批样本上实现路径过滤或答案归一化等轻量改进。
7. 设计路径数、温度等消融实验，分析 SC 的敏感性。
8. 生成统一结果表、对照图表和项目报告。

## 七、任务 A：Baseline 复现

### 7.1 任务目标

任务 A 负责完成原始 baseline 复现，建立后续 B/C 实验的对照基准。

任务 A 需要回答：

- Zero-shot CoT、Few-shot CoT 和 SC 在三个数据集上的表现如何。
- SC 是否相比单条 CoT 有准确率提升。
- 不同模型在 sampled subset 上是否呈现相似趋势。

### 7.2 实验范围

任务 A 覆盖：

- 3 个模型：Mistral、Olmo、Llama2
- 3 个数据集：GSM8K、CSQA、MMLU
- 3 种推理模式：Zero-shot CoT、Few-shot CoT、Self-Consistency

总计：

```text
3 models × 3 datasets × 3 methods = 27 configurations
```

### 7.3 方法设置

Zero-shot CoT：

- 不提供示例
- 只通过统一 prompt 引导模型逐步推理
- 每题生成一条推理路径

Few-shot CoT：

- 使用预先固定的 few-shot 示例
- 每题生成一条推理路径
- few-shot 示例对所有模型保持一致

Self-Consistency：

- 对每道题采样多条推理路径
- 默认路径数 `N = 5`
- 使用原始多数投票作为 baseline
- 保留每条路径的推理文本、抽取答案和最终投票结果

### 7.4 输出要求

任务 A 需要输出：

- 每个配置的逐题 `JSONL`
- 每个配置的汇总 `CSV`
- Baseline 主结果表
- 单条 CoT vs SC 对照表
- SC 路径统计表

建议统计指标：

- `accuracy`
- SC 路径数
- 每题唯一答案数
- 多数票占比
- 答案抽取失败比例

### 7.5 完成标准

任务 A 完成标准：

- 27 个 baseline 配置均完成 sampled-200 评测
- 结果文件可复现、可追踪
- 能明确回答 SC 是否优于单条 CoT
- 能为任务 B 和任务 C 提供统一 baseline

## 八、任务 B：Self-Consistency 轻量改进

### 8.1 任务目标

任务 B 在任务 A 的原始 SC baseline 基础上，设计低成本改进方法，重点解决原始 SC 中常见的路径质量与投票鲁棒性问题。

任务 B 不进行模型训练，也不修改模型参数。

### 8.2 改进方向一：答案归一化投票

原始 SC 通常直接对抽取出的答案字符串进行多数投票。但模型输出可能存在格式不一致问题，例如：

- 数字中包含逗号或单位
- 选项题输出选项文本而不是字母
- 答案后包含句号、括号或解释
- 同一答案存在多种等价表达

任务 B 可以实现答案归一化投票：

- 统一数字格式
- 去除多余标点和单位符号
- 将选项内容映射回选项字母
- 对归一化后的答案进行多数投票

对比对象：

- 原始 SC 多数投票
- 归一化 SC 多数投票

### 8.3 改进方向二：无效路径过滤

SC 的多条路径中可能存在无法抽取答案、格式明显错误或重复输出的路径。任务 B 可以加入简单过滤规则：

- 过滤无法抽取最终答案的路径
- 过滤空答案或明显无效答案
- 统计重复路径比例
- 对剩余有效路径投票

该方法不改变模型生成，只改变投票前的路径选择，因此属于轻量后处理改进。

### 8.4 实验范围

任务 B 使用与任务 A 完全相同的 sampled subset。

为了控制工作量，任务 B 建议选择：

- 1 到 2 个代表模型，例如 Mistral 和 Olmo
- 3 个数据集均可覆盖
- 重点对比 Self-Consistency 相关结果

### 8.5 输出要求

任务 B 需要输出：

- 原始 SC vs 改进 SC 准确率对照表
- 路径有效率统计
- 答案抽取失败比例
- 改进前后多数票变化比例
- 对改进是否有效的分析文字

### 8.6 完成标准

任务 B 完成标准：

- 至少实现一种轻量 SC 改进方法
- 在同一 sampled subset 上与任务 A 原始 SC 公平对比
- 给出准确率、路径有效性和失败样本分析

## 九、任务 C：消融实验与参数分析

### 9.1 任务目标

任务 C 负责分析 Self-Consistency 的关键超参数对结果的影响，解释 SC 在什么条件下更有效。

任务 C 关注的是实验分析，而不是提出新方法。

### 9.2 消融方向一：路径数 N

研究问题：

- SC 路径数越多是否一定越好。
- 路径数增加后是否存在边际收益递减。

建议设置：

- `N = 3`
- `N = 5`

如果算力允许，可额外加入：

- `N = 8`

统计指标：

- 准确率
- 多数票占比
- 唯一答案数
- 单题平均推理成本

### 9.3 消融方向二：采样温度

研究问题：

- 较高温度是否带来更多样的路径。
- 较高温度是否也会增加无效推理路径。

建议设置：

- `temperature = 0.3`
- `temperature = 0.7`

如果算力允许，可额外加入：

- `temperature = 1.0`

统计指标：

- 准确率
- 路径答案多样性
- 无效路径比例
- 投票稳定性

### 9.4 实验范围

任务 C 使用与任务 A 相同的 sampled subset。

为了控制工作量，任务 C 建议选择：

- 1 个主模型，例如 Mistral
- 1 到 2 个代表数据集，例如 GSM8K 和 CSQA
- 重点分析 SC，不重复完整跑全部 27 个配置

### 9.5 输出要求

任务 C 需要输出：

- 路径数消融表
- 温度消融表
- 准确率折线图或柱状图
- 路径多样性与准确率关系分析
- 对 SC 参数敏感性的结论

### 9.6 完成标准

任务 C 完成标准：

- 至少完成一个路径数消融实验
- 至少完成一个温度消融实验
- 能解释 SC 的收益是否稳定，以及收益是否随路径数增加而递减

## 十、三项任务之间的关系

任务 A、B、C 的关系如下：

- 任务 A 提供原始 baseline
- 任务 B 在任务 A 的 SC 结果上做轻量改进
- 任务 C 分析 SC 超参数对结果的影响

三项任务必须共享同一批 sampled subset。这样可以保证：

- 不同方法之间对比公平
- 不同成员结果可以合并
- 最终报告中的表格和图具有统一口径

## 十一、实验结果组织方式

建议输出目录：

```text
output/raw/A/
output/raw/B/
output/raw/C/
output/log/A/
output/log/B/
output/log/C/
```

任务 A 输出：

- `baseline_main_results.csv`
- `cot_vs_sc_results.csv`
- `sc_path_statistics.csv`

任务 B 输出：

- `sc_improvement_results.csv`
- `path_filtering_statistics.csv`
- `normalization_vote_results.csv`

任务 C 输出：

- `num_paths_ablation.csv`
- `temperature_ablation.csv`
- 对应图表文件

## 十二、报告结构建议

最终报告可以采用以下结构：

1. 引言
2. 相关工作
3. 方法
4. 实验设置
5. Baseline 复现结果
6. Self-Consistency 改进实验
7. 消融实验
8. 分析与讨论
9. 结论

其中：

- 任务 A 主要负责第 4、5 节
- 任务 B 主要负责第 3、6 节
- 任务 C 主要负责第 7、8 节

## 十三、风险与应对

### 13.1 算力风险

风险：

- 全量数据集评测需要大量 GPU 时间，可能无法在课程周期内完成。

应对：

- 使用固定随机种子的 sampled-200 评测口径
- 所有任务共享同一批样本
- 优先保证实验矩阵完整，而不是追求全量 benchmark 数值

### 13.2 输出格式风险

风险：

- 模型可能不严格输出 `The answer is ...`，导致答案抽取失败。

应对：

- 保留原始推理文本
- 统计答案抽取失败比例
- 将格式失败作为模型输出稳定性的一部分进行分析
- 在任务 B 中研究答案归一化和路径过滤

### 13.3 结果波动风险

风险：

- sampled subset 上的结果可能存在随机波动。

应对：

- 固定随机种子
- 明确说明评测口径
- 将重点放在方法趋势和相对比较，而不是宣称完整 benchmark 结论

## 十四、预期成果

本项目预期产出：

- 一套可复现的 CoT / SC 推理实验框架
- sampled-200 baseline 结果表
- 原始 SC 与改进 SC 的对比结果
- 路径数和采样温度的消融分析
- 对 Self-Consistency 方法优点与局限性的课程项目报告

最终结论将围绕以下问题展开：

- SC 是否在 sampled evaluation 上提升 CoT 推理准确率
- SC 的收益是否依赖模型、数据集和采样参数
- 简单的后处理改进是否能够提升 SC 的稳定性
- 在有限算力条件下，如何合理复现和分析大语言模型推理方法
