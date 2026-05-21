# 任务B：Self-Consistency 轻量改进计划

## 1. 目标

任务B在任务A已经完成的 Self-Consistency 结果上做轻量后处理改进，重点解决原始 SC 中的答案格式不一致、无效答案和投票鲁棒性问题。

任务B不重新训练模型，不修改模型参数，不重新生成推理路径。所有实验复用任务A的固定 sampled subset 和原始 SC JSONL。

## 2. 输入

任务B直接读取以下任务A结果：

```text
output/raw/A/lightweight_100/self_consistency/*.jsonl
```

每条样本需要包含：

- `id`
- `dataset`
- `choices`
- `gold_answer`
- `prediction`
- `correct`
- `paths`
- `vote_counts`

其中 `paths` 中每条路径包含：

- `reasoning`
- `answer`

## 3. 实验范围

主实验覆盖两个代表模型和三个数据集：

- `qwen2_5_1_5b_instruct`
- `tinyllama_1_1b_chat`
- `gsm8k`
- `csqa`
- `mmlu`

保留对 `qwen2_5_0_5b_instruct` 的可选支持，方便生成完整补充表。

## 4. 改进方法

### 4.1 答案归一化投票

对每条路径抽取出的 `answer` 做数据集相关归一化。

`GSM8K`：

- 删除逗号、美元符号和首尾标点
- 抽取最后可识别数字
- 将整数形式的小数统一成整数，例如 `18.0` 归一化为 `18`

`CSQA / MMLU`：

- 将 `option A`、`choice A`、`answer is A` 归一化为 `A`
- 将 `A. xxx` 归一化为 `A`
- 将选项文本映射回对应字母
- 最终合法答案必须属于 `A-E`

### 4.2 无效路径过滤

过滤规则：

- 空答案无效
- `GSM8K` 中无法归一化为数字的答案无效
- `CSQA / MMLU` 中无法归一化为 `A-E` 的答案无效

统计但不默认删除重复路径。原因是 SC 的多数票机制允许多条路径得到同一答案，删除重复推理文本会改变原始方法定义。

### 4.3 改进配置

任务B输出四种方法：

- `original_sc`：任务A原始 SC 结果
- `normalized_vote`：只做答案归一化后投票
- `filtered_vote`：只过滤无效路径后投票
- `normalized_filtered_vote`：先归一化，再过滤无效路径，再投票

平票规则沿用任务A口径：选择最早出现的最高票答案。

如果过滤后没有有效路径，预测记为空字符串，该题计为错误。

## 5. 输出

```text
output/raw/B/sc_improvement_results.jsonl
output/log/B/sc_improvement_results.csv
output/log/B/normalization_vote_results.csv
output/log/B/path_filtering_statistics.csv
output/log/B/vote_change_cases.csv
```

## 6. 指标

- `accuracy`
- `correct_count`
- `sample_count`
- `valid_path_rate`
- `answer_extraction_failure_rate`
- `vote_changed_rate`
- `mean_unique_answers_before`
- `mean_unique_answers_after`
- `mean_majority_share_before`
- `mean_majority_share_after`
- `wrong_to_correct_count`
- `correct_to_wrong_count`

## 7. 完成标准

- 至少完成一种轻量 SC 改进方法
- 与任务A原始 SC 在同一 sampled subset 上公平对比
- 输出准确率、路径有效性、投票变化比例
- 能分析改进有效或无效的原因
