# 任务A要求整理

## 任务定位

任务A负责完成 `Chain-of-Thought (CoT)` 与 `Self-Consistency (SC)` baseline 的复现，产出可复现、可统计、可对照论文趋势的正式实验结果。任务A只做 baseline，不引入额外改进方法。

任务A的核心目标有三个：

- 复现 `Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency`
- 比较单条 CoT 与 SC 的效果差异
- 给出与参考论文方法趋势的一致性结论

## 固定实验范围

### 固定模型

任务A正式纳入的 3 个模型是：

- `mistral_7b_instruct_v0_3`
- `olmo3_7b_instruct`
- `llama2_7b_hf`

以下模型不纳入正式 baseline：

- `qwen2.5-7b-instruct`
- `phi3.5-mini-instruct`

原因是当前环境下输出异常或不稳定，不能作为可靠正式结果。

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
2. 在 3 个模型上，`Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency` 分别得到什么结果。
3. 本项目观察到的趋势是否与参考论文一致。

## 必须完成的实验

### 正式实验组合

任务A正式实验应覆盖：

- `3 个模型`
- `3 种推理模式`
- `3 个数据集`

总计：

- `3 × 3 × 3 = 27` 个正式配置

配置命名规则固定为：

- `configs/runs/formal/<mode>_<model>_<dataset>.yaml`

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
