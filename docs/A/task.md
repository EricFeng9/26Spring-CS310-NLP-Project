# 任务A lightweight-100 批次执行清单

## 执行口径

- 复现对象：`Self-Consistency Improves Chain of Thought Reasoning` 的 CoT 与 Self-Consistency 方法框架
- 资源策略：任务A不限制模型，正式实验改用适配当前集群硬件的轻量开源指令模型
- 数据口径：每个数据集固定随机采样 100 条样本
- 随机种子：`42`
- 数据集：`GSM8K`、`CSQA`、`MMLU`
- 模型：`qwen2_5_0_5b_instruct`、`qwen2_5_1_5b_instruct`、`tinyllama_1_1b_chat`
- 推理模式：`zero_shot`、`few_shot`、`self_consistency`
- 生成长度：`zero_shot/few_shot max_new_tokens = 256`，`self_consistency max_new_tokens = 384`
- SC 路径数：`num_paths = 3`
- 拆分方式：`model × mode`，每个 Slurm 作业跑同一模型同一模式下的 3 个数据集
- 运行顺序：先全部 `zero_shot`，再全部 `few_shot`，最后全部 `self_consistency`
- 结果目录：`output/raw/A/lightweight_100/`
- 日志目录：`output/log/A/lightweight_100/`

## 状态标记

- `[ ]` 待提交
- `[~]` 已提交或运行中
- `[x]` 已完成
- `[!]` 失败，需要排查或重跑

## 第一阶段：zero_shot

- [~] `L100-qwen05-zero`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen05-zero.slurm`
  - 内容：`qwen2_5_0_5b_instruct` 的 `zero_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85255`
  - 状态：已提交，等待调度

- [~] `L100-qwen15-zero`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen15-zero.slurm`
  - 内容：`qwen2_5_1_5b_instruct` 的 `zero_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85251`
  - 状态：已提交，等待调度

- [~] `L100-tinyllama-zero`
  - 脚本：`slurm/taskA_lightweight100/run_L100-tinyllama-zero.slurm`
  - 内容：`tinyllama_1_1b_chat` 的 `zero_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85335`
  - 状态：已提交，等待调度

## 第二阶段：few_shot

- [~] `L100-qwen05-few`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen05-few.slurm`
  - 内容：`qwen2_5_0_5b_instruct` 的 `few_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

- [~] `L100-qwen15-few`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen15-few.slurm`
  - 内容：`qwen2_5_1_5b_instruct` 的 `few_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

- [~] `L100-tinyllama-few`
  - 脚本：`slurm/taskA_lightweight100/run_L100-tinyllama-few.slurm`
  - 内容：`tinyllama_1_1b_chat` 的 `few_shot`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

## 第三阶段：self_consistency

- [~] `L100-qwen05-sc`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen05-sc.slurm`
  - 内容：`qwen2_5_0_5b_instruct` 的 `self_consistency`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

- [~] `L100-qwen15-sc`
  - 脚本：`slurm/taskA_lightweight100/run_L100-qwen15-sc.slurm`
  - 内容：`qwen2_5_1_5b_instruct` 的 `self_consistency`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

- [~] `L100-tinyllama-sc`
  - 脚本：`slurm/taskA_lightweight100/run_L100-tinyllama-sc.slurm`
  - 内容：`tinyllama_1_1b_chat` 的 `self_consistency`，覆盖 `GSM8K / CSQA / MMLU`
  - 作业号：`85338`
  - 状态：已纳入流水线作业，等待自动执行

## 操作记录

- `2026-05-18`：取消 7B sampled-50 作业 `85082`、`85083`。
- `2026-05-18`：确认任务A只要求复现主论文 CoT/SC 方法，不限制具体模型。
- `2026-05-18`：正式模型切换为 `Qwen2.5-0.5B-Instruct`、`Qwen2.5-1.5B-Instruct`、`TinyLlama-1.1B-Chat-v1.0`。
- `2026-05-18`：生成 `model × mode` 的 9 个 lightweight-50 批次脚本，并提交 `85193`、`85194` 进行速度验证。
- `2026-05-18`：取消验证作业 `85193`、`85194`。
- `2026-05-18`：最终口径调整为 lightweight-100：`sample_size = 100`，`zero/few max_new_tokens = 256`，`SC max_new_tokens = 384`，`SC num_paths = 3`。
- `2026-05-18`：生成 `model × mode` 的 9 个 lightweight-100 批次脚本。

- `2026-05-18`：提交 `L100-qwen05-zero`，作业号 `85214`。
- `2026-05-18`：提交 `L100-qwen15-zero`，作业号 `85215`。
- `2026-05-18`：尝试提交 `L100-tinyllama-zero`，因 `QOSMaxSubmitJobPerUserLimit` 暂未提交成功。

- `2026-05-18`：取消 token sweep 作业 `85246`；根据 smoke 结果确定最终生成长度为 `zero/few = 256`、`SC = 384`，SC 路径数保持 `3`。

- `2026-05-18`：取消旧 L100 zero-shot 作业 `85214`、`85215`，使用最终 token 口径重新提交。
- `2026-05-18`：提交 `L100-qwen05-zero`，作业号 `85250`。
- `2026-05-18`：提交 `L100-qwen15-zero`，作业号 `85251`。
- `2026-05-18`：尝试提交 `L100-tinyllama-zero`，仍受 `QOSMaxSubmitJobPerUserLimit` 限制，等待后续补交。

- `2026-05-19`：发现 `L100-qwen05-zero` 复用了旧 token 口径的 GSM8K 结果，已取消作业 `85250`。
- `2026-05-19`：提交 `L100-q05zero-rerun` 强制重跑 `qwen2_5_0_5b_instruct` 的 zero-shot，作业号 `85255`。

- `2026-05-19`：`L100-qwen05-zero` 和 `L100-qwen15-zero` 已完成，两个模型的 zero-shot 三数据集结果均为 100 条。
- `2026-05-19`：提交 `L100-tinyllama-zero`，作业号 `85335`。

- `2026-05-19`：新增并提交剩余任务流水线 `taskA-pipeline`，作业号 `85338`；该作业会串行运行全部 few-shot 与 self-consistency 剩余配置，并自动跳过已有结果。
