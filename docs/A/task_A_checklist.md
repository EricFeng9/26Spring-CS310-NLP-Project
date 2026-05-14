# 任务A执行清单

## 1. 当前状态
- 任务A的 baseline 工程已经跑通。
- 已完成数据、配置、脚本、推理链路、结果导出、smoke test 的端到端验证。
- 当前可用的 3 个 baseline 模型为：
  - `mistral_7b_instruct_v0_3`
  - `olmo3_7b_instruct`
  - `llama2_7b_hf`
- 当前已不再使用 `qwen2_5_7b_instruct` 作为可靠 baseline 模型。
- `phi3_5_mini_instruct` 已下载，但在当前环境下推理输出异常，未纳入 baseline。

## 2. 目标完成情况
- [x] 搭建任务A完整 baseline 工程
- [x] 支持 `GSM8K`、`CommonsenseQA`、`MMLU` 三个数据集
- [x] 支持 3 个可运行 baseline 模型
- [x] 支持 `Zero-shot CoT`、`Few-shot CoT`、`Self-Consistency` 三类推理流程
- [x] 完成环境、模型、数据、脚本、配置、冒烟测试准备
- [x] 产出可落盘、可抽取、可统计的 smoke baseline 结果

## 3. 当前基线模型
- `mistral_7b_instruct_v0_3`
- `olmo3_7b_instruct`
- `llama2_7b_hf`

说明：
- `llama2_7b_hf` 使用本机现成本地目录 `/data/student/zhouziyi/VoxelMorph-torch/llama/Llama-2-7b-hf`
- 它是当前环境下能稳定接入并跑通三类 smoke baseline 的第三个模型

## 4. 已完成内容

### 4.1 环境与依赖
- [x] `fjm_CoT` conda 环境可用
- [x] 核心依赖可导入
- [x] GPU 可识别
- [x] 本地模型可加载并完成最小推理

### 4.2 数据集
- [x] `GSM8K` 下载并标准化
- [x] `CommonsenseQA` 下载并标准化
- [x] `MMLU` 下载并标准化
- [x] few-shot 示例文件已构建

已存在的数据文件：
- `data/processed/gsm8k/`
- `data/processed/csqa/`
- `data/processed/mmlu/`
- `data/processed/fewshot/`

### 4.3 模型
- [x] `mistral_7b_instruct_v0_3` 可用
- [x] `olmo3_7b_instruct` 可用
- [x] `llama2_7b_hf` 可用
- [ ] `qwen2_5_7b_instruct` 可靠可用
- [ ] `phi3_5_mini_instruct` 可靠可用

### 4.4 推理脚本
- [x] `scripts/run_zero_shot.py`
- [x] `scripts/run_few_shot.py`
- [x] `scripts/run_self_consistency.py`
- [x] `scripts/run_smoke_tests.sh`

### 4.5 结果导出
- [x] JSONL 导出
- [x] CSV 导出
- [x] Self-Consistency 多路径保存
- [x] Self-Consistency 投票结果保存

## 5. 已完成的 smoke baseline

### 5.1 Zero-shot CoT
- [x] `mistral_7b_instruct_v0_3` on `GSM8K`
- [x] `olmo3_7b_instruct` on `GSM8K`
- [x] `llama2_7b_hf` on `GSM8K`

结果文件：
- `output/raw/A/smoke_tests/zero_shot/mistral_7b_instruct_v0_3_gsm8k.jsonl`
- `output/raw/A/smoke_tests/zero_shot/olmo3_7b_instruct_gsm8k.jsonl`
- `output/raw/A/smoke_tests/zero_shot/llama2_7b_hf_gsm8k.jsonl`

### 5.2 Few-shot CoT
- [x] `mistral_7b_instruct_v0_3` on `CSQA`
- [x] `olmo3_7b_instruct` on `CSQA`
- [x] `llama2_7b_hf` on `CSQA`

结果文件：
- `output/raw/A/smoke_tests/few_shot/mistral_7b_instruct_v0_3_csqa.jsonl`
- `output/raw/A/smoke_tests/few_shot/olmo3_7b_instruct_csqa.jsonl`
- `output/raw/A/smoke_tests/few_shot/llama2_7b_hf_csqa.jsonl`

### 5.3 Self-Consistency
- [x] `mistral_7b_instruct_v0_3` on `MMLU`
- [x] `olmo3_7b_instruct` on `MMLU`
- [x] `llama2_7b_hf` on `MMLU`

结果文件：
- `output/raw/A/smoke_tests/self_consistency/mistral_7b_instruct_v0_3_mmlu.jsonl`
- `output/raw/A/smoke_tests/self_consistency/olmo3_7b_instruct_mmlu.jsonl`
- `output/raw/A/smoke_tests/self_consistency/llama2_7b_hf_mmlu.jsonl`

## 6. 关键实现修复
- [x] prompt 指令切到英文 benchmark 风格
- [x] 不同题型使用不同最终答案格式约束
- [x] 数值题答案抽取修复
- [x] 多选题答案抽取修复
- [x] 多选题支持“选项内容反推选项字母”
- [x] Self-Consistency 结果可稳定落盘
- [x] 非 chat template 模型可走纯文本 prompt
- [x] smoke test 默认 GPU 映射已修正

## 7. 当前已知问题
- `llama2_7b_hf` 是 base 模型，效果弱，只适合作为当前第三个 baseline 模型，不适合作为高质量主模型
- `mistral_7b_instruct_v0_3` 与 `llama2_7b_hf` 在 smoke 样本上存在答错现象
- `qwen2_5_7b_instruct` 在当前环境下输出异常，未修复
- `phi3_5_mini_instruct` 在当前环境下输出乱码，未修复

## 8. 还没做的事
- [ ] 跑正式全量实验
- [ ] 统计完整准确率表
- [ ] 与论文结果做系统对照
- [ ] 输出最终实验报告表格
- [ ] 清理已废弃的 `qwen/phi3` smoke 配置与说明

## 9. 现在怎么复跑

### 9.1 单独运行
```bash
python scripts/run_zero_shot.py --config configs/runs/zero_shot_smoke_mistral.yaml
python scripts/run_zero_shot.py --config configs/runs/zero_shot_smoke_olmo.yaml
python scripts/run_zero_shot.py --config configs/runs/zero_shot_smoke_llama2.yaml

python scripts/run_few_shot.py --config configs/runs/few_shot_smoke_mistral.yaml
python scripts/run_few_shot.py --config configs/runs/few_shot_smoke_olmo.yaml
python scripts/run_few_shot.py --config configs/runs/few_shot_smoke_llama2.yaml

python scripts/run_self_consistency.py --config configs/runs/self_consistency_smoke_mistral.yaml
python scripts/run_self_consistency.py --config configs/runs/self_consistency_smoke_olmo.yaml
python scripts/run_self_consistency.py --config configs/runs/self_consistency_smoke_llama2.yaml
```

### 9.2 一次性冒烟测试
```bash
bash scripts/run_smoke_tests.sh
```

说明：
- 当前 `run_smoke_tests.sh` 默认使用可用模型列表
- 当前默认 GPU 映射已经修正为可稳定运行的设置

## 10. 一句话结论
- 任务A当前已经完成到“3 个模型 × 3 种推理模式的 smoke baseline 全部跑通并可统计”的阶段。
- 当前真正未完成的只剩正式实验、结果汇总、论文对照和文档收尾。
