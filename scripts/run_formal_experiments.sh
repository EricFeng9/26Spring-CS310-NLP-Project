#!/usr/bin/env bash

set -euo pipefail

# 任务A正式实验串行运行脚本。
# 设计原则：
# 1. 全部正式实验固定按单卡串行执行，避免多模型并发争抢显存导致结果不稳定。
# 2. 运行顺序固定为：Zero-shot -> Few-shot -> Self-Consistency。
# 3. 配置文件使用显式清单，避免运行时动态拼接路径带来遗漏或顺序混乱。

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/A/final_runs
exec > >(tee -a output/log/A/final_runs/run_formal_experiments.log) 2>&1

ZERO_SHOT_CONFIGS=(
  "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_gsm8k.yaml"
  "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_csqa.yaml"
  "configs/runs/formal/zero_shot_mistral_7b_instruct_v0_3_mmlu.yaml"
  "configs/runs/formal/zero_shot_olmo3_7b_instruct_gsm8k.yaml"
  "configs/runs/formal/zero_shot_olmo3_7b_instruct_csqa.yaml"
  "configs/runs/formal/zero_shot_olmo3_7b_instruct_mmlu.yaml"
  "configs/runs/formal/zero_shot_llama2_7b_hf_gsm8k.yaml"
  "configs/runs/formal/zero_shot_llama2_7b_hf_csqa.yaml"
  "configs/runs/formal/zero_shot_llama2_7b_hf_mmlu.yaml"
)

FEW_SHOT_CONFIGS=(
  "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_gsm8k.yaml"
  "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_csqa.yaml"
  "configs/runs/formal/few_shot_mistral_7b_instruct_v0_3_mmlu.yaml"
  "configs/runs/formal/few_shot_olmo3_7b_instruct_gsm8k.yaml"
  "configs/runs/formal/few_shot_olmo3_7b_instruct_csqa.yaml"
  "configs/runs/formal/few_shot_olmo3_7b_instruct_mmlu.yaml"
  "configs/runs/formal/few_shot_llama2_7b_hf_gsm8k.yaml"
  "configs/runs/formal/few_shot_llama2_7b_hf_csqa.yaml"
  "configs/runs/formal/few_shot_llama2_7b_hf_mmlu.yaml"
)

SELF_CONSISTENCY_CONFIGS=(
  "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_gsm8k.yaml"
  "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_csqa.yaml"
  "configs/runs/formal/self_consistency_mistral_7b_instruct_v0_3_mmlu.yaml"
  "configs/runs/formal/self_consistency_olmo3_7b_instruct_gsm8k.yaml"
  "configs/runs/formal/self_consistency_olmo3_7b_instruct_csqa.yaml"
  "configs/runs/formal/self_consistency_olmo3_7b_instruct_mmlu.yaml"
  "configs/runs/formal/self_consistency_llama2_7b_hf_gsm8k.yaml"
  "configs/runs/formal/self_consistency_llama2_7b_hf_csqa.yaml"
  "configs/runs/formal/self_consistency_llama2_7b_hf_mmlu.yaml"
)

run_zero_shot_group() {
  local config
  for config in "${ZERO_SHOT_CONFIGS[@]}"; do
    echo "[formal][zero_shot] running ${config}"
    python scripts/run_zero_shot.py --config "${config}"
  done
}

run_few_shot_group() {
  local config
  for config in "${FEW_SHOT_CONFIGS[@]}"; do
    echo "[formal][few_shot] running ${config}"
    python scripts/run_few_shot.py --config "${config}"
  done
}

run_self_consistency_group() {
  local config
  for config in "${SELF_CONSISTENCY_CONFIGS[@]}"; do
    echo "[formal][self_consistency] running ${config}"
    python scripts/run_self_consistency.py --config "${config}"
  done
}

run_zero_shot_group
run_few_shot_group
run_self_consistency_group

echo "[formal] 所有正式实验运行完成。"
