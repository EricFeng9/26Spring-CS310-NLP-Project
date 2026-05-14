#!/usr/bin/env bash
set -euo pipefail

# 运行任务A的完整冒烟测试。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
# 当前 Python 里的 CUDA 序号和 `nvidia-smi` 显示并不一致。
# 实测默认值设为 `0` 时，才能稳定落到 48GB 的 A6000 上。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/verify_env.sh
conda run -n fjm_CoT python scripts/download_datasets.py
conda run -n fjm_CoT python scripts/build_fewshot_prompts.py

for model in mistral_7b_instruct_v0_3 olmo3_7b_instruct phi3_5_mini_instruct; do
  conda run -n fjm_CoT python scripts/download_models.py --model "${model}"
  if [[ "${model}" == "mistral_7b_instruct_v0_3" ]]; then
    zero_config="configs/runs/zero_shot_smoke_mistral.yaml"
    few_config="configs/runs/few_shot_smoke_mistral.yaml"
    sc_config="configs/runs/self_consistency_smoke_mistral.yaml"
  elif [[ "${model}" == "olmo3_7b_instruct" ]]; then
    zero_config="configs/runs/zero_shot_smoke_olmo.yaml"
    few_config="configs/runs/few_shot_smoke_olmo.yaml"
    sc_config="configs/runs/self_consistency_smoke_olmo.yaml"
  elif [[ "${model}" == "phi3_5_mini_instruct" ]]; then
    zero_config="configs/runs/zero_shot_smoke_phi3.yaml"
    few_config="configs/runs/few_shot_smoke_phi3.yaml"
    sc_config="configs/runs/self_consistency_smoke_phi3.yaml"
  else
    echo "未知模型：${model}"
    exit 1
  fi

  conda run -n fjm_CoT python scripts/run_zero_shot.py --config "${zero_config}"
  conda run -n fjm_CoT python scripts/run_few_shot.py --config "${few_config}"
  conda run -n fjm_CoT python scripts/run_self_consistency.py --config "${sc_config}"
done
