#!/usr/bin/env bash
set -euo pipefail

# 运行任务A的完整冒烟测试。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

bash scripts/verify_env.sh
conda run -n fjm_CoT python scripts/download_datasets.py
conda run -n fjm_CoT python scripts/build_fewshot_prompts.py

for model in qwen2_5_7b_instruct mistral_7b_instruct_v0_3 olmo3_7b_instruct; do
  conda run -n fjm_CoT python scripts/download_models.py --model "${model}"

  python - <<PY
from pathlib import Path
import yaml

files = {
    "zero_shot_smoke.yaml": ("gsm8k", "test"),
    "few_shot_smoke.yaml": ("csqa", "validation"),
    "self_consistency_smoke.yaml": ("mmlu", "test"),
}
base = Path("${PROJECT_ROOT}") / "configs" / "runs"
for file_name, (dataset_key, split_name) in files.items():
    path = base / file_name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["model_key"] = "${model}"
    data["dataset_key"] = dataset_key
    data["split"] = split_name
    if file_name == "few_shot_smoke.yaml":
        data["few_shot_examples_path"] = f"data/processed/fewshot/{dataset_key}_fewshot.jsonl"
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
PY

  conda run -n fjm_CoT python scripts/run_zero_shot.py --config configs/runs/zero_shot_smoke.yaml
  conda run -n fjm_CoT python scripts/run_few_shot.py --config configs/runs/few_shot_smoke.yaml
  conda run -n fjm_CoT python scripts/run_self_consistency.py --config configs/runs/self_consistency_smoke.yaml
done
