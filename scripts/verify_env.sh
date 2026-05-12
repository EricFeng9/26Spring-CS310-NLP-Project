#!/usr/bin/env bash
set -euo pipefail

# 验证环境依赖、CUDA 与量化组件是否可用。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda run -n fjm_CoT python - <<'PY'
import importlib
import torch

modules = [
    "transformers",
    "accelerate",
    "datasets",
    "sentencepiece",
    "scipy",
    "pandas",
    "tqdm",
    "yaml",
    "evaluate",
    "bitsandbytes",
    "huggingface_hub",
    "pytest",
]

for name in modules:
    importlib.import_module(name)

importlib.import_module("google.protobuf")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA 不可用，环境验证失败。")

print("环境验证通过。")
print("CUDA 设备数量：", torch.cuda.device_count())
print("当前 PyTorch 版本：", torch.__version__)
PY
