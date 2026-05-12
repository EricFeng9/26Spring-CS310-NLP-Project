#!/usr/bin/env bash
set -euo pipefail

# 创建任务A专用 conda 环境，并安装所有依赖。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
  echo "未找到 conda，请先安装 conda。"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
ENV_MANAGER="conda"
if command -v mamba >/dev/null 2>&1; then
  ENV_MANAGER="mamba"
fi

if conda env list | awk '{print $1}' | grep -qx "fjm_CoT"; then
  conda env remove -n fjm_CoT -y
fi

"${ENV_MANAGER}" env create -f environment.yml -y
conda run -n fjm_CoT python -m pip install --upgrade pip
