#!/usr/bin/env bash
#SBATCH --job-name=taskC-smoke
#SBATCH --account=gpulab02
#SBATCH --partition=rtx2080ti
#SBATCH --qos=rtx2080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/C/smoke output/raw/C/smoke_self_consistency
exec > >(tee -a output/log/C/smoke/run_taskC_env_smoke.log) 2>&1

echo "[taskC-smoke] started at $(date)"
echo "[taskC-smoke] node=$(hostname)"
python - <<'PY'
import torch
print("[taskC-smoke] torch", torch.__version__)
print("[taskC-smoke] cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[taskC-smoke] cuda_device", torch.cuda.get_device_name(0))
    print("[taskC-smoke] capability", torch.cuda.get_device_capability(0))
PY

python -u scripts/run_task_c_sc_ablation.py \
  --model qwen2_5_1_5b_instruct \
  --datasets gsm8k \
  --num-paths 1 \
  --temperature 0.7 \
  --output-dir output/raw/C/smoke_self_consistency \
  --log-dir output/log/C/smoke \
  --sample-size 1 \
  --sample-seed 42 \
  --max-new-tokens 16

echo "[taskC-smoke] completed at $(date)"
