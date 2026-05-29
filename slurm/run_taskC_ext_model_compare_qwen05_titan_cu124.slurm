#!/usr/bin/env bash
#SBATCH --job-name=taskC-q05-cu124
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm-cu124

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/C_titan_cu124/runs output/raw/C_titan_cu124/self_consistency
exec > >(tee -a output/log/C_titan_cu124/run_taskC_ext_model_compare_qwen05_titan_cu124.log) 2>&1

echo "[taskC-q05-cu124] started at $(date)"
echo "[taskC-q05-cu124] node=$(hostname)"
python - <<'PY'
import sys
import torch
print("[taskC-q05-cu124] torch", torch.__version__)
print("[taskC-q05-cu124] torch_cuda", torch.version.cuda)
print("[taskC-q05-cu124] cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("[taskC-q05-cu124] CUDA is not available; refusing to run formal experiment on CPU.")
print("[taskC-q05-cu124] cuda_device", torch.cuda.get_device_name(0))
print("[taskC-q05-cu124] capability", torch.cuda.get_device_capability(0))
PY

python -u scripts/run_task_c_sc_ablation.py \
  --model qwen2_5_0_5b_instruct \
  --datasets gsm8k csqa \
  --num-paths 5 \
  --temperature 0.7 \
  --output-dir output/raw/C_titan_cu124/self_consistency \
  --log-dir output/log/C_titan_cu124/runs \
  --sample-size 100 \
  --sample-seed 42

python -u scripts/run_task_c_sc_ablation.py \
  --model qwen2_5_0_5b_instruct \
  --datasets gsm8k csqa \
  --num-paths 3 \
  --temperature 0.3 \
  --output-dir output/raw/C_titan_cu124/self_consistency \
  --log-dir output/log/C_titan_cu124/runs \
  --sample-size 100 \
  --sample-seed 42

python -u scripts/summarize_task_c_results.py \
  --models qwen2_5_0_5b_instruct qwen2_5_1_5b_instruct \
  --datasets gsm8k csqa \
  --baseline-dir output/raw/A/lightweight_100/self_consistency \
  --c-dir output/raw/C_titan_cu124/self_consistency \
  --output-dir output/log/C_titan_cu124

echo "[taskC-q05-cu124] completed at $(date)"
