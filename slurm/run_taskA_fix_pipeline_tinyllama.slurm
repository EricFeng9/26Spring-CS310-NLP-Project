#!/usr/bin/env bash
#SBATCH --job-name=taskAfix-tiny
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/A/lightweight_100
exec > >(tee -a output/log/A/lightweight_100/run_taskA_fix_pipeline_tinyllama.log) 2>&1

run_group() {
  local model_key="$1"
  local mode="$2"
  echo "[taskAfix-tiny] start model=${model_key} mode=${mode} at $(date)"
  python -u scripts/run_formal_model_group.py \
    --model "${model_key}" \
    --mode "${mode}" \
    --output-root output/raw/A/lightweight_100 \
    --sample-size 100 \
    --sample-seed 42
  echo "[taskAfix-tiny] done model=${model_key} mode=${mode} at $(date)"
}

echo "[taskAfix-tiny] pipeline started at $(date)"

run_group tinyllama_1_1b_chat zero_shot
run_group tinyllama_1_1b_chat few_shot
run_group tinyllama_1_1b_chat self_consistency

echo "[taskAfix-tiny] pipeline completed at $(date)"
