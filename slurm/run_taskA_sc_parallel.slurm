#!/usr/bin/env bash
#SBATCH --job-name=taskA-sc-parallel
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
exec > >(tee -a output/log/A/lightweight_100/run_taskA_sc_parallel.log) 2>&1

run_sc_group() {
  local model_key="$1"
  echo "[taskA-sc-parallel] start model=${model_key} mode=self_consistency at $(date)"
  python -u scripts/run_formal_model_group.py \
    --model "${model_key}" \
    --mode self_consistency \
    --output-root output/raw/A/lightweight_100 \
    --sample-size 100 \
    --sample-seed 42
  echo "[taskA-sc-parallel] done model=${model_key} mode=self_consistency at $(date)"
}

echo "[taskA-sc-parallel] pipeline started at $(date)"

run_sc_group qwen2_5_1_5b_instruct
run_sc_group tinyllama_1_1b_chat

echo "[taskA-sc-parallel] pipeline completed at $(date)"
