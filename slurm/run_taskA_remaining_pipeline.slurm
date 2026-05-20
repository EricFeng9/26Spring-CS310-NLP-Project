#!/usr/bin/env bash
#SBATCH --job-name=taskA-pipeline
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
exec > >(tee -a output/log/A/lightweight_100/run_taskA_remaining_pipeline.log) 2>&1

run_group() {
  local model_key="$1"
  local mode="$2"
  echo "[taskA-pipeline] start model=${model_key} mode=${mode} at $(date)"
  python -u scripts/run_formal_model_group.py \
    --model "${model_key}" \
    --mode "${mode}" \
    --output-root output/raw/A/lightweight_100 \
    --sample-size 100 \
    --sample-seed 42
  echo "[taskA-pipeline] done model=${model_key} mode=${mode} at $(date)"
}

echo "[taskA-pipeline] pipeline started at $(date)"

for model_key in \
  qwen2_5_0_5b_instruct \
  qwen2_5_1_5b_instruct \
  tinyllama_1_1b_chat
do
  run_group "${model_key}" few_shot
done

for model_key in \
  qwen2_5_0_5b_instruct \
  qwen2_5_1_5b_instruct \
  tinyllama_1_1b_chat
do
  run_group "${model_key}" self_consistency
done

echo "[taskA-pipeline] pipeline completed at $(date)"
