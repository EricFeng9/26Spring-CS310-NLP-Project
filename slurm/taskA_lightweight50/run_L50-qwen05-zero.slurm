#!/usr/bin/env bash
#SBATCH --job-name=L50-qwen05-zero
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail
cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/A/lightweight_50
exec > >(tee -a output/log/A/lightweight_50/run_L50-qwen05-zero.log) 2>&1
python -u scripts/run_formal_model_group.py --model qwen2_5_0_5b_instruct --mode zero_shot --output-root output/raw/A/lightweight_50 --sample-size 50 --sample-seed 42
