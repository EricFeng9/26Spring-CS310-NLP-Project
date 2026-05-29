#!/usr/bin/env bash
#SBATCH --job-name=taskC-ablation
#SBATCH --account=gpulab02
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail

cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p output/log/C/runs output/raw/C/self_consistency
exec > >(tee -a output/log/C/run_taskC_minimal_ablation.log) 2>&1

echo "[taskC] minimal ablation started at $(date)"

python -u scripts/run_task_c_sc_ablation.py \
  --model qwen2_5_1_5b_instruct \
  --datasets gsm8k csqa \
  --num-paths 5 \
  --temperature 0.7 \
  --output-dir output/raw/C/self_consistency \
  --log-dir output/log/C/runs \
  --sample-size 100 \
  --sample-seed 42

python -u scripts/run_task_c_sc_ablation.py \
  --model qwen2_5_1_5b_instruct \
  --datasets gsm8k csqa \
  --num-paths 3 \
  --temperature 0.3 \
  --output-dir output/raw/C/self_consistency \
  --log-dir output/log/C/runs \
  --sample-size 100 \
  --sample-seed 42

python -u scripts/summarize_task_c_results.py \
  --models qwen2_5_1_5b_instruct \
  --datasets gsm8k csqa \
  --baseline-dir output/raw/A/lightweight_100/self_consistency \
  --c-dir output/raw/C/self_consistency \
  --output-dir output/log/C

echo "[taskC] minimal ablation completed at $(date)"
