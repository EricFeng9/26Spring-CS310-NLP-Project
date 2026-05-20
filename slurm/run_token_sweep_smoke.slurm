#!/usr/bin/env bash
#SBATCH --job-name=token-sweep
#SBATCH --account=gpulab02
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/%x-%j.out
#SBATCH --error=slurm/%x-%j.err

set -euo pipefail
cd /home/xuyang_lab/cse12311031/26Spring-CS310-NLP-Project
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate fjm
mkdir -p output/log/A/token_sweep_smoke
exec > >(tee -a output/log/A/token_sweep_smoke/run_token_sweep_smoke.log) 2>&1
python -u scripts/smoke_token_sweep.py --samples-per-dataset 1 --sample-seed 42 --num-paths 3 --quick
