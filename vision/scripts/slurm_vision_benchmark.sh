#!/usr/bin/env bash
#SBATCH --job-name=vision-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=vision-benchmark-%j.out

set -euo pipefail

# sbatch sets this to the directory from which the job was submitted.
cd "${SLURM_SUBMIT_DIR:?Submit this file from the project directory}"

bash scripts/run_gpu_benchmark.sh
