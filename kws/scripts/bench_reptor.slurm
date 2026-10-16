#!/bin/bash
#SBATCH --job-name=kws-reptor
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/reptor-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-modern

export KB=$HOME/kws_benchmark

nvidia-smi

python $KB/scripts/bench_reptor.py \
  --repo $KB/repos/RepTor-Reproduction \
  --checkpoint $KB/repos/RepTor-Reproduction/checkpoints/reptor9_gscv2_seed1337/best.pt \
  --output $KB/results/reptor9_latency.json
