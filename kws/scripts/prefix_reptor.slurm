#!/bin/bash
#SBATCH --job-name=pfx-reptor
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx3060:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/pfx-reptor-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-modern

export KB=$HOME/kws_benchmark

python $KB/scripts/bench_prefix_gsc12.py \
  --model reptor9 \
  --checkpoint $KB/repos/RepTor-Reproduction/checkpoints/reptor9_gscv2_seed1337/best.pt \
  --repo $KB/repos/RepTor-Reproduction \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest $KB/repos/RepTor-Reproduction/manifests/gsc12_test.csv \
  --output $KB/results/prefix_reptor9.json
