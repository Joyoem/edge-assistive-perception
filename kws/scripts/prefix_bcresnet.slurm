#!/bin/bash
#SBATCH --job-name=pfx-bcres
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx3060:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/pfx-bcres-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate /home/zhaoyan/data/.conda/envs/kws-bcresnet

export KB=$HOME/kws_benchmark

python $KB/scripts/bench_prefix_gsc12.py \
  --model bcresnet8 \
  --checkpoint /data/zhaoyan/checkpoints/bcresnet/bcresnet8_seed1337_ep200/best.pt \
  --repo $KB/repos/bcresnet \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest $KB/repos/RepTor-Reproduction/manifests/gsc12_test.csv \
  --output $KB/results/prefix_bcresnet8.json
