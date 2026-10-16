#!/bin/bash
#SBATCH --job-name=bcres-smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/bcres-smoke-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate /home/zhaoyan/data/.conda/envs/kws-bcresnet

export KB=$HOME/kws_benchmark

python $KB/scripts/train_bcresnet_gsc12.py \
  --repo $KB/repos/bcresnet \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest-dir $KB/repos/RepTor-Reproduction/manifests \
  --output-dir /data/zhaoyan/checkpoints/bcresnet \
  --tau 8 \
  --epochs 1 \
  --seed 1337
  --num-workers 2
