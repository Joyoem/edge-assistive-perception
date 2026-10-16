#!/bin/bash
#SBATCH --job-name=dscnn-full
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx3060:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/dscnn-full-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate /home/zhaoyan/data/.conda/envs/kws-bcresnet

export KB=$HOME/kws_benchmark

python $KB/scripts/train_dscnn_gsc12.py \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest-dir $KB/repos/RepTor-Reproduction/manifests \
  --output-dir /data/zhaoyan/checkpoints/dscnn \
  --epochs 36 \
  --num-workers 2 \
  --seed 1337
