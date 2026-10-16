#!/bin/bash
#SBATCH --job-name=bench-dscnn
#SBATCH --partition=main
#SBATCH --gres=gpu:rtx3060:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/bench-dscnn-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate /home/zhaoyan/data/.conda/envs/kws-bcresnet

export KB=$HOME/kws_benchmark

python $KB/scripts/bench_remaining_kws.py \
  --model dscnn \
  --checkpoint /data/zhaoyan/checkpoints/dscnn/dscnn_mlperfstyle_seed1337_ep36/best.pt \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest $KB/repos/RepTor-Reproduction/manifests/gsc12_test.csv \
  --output $KB/results/dscnn_gsc12_real.json
