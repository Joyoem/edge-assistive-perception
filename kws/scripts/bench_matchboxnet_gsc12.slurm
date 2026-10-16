#!/bin/bash
#SBATCH --job-name=kws-matchbox
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=/home/zhaoyan/kws_benchmark/logs/matchbox-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate /home/zhaoyan/data/.conda/envs/kws-nemo

export KB=$HOME/kws_benchmark

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

nvidia-smi

python $KB/scripts/bench_matchboxnet_gsc12.py \
  --checkpoint $KB/checkpoints/matchboxnet3x1x64_gscv2_subset.nemo \
  --data-root /data/zhaoyan/datasets/speech_commands_v0.02 \
  --manifest $KB/repos/RepTor-Reproduction/manifests/gsc12_test.csv \
  --output $KB/results/matchboxnet3x1x64_gsc12_real.json
