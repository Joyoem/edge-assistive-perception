#!/bin/bash
#SBATCH --job-name=seaformer-t
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/vision_benchmark/logs/seaformer-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-legacy
source /home/zhaoyan/vision_benchmark/paths.sh

cd $VB/repos/SeaFormer/seaformer-seg
export PYTHONPATH="$PWD:$PYTHONPATH"

python -u tools/test.py \
  local_configs/seaformer/seaformer_tiny_512x512_160k_4x8_ade20k.py \
  $VB/checkpoints/seaformer_tiny_ade20k.pth \
  --eval mIoU
