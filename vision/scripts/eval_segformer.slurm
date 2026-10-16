#!/bin/bash
#SBATCH --job-name=segformer-b0
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/vision_benchmark/logs/segformer-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-legacy
source /home/zhaoyan/vision_benchmark/paths.sh

cd $VB/repos/mmsegmentation-legacy

python -u tools/test.py \
  configs/segformer/segformer_mit-b0_512x512_160k_ade20k.py \
  $VB/checkpoints/segformer_b0_ade20k.pth \
  --eval mIoU
