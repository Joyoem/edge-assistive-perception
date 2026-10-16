#!/bin/bash
#SBATCH --job-name=effvit-b1
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/vision_benchmark/logs/efficientvit-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-modern
source /home/zhaoyan/vision_benchmark/paths.sh

cd $VB/repos/efficientvit

nvidia-smi

python -u applications/efficientvit_seg/eval_efficientvit_seg_model.py \
  --dataset ade20k \
  --crop_size 512 \
  --batch_size 1 \
  --workers 4 \
  --model efficientvit-seg-b1-ade20k \
  --weight_url $VB/checkpoints/efficientvit_seg_b1_ade20k.pt \
  --path $ADE20K_ROOT/images/validation
