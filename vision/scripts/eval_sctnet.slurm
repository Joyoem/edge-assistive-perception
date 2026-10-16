#!/bin/bash
#SBATCH --job-name=sctnet-s
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhaoyan/vision_benchmark/logs/sctnet-%j.out

source /home/zhaoyan/miniforge3/etc/profile.d/conda.sh
conda activate vision-legacy
source /home/zhaoyan/vision_benchmark/paths.sh

cd $VB/repos/SCTNet
export PYTHONPATH="$VB/repos/SCTNet:$PYTHONPATH"

echo "===== IMPORT CHECK ====="
python - <<'PY'
import mmseg
from mmseg.models.builder import SEGMENTORS

print("mmseg path:", mmseg.__file__)
print("EncoderDecoder_Distill:",
      SEGMENTORS.get("EncoderDecoder_Distill"))
PY

python -u tools/test.py \
  configs/sctnet/ADE20K/sctnet-s_8x4_160k_ade.py \
  $VB/checkpoints/SCTNet-S-ADE20K.pth \
  --eval mIoU \
  --cfg-options \
    model.backbone.init_cfg=None \
    model.auxiliary_head=None
