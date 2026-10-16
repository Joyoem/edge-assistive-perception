#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

if [[ ! -x "$BENCH_MICROMAMBA" ]]; then
  echo "Run scripts/bootstrap_micromamba.sh first." >&2
  exit 2
fi

if [[ ! -x "$BENCH_ENV_PREFIX/bin/python" ]]; then
  "$BENCH_MICROMAMBA" create -y -p "$BENCH_ENV_PREFIX" python=3.8 pip
fi

"$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" pip install --upgrade \
  "pip<25" "setuptools<70" wheel

"$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" pip install \
  torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
  --extra-index-url https://download.pytorch.org/whl/cu113

"$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" pip install \
  "numpy<2" "opencv-python-headless<4.10" pillow scipy tqdm pyyaml \
  timm==0.6.13 einops prettytable gdown fvcore pandas

"$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" pip install \
  mmcv-full==1.6.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

bench_python - <<'PY'
import torch
import mmcv
print("torch", torch.__version__)
print("mmcv", mmcv.__version__)
print("cuda available", torch.cuda.is_available())
PY

