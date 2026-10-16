#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

bench_python "$BENCH_PROJECT_ROOT/tools/fetch_assets.py" \
  --storage-root "$BENCH_STORAGE_ROOT" \
  --ade20k-root "$BENCH_ADE20K_ROOT" \
  --download-ade20k "${BENCH_DOWNLOAD_ADE20K:-0}" \
  --seaformer-url "${BENCH_SEAFORMER_CKPT_URL:-}" \
  --sctnet-url "${BENCH_SCTNET_CKPT_URL:-}" \
  --bisenet-url "${BENCH_BISENETV2_CKPT_URL:-}"

