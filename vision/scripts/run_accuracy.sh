#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

bash "$SCRIPT_DIR/prepare_data_links.sh"

export CUDA_VISIBLE_DEVICES="$BENCH_GPU_ID"

sea_root="$BENCH_REPOS_DIR/SeaFormer/seaformer-seg"
sct_root="$BENCH_REPOS_DIR/SCTNet"
bi_root="$BENCH_REPOS_DIR/BiSeNet"

sea_config="$sea_root/local_configs/seaformer/seaformer_tiny_512x512_160k_2x8_ade20k.py"
sct_config="$(bench_find_one "$sct_root/configs" '*/sctnet/ADE20K/sctnet-s*ade*.py')"
bi_config="$(bench_find_one "$bi_root/configs" '*/bisenetv2*ade*.py')"

run_logged() {
  local name="$1"
  local workdir="$2"
  shift 2
  echo "===== $name ====="
  (
    cd "$workdir"
    "$@"
  ) 2>&1 | tee "$BENCH_RESULTS_DIR/${name}_accuracy.log"
}

run_logged seaformer_tiny "$sea_root" \
  "$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" python tools/test.py \
  "$sea_config" "$BENCH_WEIGHTS_DIR/seaformer_tiny_ade20k.pth" \
  --eval mIoU

run_logged sctnet_s "$sct_root" \
  "$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" python tools/test.py \
  "$sct_config" "$BENCH_WEIGHTS_DIR/SCTNet-S-ADE20K.pth" \
  --eval mIoU

run_logged bisenetv2 "$bi_root" \
  "$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" python tools/evaluate.py \
  --config "$bi_config" \
  --weight-path "$BENCH_WEIGHTS_DIR/model_final_v2_ade20k.pth"

bench_python "$BENCH_PROJECT_ROOT/tools/collect_accuracy.py" \
  --results-dir "$BENCH_RESULTS_DIR"

echo "Accuracy summary: $BENCH_RESULTS_DIR/accuracy_summary.csv"

