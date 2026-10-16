#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

export CUDA_VISIBLE_DEVICES="$BENCH_GPU_ID"

sea_root="$BENCH_REPOS_DIR/SeaFormer/seaformer-seg"
sct_root="$BENCH_REPOS_DIR/SCTNet"
bi_root="$BENCH_REPOS_DIR/BiSeNet"

sea_config="$sea_root/local_configs/seaformer/seaformer_tiny_512x512_160k_2x8_ade20k.py"
sct_config="$(bench_find_one "$sct_root/configs" '*/sctnet/ADE20K/sctnet-s*ade*.py')"
bi_config="$(bench_find_one "$bi_root/configs" '*/bisenetv2*ade*.py')"

run_one() {
  local model="$1"
  local kind="$2"
  local repo="$3"
  local config="$4"
  local checkpoint="$5"
  local source="$6"
  local precision="$7"

  bench_python "$BENCH_PROJECT_ROOT/tools/benchmark_model.py" \
    --model-name "$model" \
    --kind "$kind" \
    --repo "$repo" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --checkpoint-source "$source" \
    --height "$BENCH_INPUT_HEIGHT" \
    --width "$BENCH_INPUT_WIDTH" \
    --warmup "$BENCH_WARMUP" \
    --runs "$BENCH_RUNS" \
    --precision "$precision" \
    --results-dir "$BENCH_RESULTS_DIR"
}

for precision in fp32 fp16; do
  run_one seaformer_tiny mmseg "$sea_root" "$sea_config" \
    "$BENCH_WEIGHTS_DIR/seaformer_tiny_ade20k.pth" official "$precision"
  run_one sctnet_s mmseg_sct "$sct_root" "$sct_config" \
    "$BENCH_WEIGHTS_DIR/SCTNet-S-ADE20K.pth" official "$precision"
  run_one bisenetv2 bisenet "$bi_root" "$bi_config" \
    "$BENCH_WEIGHTS_DIR/model_final_v2_ade20k.pth" third_party "$precision"
done

echo "Latency table: $BENCH_RESULTS_DIR/latency_summary.csv"

