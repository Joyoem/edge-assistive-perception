#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

fail=0
for path in \
  "$BENCH_ADE20K_ROOT/images/validation" \
  "$BENCH_ADE20K_ROOT/annotations/validation" \
  "$BENCH_REPOS_DIR/SeaFormer/seaformer-seg" \
  "$BENCH_REPOS_DIR/SCTNet" \
  "$BENCH_REPOS_DIR/BiSeNet" \
  "$BENCH_WEIGHTS_DIR/seaformer_tiny_ade20k.pth" \
  "$BENCH_WEIGHTS_DIR/SCTNet-S-ADE20K.pth" \
  "$BENCH_WEIGHTS_DIR/model_final_v2_ade20k.pth"; do
  if [[ -e "$path" ]]; then
    printf 'OK      %s\n' "$path"
  else
    printf 'MISSING %s\n' "$path"
    fail=1
  fi
done

if [[ ! -x "$BENCH_MICROMAMBA" || ! -x "$BENCH_ENV_PREFIX/bin/python" ]]; then
  echo "MISSING local micromamba environment" >&2
  fail=1
fi

exit "$fail"

