#!/usr/bin/env bash
set -euo pipefail

BENCH_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_CONFIG_FILE="${BENCH_CONFIG_FILE:-$BENCH_PROJECT_ROOT/benchmark.env}"

if [[ ! -f "$BENCH_CONFIG_FILE" ]]; then
  echo "Missing $BENCH_CONFIG_FILE" >&2
  echo "Run: cp benchmark.env.example benchmark.env, then edit it." >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$BENCH_CONFIG_FILE"

: "${BENCH_STORAGE_ROOT:?Set BENCH_STORAGE_ROOT in benchmark.env}"
: "${BENCH_ADE20K_ROOT:?Set BENCH_ADE20K_ROOT in benchmark.env}"

BENCH_REPOS_DIR="$BENCH_STORAGE_ROOT/repos"
BENCH_WEIGHTS_DIR="$BENCH_STORAGE_ROOT/checkpoints"
BENCH_RESULTS_DIR="$BENCH_STORAGE_ROOT/results"
BENCH_ENV_PREFIX="$BENCH_STORAGE_ROOT/.conda/envs/vision-benchmark"
BENCH_MICROMAMBA="$BENCH_PROJECT_ROOT/.runtime/bin/micromamba"

# Keep package/model caches off the 20 GB home SSD. These variables apply to
# every micromamba-run subprocess launched by this project.
export MAMBA_ROOT_PREFIX="$BENCH_STORAGE_ROOT/.micromamba"
export PIP_CACHE_DIR="$BENCH_STORAGE_ROOT/.cache/pip"
export TORCH_HOME="$BENCH_STORAGE_ROOT/.cache/torch"
export XDG_CACHE_HOME="$BENCH_STORAGE_ROOT/.cache"

BENCH_GPU_ID="${BENCH_GPU_ID:-0}"
BENCH_INPUT_HEIGHT="${BENCH_INPUT_HEIGHT:-512}"
BENCH_INPUT_WIDTH="${BENCH_INPUT_WIDTH:-512}"
BENCH_WARMUP="${BENCH_WARMUP:-100}"
BENCH_RUNS="${BENCH_RUNS:-1000}"

mkdir -p \
  "$BENCH_REPOS_DIR" \
  "$BENCH_WEIGHTS_DIR" \
  "$BENCH_RESULTS_DIR" \
  "$PIP_CACHE_DIR" \
  "$TORCH_HOME"

bench_python() {
  "$BENCH_MICROMAMBA" run -p "$BENCH_ENV_PREFIX" python "$@"
}

bench_find_one() {
  local root="$1"
  local pattern="$2"
  local result
  result="$(find "$root" -type f -path "$pattern" | sort | head -n 1)"
  if [[ -z "$result" ]]; then
    echo "Could not find file matching $pattern under $root" >&2
    return 1
  fi
  printf '%s\n' "$result"
}
