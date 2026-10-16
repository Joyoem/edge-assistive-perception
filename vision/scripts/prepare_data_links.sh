#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

if [[ ! -d "$BENCH_ADE20K_ROOT/images/validation" || \
      ! -d "$BENCH_ADE20K_ROOT/annotations/validation" ]]; then
  echo "Invalid ADE20K root: $BENCH_ADE20K_ROOT" >&2
  exit 2
fi

link_directory() {
  local target="$1"
  local link="$2"
  mkdir -p "$(dirname "$link")"
  if [[ -L "$link" ]]; then
    current="$(readlink -f "$link")"
    if [[ "$current" == "$(readlink -f "$target")" ]]; then
      return
    fi
    unlink "$link"
  elif [[ -e "$link" ]]; then
    echo "Refusing to replace existing non-symlink: $link" >&2
    exit 2
  fi
  ln -s "$target" "$link"
}

for mmseg_root in \
  "$BENCH_REPOS_DIR/SeaFormer/seaformer-seg" \
  "$BENCH_REPOS_DIR/SCTNet"; do
  mkdir -p "$mmseg_root/data/ade"
  link_directory \
    "$BENCH_ADE20K_ROOT" \
    "$mmseg_root/data/ade/ADEChallengeData2016"
done

bisenet_root="$BENCH_REPOS_DIR/BiSeNet"
mkdir -p "$bisenet_root/datasets/ade20k"
link_directory "$BENCH_ADE20K_ROOT/images" "$bisenet_root/datasets/ade20k/images"
link_directory \
  "$BENCH_ADE20K_ROOT/annotations" \
  "$bisenet_root/datasets/ade20k/annotations"

if [[ ! -f "$bisenet_root/datasets/ade20k/val.txt" ]]; then
  (
    cd "$bisenet_root"
    bench_python tools/gen_dataset_annos.py --dataset ade20k
  )
fi

echo "ADE20K links are ready."

