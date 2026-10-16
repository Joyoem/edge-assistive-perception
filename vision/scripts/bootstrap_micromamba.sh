#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/.runtime"
MAMBA_BIN="$RUNTIME_DIR/bin/micromamba"

# Only the small micromamba executable is stored with the code. The actual
# environment is created under BENCH_STORAGE_ROOT by setup_environment.sh.

if [[ -x "$MAMBA_BIN" ]]; then
  "$MAMBA_BIN" --version
  exit 0
fi

mkdir -p "$RUNTIME_DIR/bin"
archive="$RUNTIME_DIR/micromamba.tar.bz2"
curl -L --fail --retry 3 \
  https://micro.mamba.pm/api/micromamba/linux-64/latest \
  -o "$archive"
tar -xjf "$archive" -C "$RUNTIME_DIR" bin/micromamba
"$MAMBA_BIN" --version
