#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/check_setup.sh"
bash "$SCRIPT_DIR/run_accuracy.sh"
bash "$SCRIPT_DIR/run_latency.sh"

echo "All checkpoint benchmarks completed."

