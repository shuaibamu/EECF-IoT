#!/usr/bin/env bash
# Full reproduction from scratch. The shipped per-run results are moved to
# sim/results/shipped/ first, because the experiment scripts append to their output files.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
res="$here/../sim/results"
mkdir -p "$res/shipped"
mv "$res"/*.jsonl "$res/shipped/" 2>/dev/null || true
bash "$here/run_contract.sh"
bash "$here/run_network.sh"
bash "$here/run_federated.sh"
bash "$here/run_analysis.sh"
