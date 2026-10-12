#!/usr/bin/env bash
# Statistics, timings and figures from the result files in sim/results.
set -euo pipefail
cd "$(dirname "$0")/../sim"
python3 timing.py
python3 analysis.py
python3 figs_tikz.py
