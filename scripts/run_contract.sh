#!/usr/bin/env bash
# Compile the governance contract, measure gas for K = 5, 10, 15, 20 and run the negative tests.
set -euo pipefail
cd "$(dirname "$0")/../sc"
npm install --no-audit --no-fund
node compile.js
python3 gas_bench.py 5 10 15 20
python3 tamper_test.py
