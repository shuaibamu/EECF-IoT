#!/usr/bin/env bash
# Federated-learning experiments on TON_IoT and Wazihub (datasets in ../data, see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/../sim"
python3 fl_ton.py --exp main --seeds 0-9 --out results/fl_main.jsonl
python3 fl_ton.py --exp fair --seeds 0-9 --out results/fl_fair.jsonl
for e in ablation ablation_clean dpcomp dpnoscreen byz theta noniid robust; do
  python3 fl_ton.py --exp $e --seeds 0-4 --out results/fl_$e.jsonl
done
python3 fl_waz.py --seeds 0-9 --methods EECF PSO CAPSO GA AOA OOA LEACH FLAG Server-only --out results/waz_main.jsonl
python3 dp_accounting.py
