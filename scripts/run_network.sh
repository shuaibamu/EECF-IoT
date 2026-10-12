#!/usr/bin/env bash
# Network-level experiments (lifetime, energy, reliability, objective vs search, sensitivity).
set -euo pipefail
cd "$(dirname "$0")/../sim"
python3 eecf_sim.py --methods EECF EECF:static EECF:noECP EECF:noBCS EECF:noDQA EECF:single --N 300 --seeds 0-29 --curve --out results/n300_eecf.jsonl
python3 eecf_sim.py --methods PSO CAPSO GA AOA OOA LEACH FLAG --N 300 --seeds 0-29 --curve --out results/n300_base.jsonl
python3 eecf_sim.py --methods EECF PSO CAPSO GA AOA OOA LEACH FLAG --N 100 200 --seeds 0-9 --out results/scale_a.jsonl
python3 eecf_sim.py --methods EECF PSO CAPSO GA AOA OOA LEACH FLAG --N 400 500 --seeds 0-9 --out results/scale_b.jsonl
python3 eecf_sim.py --methods Greedy PSO:Fcomp EECF:baseFit --N 300 --seeds 0-29 --out results/decouple.jsonl
for k in 5 15 20; do python3 eecf_sim.py --methods EECF PSO --N 300 --seeds 0-9 --P n_CH=$k --out results/nch.jsonl; done
for e in 0.00065e-12 0.0026e-12; do python3 eecf_sim.py --methods EECF PSO --N 300 --seeds 0-9 --P eps_mp=$e --out results/radio.jsonl; done
python3 extra_exps.py
python3 placement.py
