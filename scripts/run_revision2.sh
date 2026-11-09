#!/usr/bin/env bash
# Second-round experiments: full radio accounting (data gathering + FL upload, download and local
# training), intra-cluster-cost baselines (LEACH-C, energy-weighted k-medoids), the battery-relative
# uplink term (Eq. 7a), sensitivity runs, theta_byz chosen on a validation split, and FL accuracy of
# the added topologies.
set -euo pipefail
cd "$(dirname "$0")/../sim"
FC="fl_charge=1"
python3 eecf_sim.py --methods EECF EECF:upl PSO CAPSO GA AOA OOA LEACH LEACHC EWKM FLAG --N 300 --seeds 0-29 --P $FC --out results/r2_n300_full.jsonl
python3 eecf_sim.py --methods LEACHC EWKM --N 300 --seeds 0-29 --P fl_charge=0 --out results/r2_n300_dataonly.jsonl
python3 eecf_sim.py --methods Greedy PSO:Fcomp EECF:baseFit --N 300 --seeds 0-29 --P $FC --out results/r2_decouple_full.jsonl
for k in 5 15 20; do python3 eecf_sim.py --methods EECF EECF:upl PSO --N 300 --seeds 0-9 --P $FC n_CH=$k --out results/r2_nch_full.jsonl; done
for e in 0.00065e-12 0.0026e-12; do python3 eecf_sim.py --methods EECF EECF:upl PSO --N 300 --seeds 0-9 --P $FC eps_mp=$e --out results/r2_radio_full.jsonl; done
python3 eecf_sim.py --methods EECF EECF:upl PSO --N 300 --seeds 0-9 --P $FC fl_every=100 --out results/r2_flevery_full.jsonl
python3 eecf_sim.py --methods EECF EECF:upl PSO --N 300 --seeds 0-9 --P $FC c_frac=0.25 --out results/r2_comp_full.jsonl
python3 eecf_sim.py --methods EECF PSO --N 300 --seeds 0-9 --P $FC e_mac=1e-8 --out results/r2_emac_full.jsonl
python3 eecf_sim.py --methods EECF --N 300 --seeds 0-9 --P $FC gc_norm=1 --out results/r2_gcnorm_full.jsonl
python3 eecf_sim.py --methods EECF --N 300 --seeds 0-9 --P fl_charge=0 gc_norm=1 --out results/r2_gcnorm_dataonly.jsonl
python3 eecf_sim.py --methods EECF:static EECF:noECP EECF:noBCS EECF:noDQA EECF:single --N 300 --seeds 0-29 --P $FC --out results/r2_abl_full.jsonl
python3 eecf_sim.py --methods EECF EECF:upl PSO --N 100 200 400 500 --seeds 0-9 --P $FC --out results/r2_scale_full.jsonl
python3 fl_ton.py --exp theta_val --seeds 0-4 --out results/r2_fl_theta_val.jsonl
python3 fl_ton.py --exp r2main --seeds 0-9 --out results/r2_fl_main.jsonl
python3 fl_waz.py --methods LEACHC EWKM EECF:upl --seeds 0-9 --out results/r2_waz.jsonl
python3 first_epoch_upload.py
python3 analysis_r2.py
