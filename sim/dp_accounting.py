"""Node-level DP accounting for the Gaussian mechanism of Eq. 10 (sensitivity C, noise sigma*C).
Per-participation epsilon: analytic Gaussian mechanism (Balle & Wang 2018).
Composition over a node's head rounds: Renyi DP (Mironov 2017), eps = min_a [T a/(2 s^2) + ln(1/delta)/(a-1)].
Participation counts are taken from the EECF-IoT FL campaigns (100 rounds, 300 nodes, seeds 0-9)."""
import json, os, hashlib, math, numpy as np
from statistics import NormalDist
Phi = NormalDist().cdf
here = os.path.dirname(os.path.abspath(__file__))
def agm_eps(s, delta):                      # smallest eps with delta(eps) <= delta, sensitivity 1, std s
    f = lambda e: Phi(1 / (2 * s) - e * s) - math.exp(e) * Phi(-1 / (2 * s) - e * s)
    lo, hi = 0.0, 1.0
    while f(hi) > delta: hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid) > delta: lo = mid
        else: hi = mid
    return hi
def rdp_eps(s, T, delta):
    a = np.linspace(1.01, 20000, 400000)
    return float(np.min(T * a / (2 * s * s) + np.log(1 / delta) / (a - 1)))
parts = []
for seed in range(10):
    key = hashlib.md5(json.dumps(['EECF', seed, 100, None, True], sort_keys=True).encode()).hexdigest()
    p = os.path.join(here, 'results', 'simcache', key + '.json')
    if not os.path.exists(p): continue
    log = json.load(open(p))['ch_log'][:100]
    cnt = np.zeros(300, int)
    for ep in log:
        if ep['gate']:
            for c in ep['CH']: cnt[c] += 1
    parts.append(cnt)
cnt = np.concatenate(parts)
T_mean, T_max = float(cnt[cnt > 0].mean()), int(cnt.max())
delta = 1e-5
out = dict(T_mean=T_mean, T_max=T_max, frac_ever_head=float((cnt > 0).mean()), seeds=len(parts), rows=[])
for s in [0.1, 0.5, 1.0, 2.0, 5.0]:
    out['rows'].append(dict(sigma=s, eps_round_agm=agm_eps(s, delta), eps_round_classic=math.sqrt(2 * math.log(1.25 / delta)) / s,
                            eps_total_mean=rdp_eps(s, round(T_mean), delta), eps_total_max=rdp_eps(s, T_max, delta)))
json.dump(out, open(os.path.join(here, 'results', 'dp_accounting.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
