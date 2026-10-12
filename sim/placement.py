"""First-epoch head placement at N = 300 (seeds 0-29): head-to-sink distance, member-to-head distance, Zigbee share of heads,
and RCIC(0.75) of the first backbone. Written to results/placement.json."""
import json, os, numpy as np, eecf_sim as S
out = {}
for m in ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'Greedy', 'PSO:Fcomp']:
    rows = []
    for seed in range(30):
        net = S.Net(300, seed); rng = np.random.default_rng(10_000 + seed)
        CH, _ = S.cluster_once(net, m, rng, net.E.sum(), np.array([], int))
        a = CH[net.D[:, CH].argmin(1)]; mem = np.setdiff1d(np.arange(300), CH)
        rows.append([net.dbs[CH].mean(), net.D[mem, a[mem]].mean(), net.is4[CH].mean(), S.rcic(net, CH, a, np.random.default_rng(99 + seed))])
    r = np.array(rows); out[m] = dict(d_sink=list(map(float, [r[:, 0].mean(), r[:, 0].std(ddof=1)])), d_member=list(map(float, [r[:, 1].mean(), r[:, 1].std(ddof=1)])),
                                     zig_share=list(map(float, [r[:, 2].mean(), r[:, 2].std(ddof=1)])), rcic=list(map(float, [r[:, 3].mean(), r[:, 3].std(ddof=1)])), n=30)
    print(m, {k: [round(x, 3) for x in v] for k, v in out[m].items() if k != 'n'}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'placement.json'), 'w'), indent=1)
