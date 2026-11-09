"""first_epoch_upload.py -- head-to-sink distance and Eq. 9 upload energy of the first-epoch heads (N = 300, 30 seeds)."""
import json, os
import numpy as np
import eecf_sim as S
out = {}
for m in ['EECF', 'EECF:upl', 'PSO', 'LEACHC', 'EWKM']:
    up, dch, mx = [], [], []
    for seed in range(30):
        net = S.Net(300, seed); rng = np.random.default_rng(10_000 + seed)
        CH, _ = S.cluster_once(net, m, rng, net.E.sum(), np.array([], int))
        e = S.radio_tx(S.upload_bits(), net.dbs[CH], net.is4[CH])
        up.append(e.sum()); dch.append(net.dbs[CH].mean()); mx.append(e.max())
    out[m] = dict(up=float(np.mean(up)), up_sd=float(np.std(up, ddof=1)), dch=float(np.mean(dch)), max_head=float(np.mean(mx)))
    print(m, out[m])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'r2_first_epoch_upload.json'), 'w'), indent=1)
