"""
fl_waz.py -- irrigation-decision FL on the Wazihub soil-moisture dataset.

Samples: for each field k in 1..4, a 12-step (1 h, 5-min cadence) window of the seven features
[soil humidity k, air temperature, air humidity, pressure, wind speed, wind gust, wind direction]
predicts the irrigation state of field k at the end of the window. Missing values -> sentinel -1
after min-max scaling. Chronological 80/20 split per field. Each of the N=300 sensor nodes holds
a contiguous block of training windows from one field; FL clients are the elected heads (FLAG:
K random alive nodes; Server-only: centralised training on all training windows).
"""
import argparse, json, time, math, os, hashlib
import numpy as np, pandas as pd, torch, torch.nn as nn
import eecf_sim as S
import fl_ton as T

torch.set_num_threads(1)
DATA = os.environ.get('WAZIHUB_CSV', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'Train.csv'))
W = 12
WEATHER = ['Air temperature (C)', 'Air humidity (%)', 'Pressure (KPa)', 'Wind speed (Km/h)', 'Wind gust (Km/h)', 'Wind direction (Deg)']

def load_waz(cache={}):
    if 'd' in cache: return cache['d']
    d = pd.read_csv(DATA).sort_values('timestamp').reset_index(drop=True)
    Xtr, ytr, ftr, Xte, yte = [], [], [], [], []
    feats_all = []
    for k in range(1, 5):
        cols = [f'Soil humidity {k}'] + WEATHER
        sub = d[cols + [f'Irrigation field {k}']].copy()
        ok = sub[f'Soil humidity {k}'].notna() & sub[f'Irrigation field {k}'].notna()
        sub = sub[ok].reset_index(drop=True)
        feats_all.append(sub[cols].values)
    allf = np.vstack(feats_all)
    lo, hi = np.nanmin(allf, 0), np.nanmax(allf, 0)
    stats = []
    for k in range(1, 5):
        cols = [f'Soil humidity {k}'] + WEATHER
        sub = d[cols + [f'Irrigation field {k}']].copy()
        ok = sub[f'Soil humidity {k}'].notna() & sub[f'Irrigation field {k}'].notna()
        sub = sub[ok].reset_index(drop=True)
        X = (sub[cols].values - lo) / (hi - lo + 1e-8)
        dh = np.r_[0.0, np.diff(sub[f'Soil humidity {k}'].values)] / 2.0       # humidity rate (per 5 min)
        X = np.column_stack([np.where(np.isnan(X), -1.0, X), np.clip(dh, -1, 1)]).astype(np.float32)
        y = sub[f'Irrigation field {k}'].values.astype(int)
        n = len(y) - W + 1
        Xw = np.stack([X[i:i + W] for i in range(n)]); yw = y[W - 1:]
        cut = int(0.8 * n)
        Xtr.append(Xw[:cut - W]); ytr.append(yw[:cut - W]); ftr.append(np.full(cut - W, k))   # gap avoids overlap
        Xte.append(Xw[cut:]); yte.append(yw[cut:])
        stats.append(dict(field=k, samples=int(len(y)), windows=int(n), irr_rate=float(y.mean())))
    cache['d'] = dict(Xtr=np.concatenate(Xtr), ytr=np.concatenate(ytr), ftr=np.concatenate(ftr),
                      Xte=np.concatenate(Xte), yte=np.concatenate(yte), stats=stats)
    return cache['d']

class LSTMNet(nn.Module):
    def __init__(self, nf=8, h=32):
        super().__init__(); self.lstm = nn.LSTM(nf, h, batch_first=True); self.fc = nn.Linear(h, 2)
    def forward(self, x):
        o, _ = self.lstm(x); return self.fc(o[:, -1])

CW = torch.tensor([1.0, 3.0])                       # class weights (irrigation events are ~20 %)

def evaluate(model, w, X, y):
    T.set_vec(model, w)
    with torch.no_grad():
        pr = torch.cat([model(torch.from_numpy(X[i:i + 4096])).argmax(1) for i in range(0, len(X), 4096)]).numpy()
    acc = (pr == y).mean()
    f1s = []
    for c in (0, 1):
        tp = ((pr == c) & (y == c)).sum(); fp = ((pr == c) & (y != c)).sum(); fn = ((pr != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(acc), float(np.mean(f1s))

def node_blocks(d, N, seed):
    """Contiguous training blocks: nodes split across fields proportionally to training size."""
    rng = np.random.default_rng(seed + 31)
    idx_f = [np.flatnonzero(d['ftr'] == k) for k in range(1, 5)]
    sizes = np.array([len(x) for x in idx_f]); alloc = np.maximum(1, np.round(N * sizes / sizes.sum())).astype(int)
    alloc[-1] = N - alloc[:-1].sum()
    blocks = []
    for k, n_k in enumerate(alloc):
        parts = np.array_split(idx_f[k], n_k)
        blocks += parts
    order = rng.permutation(N)
    blocks = [blocks[i] for i in order]
    props = np.array([[1 - d['ytr'][b].mean(), d['ytr'][b].mean()] for b in blocks])
    return blocks, props

def run(method, seed, cfg, d):
    blocks, props = node_blocks(d, 300, seed)
    torch.manual_seed(seed); model = LSTMNet(); w = T.get_vec(model).clone()
    gen = torch.Generator().manual_seed(seed + 11)
    if method == 'Server-only':
        Xt, yt = d['Xtr'], d['ytr']
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr']); lossf = nn.CrossEntropyLoss(weight=CW)
        for ep in range(cfg['central_epochs']):
            perm = torch.randperm(len(yt), generator=gen)
            for b in range(0, len(yt), 256):
                j = perm[b:b + 256].numpy()
                opt.zero_grad(); lossf(model(torch.from_numpy(Xt[j])), torch.from_numpy(yt[j]).long()).backward(); opt.step()
        acc, f1 = evaluate(model, T.get_vec(model), d['Xte'], d['yte'])
        return dict(method=method, seed=seed, acc=acc, f1=f1)
    key = hashlib.md5(json.dumps(['waz', method, seed, cfg['R']]).encode()).hexdigest()
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simcache', key + '.json')
    if os.path.exists(cpath): sim = json.load(open(cpath))
    else:
        keep = S.P['recluster_every']; S.P['recluster_every'] = 1
        sim = S.simulate(method, 300, seed, record_ch=True, max_rounds=cfg['R'], stop_at_hmr=False,
                         net_kwargs=dict(label_props=props), gate_retry=True)
        S.P['recluster_every'] = keep
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        json.dump(dict(ch_log=sim['ch_log']), open(cpath, 'w'))
    rng = np.random.default_rng(seed + 5)
    lossf = nn.CrossEntropyLoss(weight=CW); deferred = 0
    for ep in sim['ch_log'][:cfg['R']]:
        if method.startswith('EECF') and not ep['gate']:
            deferred += 1; continue
        clients = rng.choice(ep['alive'], cfg['K'], replace=False) if method == 'FLAG' else ep['CH']
        deltas, ns = [], []
        for c in clients:
            b = blocks[c]
            T.set_vec(model, w.clone()); opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            Xc, yc = torch.from_numpy(d['Xtr'][b]), torch.from_numpy(d['ytr'][b]).long()
            for _ in range(cfg['E']):
                perm = torch.randperm(len(yc), generator=gen)
                for s_ in range(0, len(yc), cfg['B']):
                    j = perm[s_:s_ + cfg['B']]; opt.zero_grad(); lossf(model(Xc[j]), yc[j]).backward(); opt.step()
            deltas.append(T.get_vec(model) - w); ns.append(len(b))
        tot = sum(ns); w = w + sum(x * n / tot for x, n in zip(deltas, ns))
    acc, f1 = evaluate(model, w, d['Xte'], d['yte'])
    return dict(method=method, seed=seed, acc=acc, f1=f1, deferred=deferred, n_params=int(w.numel()))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0-9'); ap.add_argument('--out', required=True)
    ap.add_argument('--methods', nargs='+', default=['EECF', 'PSO', 'CAPSO', 'LEACH', 'FLAG', 'Server-only'])
    a = ap.parse_args()
    d = load_waz()
    print('stats', d['stats'], 'train', len(d['ytr']), 'test', len(d['yte']), flush=True)
    cfg = dict(R=100, K=10, E=2, B=64, lr=3e-3, central_epochs=5)
    for m in a.methods:
        for s in S.parse_seeds(a.seeds):
            t0 = time.time(); r = run(m, s, cfg, d); r['wall_s'] = time.time() - t0; r['cfg'] = cfg
            with open(a.out, 'a') as fo: fo.write(json.dumps(r) + '\n')
            print(m, s, round(r['acc'], 4), round(r['f1'], 4), round(r['wall_s'], 1), flush=True)
