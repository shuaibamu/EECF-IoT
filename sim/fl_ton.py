"""
fl_ton.py -- federated learning on TON-IoT driven by the clustering topology of eecf_sim.

During an FL campaign the heads are re-elected every round; the FL clients of a round are the cluster heads elected for the epoch (FLAG: K alive nodes drawn
uniformly, no clustering). Each sensor node holds a local non-IID shard whose label proportions
are the node's label distribution in the simulator, so the diversity term f_div of Eq. 6 is
computed on exactly the data the head trains on. EECF-IoT defers aggregation when the reliability
gate fails (Eq. 16); baselines aggregate unconditionally unless configured otherwise.

Usage: python fl_ton.py --exp main --seeds 0-9 --out results/fl_main.jsonl
"""
import argparse, json, time, math, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
import eecf_sim as S

torch.set_num_threads(1)
DATA = os.environ.get('TON_IOT_CSV', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'train_test_network.csv'))
CLASSES = ['normal', 'backdoor', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']

# ------------------------------------------------------------------ data
def load_ton(cache={}):
    if 'd' in cache: return cache['d']
    d = pd.read_csv(DATA, low_memory=False)
    y = d['type'].map({c: i for i, c in enumerate(CLASSES)}).values
    X = d.drop(columns=['src_ip', 'dst_ip', 'label', 'type'])
    num = {}
    for c in X.columns:
        v = pd.to_numeric(X[c], errors='coerce')
        if v.notna().mean() > 0.95 and X[c].astype(str).nunique() > 2 or v.notna().all():
            num[c] = v
    cat = [c for c in X.columns if c not in num]
    drop = [c for c in cat if X[c].astype(str).nunique() > 50]                    # free-text fields
    cols = []
    for c in X.columns:
        if c in drop: continue
        if c in num:
            v = num[c].astype(float)
            if (v.dropna() >= 0).all(): v = np.log1p(v)                         # heavy-tailed counts
            cols.append(v.fillna(-1).values)                                     # sentinel -1
        else:
            cols.append(X[c].astype(str).astype('category').cat.codes.astype(float).values)  # label encoding
    X = np.column_stack(cols).astype(np.float32)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y)); ntest = int(0.2 * len(y))
    te, tr = idx[:ntest], idx[ntest:]
    lo, hi = X[tr].min(0), X[tr].max(0)
    X = (X - lo) / (hi - lo + 1e-8)                                                  # Eq. 28 min-max
    cache['d'] = dict(Xtr=X[tr], ytr=y[tr], Xte=X[te], yte=y[te], n_feat=X.shape[1], dropped=drop)
    return cache['d']

def node_shards(p, ytr, n_per_node, seed):
    """Local shard of each node: multinomial class counts from its label distribution p_i."""
    rng = np.random.default_rng(seed + 991)
    pools = [np.flatnonzero(ytr == c) for c in range(len(CLASSES))]
    shards = []
    for i in range(p.shape[0]):
        cnt = rng.multinomial(n_per_node, p[i])
        idx = np.concatenate([rng.choice(pools[c], k, replace=True) for c, k in enumerate(cnt) if k > 0])
        shards.append(idx)
    return shards

# ------------------------------------------------------------------ model / training
def make_model(n_feat):
    return nn.Sequential(nn.Linear(n_feat, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, len(CLASSES)))

def get_vec(m): return torch.cat([p.data.view(-1) for p in m.parameters()])
def set_vec(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n

def local_train(model, w_glob, X, y, cfg, rng_t):
    set_vec(model, w_glob.clone())
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    lossf = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y).long()
    n = len(yt)
    for _ in range(cfg['E']):
        perm = torch.randperm(n, generator=rng_t)
        for b in range(0, n, cfg['B']):
            j = perm[b:b + cfg['B']]
            opt.zero_grad()
            loss = lossf(model(Xt[j]), yt[j])
            if cfg.get('mu', 0) > 0:                                   # FedProx proximal term
                loss = loss + cfg['mu'] / 2 * (get_vec_grad(model) - w_glob).pow(2).sum()
            loss.backward(); opt.step()
    return get_vec(model) - w_glob

def get_vec_grad(m): return torch.cat([p.view(-1) for p in m.parameters()])

def evaluate(model, w, X, y):
    set_vec(model, w)
    with torch.no_grad():
        pr = model(torch.from_numpy(X)).argmax(1).numpy()
    acc = (pr == y).mean()
    f1s = []
    for c in range(len(CLASSES)):
        tp = ((pr == c) & (y == c)).sum(); fp = ((pr == c) & (y != c)).sum(); fn = ((pr != c) & (y == c)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(acc), float(np.mean(f1s)), pr

# ------------------------------------------------------------------ privacy / compression / screening
def dp(delta, cfg, gen):
    if cfg['dp_sigma'] <= 0: return delta
    C = cfg['clip']
    delta = delta * min(1.0, C / (delta.norm().item() + 1e-12))                    # clip (sensitivity C)
    return delta + torch.randn(delta.shape, generator=gen) * cfg['dp_sigma'] * C   # Gaussian mechanism (Eq. 10)

def compress(delta, c):
    if c >= 1.0: return delta
    k = max(1, int(c * delta.numel()))
    idx = delta.abs().topk(k).indices
    out = torch.zeros_like(delta); out[idx] = delta[idx]
    return out

class Sketch:
    """64-dim Gaussian random projection quantised to int16 (the on-chain sketch of verifyBatch())."""
    def __init__(self, dim, seed=7, D=64):
        g = torch.Generator().manual_seed(seed)
        self.M = torch.randn(D, dim, generator=g) / math.sqrt(D)
    def __call__(self, v):
        s = self.M @ v
        sc = 30000 / (s.abs().max().item() + 1e-12)
        return torch.round(s * sc)
def cos(a, b): return float(a @ b / (a.norm() * b.norm() + 1e-12))

# ------------------------------------------------------------------ one FL run
def run(method, seed, cfg, data, log_cache=None):
    topo = cfg.get('topology', method)
    import os, hashlib
    key = hashlib.md5(json.dumps([topo, seed, cfg['R'], cfg.get('net_kwargs'), cfg['gate']] + ([True] if cfg.get('gate_any') else []), sort_keys=True).encode()).hexdigest()
    cpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'simcache', key + '.json')
    if os.path.exists(cpath):
        sim = json.load(open(cpath))
    else:
        keep_rc = S.P['recluster_every']; S.P['recluster_every'] = 1        # FL campaign: re-elect every FL round
        S.P['retry_all'] = bool(cfg.get('gate_any'))
        sim = S.simulate(topo, 300, seed, record_ch=True, max_rounds=cfg['R'], stop_at_hmr=False,
                         net_kwargs=cfg.get('net_kwargs'), gate=cfg['gate'], gate_retry=cfg['gate'])
        S.P['recluster_every'] = keep_rc; S.P['retry_all'] = False
        os.makedirs(os.path.dirname(cpath), exist_ok=True)
        json.dump(dict(ch_log=sim['ch_log'], gate_fails=sim['gate_fails'], retries=sim['retries'], deferred=sim['deferred']), open(cpath, 'w'))
    net = S.Net(300, seed, **(cfg.get('net_kwargs') or {}))
    shards = node_shards(net.p, data['ytr'], cfg['n_per_node'], seed)
    rng = np.random.default_rng(seed + 5)
    malicious = set(rng.choice(300, int(cfg['byz_frac'] * 300), replace=False).tolist()) if cfg['byz_frac'] > 0 else set()
    torch.manual_seed(seed)
    model = make_model(data['n_feat'])
    w = get_vec(model).clone()
    sk = Sketch(w.numel())
    gen = torch.Generator().manual_seed(seed + 11)
    prev_g = None
    hist = []; deferred = 0; rejected = 0; rej_honest = 0; submitted = 0; byz_in = 0
    use_gate = cfg['gate'] and (topo.startswith('EECF') or bool(cfg.get('gate_any')))
    for k, ep in enumerate(sim['ch_log'][:cfg['R']]):
        if use_gate and not ep['gate']:
            deferred += 1
            hist.append(hist[-1] if hist else (0.0, 0.0)); continue
        if topo == 'FLAG':
            alive = np.array(ep['alive']);
            clients = rng.choice(alive, min(cfg['K'], len(alive)), replace=False)
        else:
            clients = [c for c in ep['CH'] if c in set(ep['alive'])] or ep['CH']
        deltas, ns, ids = [], [], []
        for c in clients:
            idx = shards[c]
            d = local_train(model, w, data['Xtr'][idx], data['ytr'][idx], cfg, gen)
            if c in malicious and cfg.get('attack', 'signflip') == 'signflip': d = -cfg['byz_scale'] * d
            if c in malicious: byz_in += 1
            deltas.append(d); ns.append(len(idx)); ids.append(c)
        mal_i = [i for i, c in enumerate(ids) if c in malicious]
        if cfg.get('attack') == 'alie' and mal_i:                       # "a little is enough" (Baruch et al.)
            from statistics import NormalDist
            n_, m_ = len(deltas), len(mal_i); s_ = n_ // 2 + 1 - m_
            z = cfg.get('alie_z') or NormalDist().inv_cdf(min(max((n_ - s_) / n_, 0.51), 0.99))
            st = torch.stack(deltas); mu, sd = st.mean(0), st.std(0)
            for i in mal_i: deltas[i] = mu - z * sd
        deltas = [compress(dp(d, cfg, gen), cfg['c']) for d in deltas]
        submitted += len(deltas)
        keep = list(range(len(deltas)))
        if cfg['screen'] == 'cosine':
            sks = torch.stack([sk(dd) for dd in deltas])
            ref = sks.median(0).values                                          # coordinate-wise median
            keep = [i for i in keep if cos(sks[i], ref) >= cfg['theta']]
            for i in range(len(deltas)):
                if i not in keep:
                    rejected += 1; rej_honest += ids[i] not in malicious
        if cfg['screen'] in ('krum', 'median', 'trimmed'):                 # off-chain robust aggregators
            st = torch.stack(deltas); n_ = len(deltas); f_ = max(1, int(round(cfg['byz_frac'] * n_)))
            if cfg['screen'] == 'median': g = st.median(0).values
            elif cfg['screen'] == 'trimmed':
                b_ = min(f_, (n_ - 1) // 2); srt = st.sort(0).values; g = srt[b_:n_ - b_].mean(0)
            else:
                dd = torch.cdist(st, st) ** 2; kk = max(1, n_ - f_ - 2)
                sc_ = torch.stack([dd[i][torch.arange(n_) != i].sort().values[:kk].sum() for i in range(n_)])
                ch_ = int(sc_.argmin()); g = st[ch_]
                rejected += n_ - 1; rej_honest += sum(1 for i in range(n_) if i != ch_ and ids[i] not in malicious)
        elif cfg['screen'] == 'fltrust':
            ridx = cfg['root_idx']
            d0 = local_train(model, w, data['Xtr'][ridx], data['ytr'][ridx], cfg, gen)
            ts = [max(0.0, cos(d, d0)) for d in deltas]
            if sum(ts) == 0: g = torch.zeros_like(w)
            else: g = sum(t * d * (d0.norm() / (d.norm() + 1e-12)) for t, d in zip(ts, deltas)) / sum(ts)
        if cfg['screen'] not in ('krum', 'median', 'trimmed', 'fltrust'):
            if not keep:
                hist.append(hist[-1] if hist else (0.0, 0.0)); continue
            tot = sum(ns[i] for i in keep)
            g = sum(deltas[i] * ns[i] / tot for i in keep)                           # FedAvg (Eq. 2)
        w = w + g; prev_g = g
        if (k + 1) % cfg['eval_every'] == 0 or k + 1 == cfg['R']:
            acc, f1, _ = evaluate(model, w, data['Xte'], data['yte']); hist.append((acc, f1))
        else:
            hist.append(None)
    # fill non-evaluated
    last = (0.0, 0.0); filled = []
    for h in hist:
        if h is None: filled.append(last)
        else: filled.append(h); last = h
    acc, f1, pr = evaluate(model, w, data['Xte'], data['yte'])
    cm = np.zeros((10, 10), int)
    for t, p_ in zip(data['yte'], pr): cm[t, p_] += 1
    return dict(method=method, seed=seed, acc=acc, f1=f1, sim_gate_fails=sim['gate_fails'], sim_deferred=sim['deferred'], curve=[x[0] for x in filled], deferred=deferred,
                rejected=rejected, rejected_honest=rej_honest, submitted=submitted, byz_updates=byz_in,
                n_params=int(w.numel()), cm=cm.tolist() if cfg.get('keep_cm') else None,
                mean_fdiv_heads=float(np.mean([net.fdiv[c] for e in sim['ch_log'][:cfg['R']] for c in e['CH']])) if topo != 'FLAG' else None)

BASE = dict(R=100, K=10, E=2, B=64, lr=3e-3, n_per_node=500, dp_sigma=0.0, clip=1.0, c=1.0,
            screen='cosine', theta=0.20, byz_frac=0.0, byz_scale=3.0, gate=True, mu=0.0, eval_every=5)

def experiments(name):
    """Returns list of (label, method/topology, cfg overrides)."""
    if name == 'main':
        return [('EECF-IoT', 'EECF', {}), ('PSO-FedAvg', 'PSO', dict(screen='none')),
                ('CAPSO-FedAvg', 'CAPSO', dict(screen='none')), ('GA-FedAvg', 'GA', dict(screen='none')),
                ('AOA-FedAvg', 'AOA', dict(screen='none')), ('OOA-FedAvg', 'OOA', dict(screen='none')),
                ('LEACH-FedAvg', 'LEACH', dict(screen='none')), ('FedProx (PSO)', 'PSO', dict(screen='none', mu=0.01)),
                ('FLAG (star)', 'FLAG', dict(screen='none'))]
    if name == 'ablation':
        return [('w/o adaptive weights', 'EECF:static', {}), ('w/o ECP', 'EECF:noECP', {}),
                ('w/o BCS', 'EECF:noBCS', {}), ('w/o DQA', 'EECF:noDQA', {}),
                ('Single-objective', 'EECF:single', {}), ('w/o TMCRA gate', 'EECF', dict(gate=False))]
    if name == 'dpcomp':
        return [('DP off', 'EECF', dict(dp_sigma=0.0)), ('DP sigma=0.1', 'EECF', dict(dp_sigma=0.1)),
                ('DP sigma=0.5', 'EECF', dict(dp_sigma=0.5)), ('DP sigma=1.0', 'EECF', dict(dp_sigma=1.0)),
                ('c=0.5', 'EECF', dict(c=0.5)), ('c=0.25', 'EECF', dict(c=0.25)), ('c=0.1', 'EECF', dict(c=0.1))]
    if name == 'dpnoscreen':
        return [(f'DP sigma={sg} no screening', 'EECF', dict(dp_sigma=sg, screen='none')) for sg in [0.0, 0.1, 0.5, 1.0]]
    if name == 'byz':
        out = []
        for f in [0.1, 0.2, 0.3]:
            out += [(f'EECF cosine f={f}', 'EECF', dict(byz_frac=f)),
                    (f'EECF no screening f={f}', 'EECF', dict(byz_frac=f, screen='none')),
                    (f'EECF FLTrust f={f}', 'EECF', dict(byz_frac=f, screen='fltrust'))]
        return out
    if name == 'noniid':
        out = []
        for k, lab in [(5, 'mild'), (3, 'moderate'), (1, 'extreme')]:
            nk = dict(net_kwargs=dict(partition=k))
            out += [(f'EECF-IoT {lab}', 'EECF', nk), (f'PSO-FedAvg {lab}', 'PSO', dict(screen='none', **nk)),
                    (f'FedProx {lab}', 'PSO', dict(screen='none', mu=0.01, **nk))]
        return out
    if name == 'fair':
        return [('EECF-IoT, no gate, no screening', 'EECF', dict(gate=False, screen='none')),
                ('PSO-FedAvg + gate + screening', 'PSO', dict(screen='cosine', gate_any=True))]
    if name == 'ablation_clean':
        nc = dict(gate=False, screen='none')
        return [(f'{lab} [no gate/screen]', m, nc) for lab, m in [('Full', 'EECF'), ('w/o adaptive weights', 'EECF:static'), ('w/o ECP', 'EECF:noECP'),
                ('w/o BCS', 'EECF:noBCS'), ('w/o DQA', 'EECF:noDQA'), ('Single-objective', 'EECF:single')]]
    if name == 'robust':
        out = [(f'{agg} signflip f=0.2', 'EECF', dict(byz_frac=0.2, screen=agg)) for agg in ['krum', 'median', 'trimmed']]
        out += [(f'{agg} alie f=0.2', 'EECF', dict(byz_frac=0.2, screen=agg, attack='alie')) for agg in ['none', 'cosine', 'fltrust', 'krum', 'median', 'trimmed']]
        out += [(f'{agg} alie1.5 f=0.2', 'EECF', dict(byz_frac=0.2, screen=agg, attack='alie', alie_z=1.5)) for agg in ['none', 'cosine', 'fltrust', 'krum', 'median', 'trimmed']]
        return out
    if name == 'theta':
        return [(f'theta={t}', 'EECF', dict(theta=t, byz_frac=0.2)) for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]]
    raise ValueError(name)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True); ap.add_argument('--seeds', default='0-9')
    ap.add_argument('--out', required=True); ap.add_argument('--only', nargs='*')
    ap.add_argument('--set', nargs='*', default=[])
    a = ap.parse_args()
    data = load_ton()
    seeds = S.parse_seeds(a.seeds)
    for lab, m, ov in experiments(a.exp):
        if a.only and lab not in a.only: continue
        for s in seeds:
            cfg = dict(BASE); cfg.update(ov)
            for kv in a.set:
                k, v = kv.split('=')
                cfg[k] = (v.lower() in ('1', 'true')) if isinstance(BASE.get(k), bool) else (type(BASE[k])(v) if k in BASE else v)
            if cfg['screen'] == 'fltrust':
                cfg['root_idx'] = np.random.default_rng(123).choice(len(data['ytr']), 200, replace=False)
            t0 = time.time()
            res = run(m, s, cfg, data)
            res['label'] = lab; res['wall_s'] = time.time() - t0
            res['cfg'] = {k: v for k, v in cfg.items() if k not in ('root_idx',)}
            with open(a.out, 'a') as fo: fo.write(json.dumps(res) + '\n')
            print(lab, s, round(res['acc'], 4), round(res['f1'], 4), 'def', res['deferred'], 'rej', res['rejected'], res['rejected_honest'], round(res['wall_s'], 1), flush=True)
