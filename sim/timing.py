"""timing.py -- wall-clock cost of each software component per FL round (20 repetitions)."""
import time, json, numpy as np, torch, sys, os
import eecf_sim as S, fl_ton as T, fl_waz as Wz
torch.set_num_threads(1)
REP = 20
def tm(fn, rep=REP):
    fn()                                   # warm-up call (allocations, JIT paths) excluded
    v = []
    for _ in range(rep):
        t0 = time.perf_counter(); fn(); v.append(time.perf_counter() - t0)
    return np.mean(v), np.std(v, ddof=1)
rows = []
def ms(m, s): return f'{1e3 * m:.1f} ± {1e3 * s:.1f} ms' if m < 1 else f'{m:.2f} ± {s:.2f} s'
# head election + gate
for N in [100, 300, 500]:
    net = S.Net(N, 0); rng = np.random.default_rng(1)
    def el():
        CH, _ = S.cluster_once(net, 'EECF', rng, N * S.P['E0'], np.array([], int))
    m, s = tm(el, 10); rows.append(('Head election (Algorithm 1)', f'N = {N}, n = 40, T = 150', ms(m, s)))
    CH, _ = S.cluster_once(net, 'EECF', rng, N * S.P['E0'], np.array([], int)); a = CH[net.D[:, CH].argmin(1)]
    m, s = tm(lambda: S.rcic(net, CH, a, rng), 10); rows.append(('Reliability gate (Eq. 16)', f'N = {N}, M = 200 samples', ms(m, s)))
# local training
d = T.load_ton(); model = T.make_model(d['n_feat']); w = T.get_vec(model).clone(); gen = torch.Generator().manual_seed(0)
idx = np.random.default_rng(0).choice(len(d['ytr']), 500, replace=False); cfg = dict(T.BASE)
m, s = tm(lambda: T.local_train(model, w, d['Xtr'][idx], d['ytr'][idx], cfg, gen))
rows.append(('Local training per head', 'TON_IoT MLP, 500 records, E = 2', ms(m, s)))
dw = Wz.load_waz(); lm = Wz.LSTMNet(); wl = T.get_vec(lm).clone(); bi = np.arange(207)
def lstm_train():
    T.set_vec(lm, wl.clone()); opt = torch.optim.Adam(lm.parameters(), lr=3e-3); lf = torch.nn.CrossEntropyLoss(weight=Wz.CW)
    X, y = torch.from_numpy(dw['Xtr'][bi]), torch.from_numpy(dw['ytr'][bi]).long()
    for _ in range(2):
        for b in range(0, len(y), 64): opt.zero_grad(); lf(lm(X[b:b + 64]), y[b:b + 64]).backward(); opt.step()
m, s = tm(lstm_train); rows.append(('Local training per head', 'Wazihub LSTM(32), 207 windows, E = 2', ms(m, s)))
# screening + aggregation
deltas = [torch.randn(w.numel()) for _ in range(10)]; sk = T.Sketch(w.numel())
def screen():
    sks = torch.stack([sk(x) for x in deltas]); ref = sks.median(0).values
    return [T.cos(sks[i], ref) >= 0.2 for i in range(10)]
m, s = tm(screen); rows.append(('Sketching and screening (Eq. 15)', 'K = 10, 64-dim sketches', ms(m, s)))
m, s = tm(lambda: sum(x * 0.1 for x in deltas)); rows.append(('FedAvg aggregation', f'K = 10, |w| = {w.numel():,}', ms(m, s)))
# contract round on py-evm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sc'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sc'))
import importlib.util
spec = importlib.util.spec_from_file_location('gb', 'gas_bench_timed.py')
if os.path.exists('gas_bench_timed.py'):
    gb = importlib.util.module_from_spec(spec); spec.loader.exec_module(gb)
    m, s = gb.timed_round(10, REP); rows.append(('Contract round on a Python EVM (py-evm)', 'elect + 10 × submit + verifyBatch + distribute', ms(m, s)))
json.dump(dict(rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'timing.json'), 'w'), indent=1)
for r in rows: print(r)
