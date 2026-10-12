"""analysis.py -- statistics (mean, SD, 95% CI, Wilcoxon signed-rank + Holm, Vargha-Delaney A12)
for all reported results. Writes results/stats.json and figs/*.png"""
import json, os, glob, math, collections
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
ORDER = ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'FLAG']
LABEL = dict(EECF='EECF-IoT', PSO='PSO', CAPSO='CAPSO', GA='GA', AOA='AOA', OOA='OOA', LEACH='LEACH', FLAG='FLAG (star)')
COL = {m: PAL[i] for i, m in enumerate(ORDER)}
MARK = dict(EECF='o', PSO='s', CAPSO='^', GA='v', AOA='D', OOA='P', LEACH='X', FLAG='*')
LS = dict(EECF='-', PSO='--', CAPSO='-.', GA=':', AOA='--', OOA='-.', LEACH=':', FLAG='--')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'savefig.dpi': 300,
                     'axes.titlesize': 9.5, 'axes.titleweight': 'bold'})

def load(pattern):
    rows = []
    for f in glob.glob(os.path.join(R, pattern)):
        for line in open(f):
            if line.strip(): rows.append(json.loads(line))
    return rows

def ci(x):
    x = np.asarray([v for v in x if v is not None], float)
    n = len(x)
    if n < 2: return float(x.mean()) if n else float('nan'), 0.0, 0.0, n
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return float(m), float(sd), float(h), n

def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).sum(); eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))

def holm(ps):
    idx = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def paired_tests(by_seed, ref, others, key):
    """Wilcoxon signed-rank (two-sided) of ref vs each other method, paired on seed; Holm-adjusted."""
    out = {}; ps = []
    for o in others:
        seeds = sorted(set(by_seed[ref]) & set(by_seed[o]))
        a = np.array([by_seed[ref][s][key] for s in seeds], float); b = np.array([by_seed[o][s][key] for s in seeds], float)
        ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
        if len(a) < 5 or np.all(a == b): p = 1.0
        else: p = float(stats.wilcoxon(a, b).pvalue)
        out[o] = dict(p=p, A12=a12(a, b), n=int(len(a)), diff_mean=float(np.mean(a - b)), ratio=float(np.mean(a) / np.mean(b)) if np.mean(b) else None)
        ps.append(p)
    for o, pa in zip(others, holm(ps)): out[o]['p_holm'] = pa
    return out

def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'

S = {}

# ---------------------------------------------------------------- network-level (N=300)
net = load('n300_*.jsonl')
def gate_freq(r, alive_min=0.9):
    tr = [x for x in (r.get('rcic_trace') or []) if x[1] >= alive_min]
    return (sum(x[2] < 0.9 for x in tr) / len(tr)) if tr else None
by = collections.defaultdict(dict)
for r in net:
    r['gate_freq'] = gate_freq(r)
    r['IMR'] = r['IMR'] if r['IMR'] is not None else float('nan')
    by[r['method']][r['seed']] = r
S['n300'] = {}
for m, d in by.items():
    rs = list(d.values())
    S['n300'][m] = {k: ci([x[k] for x in rs]) for k in ['IMR', 'HMR', 'E_round_mJ', 'E_lora_mJ', 'E_zig_mJ', 'CH_std_mJ', 'gate_freq', 'rcic_first', 'recluster_s', 'Fcomp_sel']}
    S['n300'][m]['sub'] = {k: ci([x['sub'][k] for x in rs if x.get('sub')]) for k in ['fe', 'fc', 'fd', 'fx']} if rs[0].get('sub') else None
base_m = [m for m in ORDER[1:] if m in by]
S['tests_n300'] = {k: paired_tests(by, 'EECF', [m for m in base_m if not (m == 'FLAG' and k in ('CH_std_mJ', 'gate_freq'))], k) for k in ['IMR', 'HMR', 'E_round_mJ', 'CH_std_mJ', 'gate_freq']} if 'EECF' in by else {}
# relative to PSO (paired per seed)
def ratio_ci(a, b, B=4000, seed=0):
    """Ratio of means in % (100*(mean a / mean b - 1)) with paired percentile-bootstrap 95% CI.
    Returned as (estimate, sd_boot, halfwidth) for compatibility with ci()."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
    if len(a) < 2 or b.mean() == 0: return (float('nan'), 0.0, 0.0, len(a))
    est = 100 * (a.mean() / b.mean() - 1)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(a), (B, len(a)))
    bs = 100 * (a[idx].mean(1) / b[idx].mean(1) - 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return (float(est), float(bs.std()), float(max(est - lo, hi - est)), int(len(a)), float(lo), float(hi))
def paired_pct(a_m, b_m, key, inv=False):
    seeds = sorted(set(by[a_m]) & set(by[b_m]))
    return ratio_ci([by[a_m][s][key] for s in seeds], [by[b_m][s][key] for s in seeds])
if 'EECF' in by:
    S['rel_vs'] = {o: {k: paired_pct('EECF', o, k) for k in ['IMR', 'HMR', 'E_round_mJ', 'E_lora_mJ', 'E_zig_mJ']} for o in base_m}
abl_net = {}
for v in ['EECF:static', 'EECF:noECP', 'EECF:noBCS', 'EECF:noDQA', 'EECF:single']:
    if v in by:
        abl_net[v] = {k: paired_pct(v, 'EECF', k) for k in ['E_round_mJ', 'IMR', 'HMR', 'CH_std_mJ']}
        abl_net[v]['gate_freq'] = ci([x['gate_freq'] for x in by[v].values()])
        abl_net[v]['CH_std_mJ_abs'] = ci([x['CH_std_mJ'] for x in by[v].values()])
S['abl_net'] = abl_net

S['daily'] = {m: ci([x['E_round_mJ'] * 288 / 1000 for x in d.values()]) for m, d in by.items()}   # J/day at 5-min sensing
S['recluster'] = {m: ci([x['recluster_s'] for x in d.values() if x.get('recluster_s')]) for m, d in by.items()}
# ---------------------------------------------------------------- scaling
sc = load('scale_*.jsonl') + [r for r in net if r['method'] in ORDER]
S['scale'] = collections.defaultdict(dict)
bys = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in sc:
    if r['seed'] > 9: continue
    bys[r['N']][r['method']][r['seed']] = r
for N, d in bys.items():
    for m, rr in d.items():
        rs = list(rr.values())
        S['scale'][N][m] = {k: ci([x[k] if x[k] is not None else float('nan') for x in rs]) for k in ['IMR', 'HMR', 'E_round_mJ']}
    if 'EECF' in d and 'PSO' in d:
        seeds = sorted(set(d['EECF']) & set(d['PSO']))
        S['scale'][N]['gain_vs_PSO'] = {k: ratio_ci([d['EECF'][s][k] if d['EECF'][s][k] is not None else np.nan for s in seeds], [d['PSO'][s][k] if d['PSO'][s][k] is not None else np.nan for s in seeds]) for k in ['IMR', 'HMR', 'E_round_mJ']}
S['scale'] = {str(k): v for k, v in S['scale'].items()}

# ---------------------------------------------------------------- FL (TON-IoT)
def fl_summary(rows):
    g = collections.defaultdict(list)
    for r in rows: g[r['label']].append(r)
    out = {}
    for lab, rs in g.items():
        out[lab] = dict(acc=ci([100 * x['acc'] for x in rs]), f1=ci([100 * x['f1'] for x in rs]),
                        deferred=ci([x.get('deferred', 0) for x in rs]), rej=ci([x.get('rejected', 0) for x in rs]),
                        rej_honest=ci([x.get('rejected_honest', 0) for x in rs]), byz=ci([x.get('byz_updates', 0) for x in rs]),
                        submitted=ci([x.get('submitted', 0) for x in rs]), n_params=rs[0].get('n_params'),
                        fdiv=ci([x.get('mean_fdiv_heads') for x in rs if x.get('mean_fdiv_heads') is not None]),
                        curve=np.mean([x['curve'] for x in rs], 0).tolist() if rs[0].get('curve') else None,
                        curve_h=(stats.t.ppf(.975, len(rs) - 1) * np.std([x['curve'] for x in rs], 0, ddof=1) / math.sqrt(len(rs))).tolist() if rs[0].get('curve') and len(rs) > 1 else None,
                        seeds={x['seed']: x for x in rs})
    return out
for name in ['main', 'ablation', 'dpcomp', 'byz', 'theta', 'noniid', 'dpnoscreen']:
    S['fl_' + name] = fl_summary(load(f'fl_{name}.jsonl'))
def fl_tests(summ, ref, others, key='acc'):
    by_ = {lab: {s: {key: v[key]} for s, v in summ[lab]['seeds'].items()} for lab in [ref] + others if lab in summ}
    oth = [o for o in others if o in by_]
    return paired_tests(by_, ref, oth, key) if ref in by_ and oth else {}
S['tests_fl_main'] = fl_tests(S['fl_main'], 'EECF-IoT', [l for l in S['fl_main'] if l != 'EECF-IoT'])
S['tests_fl_main_f1'] = fl_tests(S['fl_main'], 'EECF-IoT', [l for l in S['fl_main'] if l != 'EECF-IoT'], 'f1')

# ---------------------------------------------------------------- Wazihub
waz = load('waz_main.jsonl')
g = collections.defaultdict(list)
for r in waz: g[r['method']].append(r)
S['waz'] = {m: dict(acc=ci([100 * x['acc'] for x in rs]), f1=ci([100 * x['f1'] for x in rs]), seeds={x['seed']: x for x in rs}) for m, rs in g.items()}
if 'EECF' in g:
    byw = {m: {x['seed']: {'acc': x['acc'], 'f1': x['f1']} for x in rs} for m, rs in g.items()}
    S['tests_waz'] = paired_tests(byw, 'EECF', [m for m in byw if m != 'EECF'], 'f1')

# ---------------------------------------------------------------- gas
gas = json.load(open(os.path.join(HERE, '..', 'sc', 'gas_results.json')))
S['gas'] = {}
for K, v in gas.items():
    rr = v['rounds'][1:]          # steady state (round >= 2)
    S['gas'][K] = dict(deploy=v['deploy_gas'],
                       elect=float(np.mean([x['elect'][0] for x in rr])), submit=float(np.mean([x['submit'][0] for x in rr])),
                       verify=float(np.mean([x['verify'][0] for x in rr])), distribute=float(np.mean([x['distribute'][0] for x in rr])),
                       total=float(np.mean([x['total_gas'] for x in rr])), bytes=float(np.mean([x['total_bytes'] for x in rr])),
                       first_total=v['rounds'][0]['total_gas'], accepted=rr[0]['accepted'])

# ---------------------------------------------------------------- objective vs search, sensitivity, extended FL studies
def paired_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b
    if len(d) < 2: return None
    h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return dict(diff=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), p=float(stats.ttest_rel(a, b).pvalue) if d.std() > 0 else 1.0, n=int(len(d)))
# (1) objective vs search: same seeds as the N = 300 runs
dec = load('decouple.jsonl')
for r in dec:
    r['gate_freq'] = gate_freq(r); r['IMR'] = r['IMR'] if r['IMR'] is not None else float('nan'); by[r['method']][r['seed']] = r
DEC = ['Greedy', 'PSO:Fcomp', 'EECF:baseFit']
S['decouple'] = {}
for m in DEC + ['EECF', 'PSO']:
    if m not in by: continue
    rs = list(by[m].values())
    S['decouple'][m] = {k: ci([x[k] for x in rs]) for k in ['IMR', 'HMR', 'E_round_mJ', 'CH_std_mJ', 'gate_freq']}
    S['decouple'][m]['vs_EECF'] = {k: paired_pct(m, 'EECF', k) for k in ['IMR', 'HMR', 'E_round_mJ']} if m != 'EECF' else None
    S['decouple'][m]['vs_PSO'] = {k: paired_pct(m, 'PSO', k) for k in ['IMR', 'HMR', 'E_round_mJ']} if m != 'PSO' else None
if all(m in by for m in DEC):
    S['tests_decouple'] = {k: paired_tests(by, 'EECF', [m for m in DEC], k) for k in ['IMR', 'HMR', 'E_round_mJ']}
    S['tests_merit_pso'] = {k: paired_tests(by, 'PSO:Fcomp', ['PSO'], k) for k in ['IMR', 'HMR', 'E_round_mJ']}
# (2) n_CH and radio-parameter sensitivity (10 seeds; n_CH = 10 / nominal radio from the N = 300 runs, seeds 0-9)
def sens(rows, key):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows: d[key(r)][r['method']][r['seed']] = r
    for m in ['EECF', 'PSO']:
        for sd in range(10):
            if sd in by.get(m, {}): d['nominal'][m][sd] = by[m][sd]
    out = {}
    for k, dd in d.items():
        if 'EECF' in dd and 'PSO' in dd:
            seeds = sorted(set(dd['EECF']) & set(dd['PSO']))
            out[str(k)] = {'EECF': {q: ci([dd['EECF'][s][q] for s in seeds]) for q in ['IMR', 'E_round_mJ']},
                           'PSO': {q: ci([dd['PSO'][s][q] for s in seeds]) for q in ['IMR', 'E_round_mJ']},
                           'gain': {q: ratio_ci([dd['EECF'][s][q] for s in seeds], [dd['PSO'][s][q] for s in seeds]) for q in ['IMR', 'HMR', 'E_round_mJ']},
                           'p': {q: float(stats.wilcoxon([dd['EECF'][s][q] for s in seeds], [dd['PSO'][s][q] for s in seeds]).pvalue) for q in ['IMR', 'E_round_mJ']}, 'n': len(seeds)}
    return out
S['nch'] = sens(load('nch.jsonl'), lambda r: dict(kv.split('=') for kv in r.get('Pset', []))['n_CH'] if r.get('Pset') else 'nominal')
S['radio'] = sens(load('radio.jsonl'), lambda r: dict(kv.split('=') for kv in r.get('Pset', []))['eps_mp'] if r.get('Pset') else 'nominal')
# (3) Holm-corrected tests of the scaling gains
ps = {}
for N in ['100', '200', '400', '500']:
    d = bys.get(int(N), {})
    if 'EECF' in d and 'PSO' in d:
        sd_ = sorted(set(d['EECF']) & set(d['PSO']))
        for q in ['IMR', 'E_round_mJ']:
            a_ = [d['EECF'][s][q] for s in sd_]; b_ = [d['PSO'][s][q] for s in sd_]
            ps[(N, q)] = float(stats.wilcoxon(a_, b_).pvalue)
hp = holm(list(ps.values())) if ps else []
S['tests_scale'] = {f'{N}|{q}': dict(p=p_, p_holm=float(h_)) for ((N, q), p_), h_ in zip(ps.items(), hp)}
# (4) pipeline swap, robust aggregation, ablation without gate/screening
for name in ['fair', 'robust', 'ablation_clean']:
    if os.path.exists(os.path.join(R, f'fl_{name}.jsonl')): S['fl_' + name] = fl_summary(load(f'fl_{name}.jsonl'))
def seedvals(summ, lab, key='acc'):
    return {s: 100 * v[key] for s, v in summ[lab]['seeds'].items()} if lab in summ else {}
def ptest(sa, sb):
    k = sorted(set(sa) & set(sb)); return paired_t([sa[i] for i in k], [sb[i] for i in k]) if len(k) > 1 else None
FM = S['fl_main']; FF = S.get('fl_fair', {}); FB = S.get('fl_byz', {}); FR = S.get('fl_robust', {}); FA = S.get('fl_ablation', {}); FC = S.get('fl_ablation_clean', {})
S['tests_fair'] = {}
if 'EECF-IoT, no gate, no screening' in FF:
    for o in ['PSO-FedAvg', 'CAPSO-FedAvg', 'GA-FedAvg', 'AOA-FedAvg', 'OOA-FedAvg', 'FedProx (PSO)', 'FLAG (star)']:
        S['tests_fair'][o] = ptest(seedvals(FF, 'EECF-IoT, no gate, no screening'), seedvals(FM, o))
    S['tests_fair']['EECF-IoT (full)'] = ptest(seedvals(FF, 'EECF-IoT, no gate, no screening'), seedvals(FM, 'EECF-IoT'))
if 'PSO-FedAvg + gate + screening' in FF:
    S['tests_fair']['PSO gate+screen vs PSO'] = ptest(seedvals(FF, 'PSO-FedAvg + gate + screening'), seedvals(FM, 'PSO-FedAvg'))
    S['tests_fair']['PSO gate+screen vs EECF full'] = ptest(seedvals(FF, 'PSO-FedAvg + gate + screening'), seedvals(FM, 'EECF-IoT'))
if 'EECF-IoT, no gate, no screening' in FF:
    _b = {'EECF-clean': {s_: {'acc': v['acc']} for s_, v in FF['EECF-IoT, no gate, no screening']['seeds'].items()}}
    for o in ['PSO-FedAvg', 'CAPSO-FedAvg', 'GA-FedAvg', 'AOA-FedAvg', 'OOA-FedAvg', 'FedProx (PSO)', 'FLAG (star)', 'LEACH-FedAvg']:
        _b[o] = {s_: {'acc': v['acc']} for s_, v in FM[o]['seeds'].items()}
    S['tests_fair_w'] = paired_tests(_b, 'EECF-clean', [o for o in _b if o != 'EECF-clean'], 'acc')
S['tests_ablation'] = {lab: ptest(seedvals(FA, lab), seedvals(FM, 'EECF-IoT')) for lab in FA}
S['tests_ablation_clean'] = {lab: ptest(seedvals(FC, lab), seedvals(FC, 'Full [no gate/screen]')) for lab in FC if lab != 'Full [no gate/screen]'}
S['tests_byz'] = {}
for f_ in ['0.1', '0.2', '0.3']:
    S['tests_byz'][f_] = dict(cos_vs_none=ptest(seedvals(FB, f'EECF cosine f={f_}'), seedvals(FB, f'EECF no screening f={f_}')),
                             cos_vs_fltrust=ptest(seedvals(FB, f'EECF cosine f={f_}'), seedvals(FB, f'EECF FLTrust f={f_}')))
S['tests_robust'] = {lab: ptest(seedvals(FR, lab), seedvals(FB, 'EECF cosine f=0.2') if 'signflip' in lab else seedvals(FR, 'cosine alie f=0.2')) for lab in FR}
S['tests_dp'] = {k: ptest(seedvals(S['fl_dpcomp'], k), seedvals(S['fl_dpcomp'], 'DP off')) for k in S['fl_dpcomp'] if k != 'DP off'}
NI_ = S['fl_noniid']
S['tests_noniid'] = {r: ptest(seedvals(NI_, f'EECF-IoT {r}'), seedvals(NI_, f'PSO-FedAvg {r}')) for r in ['mild', 'moderate', 'extreme']}
for extra in ['placement', 'dp_accounting']:
    pth = os.path.join(R, extra + '.json')
    if os.path.exists(pth): S[extra] = json.load(open(pth))

json.dump(S, open(os.path.join(R, 'stats.json'), 'w'), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
print('stats written')
