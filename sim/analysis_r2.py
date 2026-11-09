"""analysis_r2.py -- statistics for the full radio accounting (data gathering + FL upload, download and
local training), the intra-cluster-cost baselines, the validation-based choice of theta_byz, search
convergence statistics and the connectivity term. Writes results/stats_r2.json."""
import json, os, glob, math, collections
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')

def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        for line in open(f):
            if line.strip(): rows.append(json.loads(line))
    return rows

def ci(x):
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    n = len(x)
    if n < 2: return [float(x.mean()) if n else float('nan'), 0.0, 0.0, n]
    m, sd = x.mean(), x.std(ddof=1)
    return [float(m), float(sd), float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)), n]

def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y)))

def holm(ps):
    ps = list(ps); idx = np.argsort(ps); m = len(ps); adj = [0.0] * m; run = 0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 5 or np.all(a == b): return 1.0
    return float(stats.wilcoxon(a, b).pvalue)

def ratio_ci(a, b, B=4000, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or b.mean() == 0: return [float('nan')] * 3
    est = 100 * (a.mean() / b.mean() - 1)
    idx = np.random.default_rng(seed).integers(0, len(a), (B, len(a)))
    bs = 100 * (a[idx].mean(1) / b[idx].mean(1) - 1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return [float(est), float(lo), float(hi)]

def bysd(rows):
    d = collections.defaultdict(dict)
    for r in rows:
        r['E_FL_J'] = r.get('FL_up_J', 0) + r.get('FL_dn_J', 0) + r.get('FL_tr_J', 0)
        d[r['method']][r['seed']] = r
    return d

def pair(d, a, b, key):
    s = sorted(set(d[a]) & set(d[b]))
    return [d[a][x][key] for x in s], [d[b][x][key] for x in s]

KEYS = ['IMR', 'HMR', 'E_round_mJ', 'E_total_mJ', 'E_FL_J', 'FL_up_J', 'FL_dn_J', 'FL_tr_J', 'FL_up_zig_J', 'd_CH_first', 'zig_CH_first', 'E_zig_mJ', 'E_lora_mJ']
S = {}

# ------------------------------------------------------------- Table 3: full accounting at N = 300
full = bysd(load('r2_n300_full.jsonl'))
S['full'] = {m: {k: ci([r.get(k) for r in d.values()]) for k in KEYS} for m, d in full.items()}
ref = 'EECF'
others = [m for m in ['EECF:upl', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'LEACHC', 'EWKM', 'FLAG'] if m in full]
S['tests_full'] = {}
for k in ['IMR', 'HMR', 'E_total_mJ', 'E_round_mJ']:
    ps, out = [], {}
    for o in others:
        a, b = pair(full, ref, o, k); p = wil(a, b); ps.append(p)
        out[o] = dict(p=p, A12=a12(a, b), n=len(a), rel=ratio_ci(a, b))
    for o, pa in zip(others, holm(ps)): out[o]['p_holm'] = pa
    S['tests_full'][k] = out
# the upload-aware merit (Eq. 7a) against every baseline
others_u = [m for m in ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'LEACHC', 'EWKM', 'FLAG'] if m in full]
S['tests_full_upl'] = {}
if 'EECF:upl' in full:
    for k in ['IMR', 'HMR', 'E_total_mJ', 'E_round_mJ']:
        ps, out = [], {}
        for o in others_u:
            a, b = pair(full, 'EECF:upl', o, k); p = wil(a, b); ps.append(p)
            out[o] = dict(p=p, A12=a12(a, b), n=len(a), rel=ratio_ci(a, b))
        for o, pa in zip(others_u, holm(ps)): out[o]['p_holm'] = pa
        S['tests_full_upl'][k] = out
# data gathering only for the two new baselines (same protocol as the submitted Table 3)
old = bysd(load('n300_eecf.jsonl') + load('n300_base.jsonl'))
dat = bysd(load('r2_n300_dataonly.jsonl'))
S['dataonly_new'] = {m: {k: ci([r.get(k) for r in d.values()]) for k in ['IMR', 'HMR', 'E_round_mJ', 'd_CH_first']} for m, d in dat.items()}
S['dataonly_vs_eecf'] = {}
for m in dat:
    comb = {'EECF': old['EECF'], m: dat[m]}
    S['dataonly_vs_eecf'][m] = {k: dict(rel=ratio_ci(*pair(comb, 'EECF', m, k)), p=wil(*pair(comb, 'EECF', m, k))) for k in ['IMR', 'HMR', 'E_round_mJ']}
# first-epoch head distance and upload energy in the data-only runs (submitted protocol)
S['first_epoch_dataonly'] = {m: dict(d_CH=ci([r.get('d_CH_first') for r in d.values()])) for m, d in old.items() if m in ('EECF', 'PSO')}

# ------------------------------------------------------------- Table 4: objective vs search, full accounting
dec = bysd(load('r2_decouple_full.jsonl'))
for m in ['EECF', 'PSO']:
    if m in full: dec[m] = full[m]
S['dec_full'] = {m: {k: ci([r.get(k) for r in d.values()]) for k in KEYS} for m, d in dec.items()}
S['dec_full_rel'] = {}
for m in dec:
    if m == 'EECF': continue
    S['dec_full_rel'][m] = {k: ratio_ci(*pair(dec, m, 'EECF', k)) for k in ['IMR', 'E_total_mJ', 'E_round_mJ']}
    S['dec_full_rel'][m]['p_IMR'] = wil(*pair(dec, 'EECF', m, 'IMR'))
    S['dec_full_rel'][m]['p_E'] = wil(*pair(dec, 'EECF', m, 'E_total_mJ'))
ms = [m for m in S['dec_full_rel']]
for key in ['p_IMR', 'p_E']:
    for m, pa in zip(ms, holm([S['dec_full_rel'][m][key] for m in ms])): S['dec_full_rel'][m][key + '_holm'] = pa
if 'PSO:Fcomp' in dec and 'PSO' in dec:
    S['merit_pso_full'] = {k: ratio_ci(*pair(dec, 'PSO:Fcomp', 'PSO', k)) for k in ['IMR', 'E_total_mJ', 'E_round_mJ']}

# ------------------------------------------------------------- Table 5: sensitivity (10 seeds), Holm over the table
def sub10(d):
    return {m: {s: r for s, r in v.items() if s < 10} for m, v in d.items()}
sens_rows = []
base10 = sub10(full)
sens_rows.append(('n_CH = 10 (default)', base10))
for nch in [5, 15, 20]:
    rows = [r for r in load('r2_nch_full.jsonl') if f'n_CH={nch}' in r.get('Pset', [])]
    sens_rows.append((f'n_CH = {nch}', bysd(rows)))
for eps, lab in [('0.00065e-12', 'eps_mp x 0.5'), ('0.0026e-12', 'eps_mp x 2')]:
    rows = [r for r in load('r2_radio_full.jsonl') if f'eps_mp={eps}' in r.get('Pset', [])]
    sens_rows.append((lab, bysd(rows)))
sens_rows.append(('FL every 100 rounds', bysd(load('r2_flevery_full.jsonl'))))
sens_rows.append(('top-k c = 0.25', bysd(load('r2_comp_full.jsonl'))))
sens_rows.append(('e_MAC = 10 nJ', bysd(load('r2_emac_full.jsonl'))))
S['sens'] = []; plist = []
for lab, d in sens_rows:
    row = dict(label=lab)
    for m in ['EECF', 'EECF:upl']:
        if m in d and 'PSO' in d and len(set(d[m]) & set(d['PSO'])) >= 5:
            for k in ['IMR', 'E_total_mJ', 'E_round_mJ']:
                a, b = pair(d, m, 'PSO', k)
                row[f'{m}|{k}'] = dict(rel=ratio_ci(a, b), p=wil(a, b), n=len(a), mean=float(np.mean(a)), mean_pso=float(np.mean(b)))
                if k != 'E_round_mJ': plist.append((len(S['sens']), f'{m}|{k}', row[f'{m}|{k}']['p']))
    S['sens'].append(row)
for col in set(k for _, k, _ in plist):                      # Holm within each column (method x metric) over the settings
    sub_ = [(i, k, p) for i, k, p in plist if k == col]
    for (i, key, _), pa in zip(sub_, holm([p for *_, p in sub_])): S['sens'][i][key]['p_holm'] = pa
# Holm-adjusted p-values of the submitted (data-gathering) sensitivity table
oldsens = []
nch_old = load('nch.jsonl'); radio_old = load('radio.jsonl')
def old_rows(rows, tag):
    return bysd([r for r in rows if tag in r.get('Pset', [])])
old10 = sub10(old)
cases = [('n_CH = 5', old_rows(nch_old, 'n_CH=5')), ('n_CH = 10 (default)', old10), ('n_CH = 15', old_rows(nch_old, 'n_CH=15')),
         ('n_CH = 20', old_rows(nch_old, 'n_CH=20')), ('eps_mp x 0.5', old_rows(radio_old, 'eps_mp=0.00065e-12')),
         ('eps_mp x 2', old_rows(radio_old, 'eps_mp=0.0026e-12'))]
pl = []
for lab, d in cases:
    row = dict(label=lab)
    for k in ['IMR', 'E_round_mJ']:
        if 'EECF' in d and 'PSO' in d:
            a, b = pair(d, 'EECF', 'PSO', k); row[k] = dict(rel=ratio_ci(a, b), p=wil(a, b), n=len(a)); pl.append((len(oldsens), k, row[k]['p']))
    oldsens.append(row)
for col in ['IMR', 'E_round_mJ']:
    sub_ = [(i, k, p) for i, k, p in pl if k == col]
    for (i, k, _), pa in zip(sub_, holm([p for *_, p in sub_])): oldsens[i][k]['p_holm'] = pa
S['sens_dataonly'] = oldsens
# normalised connectivity term
gn = bysd(load('r2_gcnorm_full.jsonl'))
if 'EECF' in gn:
    comb = {'g': gn['EECF'], 'EECF': base10['EECF'], 'PSO': base10.get('PSO', {})}
    S['gcnorm_full'] = {k: dict(vs_default=ratio_ci(*pair(comb, 'g', 'EECF', k)), vs_pso=ratio_ci(*pair(comb, 'g', 'PSO', k)),
                                p_default=wil(*pair(comb, 'g', 'EECF', k))) for k in ['IMR', 'HMR', 'E_total_mJ', 'E_round_mJ']}
gd = bysd(load('r2_gcnorm_dataonly.jsonl'))
if 'EECF' in gd:
    comb = {'g': gd['EECF'], 'EECF': old10['EECF'], 'PSO': old10['PSO']}
    S['gcnorm_dataonly'] = {k: dict(vs_default=ratio_ci(*pair(comb, 'g', 'EECF', k)), vs_pso=ratio_ci(*pair(comb, 'g', 'PSO', k)),
                                    p_default=wil(*pair(comb, 'g', 'EECF', k))) for k in ['IMR', 'HMR', 'E_round_mJ']}

# ------------------------------------------------------------- scaling, full accounting
sc = bysd(load('r2_scale_full.jsonl'))
S['scale_full'] = {}
byN = collections.defaultdict(lambda: collections.defaultdict(dict))
for r in load('r2_scale_full.jsonl'):
    r['E_FL_J'] = r['FL_up_J'] + r['FL_dn_J'] + r['FL_tr_J']; byN[r['N']][r['method']][r['seed']] = r
byN[300] = sub10(full)
pl = []
for N in sorted(byN):
    d = byN[N]; S['scale_full'][N] = {}
    for m in ['EECF', 'EECF:upl']:
        if m in d and 'PSO' in d:
            e = {}
            for k in ['IMR', 'HMR', 'E_total_mJ']:
                a, b = pair(d, m, 'PSO', k); e[k] = dict(rel=ratio_ci(a, b), p=wil(a, b)); pl.append((N, m, k, e[k]['p']))
            S['scale_full'][N][m] = e
    S['scale_full'][N]['means'] = {m: {k: ci([r.get(k) for r in d[m].values()]) for k in ['IMR', 'HMR', 'E_total_mJ']} for m in d}
for col in set((m, k) for _, m, k, _ in pl):                 # Holm over the five sizes for each method and metric
    sub_ = [x for x in pl if (x[1], x[2]) == col]
    for (N, m, k, _), pa in zip(sub_, holm([x[3] for x in sub_])): S['scale_full'][N][m][k]['p_holm'] = pa

# ------------------------------------------------------------- ablation, full accounting
ab = bysd(load('r2_abl_full.jsonl'))
S['abl_full'] = {}
for v in ab:
    comb = {v: ab[v], 'EECF': full['EECF']}
    S['abl_full'][v] = {k: ratio_ci(*pair(comb, v, 'EECF', k)) for k in ['E_total_mJ', 'E_round_mJ', 'IMR', 'HMR']}

# ------------------------------------------------------------- search convergence (submitted network runs)
it = {}
for m, d in old.items():
    vals, imp = [], []
    for r in d.values():
        c = r.get('curve')
        if not c: continue
        c = np.asarray(c); fin = c[-1]
        vals.append(int(np.argmax(c >= fin - 1e-12)) + 1)            # iteration (1..T) at which the final best is reached
        imp.append(100 * (fin - c[0]) / abs(c[0]))                     # improvement after the first iteration (%)
    if vals:
        v = np.asarray(vals)
        it[m] = dict(median=float(np.median(v)), mean=float(v.mean()), min=int(v.min()), max=int(v.max()),
                     q75=float(np.percentile(v, 75)), share_le10=float((v <= 10).mean()),
                     imp_median=float(np.median(imp)), imp_mean=float(np.mean(imp)), imp_max=float(np.max(imp)))
S['iters'] = it

# ------------------------------------------------------------- connectivity term (first epoch, N = 300)
import eecf_sim as SIM
degs, fcs, fns = [], [], []
for seed in range(30):
    net = SIM.Net(300, seed); dg = net.degree()
    degs.append(dg); fcs.append(1 - np.exp(-dg / 3.0)); fns.append(1 - np.exp(-dg / dg.mean()))
dg, fc, fn = np.concatenate(degs), np.concatenate(fcs), np.concatenate(fns)
S['conn'] = dict(deg_mean=float(dg.mean()), deg_sd=float(dg.std()), deg_p5=float(np.percentile(dg, 5)), deg_min=int(dg.min()),
                 share_fc_ge_099=float((fc >= 0.99).mean()), share_fc_ge_0999=float((fc >= 0.999).mean()),
                 share_fc_lt_095=float((fc < 0.95).mean()), share_deg_lt_14=float((dg < 14).mean()),
                 fn_p5=float(np.percentile(fn, 5)), fn_p50=float(np.percentile(fn, 50)), fn_p95=float(np.percentile(fn, 95)),
                 fc_p5=float(np.percentile(fc, 5)), fc_p50=float(np.percentile(fc, 50)))

# ------------------------------------------------------------- theta_byz chosen on a validation split
tv = load('r2_fl_theta_val.jsonl')
if tv:
    g = collections.defaultdict(dict)
    for r in tv: g[r['cfg']['theta']][r['seed']] = r
    th = sorted(g)
    S['theta_val'] = {str(t): dict(val=ci([r['val_acc'] * 100 for r in g[t].values()]), test=ci([r['acc'] * 100 for r in g[t].values()]),
                                   rej_h=ci([100 * r['rejected_honest'] / max(r['submitted'] - r['byz_updates'], 1) for r in g[t].values()]),
                                   rej_m=ci([100 * (r['rejected'] - r['rejected_honest']) / max(r['byz_updates'], 1) for r in g[t].values()]))
                      for t in th}
    seeds = sorted(set.intersection(*[set(g[t]) for t in th]))
    best_mean = max(th, key=lambda t: np.mean([g[t][s]['val_acc'] for s in seeds]))
    nested = []
    for s in seeds:
        tb = max(th, key=lambda t: g[t][s]['val_acc']); nested.append((s, tb, g[tb][s]['acc'] * 100, g[0.2][s]['acc'] * 100 if 0.2 in g else None))
    S['theta_sel'] = dict(best_mean=best_mean, test_at_best=ci([g[best_mean][s]['acc'] * 100 for s in seeds]),
                          nested=nested, nested_test=ci([x[2] for x in nested]),
                          test_at_020=ci([g[0.2][s]['acc'] * 100 for s in seeds]) if 0.2 in g else None,
                          val_at_020=ci([g[0.2][s]['val_acc'] * 100 for s in seeds]) if 0.2 in g else None,
                          n_seeds=len(seeds))

# ------------------------------------------------------------- FL accuracy of the added topologies
fm = load('r2_fl_main.jsonl'); old_main = load('fl_main.jsonl')
if fm:
    lab = collections.defaultdict(dict)
    for r in fm + old_main: lab[r['label']][r['seed']] = r
    S['fl_new'] = {}
    ref = lab['EECF-IoT']
    labs = [l for l in ['LEACH-C-FedAvg', 'EWKM-FedAvg', 'EECF-IoT (Eq. 7a)'] if l in lab]
    for l in labs:
        s_ = sorted(set(lab[l]) & set(ref))
        a = [ref[s]['acc'] * 100 for s in s_]; b = [lab[l][s]['acc'] * 100 for s in s_]
        S['fl_new'][l] = dict(acc=ci(b), f1=ci([lab[l][s]['f1'] * 100 for s in s_]), p_vs_eecf=wil(a, b), n=len(s_),
                              deferred=ci([100 * lab[l][s]['deferred'] / 100 for s in s_]),
                              fdiv=ci([lab[l][s].get('mean_fdiv_heads') for s in s_]))
wz = load('r2_waz.jsonl'); old_wz = load('waz_main.jsonl')
if wz:
    d = collections.defaultdict(dict)
    for r in wz + old_wz: d[r['method']][r['seed']] = r
    S['waz_new'] = {}
    for m in ['LEACHC', 'EWKM', 'EECF:upl']:
        if m in d:
            s_ = sorted(set(d[m]) & set(d['EECF']))
            S['waz_new'][m] = dict(acc=ci([d[m][s]['acc'] * 100 for s in s_]), f1=ci([d[m][s]['f1'] * 100 for s in s_]),
                                   p_f1_vs_eecf=wil([d['EECF'][s]['f1'] for s in s_], [d[m][s]['f1'] for s in s_]), n=len(s_))

# ------------------------------------------------------------- daily radio energy, full accounting (Section 7F)
if 'EECF' in full and 'FLAG' in full:
    S['daily_full'] = {m: ci([r['E_total_mJ'] * 288 / 1000 for r in full[m].values()]) for m in full}
    S['daily_full_rel'] = {m: ratio_ci(*pair(full, m, 'FLAG', 'E_total_mJ')) for m in ['EECF', 'EECF:upl'] if m in full}

# ------------------------------------------------------------- gas with deferral logging
gpath = os.path.join(HERE, '..', 'sc', 'gas_results.json')
if os.path.exists(gpath):
    G = json.load(open(gpath)); S['gas'] = {}
    for K, v in G.items():
        rr = [r for r in v['rounds'] if not r.get('deferred') and r['round'] >= 2]
        dfr = [r for r in v['rounds'] if r.get('deferred')]
        S['gas'][K] = dict(elect=float(np.mean([r['elect'][0] for r in rr])), submit=float(np.mean([r['submit'][0] for r in rr])),
                           verify=float(np.mean([r['verify'][0] for r in rr])), distribute=float(np.mean([r['distribute'][0] for r in rr])),
                           total=float(np.mean([r['total_gas'] for r in rr])), round1=float([r for r in v['rounds'] if r['round'] == 1][0]['total_gas']),
                           bytes=float(np.mean([r['total_bytes'] for r in rr])), deploy=v['deploy_gas'],
                           log_deferral=float(dfr[0]['logDeferral'][0]) if dfr else None,
                           deferred_round_total=float(dfr[0]['total_gas']) if dfr else None)

fp = os.path.join(R, 'r2_first_epoch_upload.json')
if os.path.exists(fp): S['first_epoch_upload'] = json.load(open(fp))
# Holm over all TON_IoT baselines including the added topologies (accuracy vs EECF-IoT)
if fm:
    raw = {}
    for l in ['PSO-FedAvg', 'CAPSO-FedAvg', 'GA-FedAvg', 'AOA-FedAvg', 'OOA-FedAvg', 'LEACH-FedAvg', 'FedProx (PSO)', 'FLAG (star)', 'LEACH-C-FedAvg', 'EWKM-FedAvg', 'EECF-IoT (Eq. 7a)']:
        if l in lab:
            s_ = sorted(set(lab[l]) & set(ref)); raw[l] = wil([ref[x]['acc'] for x in s_], [lab[l][x]['acc'] for x in s_])
    S['fl_holm_ext'] = dict(zip(raw, holm(list(raw.values()))))
json.dump(S, open(os.path.join(R, 'stats_r2.json'), 'w'), indent=1, default=float)
print('written stats_r2.json; keys', list(S))
