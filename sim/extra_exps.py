"""extra_exps.py -- (i) temperature-decay sensitivity, (ii) adaptive vs static weights on a depleted
network, (iii) RCIC(ACR) curves of the initial topology of every method. 30 seeds, N=300."""
import json, sys, numpy as np, eecf_sim as S

out = {}
seeds = range(30)
# (i) rho sensitivity: merit of the selected heads after T=150 on a fresh network
res = {}
for rho in [0.90, 0.93, 0.95, 0.97, 0.99]:
    S.P['rho'] = rho; vals = []
    for s in seeds:
        net = S.Net(300, s); rng = np.random.default_rng(10_000 + s)
        F = S.fitness_eecf(net, S.P['alpha0'])
        Sx = S.run_eecf(net, F, rng); CH = Sx.decode()
        vals.append(float(F[CH].mean()))
    res[rho] = vals; print('rho', rho, np.mean(vals), flush=True)
S.P['rho'] = 0.97
out['rho'] = res
# (ii) adaptive vs static weights at 50 % residual network energy (heterogeneous depletion)
res = {'adaptive': [], 'static': [], 'adaptive_fe': [], 'static_fe': []}
for s in seeds:
    net = S.Net(300, s); r = np.random.default_rng(s + 99)
    net.E = net.E * r.uniform(0.2, 0.8, 300)
    E0 = 300 * S.P['E0']
    for mode in ['adaptive', 'static']:
        alpha = net.alpha_t(E0) if mode == 'adaptive' else 0.35
        F = S.fitness_eecf(net, alpha); rng = np.random.default_rng(10_000 + s)
        CH = S.run_eecf(net, F, rng).decode()
        Fa = S.fitness_eecf(net, net.alpha_t(E0))
        res[mode].append(float(Fa[CH].mean())); res[mode + '_fe'].append(float((net.E[CH] / S.P['E0']).mean()))
out['annealing'] = res; print('annealing', {k: np.mean(v) for k, v in res.items()}, flush=True)
# (iii) RCIC vs ACR for the initial topology
ACRS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
res = {}
for m in ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH']:
    curves = []
    for s in seeds:
        net = S.Net(300, s); rng = np.random.default_rng(10_000 + s)
        if m == 'LEACH':
            p = S.P['n_CH'] / 300; CH = np.flatnonzero(rng.random(300) < p)
            if CH.size == 0: CH = rng.choice(300, 1)
        else:
            CH, _ = S.cluster_once(net, m, rng, 300 * S.P['E0'], np.array([], int))
        a = CH[net.D[:, CH].argmin(1)]
        row = []
        for acr in ACRS:
            S.P['ACR'] = acr; row.append(S.rcic(net, CH, a, np.random.default_rng(s + 5)))
        S.P['ACR'] = 0.75; curves.append(row)
    res[m] = curves; print('rcic', m, np.round(np.mean(curves, 0), 3), flush=True)
out['rcic_acr'] = dict(ACR=ACRS, curves=res)
json.dump(out, open('results/extra.json', 'w'))
