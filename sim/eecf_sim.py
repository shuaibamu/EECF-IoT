"""
eecf_sim.py -- EECF-IoT network-level simulator (clustering, energy, lifetime, reliability).

Implements Eqs. 1-8a (energy model and composite merit), Eqs. 17-19 (three search phases),
the RCIC(ACR) reliability gate (Eq. 16, Monte Carlo estimator) and the baselines
PSO, CAPSO, GA, AOA, OOA, LEACH and FLAG-style direct (star) transmission.

All search methods share the same encoding (agent = point in the deployment plane, snapped
to the nearest alive node), the same population/iteration budget (n=40, T=150), the same
archive-based decoding with minimum inter-head separation, and the same re-clustering period.

Usage:  python eecf_sim.py --methods EECF PSO --N 300 --seeds 0-29 --out results/n300.jsonl
"""
import argparse, json, math, time, sys
import numpy as np

# ---------------------------------------------------------------- parameters
P = dict(
    W=500.0, H=500.0, BS=(250.0, 250.0),
    E0=2.0,                 # J per node
    l=4000,                 # bits per data packet
    E_elec=50e-9,           # J/bit
    eps_fs=10e-12,          # J/bit/m^2  (alpha=2, LoRaWAN-class link)
    eps_mp=0.0013e-12,      # J/bit/m^4  (alpha=4, Zigbee-class link)
    E_agg=5e-9,             # J/bit/signal
    frac_lora=0.5,          # share of LoRaWAN-class nodes
    R_comm=80.0,            # m, one-hop radio range (degree, relay)
    r_sense=40.0,           # m, sensing radius for coverage
    n_CH=10, n_agents=40, T=150,
    recluster_every=20,     # rounds
    d_sep=60.0,             # m, minimum inter-head separation at decoding
    # composite merit (Eq. 8/8a): beta,gamma,delta are the values at alpha=0.35;
    # they are rescaled by (1-alpha(t))/0.65 so that alpha+beta+gamma+delta=1 always
    alpha0=0.25, alpha_max=0.55, beta=0.30, gamma=0.20, delta=0.15,
    gamma_c=3.0,
    tau0=1.0, rho=0.97, sigma=0.05, beta_r=0.40, xi=0.50, dqa_step=0.3,
    n_labels=10, dir_alpha=0.5, gini_tol=0.15,
    # reliability gate
    P_on=0.95, ACR=0.75, rho_min=0.90, mc_trials=200, grid=25,
    max_rounds=6000,
    rotate=True,            # previous-epoch heads are ineligible (all metaheuristics)
)

def radio_tx(l, d, alpha_is4):
    """Eq. 1: technology-specific transmit energy."""
    amp = np.where(alpha_is4, P['eps_mp'] * d**4, P['eps_fs'] * d**2)
    return P['E_elec'] * l + amp * l

def radio_rx(l):
    return P['E_elec'] * l   # Eq. 2

# ---------------------------------------------------------------- network
class Net:
    def __init__(self, N, seed, partition=None, label_props=None):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.N = N
        self.xy = np.column_stack([r.uniform(0, P['W'], N), r.uniform(0, P['H'], N)])
        self.is4 = r.random(N) >= P['frac_lora']           # True -> Zigbee-class (alpha=4)
        self.E = np.full(N, P['E0'])
        self.alive = np.ones(N, bool)
        self.bs = np.array(P['BS'])
        self.dbs = np.linalg.norm(self.xy - self.bs, axis=1)
        D = np.linalg.norm(self.xy[:, None, :] - self.xy[None, :, :], axis=2)
        self.D = D
        self.dmax = D.max()
        # local label distributions (Dirichlet non-IID) -> f_div (Eq. 6) with Gini cross-check
        L = P['n_labels']
        p = r.dirichlet([P['dir_alpha']] * L, N)
        if partition is not None:                    # k-label partition: each node holds k labels
            pr = np.random.default_rng(seed + 777)
            p = np.zeros((N, L))
            for i in range(N):
                labs = pr.choice(L, partition, replace=False); p[i, labs] = 1.0 / partition
            p = np.clip(p, 1e-9, None); p /= p.sum(1, keepdims=True)
        if label_props is not None:
            p = np.clip(label_props, 1e-9, None); p = p / p.sum(1, keepdims=True); L = p.shape[1]
        self.p = p
        H = -(p * np.log(p + 1e-12)).sum(1) / math.log(L)
        G = (1 - (p**2).sum(1)) / (1 - 1 / L)
        self.fdiv = np.where(np.abs(H - G) > P['gini_tol'], 0.0, H)   # rejected anchors score 0
        self.fdiv_raw = H
        # coverage grid
        g = P['grid']
        gx, gy = np.meshgrid((np.arange(g) + .5) * P['W'] / g, (np.arange(g) + .5) * P['H'] / g)
        self.grid = np.column_stack([gx.ravel(), gy.ravel()])
        self.cov = (np.linalg.norm(self.grid[:, None, :] - self.xy[None, :, :], axis=2) <= P['r_sense'])

    # ------------------------------------------------ sub-objectives (Eqs. 4-7)
    def degree(self):
        A = (self.D <= P['R_comm']) & self.alive[None, :] & self.alive[:, None]
        np.fill_diagonal(A, False)
        return A.sum(1)

    def subobjectives(self):
        fe = self.E / P['E0']
        fc = 1 - np.exp(-self.degree() / P['gamma_c'])
        fd = self.fdiv
        # Eq. 7 (technology-aware): CH-to-sink transmit energy normalised over alive nodes
        etx = radio_tx(P['l'], self.dbs, self.is4)
        fx = etx / etx[self.alive].max()
        return fe, fc, fd, fx

    def alpha_t(self, E_net0):
        return P['alpha0'] + (P['alpha_max'] - P['alpha0']) * (1 - self.E[self.alive].sum() / E_net0)

# ---------------------------------------------------------------- fitness definitions
def weights(alpha):
    s = (1 - alpha) / (P['beta'] + P['gamma'] + P['delta'])
    return alpha, P['beta'] * s, P['gamma'] * s, P['delta'] * s

def fitness_eecf(net, alpha, use_div=True):
    fe, fc, fd, fx = net.subobjectives()
    a, b, g, d = weights(alpha)
    if not use_div:
        g = 0.0
    F = a * fe + b * fc + g * fd - d * fx           # Eq. 8
    return np.where(net.alive, F, -np.inf)

def fitness_baseline(net):
    """Energy + distance-to-sink fitness used by the metaheuristic baselines
    (residual-energy ratio and normalised sink distance, equal weights)."""
    fe = net.E / P['E0']
    fx = net.dbs / net.dmax
    F = 0.5 * fe + 0.5 * (1 - fx)
    return np.where(net.alive, F, -np.inf)

def fitness_energy_only(net):
    fe = net.E / P['E0']
    return np.where(net.alive, fe, -np.inf)

# ---------------------------------------------------------------- search engine
class Search:
    """Common machinery: agents live in [0,1]^2 (normalised plane), snap to nearest alive node."""
    def __init__(self, net, F, rng):
        self.net, self.F, self.rng = net, F, rng
        self.idx_alive = np.flatnonzero(net.alive)
        self.xy_alive = net.xy[self.idx_alive] / np.array([P['W'], P['H']])
        self.archive = {}
        self.curve = []

    def snap(self, X):
        d = ((X[:, None, :] - self.xy_alive[None, :, :])**2).sum(2)
        return self.idx_alive[d.argmin(1)]

    def evaluate(self, X):
        nodes = self.snap(X)
        f = self.F[nodes]
        for n, v in zip(nodes, f):
            self.archive[int(n)] = float(v)
        return f, nodes

    def decode(self):
        cand = sorted(self.archive.items(), key=lambda kv: -kv[1])
        chosen = []
        for n, v in cand:
            if not np.isfinite(v):
                continue
            if all(self.net.D[n, c] >= P['d_sep'] for c in chosen):
                chosen.append(n)
            if len(chosen) == P['n_CH']:
                break
        if len(chosen) < P['n_CH']:                 # fill from all alive nodes, same rule
            for n in np.argsort(-self.F):
                if not self.net.alive[n] or n in chosen or not np.isfinite(self.F[n]):
                    continue
                if all(self.net.D[n, c] >= P['d_sep'] for c in chosen):
                    chosen.append(int(n))
                if len(chosen) == P['n_CH']:
                    break
        return np.array(chosen, int)

def init_pop(rng):
    return rng.random((P['n_agents'], 2))

def run_eecf(net, F, rng, phases=(True, True, True)):
    """Algorithm 1: rotating ECP / BCS / DQA phases (Eqs. 17-19).
    Targets are evaluated in each agent's one-hop neighbourhood N_R(X_i)."""
    S = Search(net, F, rng)
    X = init_pop(rng)
    f, nodes = S.evaluate(X)
    best = f.max()
    fe, fc, fd, fx = net.subobjectives()
    scale = np.array([P['W'], P['H']])
    xyN = net.xy / scale
    alive = net.alive
    q75 = np.quantile(fe[alive], 0.75)
    hiE = alive & (fe >= q75)
    A = (net.D <= P['R_comm'])
    active = [i for i in range(3) if phases[i]]
    for t in range(1, P['T'] + 1):
        if active:
            stage = active[(t - 1) % len(active)]
            tau = P['tau0'] * P['rho']**t
            if stage == 0:                                     # ECP, Eq. 17
                tgt = np.empty_like(X)
                for k, n in enumerate(nodes):
                    m = A[n] & hiE
                    tgt[k] = xyN[m].mean(0) if m.any() else X[k]
                X = X + tau * (tgt - X) + P['sigma'] * rng.standard_normal(X.shape)
            elif stage == 1:                                   # BCS, Eq. 18
                tgt = np.empty_like(X)
                for k, n in enumerate(nodes):
                    m = A[n] & alive
                    cand = np.flatnonzero(m)
                    tgt[k] = xyN[cand[fc[cand].argmax()]] if cand.size else X[k]
                X = X + P['beta_r'] * (tgt - X)
            else:                                              # DQA, Eq. 19
                tgt = np.empty_like(X)
                for k, n in enumerate(nodes):
                    m = A[n] & alive
                    cand = np.flatnonzero(m)
                    tgt[k] = xyN[cand[fd[cand].argmax()]] if cand.size else X[k]
                X = X + P['dqa_step'] * (tgt - X) + P['xi'] * tau * 0.1 * rng.standard_normal(X.shape)
        else:                                                  # no guidance: random walk
            X = X + P['sigma'] * rng.standard_normal(X.shape)
        X = np.clip(X, 0, 1)
        f, nodes = S.evaluate(X)
        best = max(best, f.max())
        S.curve.append(best)
    return S

def run_pso(net, F, rng):
    S = Search(net, F, rng)
    X = init_pop(rng); V = 0.1 * (rng.random(X.shape) - .5)
    f, _ = S.evaluate(X); pb, pf = X.copy(), f.copy(); g = pb[pf.argmax()].copy(); gf = pf.max()
    for t in range(1, P['T'] + 1):
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = 0.729 * V + 1.49445 * r1 * (pb - X) + 1.49445 * r2 * (g - X)
        V = np.clip(V, -0.1, 0.1); X = np.clip(X + V, 0, 1)
        f, _ = S.evaluate(X)
        m = f > pf; pb[m], pf[m] = X[m], f[m]
        if pf.max() > gf: gf = pf.max(); g = pb[pf.argmax()].copy()
        S.curve.append(gf)
    return S

def run_capso(net, F, rng):
    """Chaos-adaptive PSO (adaptive inertia/acceleration + logistic-map chaotic search around gbest)."""
    S = Search(net, F, rng)
    X = init_pop(rng); V = 0.1 * (rng.random(X.shape) - .5)
    f, _ = S.evaluate(X); pb, pf = X.copy(), f.copy(); g = pb[pf.argmax()].copy(); gf = pf.max()
    z = rng.uniform(0.1, 0.9)
    for t in range(1, P['T'] + 1):
        z = 4 * z * (1 - z)
        w = 0.4 + 0.5 * (1 - t / P['T']) * (0.5 + 0.5 * z)
        c1 = 2.5 - 2.0 * t / P['T']; c2 = 0.5 + 2.0 * t / P['T']
        r1, r2 = rng.random(X.shape), rng.random(X.shape)
        V = w * V + c1 * r1 * (pb - X) + c2 * r2 * (g - X)
        V = np.clip(V, -0.1, 0.1); X = np.clip(X + V, 0, 1)
        f, _ = S.evaluate(X)
        m = f > pf; pb[m], pf[m] = X[m], f[m]
        # chaotic local search around gbest, range shrinking with t
        zz = rng.random((5, 2)); zz = 4 * zz * (1 - zz)
        C = np.clip(g + (0.2 * (1 - t / P['T'])) * (2 * zz - 1), 0, 1)
        fc_, _ = S.evaluate(C)
        if pf.max() > gf: gf = pf.max(); g = pb[pf.argmax()].copy()
        if fc_.max() > gf: gf = fc_.max(); g = C[fc_.argmax()].copy()
        S.curve.append(gf)
    return S

def run_ga(net, F, rng):
    S = Search(net, F, rng)
    X = init_pop(rng); f, _ = S.evaluate(X); gf = f.max()
    n = P['n_agents']
    for t in range(1, P['T'] + 1):
        order = np.argsort(-f); elite = X[order[:2]].copy()
        def tour():
            a, b = rng.integers(0, n, 2); return X[a] if f[a] > f[b] else X[b]
        kids = []
        while len(kids) < n - 2:
            p1, p2 = tour(), tour(); lam = rng.random()
            c = lam * p1 + (1 - lam) * p2
            if rng.random() < 0.2: c = c + 0.05 * rng.standard_normal(2)
            kids.append(c)
        X = np.clip(np.vstack([elite, np.array(kids)]), 0, 1)
        f, _ = S.evaluate(X); gf = max(gf, f.max()); S.curve.append(gf)
    return S

def levy(rng, D, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
    return rng.standard_normal(D) * sig / np.abs(rng.standard_normal(D))**(1 / beta)

def run_aoa(net, F, rng):
    """Aquila Optimizer (four hunting phases; synchronous population update, greedy acceptance)."""
    S = Search(net, F, rng)
    n = P['n_agents']; T = P['T']
    X = init_pop(rng); f, _ = S.evaluate(X); b = X[f.argmax()].copy(); bf = f.max()
    for t in range(1, T + 1):
        XM = X.mean(0); r = rng.random(n)
        Y = np.empty_like(X)
        L = np.array([levy(rng, 2) for _ in range(n)]) * 0.01
        R_ = X[rng.integers(0, n, n)]
        if t <= 2 * T / 3:
            y1 = b * (1 - t / T) + (XM - b * rng.random((n, 1)))
            y2 = b * L + R_ + (rng.random((n, 1)) - .5) * 0.1
            Y = np.where((r < .5)[:, None], y1, y2)
        else:
            y3 = (b - XM) * 0.1 - rng.random((n, 1)) * 0.1 + rng.random((n, 2)) * 0.1 + 0.5 * rng.random((n, 2))
            QF = t ** ((2 * rng.random((n, 1)) - 1) / (1 - T) ** 2)
            G1 = 2 * rng.random((n, 1)) - 1
            y4 = QF * b - G1 * X * rng.random((n, 1)) - 2 * (1 - t / T) * L + rng.random((n, 1)) * G1
            Y = np.where((r < .5)[:, None], y3, y4)
        Y = np.clip(Y, 0, 1)
        fy, _ = S.evaluate(Y)
        m = fy > f; X[m], f[m] = Y[m], fy[m]
        if f.max() > bf: bf = f.max(); b = X[f.argmax()].copy()
        S.curve.append(bf)
    return S

def run_ooa(net, F, rng):
    """Osprey Optimization Algorithm (fish identification + carrying phases, greedy acceptance)."""
    S = Search(net, F, rng)
    n = P['n_agents']; T = P['T']
    X = init_pop(rng); f, _ = S.evaluate(X); bf = f.max()
    for t in range(1, T + 1):
        fish = np.empty_like(X)
        for i in range(n):
            better = np.flatnonzero(f > f[i])
            fish[i] = X[rng.choice(better)] if better.size else X[f.argmax()]
        I = rng.integers(1, 3, (n, 1))
        Y = np.clip(X + rng.random((n, 2)) * (fish - I * X), 0, 1)
        fy, _ = S.evaluate(Y); m = fy > f; X[m], f[m] = Y[m], fy[m]
        Y = np.clip(X + rng.random((n, 2)) / t * 0.1, 0, 1)
        fy, _ = S.evaluate(Y); m = fy > f; X[m], f[m] = Y[m], fy[m]
        bf = max(bf, f.max()); S.curve.append(bf)
    return S

# ---------------------------------------------------------------- reliability (Eq. 16, Monte Carlo)
def rcic(net, CH, assign, rng, trials=None):
    """P[ covered-and-connected fraction of grid >= ACR ] under node on/off states (P_on).
    A node contributes coverage if it is alive, on, and connected to an on cluster head either
    directly (d <= R_comm) or via one on relay within range of both; cluster heads reach the
    edge server over the long-range uplink."""
    trials = trials or P['mc_trials']
    N = net.N
    alive = net.alive
    if len(CH) == 0:
        return 0.0
    ch_of = assign
    direct = np.zeros(N, bool); relay_need = np.zeros(N, bool)
    dch = net.D[np.arange(N), ch_of]
    direct = alive & (dch <= P['R_comm'])
    Rm = (net.D <= P['R_comm']) & alive[None, :]
    np.fill_diagonal(Rm, False)
    # relay j valid for i if j within R of i and j within R of CH(i)
    relayOK = Rm & (net.D[ch_of][:, :] <= P['R_comm'])     # (i, j): D[CH(i), j] <= R
    on = rng.random((trials, N)) < P['P_on']
    on &= alive[None, :]
    is_ch = np.zeros(N, bool); is_ch[CH] = True
    ch_on = on[:, ch_of]                                   # (trials, N)
    via_relay = (on.astype(np.float32) @ relayOK.T.astype(np.float32)) > 0
    conn = on & ch_on & (direct[None, :] | via_relay)
    conn[:, is_ch] = on[:, is_ch]
    covered = (conn.astype(np.float32) @ net.cov.T.astype(np.float32)) > 0
    frac = covered.mean(1)
    return float((frac >= P['ACR']).mean())


# ---------------------------------------------------------------- routing / per-round energy
def build_routes(net, CH, method):
    """Next hop and per-node packet load for one round.
    Members with d(i,CH) <= R_comm send directly; otherwise they forward to the alive neighbour
    (within R_comm) that is closest to their head (greedy geographic forwarding). Nodes closer to
    the sink than to any head, and every node under FLAG (star), send towards the sink with the
    same greedy rule (sink treated as the head)."""
    N = net.N; alive = net.alive; R = P['R_comm']
    dest = np.full(N, -1)                       # -1 = sink
    if CH.size and method != 'FLAG':
        dc = net.D[:, CH]; dest = CH[dc.argmin(1)]
        d_dest = net.D[np.arange(N), dest]
        to_sink = net.dbs < d_dest
        dest[to_sink] = -1
        dest[CH] = -2                           # heads
    def dist_to(i, t):
        return net.dbs[i] if t < 0 else net.D[i, t]
    nxt = np.full(N, -1); hopd = np.zeros(N)
    idx = np.flatnonzero(alive)
    for i in idx:
        t = dest[i]
        if t == -2:
            nxt[i] = -1; hopd[i] = net.dbs[i]; continue
        di = dist_to(i, t)
        if di <= R:
            nxt[i] = t; hopd[i] = di; continue
        nb = idx[(net.D[i, idx] <= R) & (idx != i)]
        if t >= 0:
            nb = nb[(dest[nb] == t) | (nb == t)]
            dn = net.D[nb, t]
        else:
            nb = nb[dest[nb] == -1]
            dn = net.dbs[nb]
        ok = dn < di
        if ok.any():
            j = nb[ok][dn[ok].argmin()]; nxt[i] = j; hopd[i] = net.D[i, j]
        else:
            nxt[i] = t; hopd[i] = di
    # packet loads: process farthest-first along decreasing distance to destination
    load = np.zeros(N); load[idx] = 1
    key = np.array([dist_to(i, dest[i]) if dest[i] != -2 else -1 for i in range(N)])
    for i in idx[np.argsort(-key[idx])]:
        j = nxt[i]
        if dest[i] != -2 and j >= 0 and j != i:
            load[j] += load[i]
    return dict(nxt=nxt, hopd=hopd, load=load, dest=dest)

def round_energy_from_routes(net, CH, rt):
    l = P['l']; alive = net.alive
    e = np.zeros(net.N)
    load, hopd, dest = rt['load'], rt['hopd'], rt['dest']
    a = alive
    heads = a & (dest == -2)
    nonh = a & ~heads
    # non-heads: transmit own + relayed packets one hop, receive relayed packets
    e[nonh] = load[nonh] * radio_tx(l, hopd[nonh], net.is4[nonh]) + (load[nonh] - 1) * radio_rx(l)
    # heads: receive cluster packets, aggregate, send one fused packet to the sink (Eq. 3)
    k = load[heads] - 1
    e[heads] = k * radio_rx(l) + P['E_agg'] * l * (k + 1) + radio_tx(l, net.dbs[heads], net.is4[heads])
    e[~a] = 0
    return e


def cluster_once(net, method, rng, E_net0, excl):
    variant = method.split(':'); base = variant[0]
    alpha = net.alpha_t(E_net0)
    if base == 'EECF':
        opts = variant[1] if len(variant) > 1 else ''
        if opts == 'static': alpha = 0.35
        if opts == 'baseFit':                   # EECF search operators with the baselines' fitness
            F = fitness_baseline(net); F[excl] = -np.inf; Sx = run_eecf(net, F, rng); return Sx.decode(), Sx
        if opts == 'single':
            F = fitness_energy_only(net); F[excl] = -np.inf; Sx = run_eecf(net, F, rng, phases=(False, False, False))
        else:
            F = fitness_eecf(net, alpha, use_div=(opts != 'noDQA')); F[excl] = -np.inf
            ph = (opts != 'noECP', opts != 'noBCS', opts != 'noDQA')
            Sx = run_eecf(net, F, rng, phases=ph)
    elif base == 'Greedy':                      # no search: rank every alive node by the EECF merit
        F = fitness_eecf(net, alpha); F[excl] = -np.inf; Sx = Search(net, F, rng)
    elif len(variant) > 1 and variant[1] == 'Fcomp':   # metaheuristic operators with the EECF merit (Eq. 8)
        F = fitness_eecf(net, alpha); F[excl] = -np.inf; Sx = SEARCH[base](net, F, rng)
    else:
        F = fitness_baseline(net); F[excl] = -np.inf; Sx = SEARCH[base](net, F, rng)
    return Sx.decode(), Sx

# ---------------------------------------------------------------- one full simulation
SEARCH = dict(EECF=run_eecf, PSO=run_pso, CAPSO=run_capso, GA=run_ga, AOA=run_aoa, OOA=run_ooa)

def simulate(method, N, seed, record_curve=False, gate=True, max_rounds=None, record_ch=False, net_kwargs=None, stop_at_hmr=True, gate_retry=False):
    net = Net(N, seed, **(net_kwargs or {}))
    rng = np.random.default_rng(10_000 + seed)
    ch_log = []; retries = 0; deferred = 0; passed = True
    E_net0 = net.E.sum()
    max_rounds = max_rounds or P['max_rounds']
    l = P['l']
    CH = np.array([], int); assign = None
    imr = hmr = None
    round_energy, lora_e, zig_e, ch_std = [], [], [], []
    gate_checks = gate_fails = 0
    rc_latency = []; rcic_vals = []; rcic_trace = []
    curve = None; sub = None; fcomp_sel = None
    leach_last = np.full(N, -10**9)
    prev_CH = np.array([], int)
    route = None; died_flag = False
    p_leach = P['n_CH'] / N
    r = 0
    while r < max_rounds:
        r += 1
        alive = net.alive
        if alive.sum() == 0: break
        need = (method == 'LEACH') or ((r - 1) % P['recluster_every'] == 0) or \
               (len(CH) and not alive[CH].all())
        if method == 'FLAG':
            CH = np.array([], int)
            if record_ch and (r - 1) % P['recluster_every'] == 0:
                ch_log.append(dict(round=r, CH=[], alive=[int(i) for i in np.flatnonzero(net.alive)], gate=True, rcic=None))
        elif need:
            t0 = time.perf_counter()
            if method == 'LEACH':
                epoch = int(round(1 / p_leach))
                eligible = alive & (r - leach_last >= epoch)
                thr = p_leach / (1 - p_leach * (r % epoch))
                sel = eligible & (rng.random(N) < thr)
                CH = np.flatnonzero(sel)
                if CH.size == 0:
                    cand = np.flatnonzero(alive); CH = rng.choice(cand, 1)
                leach_last[CH] = r
            else:
                excl = prev_CH if P['rotate'] else np.array([], int)
                CH, Sx = cluster_once(net, method, rng, E_net0, excl)
                prev_CH = CH.copy()
                if curve is None:
                    curve = list(Sx.curve)
                    fe, fc, fd, fx = net.subobjectives()
                    sub = dict(fe=float(fe[CH].mean()), fc=float(fc[CH].mean()),
                               fd=float(fd[CH].mean()), fx=float(fx[CH].mean()))
                    Fc = fitness_eecf(net, net.alpha_t(E_net0))
                    fcomp_sel = float(Fc[CH].mean())
            # member assignment
            if CH.size:
                dc = net.D[:, CH]; assign = CH[dc.argmin(1)]
            # reliability gate (Eq. 16); on failure EECF re-clusters once (failed heads excluded)
            passed = True
            if gate and method != 'FLAG' and CH.size and (r - 1) % P['recluster_every'] == 0:
                val = rcic(net, CH, assign, rng)
                gate_checks += 1; rcic_vals.append(val)
                rcic_trace.append((r, float(net.alive.mean()), val))
                if val < P['rho_min']:
                    gate_fails += 1; passed = False
                    if gate_retry and (method.startswith('EECF') or P.get('retry_all')):
                        CH2, _ = cluster_once(net, method, rng, E_net0, np.union1d(prev_CH, CH))
                        a2 = CH2[net.D[:, CH2].argmin(1)]
                        v2 = rcic(net, CH2, a2, rng)
                        retries += 1
                        if v2 >= P['rho_min']:
                            CH, assign, passed = CH2, a2, True; prev_CH = CH.copy(); rcic_vals[-1] = v2
                deferred += (not passed)
            if record_ch and (r - 1) % P['recluster_every'] == 0:
                ch_log.append(dict(round=r, CH=[int(c) for c in CH], alive=[int(i) for i in np.flatnonzero(net.alive)],
                                   gate=passed, rcic=rcic_vals[-1] if rcic_vals else None))
            rc_latency.append(time.perf_counter() - t0)
        # ---- energy for this round (Eq. 3 with greedy multi-hop intra-cluster forwarding)
        if need or method == 'LEACH' or r == 1 or route is None or died_flag:
            route = build_routes(net, CH, method)
        e = round_energy_from_routes(net, CH, route)
        if CH.size:
            ch_std.append(float(np.std(e[CH][alive[CH]]) * 1e3) if alive[CH].any() else 0.0)
        e[~alive] = 0
        net.E = net.E - e
        died = alive & (net.E <= 0)
        net.E[died] = 0; net.alive = alive & ~died
        died_flag = bool(died.any())
        round_energy.append(float(e.sum()))
        lora_e.append(float(e[~net.is4].sum())); zig_e.append(float(e[net.is4].sum()))
        na = net.alive.sum()
        if imr is None and na < N: imr = r
        if hmr is None and na <= N / 2:
            hmr = r
            if stop_at_hmr: break
    W0 = 300
    return dict(method=method, N=N, seed=seed, IMR=imr, HMR=hmr,
                E_round_mJ=float(np.mean(round_energy[:W0]) * 1e3),
                E_lora_mJ=float(np.mean(lora_e[:W0]) * 1e3), E_zig_mJ=float(np.mean(zig_e[:W0]) * 1e3),
                CH_std_mJ=float(np.mean(ch_std[:W0])) if ch_std else None,
                gate_checks=gate_checks, gate_fails=gate_fails, retries=retries, deferred=deferred,
                rcic_mean=float(np.mean(rcic_vals)) if rcic_vals else None,
                rcic_first=float(rcic_vals[0]) if rcic_vals else None,
                recluster_s=float(np.mean(rc_latency)) if rc_latency else None,
                curve=curve if record_curve else None, rcic_trace=rcic_trace, ch_log=ch_log if record_ch else None, sub=sub, Fcomp_sel=fcomp_sel)

def parse_seeds(s):
    if '-' in s:
        a, b = s.split('-'); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',')]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--N', type=int, nargs='+', default=[300])
    ap.add_argument('--seeds', default='0-29')
    ap.add_argument('--out', required=True)
    ap.add_argument('--curve', action='store_true')
    ap.add_argument('--P', nargs='*', default=[])
    a = ap.parse_args()
    for kv in a.P:
        k, v = kv.split('='); P[k] = type(P[k])(float(v)) if not isinstance(P[k], int) else int(float(v))
    with open(a.out, 'a') as fo:
        for N in a.N:
            for m in a.methods:
                for s in parse_seeds(a.seeds):
                    t0 = time.time()
                    res = simulate(m, N, s, record_curve=a.curve)
                    res['wall_s'] = time.time() - t0; res['Pset'] = a.P
                    fo.write(json.dumps(res) + '\n'); fo.flush()
                    print(m, N, s, res['IMR'], res['HMR'], round(res['E_round_mJ'], 2), res['gate_fails'], '/', res['gate_checks'], round(res['wall_s'], 1), flush=True)
