"""figs_tikz.py -- all manuscript figures as pgfplots/TikZ sources, compiled to PDF and 300-dpi PNG.
Run analysis.py first. Output: ../figures/tikz/<name>.tex, .pdf and ../figures/png/<name>.png"""
import json, os, math, subprocess, collections
import numpy as np
from scipy import stats
from analysis import load, ORDER, LABEL, R

HERE = os.path.dirname(os.path.abspath(__file__))
TZ = os.path.join(HERE, '..', 'figures', 'tikz'); PNG = os.path.join(HERE, '..', 'figures', 'png'); os.makedirs(TZ, exist_ok=True); os.makedirs(PNG, exist_ok=True)
S = json.load(open(os.path.join(R, 'stats.json')))
EX = json.load(open(os.path.join(R, 'extra.json'))) if os.path.exists(os.path.join(R, 'extra.json')) else None
N3 = S['n300']

HEX = ['2A78D6', 'EB6834', '1BAF7A', 'EDA100', 'E87BA4', '008300', '6250D6', 'E34948']
CN = {m: f'c{i + 1}' for i, m in enumerate(ORDER)}
MK = dict(EECF='*', PSO='square*', CAPSO='triangle*', GA='diamond*', AOA='pentagon*', OOA='otimes*', LEACH='x', FLAG='star')
DASH = dict(EECF='solid', PSO='dashed', CAPSO='dashdotted', GA='dotted', AOA='densely dashed', OOA='densely dashdotted', LEACH='densely dotted', FLAG='loosely dashed')
SHORT = dict(EECF='EECF-IoT', PSO='PSO', CAPSO='CAPSO', GA='GA', AOA='AOA', OOA='OOA', LEACH='LEACH', FLAG='FLAG-style')

PRE = r"""\documentclass[border=3pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots,fillbetween}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric,patterns}
""" + ''.join(f'\\definecolor{{c{i + 1}}}{{HTML}}{{{h}}}\n' for i, h in enumerate(HEX)) + r"""
\definecolor{ink}{HTML}{0B0B0B}\definecolor{ink2}{HTML}{52514E}\definecolor{gridc}{HTML}{E4E3DF}
\definecolor{s1}{HTML}{1C5CAB}\definecolor{s2}{HTML}{5598E7}\definecolor{s3}{HTML}{9EC5F4}\definecolor{s4}{HTML}{D0E3FA}
\pgfplotsset{every axis/.append style={font=\footnotesize, grid=major, major grid style={gridc}, axis line style={ink2},
  tick style={ink2}, title style={font=\footnotesize\bfseries, yshift=-2pt}, label style={font=\footnotesize},
  legend style={font=\scriptsize, draw=none, fill=white, fill opacity=0.9, text opacity=1}, legend cell align=left,
  scaled y ticks=false, yticklabel style={/pgf/number format/fixed}},
  errbar/.style={error bars/.cd, y dir=both, y explicit, error bar style={ink, line width=0.5pt, solid}, error mark options={rotate=90, mark size=1.8pt, ink}}}
\begin{document}
"""
END = '\\end{document}\n'

def compile_(name, body):
    tex = os.path.join(TZ, name + '.tex')
    open(tex, 'w').write(PRE + body + END)
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'], cwd=TZ, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2500:]); raise RuntimeError('latex failed: ' + name)
    subprocess.run(['pdftoppm', '-png', '-r', '300', '-singlefile', name + '.pdf', os.path.join(PNG, name)], cwd=TZ, check=True)
    for ext in ['.aux', '.log']:
        try: os.remove(os.path.join(TZ, name + ext))
        except OSError: pass
    print('ok', name)

def ci(a):
    a = np.asarray(a, float); n = len(a)
    return a.mean(0), (stats.t.ppf(.975, n - 1) * a.std(0, ddof=1) / math.sqrt(n)) if n > 1 else np.zeros_like(a.mean(0))

def bars(items, ylabel, title, ymin=None, ymax=None, fmt='{:.0f}', width=5.2, height=4.4, labels=False, rot=35, extra=''):
    """items: list of (xlabel, mean, halfwidth, color)."""
    xs = ','.join('{' + it[0] + '}' for it in items)
    s = (f'\\nextgroupplot[ybar, bar width=9pt, symbolic x coords={{{xs}}}, xtick={{{xs}}}, x tick label style={{rotate={rot}, anchor={"north east" if rot else "north"}, font=\\scriptsize}},'
         f' enlarge x limits=0.08, ylabel={{{ylabel}}}, title={{{title}}}, xmajorgrids=false'
         + (f', ymin={ymin}' if ymin is not None else '') + (f', ymax={ymax}' if ymax is not None else '') + extra
         + (', nodes near coords, nodes near coords style={font=\\tiny, color=ink2, yshift=2pt}, point meta=explicit symbolic' if labels else '') + ']\n')
    for x, m, h, c in items:
        meta = f' [{fmt.format(m)}]' if labels else ''
        s += f'\\addplot+[bar shift=0pt, area legend, fill={c}, draw=white, line width=0.6pt, errbar] coordinates {{({{{x}}},{m:.4f}) +- (0,{h:.4f}){meta}}};\n'
    return s

def group(n, body, w='5.2cm', h='4.4cm', hsep='1.95cm', vsep='2.2cm', rows=1):
    return (f'\\begin{{tikzpicture}}\\begin{{groupplot}}[group style={{group size={n} by {rows}, horizontal sep={hsep}, vertical sep={vsep}}}, width={w}, height={h}]\n'
            + body + '\\end{groupplot}\\end{tikzpicture}\n')

def coords(xs, ys, step=1):
    return ' '.join(f'({x:.4g},{y:.5g})' for i, (x, y) in enumerate(zip(xs, ys)) if i % step == 0 or i == len(xs) - 1)

def band(xs, mu, h, color, step=1, name=''):
    return (f'\\addplot[name path={name}u, draw=none, forget plot] coordinates {{{coords(xs, mu + h, step)}}};\n'
            f'\\addplot[name path={name}l, draw=none, forget plot] coordinates {{{coords(xs, mu - h, step)}}};\n'
            f'\\addplot[{color}, opacity=0.15, forget plot] fill between[of={name}u and {name}l];\n')

meth = [m for m in ORDER if m in N3]

# ------------------------------------------------------------------ Fig. convergence
def fig_convergence():
    rows = [r for r in load('n300_*.jsonl') if r.get('curve') and r['method'] in ORDER and r['method'] not in ('LEACH', 'FLAG')]
    g = collections.defaultdict(list)
    for r in rows:
        c = np.array(r['curve'], float); g[r['method']].append(c / c[-1])
    b = '\\nextgroupplot[xlabel={Iteration $t$}, ylabel={Best-so-far / final fitness}, title={(a) Search convergence ($T=150$)}, xmin=1, xmax=150, legend pos=south east, legend columns=2, yticklabel style={/pgf/number format/.cd, fixed, precision=3}]\n'
    for m in [m for m in ORDER if m in g]:
        mu, h = ci(np.array(g[m])); t = np.arange(1, len(mu) + 1)
        b += band(t, mu, h, CN[m], 2, m)
        b += f'\\addplot[{CN[m]}, {DASH[m]}, line width={1.3 if m == "EECF" else 0.9}pt] coordinates {{{coords(t, mu, 2)}}};\\addlegendentry{{{SHORT[m]}}}\n'
    mm = [m for m in ORDER if m in N3 and N3[m].get('sub')]
    xs = ','.join(SHORT[m] for m in mm)
    b += (f'\\nextgroupplot[ybar=0pt, bar width=4.2pt, symbolic x coords={{{xs}}}, xtick=data, ymin=0, ymax=1.12, ylabel={{Mean score of heads}}, x tick label style={{rotate=35, anchor=north east}},'
          f' title={{(b) Sub-objective decomposition}}, legend style={{at={{(0.5,-0.55)}}, anchor=north, draw=none, font=\\scriptsize}}, legend columns=4, xmajorgrids=false, enlarge x limits=0.12]\n')
    for k, lab, col in [('fe', '$f_{energy}$', 's1'), ('fc', '$f_{conn}$', 's2'), ('fd', '$f_{div}$', 's3'), ('fx', '$f_{dist}$', 's4')]:
        pts = ' '.join(f'({SHORT[m]},{N3[m]["sub"][k][0]:.4f}) +- (0,{N3[m]["sub"][k][2]:.4f})' for m in mm)
        b += f'\\addplot+[area legend, fill={col}, draw=white, errbar] coordinates {{{pts}}};\\addlegendentry{{{lab}}}\n'
    if EX:
        rh = sorted(EX['rho'], key=float); mu = [np.mean(EX['rho'][r]) for r in rh]; h = [stats.t.ppf(.975, 29) * np.std(EX['rho'][r], ddof=1) / math.sqrt(30) for r in rh]
        lo, hi = min(mu) - 0.004, max(mu) + 0.004
        b += (f'\\nextgroupplot[xlabel={{Temperature decay $\\rho$}}, ylabel={{Final $F_{{comp}}$ of heads}}, title={{(c) Sensitivity to $\\rho$}}, ymin={lo:.4f}, ymax={hi:.4f},'
              f' xtick={{0.90,0.93,0.95,0.97,0.99}}, xticklabel style={{/pgf/number format/.cd, fixed, precision=2}}, yticklabel style={{/pgf/number format/.cd, fixed, precision=3}}]\n')
        b += '\\addplot+[c1, mark=*, line width=1pt, errbar] coordinates {' + ' '.join(f'({float(r)},{m:.5f}) +- (0,{x:.5f})' for r, m, x in zip(rh, mu, h)) + '};\n'
    compile_('fig_convergence', group(3, b, w='5.4cm'))

# ------------------------------------------------------------------ Fig. lifetime
def fig_lifetime():
    xs = ','.join(SHORT[m] for m in meth)
    b = (f'\\nextgroupplot[ybar=0pt, bar width=5pt, symbolic x coords={{{xs}}}, xtick=data, x tick label style={{rotate=35, anchor=north east, font=\\scriptsize}},'
         f' ymax=2900, ylabel={{Round}}, title={{(a) Lifetime milestones, $N=300$}}, ymin=0, xmajorgrids=false, legend pos=north east, enlarge x limits=0.08]\n')
    for k, lab, op in [('IMR', 'IMR', 1.0), ('HMR', 'HMR', 0.45)]:
        pts = ' '.join(f'({SHORT[m]},{N3[m][k][0]:.2f}) +- (0,{N3[m][k][2]:.2f})' for m in meth)
        b += f'\\addplot+[fill=s1, fill opacity={op}, draw=white, errbar] coordinates {{{pts}}};\\addlegendentry{{{lab}}}\n'
    Ns = sorted(int(n) for n in S['scale'])
    for k, ttl in [('IMR', '(b) IMR vs network size'), ('HMR', '(c) HMR vs network size')]:
        b += f'\\nextgroupplot[xlabel={{Number of nodes $N$}}, ylabel={{{k} (round)}}, title={{{ttl}}}, xtick={{{",".join(map(str, Ns))}}}' + (', legend style={at={(1.03,1)}, anchor=north west}' if k == 'HMR' else '') + ']\n'
        for m in meth:
            pts = [(n, S['scale'][str(n)][m][k]) for n in Ns if m in S['scale'][str(n)] and S['scale'][str(n)][m][k][0] == S['scale'][str(n)][m][k][0]]
            if not pts: continue
            b += (f'\\addplot+[{CN[m]}, {DASH[m]}, mark={MK[m]}, mark size=1.6pt, mark options={{solid}}, line width={1.3 if m == "EECF" else 0.8}pt, errbar] coordinates {{'
                  + ' '.join(f'({n},{v[0]:.2f}) +- (0,{v[2]:.2f})' for n, v in pts) + '};' + (f'\\addlegendentry{{{SHORT[m]}}}' if k == 'HMR' else '') + '\n')
    compile_('fig_lifetime', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. energy and gas
def fig_energy():
    b = bars([(SHORT[m], N3[m]['E_round_mJ'][0], N3[m]['E_round_mJ'][2], CN[m]) for m in meth], 'Network energy per round (mJ)', '(a) Per-round energy, $N=300$', ymin=0)
    rv = S.get('rel_vs', {}); others = [m for m in ['PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'FLAG'] if m in rv]
    xs = ','.join(SHORT[o] for o in others)
    b += (f'\\nextgroupplot[ybar=0pt, bar width=5pt, symbolic x coords={{{xs}}}, xtick=data, x tick label style={{rotate=35, anchor=north east, font=\\scriptsize}},'
          f' ylabel={{Saving of EECF-IoT (\\%)}}, title={{(b) Saving by radio technology}}, xmajorgrids=false, legend style={{at={{(0.5,-0.5)}}, anchor=north, draw=none, font=\\scriptsize}}, legend columns=1, enlarge x limits=0.1, ymin=0]\n')
    for k, lab, col in [('E_lora_mJ', 'LoRaWAN-class ($\\alpha=2$)', 's1'), ('E_zig_mJ', 'Zigbee-class ($\\alpha=4$)', 's3')]:
        pts = ' '.join(f'({SHORT[o]},{-rv[o][k][0]:.3f}) +- (0,{rv[o][k][2]:.3f})' for o in others)
        b += f'\\addplot+[area legend, fill={col}, draw=white, errbar] coordinates {{{pts}}};\\addlegendentry{{{lab}}}\n'
    Ks = sorted(int(k) for k in S['gas']); xs = ','.join(f'K={K}' for K in Ks)
    b += (f'\\nextgroupplot[ybar stacked, bar width=14pt, symbolic x coords={{{xs}}}, xtick=data, ylabel={{Gas per FL round ($\\times 10^6$)}}, title={{(c) Measured contract gas}},'
          f' ymin=0, ymax=6.3, xmajorgrids=false, legend style={{at={{(0.5,-0.2)}}, anchor=north, draw=none, font=\\scriptsize}}, legend columns=2, enlarge x limits=0.18, reverse legend]\n')
    for k, lab, col in [('elect', 'elect()', 's4'), ('submit', 'submit() $\\times K$', 's3'), ('verify', 'verifyBatch()', 's2'), ('distribute', 'distribute()', 's1')]:
        pts = ' '.join(f'(K={K},{S["gas"][str(K)][k] * (K if k == "submit" else 1) / 1e6:.4f})' for K in Ks)
        b += f'\\addplot+[area legend, fill={col}, draw=white] coordinates {{{pts}}};\\addlegendentry{{{lab}}}\n'
    for K in Ks:
        b += f'\\node[font=\\tiny, color=ink2, above] at (axis cs:K={K},{S["gas"][str(K)]["total"] / 1e6:.3f}) {{{S["gas"][str(K)]["total"] / 1e6:.2f}}};\n'
    compile_('fig_energy', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. reliability
def fig_reliability():
    b = ''
    if EX:
        A = EX['rcic_acr']['ACR']
        b += '\\nextgroupplot[xlabel={Coverage requirement ACR}, ylabel={RCIC(ACR)}, title={(a) Reliability of the elected backbone}, ymin=0, ymax=1.05, legend style={at={(0.5,-0.33)}, anchor=north, draw=none, font=\\tiny}, legend columns=3]\n'
        for m in [m for m in ORDER if m in EX['rcic_acr']['curves']]:
            mu, h = ci(np.array(EX['rcic_acr']['curves'][m]))
            b += (f'\\addplot+[{CN[m]}, {DASH[m]}, mark={MK[m]}, mark size=1.4pt, mark options={{solid}}, line width={1.3 if m == "EECF" else 0.8}pt, errbar] coordinates {{'
                  + ' '.join(f'({a},{x:.4f}) +- (0,{y:.4f})' for a, x, y in zip(A, mu, h)) + f'}};\\addlegendentry{{{SHORT[m]}}}\n')
        b += '\\addplot[ink2, densely dotted, forget plot, domain=0.5:0.9] {0.9}; \\node[font=\\tiny, ink2, anchor=north west] at (axis cs:0.5,0.89) {$\\rho_{min}=0.90$};\n'
    mm = [m for m in meth if m != 'FLAG']
    b += bars([(SHORT[m], 100 * N3[m]['gate_freq'][0], 100 * N3[m]['gate_freq'][2], CN[m]) for m in mm], 'Gate failures (\\% of checks)', '(b) Gate-failure frequency', ymin=0, fmt='{:.1f}')
    b += bars([(SHORT[m], N3[m]['CH_std_mJ'][0], N3[m]['CH_std_mJ'][2], CN[m]) for m in mm], 'Std.\\ of head energy (mJ)', '(c) Head load balance', ymin=0, fmt='{:.2f}')
    compile_('fig_reliability', group(3, b, w='5.3cm', hsep='2.3cm'))

# ------------------------------------------------------------------ Fig. FL TON-IoT
FLMAP = [('EECF-IoT', 'EECF'), ('PSO-FedAvg', 'PSO'), ('CAPSO-FedAvg', 'CAPSO'), ('GA-FedAvg', 'GA'), ('AOA-FedAvg', 'AOA'),
         ('OOA-FedAvg', 'OOA'), ('LEACH-FedAvg', 'LEACH'), ('FLAG (star)', 'FLAG'), ('FedProx (PSO)', None)]
def fig_fl():
    F = S['fl_main']
    b = '\\nextgroupplot[xlabel={FL round}, ylabel={Test accuracy (\\%)}, title={(a) TON\\_IoT accuracy (10 classes)}, xmin=5, xmax=100, ymin=0, ymax=100, legend style={font=\\tiny, at={(0.5,-0.3)}, anchor=north, draw=none, inner sep=1pt, /tikz/every even column/.append style={column sep=3pt}}, legend image code/.code={\\draw[#1] (0cm,0cm) -- (0.3cm,0cm);}, legend columns=3]\n'
    for lab, m in FLMAP:
        if lab not in F or not F[lab]['curve']: continue
        c = 100 * np.array(F[lab]['curve']); h = 100 * np.array(F[lab]['curve_h'] or np.zeros_like(c)); t = np.arange(1, len(c) + 1)
        c, h, t = c[4:], h[4:], t[4:]   # accuracy is evaluated every 5 rounds; first evaluation at round 5
        col = CN[m] if m else 'ink2'; d = DASH.get(m, 'dashed')
        if m == 'EECF' or m == 'PSO': b += band(t, c, h, col, 1, 'f' + (m or 'fp'))
        b += f'\\addplot[{col}, {d}, line width={1.3 if m == "EECF" else 0.7}pt] coordinates {{{coords(t, c, 1)}}};\\addlegendentry{{{lab.replace("-FedAvg", "").replace(" (star)", "-style").replace(" (PSO)", "")}}}\n'
    labs = [l for l, _ in FLMAP if l in F]
    cols = {l: (CN[m] if m else 'ink2') for l, m in FLMAP}
    f1min = min(F[l]['f1'][0] for l in labs)
    b += bars([(l.replace('-FedAvg', '').replace(' (star)', '-style').replace(' (PSO)', ''), F[l]['f1'][0], F[l]['f1'][2], cols[l]) for l in labs],
              'Macro-F1 (\\%)', '(b) Macro-F1 after 100 rounds', ymin=max(0, math.floor(f1min - 10)), ymax=100, fmt='{:.1f}')
    NI = S.get('fl_noniid', {}); regs = ['mild', 'moderate', 'extreme']
    vals = [NI[k]['acc'][0] for k in NI]
    b += (f'\\nextgroupplot[ybar=0pt, bar width=6pt, symbolic x coords={{Mild,Moderate,Extreme}}, xtick=data, ylabel={{Test accuracy (\\%)}}, title={{(c) Label-skew regimes}},'
          f' xmajorgrids=false, legend style={{at={{(0.98,0.98)}}, anchor=north east, font=\\tiny}}, legend columns=1, enlarge x limits=0.25, ymin={max(0, math.floor(min(vals) - 12)) if vals else 0}, ymax=118, ytick={{0,20,40,60,80,100}},'
          f' xticklabels={{{{Mild\\\\(5 labels)}},{{Moderate\\\\(3 labels)}},{{Extreme\\\\(1 label)}}}}, x tick label style={{font=\\scriptsize, align=center}}]\n')
    for g_, col in [('EECF-IoT', 'c1'), ('PSO-FedAvg', 'c2'), ('FedProx', 'ink2')]:
        pts = ' '.join(f'({r.capitalize()},{NI[f"{g_} {r}"]["acc"][0]:.3f}) +- (0,{NI[f"{g_} {r}"]["acc"][2]:.3f})' for r in regs if f'{g_} {r}' in NI)
        if pts: b += f'\\addplot+[area legend, fill={col}, draw=white, errbar] coordinates {{{pts}}};\\addlegendentry{{{g_}}}\n'
    compile_('fig_fl', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. Wazihub
def fig_waz():
    W = S.get('waz', {}); order = [m for m in ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'FLAG', 'Server-only'] if m in W]
    if not order: return
    col = lambda m: CN.get(m, 'ink2'); lab = lambda m: SHORT.get(m, 'Server')
    b = bars([(lab(m), W[m]['acc'][0], W[m]['acc'][2], col(m)) for m in order], 'Accuracy (\\%)', '(a) Irrigation-decision accuracy',
             ymin=math.floor(min(W[m]['acc'][0] for m in order) - 4), ymax=100, fmt='{:.1f}')
    b += bars([(lab(m), W[m]['f1'][0], W[m]['f1'][2], col(m)) for m in order], 'Macro-F1 (\\%)', '(b) Irrigation-decision macro-F1',
              ymin=math.floor(min(W[m]['f1'][0] for m in order) - 6), ymax=100, fmt='{:.1f}')
    E = S.get('daily', {}); om = [m for m in ['EECF', 'PSO', 'CAPSO', 'GA', 'AOA', 'OOA', 'LEACH', 'FLAG'] if m in E]
    if om:
        b += bars([(('Star (FLAG-style)' if m == 'FLAG' else SHORT[m]), E[m][0], E[m][2], col(m)) for m in om], 'Sensing-network energy (J/day)', '(c) Daily network energy', ymin=0, fmt='{:.0f}')
    compile_('fig_waz', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. Byzantine
def fig_byz():
    B = S.get('fl_byz', {}); Th = S.get('fl_theta', {})
    if not B: return
    b = '\\nextgroupplot[xlabel={Malicious nodes (\\%)}, ylabel={Test accuracy (\\%)}, title={(a) Sign-flip attack ($\\times 3$)}, xtick={10,20,30}, xmin=5, xmax=35, ymin=0, ymax=100, legend style={at={(0.5,-0.33)}, anchor=north, draw=none, font=\\scriptsize}, legend columns=3]\n'
    for key, lab, c, mk in [('cosine', 'Median-cosine', 'c1', '*'), ('FLTrust', 'FLTrust', 'c3', 'square*'), ('no screening', 'None', 'c2', 'x')]:
        pts = [(f, B.get(f'EECF {key} f={f}')) for f in [0.1, 0.2, 0.3]]; pts = [(f, v) for f, v in pts if v]
        if pts: b += f'\\addplot+[{c}, mark={mk}, line width=1pt, errbar] coordinates {{' + ' '.join(f'({100 * f:.0f},{v["acc"][0]:.3f}) +- (0,{v["acc"][2]:.3f})' for f, v in pts) + f'}};\\addlegendentry{{{lab}}}\n'
    ref = S.get('fl_dpcomp', {}).get('DP off', {}).get('acc')   # attack-free, same seeds 0-4
    if ref: b += f'\\addplot[ink2, densely dotted, forget plot, domain=10:30] {{{ref[0]:.3f}}}; \\node[font=\\tiny, ink2, anchor=south east] at (axis cs:30,{ref[0]:.3f}) {{attack-free}};\n'
    ths = sorted(Th, key=lambda s: float(s.split('=')[1]))
    if ths:
        tv = [float(k.split('=')[1]) for k in ths]
        b += '\\nextgroupplot[xlabel={Threshold $\\theta_{byz}$}, ylabel={Test accuracy (\\%)}, title={(b) Accuracy vs threshold (20\\% attackers)}]\n'
        b += '\\addplot+[c1, mark=*, line width=1pt, errbar] coordinates {' + ' '.join(f'({t},{Th[k]["acc"][0]:.3f}) +- (0,{Th[k]["acc"][2]:.3f})' for t, k in zip(tv, ths)) + '};\n'
        hon = [100 * Th[k]['rej_honest'][0] / max(Th[k]['submitted'][0] - Th[k]['byz'][0], 1) for k in ths]
        mal = [100 * (Th[k]['rej'][0] - Th[k]['rej_honest'][0]) / max(Th[k]['byz'][0], 1) for k in ths]
        b += '\\nextgroupplot[xlabel={Threshold $\\theta_{byz}$}, ylabel={Updates rejected (\\%)}, title={(c) Screening selectivity}, ymin=-3, ymax=103, legend style={at={(0.03,0.45)}, anchor=west, font=\\tiny}]\n'
        b += '\\addplot+[c2, mark=square*, line width=1pt] coordinates {' + ' '.join(f'({t},{v:.2f})' for t, v in zip(tv, hon)) + '};\\addlegendentry{Honest}\n'
        b += '\\addplot+[c1, mark=*, line width=1pt] coordinates {' + ' '.join(f'({t},{v:.2f})' for t, v in zip(tv, mal)) + '};\\addlegendentry{Malicious}\n'
    compile_('fig_byz', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. ablation
def fig_ablation():
    A = S.get('fl_ablation', {}); M = S['fl_main']; AN = S.get('abl_net', {}); D = S.get('fl_dpcomp', {})
    var = [('Full', 'EECF-IoT', None), ('No adapt. w.', 'w/o adaptive weights', 'EECF:static'), ('No ECP', 'w/o ECP', 'EECF:noECP'),
           ('No BCS', 'w/o BCS', 'EECF:noBCS'), ('No LDA', 'w/o DQA', 'EECF:noDQA'), ('Single-obj.', 'Single-objective', 'EECF:single'), ('No gate', 'w/o TMCRA gate', None)]
    C = S.get('fl_ablation_clean', {})
    sd5 = [100 * v['acc'] for k, v in M['EECF-IoT']['seeds'].items() if int(k) < 5]
    m5, h5 = ci(sd5)
    full_ = []; clean_ = []
    for lab, key, _ in var:
        if key == 'EECF-IoT':
            full_.append((lab, m5, h5)); cv = C.get('Full [no gate/screen]')
        else:
            if key in A: full_.append((lab, A[key]['acc'][0], A[key]['acc'][2]))
            cv = C.get(f'{key} [no gate/screen]')
        if cv: clean_.append((lab, cv['acc'][0], cv['acc'][2]))
    labs = [l for l, _, _ in full_]
    xs = ','.join('{' + l + '}' for l in labs)
    allv = [v for _, v, _ in full_ + clean_]
    b = (f'\\nextgroupplot[ybar=0pt, bar width=4.5pt, symbolic x coords={{{xs}}}, xtick=data, x tick label style={{rotate=35, anchor=north east, font=\\scriptsize}},'
         f' ylabel={{TON\\_IoT accuracy (\\%)}}, title={{(a) Accuracy per variant}}, xmajorgrids=false, ymin={math.floor(min(allv) - 8)}, ymax=100, enlarge x limits=0.1,'
         f' legend style={{at={{(0.5,-0.55)}}, anchor=north, draw=none, font=\\scriptsize}}, legend columns=1]\n')
    b += '\\addplot+[area legend, fill=c1, draw=white, errbar] coordinates {' + ' '.join(f'({{{l}}},{v:.3f}) +- (0,{h:.3f})' for l, v, h in full_) + '};\\addlegendentry{Gate + screening}\n'
    if clean_:
        b += '\\addplot+[area legend, fill=s3, draw=white, errbar] coordinates {' + ' '.join(f'({{{l}}},{v:.3f}) +- (0,{h:.3f})' for l, v, h in clean_) + '};\\addlegendentry{No gate, no screening}\n'
    vv = [(lab, k) for lab, _, k in var if k and k in AN]
    if vv:
        xs = ','.join(l for l, _ in vv)
        b += (f'\\nextgroupplot[ybar=0pt, bar width=5pt, symbolic x coords={{{xs}}}, xtick=data, x tick label style={{rotate=35, anchor=north east, font=\\scriptsize}},'
              f' ylabel={{Change vs full EECF-IoT (\\%)}}, title={{(b) Network effect, $N=300$}}, xmajorgrids=false, legend style={{at={{(0.5,-0.62)}}, anchor=north, draw=none, font=\\scriptsize}}, legend columns=2, enlarge x limits=0.1]\n')
        for met, nm, col in [('E_round_mJ', 'Energy per round', 's1'), ('IMR', 'IMR', 's3')]:
            b += f'\\addplot+[area legend, fill={col}, draw=white, errbar] coordinates {{' + ' '.join(f'({l},{AN[k][met][0]:.3f}) +- (0,{AN[k][met][2]:.3f})' for l, k in vv) + f'}};\\addlegendentry{{{nm}}}\n'
    rows = [('No DP', 'DP off', 's1'), ('$\\sigma$=0.1', 'DP sigma=0.1', 's2'), ('$\\sigma$=0.5', 'DP sigma=0.5', 's2'), ('$\\sigma$=1.0', 'DP sigma=1.0', 's2'),
            ('$c$=0.5', 'c=0.5', 's3'), ('$c$=0.25', 'c=0.25', 's3'), ('$c$=0.1', 'c=0.1', 's3')]
    rows = [r for r in rows if r[1] in D]
    if rows:
        b += bars([(a, D[k]['acc'][0], D[k]['acc'][2], c) for a, k, c in rows], 'TON\\_IoT accuracy (\\%)', '(c) DP noise and compression',
                  ymin=math.floor(min(D[k]['acc'][0] for _, k, _ in rows) - 6), ymax=100, fmt='{:.1f}')
    compile_('fig_ablation', group(3, b, w='5.3cm'))

# ------------------------------------------------------------------ Fig. architecture (TikZ)
def fig_arch():
    tiers = [('Tier 5: Cloud', 'c7', 'global FedAvg, round and privacy scheduling'),
             ('Tier 4: Edge servers', 'c3', 'aggregation, reliability gate'),
             ('Tier 3: Smart contract', 'c1', '\\texttt{elect}, \\texttt{submit}, \\texttt{verifyBatch}, \\texttt{distribute}'),
             ('Tier 2: Cluster heads', 'c2', 'merit-based election, local training'),
             ('Tier 1: IoT sensor nodes', 'ink2', 'LoRaWAN-class and Zigbee-class nodes')]
    up = ['aggregated model', 'accept/reject decisions', 'hashes, commitments', 'sensor data']; dn = ['global model', 'global-model hash', 'election, incentives', 'cluster join']
    b = '\\begin{tikzpicture}[font=\\small, box/.style={draw=#1, line width=0.9pt, rounded corners=4pt, fill=#1!7, text width=10.4cm, minimum height=0.78cm, inner sep=3pt, align=center}]\n'
    for i, (t, c, l1) in enumerate(tiers):
        y = -i * 1.45
        b += f'\\node[box={c}, anchor=north west] (t{i}) at (0,{y}) {{\\textbf{{{t}}}\\quad{{\\color{{ink2}}{l1}}}}};\n'
    for i in range(4):
        b += (f'\\draw[-{{Stealth[length=5pt]}}, c6, line width=1pt] ([xshift=-1.9cm]t{i + 1}.north) -- node[right, font=\\scriptsize, c6] {{{up[i]}}} ([xshift=-1.9cm]t{i}.south);\n'
              f'\\draw[-{{Stealth[length=5pt]}}, s1, line width=1pt] ([xshift=1.5cm]t{i}.south) -- node[right, font=\\scriptsize, s1] {{{dn[i]}}} ([xshift=1.5cm]t{i + 1}.north);\n')
    b += ('\\draw[-{Stealth[length=5pt]}, c6, line width=1pt, densely dotted, rounded corners=8pt] (t3.west) -- ++(-0.5,0) |- node[pos=0.25, left, align=right, font=\\scriptsize, c6] {updates\\\\(off-chain)} (t1.west);\n')
    b += ('\\draw[-{Stealth[length=6pt]}, c8, line width=1pt, dashed, rounded corners=8pt] (t1.east) -- ++(0.6,0) |- node[pos=0.25, right, align=left, font=\\scriptsize, c8] {RCIC $<\\rho_{min}$:\\\\re-cluster} (t3.east);\n'
          '\\end{tikzpicture}\n')
    compile_('fig_arch', b)

if __name__ == '__main__':
    import sys
    todo = sys.argv[1:] or ['arch', 'convergence', 'lifetime', 'energy', 'reliability', 'fl', 'waz', 'byz', 'ablation']
    for n in todo:
        try: globals()['fig_' + n]()
        except Exception as e: print('FAIL', n, repr(e)[:300])
