"""DWAC vs DWAC-M on synthetic data (manuscript Section IV-D, Fig. 4, Supplementary Appendix A.3). Added in v1.1.0.

Uses the DWAC and DWAC-M classes extracted verbatim from wids_calibration_study.ipynb.
1. 100 synthetic calibration sets (seed 0; same generator as tests/test_dwac_monotonicity.py):
   decreasing steps of DWAC (8,000 logits from -12 to 12, decrease > 1e-12), DWAC-M violations,
   and the size of the monotone projection measured on DWAC-M's own 4,000-point grid.
2. Three study-sized sets (n = 10,857, prevalence ~8.5%; seeds 0-2): location and size of DWAC's decreases.
3. Fig. 4.
Run from the repository root:  python analysis_v1_1/dwac_simulation.py   (about 5-10 minutes on CPU)
Outputs: outputs/tables/table14_dwac_simulation.csv, outputs/figures/fig4_dwac_monotonicity.png/.pdf
"""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
nb = json.load(open(os.path.join(R, 'wids_calibration_study.ipynb')))
src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
g = {'np': np, 'stats': stats, 'IsotonicRegression': IsotonicRegression, 'average_precision_score': average_precision_score}
exec(src[src.index('EPS = 1e-6'):src.index('CALIBRATORS = {')], g)
DWAC, DWACM = g['DWAC'], g['DWACMonotone']
sig = lambda x: 1 / (1 + np.exp(-x))
def viol(f, lo=-12, hi=12, n=8000):
    return int((np.diff(f.transform(sig(np.linspace(lo, hi, n)))) < -1e-12).sum())

# ---- 1. 100 synthetic calibration sets
r = np.random.default_rng(0); rows = []; sets = {}
for t in range(100):
    n = int(r.integers(300, 4000)); prev = r.uniform(0.02, 0.4)
    z = r.normal(np.log(prev / (1 - prev)), r.uniform(0.8, 2.5), n)
    y = (r.random(n) < sig(r.uniform(0.5, 1.5) * z + r.normal(0, 0.5))).astype(int)
    p = sig(z); d = DWAC().fit(p, y); m = DWACM().fit(p, y)
    raw = DWAC.transform(m, sig(m.grid)); inr = (m.grid >= z.min()) & (m.grid <= z.max()); ch = np.abs(m.mono - raw)
    rows.append(dict(set=t, n=n, prevalence=prev, dwac_decreasing_steps=viol(d), dwacm_decreasing_steps=viol(m),
                     grid_points_changed=int((ch > 1e-9).sum()), max_projection_change_in_range=float(ch[inr].max())))
    sets[t] = (p, y)
S = pd.DataFrame(rows); os.makedirs(os.path.join(R, 'outputs', 'figures'), exist_ok=True)
S.to_csv(os.path.join(R, 'outputs', 'tables', 'table14_dwac_simulation.csv'), index=False)
q = lambda c: (S[c].median(), *S[c].quantile([.25, .75]), S[c].min(), S[c].max())
print('n: median %.1f IQR %.1f-%.1f range %d-%d' % q('n'))
print('DWAC non-monotone in %d/100; steps median %.1f IQR %.1f-%.1f range %d-%d' % ((S.dwac_decreasing_steps > 0).sum(), *q('dwac_decreasing_steps')))
print('DWAC-M non-monotone in %d/100' % (S.dwacm_decreasing_steps > 0).sum())
print('grid points changed: median %.1f IQR %.1f-%.1f range %d-%d' % q('grid_points_changed'))
print('max projection change (in range): median %.4f IQR %.4f-%.4f range %.4f-%.4f' % q('max_projection_change_in_range'))

# ---- 2. study-sized sets: where DWAC decreases
for seed in range(3):
    rr = np.random.default_rng(seed); n = 10857
    z = rr.normal(-3.2, 1.9, n); y = (rr.random(n) < sig(0.95 * z + 0.05)).astype(int)
    d = DWAC().fit(sig(z), y); gz = np.linspace(-9, 9, 20000); o = d.transform(sig(gz)); dd = np.diff(o); idx = np.where(dd < -1e-9)[0]
    print(f'study-sized seed {seed}: {len(idx)} decreasing steps, largest {(-dd).max():.1e}, logits {gz[idx].min():.2f} to {gz[idx].max():.2f}')

# ---- 3. Fig. 4
t = int(S.max_projection_change_in_range.idxmax()); p, y = sets[t]; d = DWAC().fit(p, y); m = DWACM().fit(p, y)
lg = lambda v: np.log(v / (1 - v)); zz = np.linspace(lg(p).min(), lg(p).max(), 6000); s = sig(zz)
G = m.grid; rawG = DWAC.transform(m, sig(G)); dG = m.mono - rawG; j = int(np.argmax(np.abs(dG))); L = H = j
while L > 0 and abs(dG[L - 1]) > 1e-9: L -= 1
while H < len(G) - 1 and abs(dG[H + 1]) > 1e-9: H += 1
pad = max(int((H - L) * 0.15), 6); kk = slice(max(L - pad, 0), min(H + pad + 1, len(G)))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(3.5, 4.5)); gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.6)
A, B, C = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
A.plot(zz, d.platt.transform(s), color='#2b6cb0', lw=1, label='Platt'); A.plot(zz, d.iso.transform(s), color='#2f855a', lw=0.8, label='Isotonic')
A.plot(zz, d.transform(s), color='#dd6b20', lw=1.4, label='DWAC'); A.plot(zz, m.transform(s), color='k', lw=1, ls='--', label='DWAC-M')
A.axvspan(G[kk][0], G[kk][-1], color='#999', alpha=0.35, lw=0)
A.set_xlabel('Raw score (logit)'); A.set_ylabel('Calibrated probability'); A.legend(frameon=False, fontsize=6)
A.set_title('(a) Calibration maps', loc='left', fontsize=8, fontweight='bold')
B.plot(G[kk], rawG[kk], '-', color='#dd6b20', lw=1.8); B.plot(G[kk], m.mono[kk], '--', color='k', lw=1.1)
yy = np.r_[rawG[L:H + 1], m.mono[L:H + 1]]; B.set_ylim(yy.min() - 0.004, yy.max() + 0.003)
B.set_xlabel('Raw score (logit)'); B.set_title('(b) Pooled block', loc='left', fontsize=8, fontweight='bold')
C.hist(S.dwac_decreasing_steps, bins=np.arange(0, 520, 40), color='#dd6b20', edgecolor='white')
C.set_xlabel('Decreasing steps of DWAC'); C.set_ylabel('Simulated sets (of 100)')
C.set_title('(c) 100 synthetic calibration sets', loc='left', fontsize=8, fontweight='bold')
C.text(0.97, 0.9, 'DWAC-M: 0 of 100\nnon-monotone', transform=C.transAxes, ha='right', va='top', fontsize=7)
for x in (A, B, C): x.grid(color='#eee', lw=0.6); x.set_axisbelow(True)
for ext in ('png', 'pdf'): fig.savefig(os.path.join(R, 'outputs', 'figures', f'fig4_dwac_monotonicity.{ext}'), dpi=300, bbox_inches='tight')
print('Fig. 4 uses synthetic set', t, 'n =', len(p))
