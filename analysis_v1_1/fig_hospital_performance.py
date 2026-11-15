"""Manuscript Fig. 5: per-hospital AUROC and calibration intercept (20 estimable held-out hospitals).
Run from the repository root:  python analysis_v1_1/fig_hospital_performance.py
Input: outputs/tables/tableS1_per_hospital_external_validation.csv  ->  outputs/figures/fig5_hospital_performance.png/.pdf
Pooled AUROC (0.895), its 95% prediction interval (0.866-0.918) and the overall intercept CIs are those printed by
cluster_addon.ipynb (see outputs/cluster_addon_executed_on_study_data.ipynb)."""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.spines.top': False, 'axes.spines.right': False})
d = pd.read_csv(os.path.join(R, 'outputs', 'tables', 'tableS1_per_hospital_external_validation.csv'))
e = d[d.auroc_lo.notna()].sort_values('n_patients').reset_index(drop=True)
y = np.arange(len(e)); lab = [f'H{int(h)} ({n:,}; {k})' for h, n, k in zip(e.hospital_id, e.n_patients, e.n_deaths)]
fig, ax = plt.subplots(2, 1, figsize=(3.5, 7.6), gridspec_kw=dict(hspace=0.42))
a = ax[0]
a.axvspan(0.866, 0.918, color='#2b6cb0', alpha=0.12, lw=0, label='95% prediction interval')
a.axvline(0.895, color='k', ls='--', lw=0.9, label='Pooled AUROC 0.895')
a.errorbar(e.auroc, y, xerr=[e.auroc - e.auroc_lo, e.auroc_hi - e.auroc], fmt='o', ms=3.5, color='#2b6cb0', elinewidth=1)
a.set_yticks(y); a.set_yticklabels(lab, fontsize=6.5); a.set_xlabel('AUROC (95% CI)'); a.set_xlim(0.70, 1.0)
a.set_title('(a) Discrimination', loc='left', fontsize=8, fontweight='bold')
a.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.35, -0.13), frameon=False)
b = ax[1]
b.axvspan(-0.285, 0.202, color='#c05621', alpha=0.12, lw=0, label='Hospital-cluster 95% CI, all 29 hospitals')
b.axvspan(-0.043, 0.027, color='#c05621', alpha=0.30, lw=0, label='Patient-level 95% CI')
b.axvline(0, color='k', ls='--', lw=0.9, label='Ideal (0)')
b.scatter(e.cal_intercept, y, s=6 + e.n_patients / 60, color='#c05621', zorder=3, edgecolor='white', lw=0.4)
b.set_yticks(y); b.set_yticklabels(lab, fontsize=6.5); b.set_xlabel('Calibration intercept'); b.set_xlim(-0.8, 1.0)
b.set_title('(b) Calibration-in-the-large', loc='left', fontsize=8, fontweight='bold')
b.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.35, -0.13), frameon=False)
for x in ax: x.grid(axis='x', color='#e5e5e5', lw=0.6); x.set_axisbelow(True)
os.makedirs(os.path.join(R, 'outputs', 'figures'), exist_ok=True)
for ext in ('png', 'pdf'): fig.savefig(os.path.join(R, 'outputs', 'figures', f'fig5_hospital_performance.{ext}'), dpi=300, bbox_inches='tight')
print('saved outputs/figures/fig5_hospital_performance.png')
