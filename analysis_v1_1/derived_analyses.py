"""Analyses derived from the archived aggregate outputs (added in v1.1.0).

1. System-level sensitivity of clinician-plus-model referral (manuscript Table VI, eq. 5)
   and the break-even clinician sensitivity.
2. AUROC differences with DeLong 95% CIs, obtained by inverting the stored two-sided DeLong
   p-values: z = Phi^-1(1 - p/2), SE = delta / z, CI = delta +/- 1.96 SE (manuscript Table III
   and Section IV-B), including the five further hospital draws.

Run from the repository root:  python analysis_v1_1/derived_analyses.py
Inputs:  outputs/tables/*.csv (aggregate results of the published run; no patient data)
Outputs: outputs/tables/table12_system_level_referral.csv
         outputs/tables/table13_delta_auroc_delong.csv
"""
import os
import numpy as np, pandas as pd
from scipy import stats

T = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'tables')
D = 1738                                   # deaths in the external test set

# ---- 1. system-level referral (external, gradient boosting + DWAC-M, alert threshold 0.20)
r = pd.read_csv(os.path.join(T, 'table6_referral_performance.csv'))
e = r[(r.test_set == 'external_test') & (r.method == 'dwac_m') & (r.model == 'gradient_boosting')]
base = e[np.isclose(e.coverage, 1.0)].iloc[0]
TP0 = round(base.sensitivity * D)          # deaths flagged by the model alone
rows = []
for sc, lab in [('entropy', 'Entropy'), ('ensemble_sd', 'Ensemble SD'), ('threshold_distance', 'Threshold distance')]:
    for cov in (0.9, 0.8):
        x = e[(e.referral_score == sc) & np.isclose(e.coverage, cov)].iloc[0]
        dref = int(x.deaths_referred); tp_ret = round(x.sensitivity * (D - dref))
        rows.append(dict(rule=lab, referred_pct=round((1 - cov) * 100), n_referred=int(x.n_referred),
                         deaths_referred=dref, model_alerts_retained=round(tp_ret / x.ppv),
                         breakeven_clinician_sensitivity=(TP0 - tp_ret) / dref,
                         **{f'system_sensitivity_sc_{s:.2f}': (tp_ret + s * dref) / D for s in (0.80, 0.90, 0.95)}))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(T, 'table12_system_level_referral.csv'), index=False)
print(f'Model alone: {TP0} of {D} deaths flagged (sensitivity {base.sensitivity:.3f})')
print(S.round(3).to_string(index=False))

# ---- 2. DeLong CIs for AUROC differences
out = []
b = pd.read_csv(os.path.join(T, 'table5_apache_iv_benchmark.csv'))
for _, x in b[b.predictor != 'APACHE IV'].iterrows():
    a = b[(b.test_set == x.test_set) & (b.predictor == 'APACHE IV')].iloc[0]
    d = x.auroc - a.auroc; se = d / stats.norm.isf(x.delong_p_vs_apache / 2)
    out.append(dict(analysis='vs APACHE IV (matched)', test_set=x.test_set, comparison=f'{x.predictor} - APACHE IV', delta=d, lo=d - 1.96 * se, hi=d + 1.96 * se))
m = pd.read_csv(os.path.join(T, 'table4b_delong_model_comparison.csv'))
for _, x in m.iterrows():
    d = x.auroc_b - x.auroc_a; se = abs(d / x.z)
    out.append(dict(analysis='between models', test_set=x.test_set, comparison=f'{x.model_b} - {x.model_a}', delta=d, lo=d - 1.96 * se, hi=d + 1.96 * se))
s = pd.read_csv(os.path.join(T, 'table11_repeated_hospital_splits.csv'))
for seed, g in s.groupby('split_seed'):
    gb = g[g.predictor == 'gradient_boosting / dwac_m'].iloc[0]; ap = g[g.predictor == 'APACHE IV (original)'].iloc[0]
    d = gb.auroc_on_apache_subset - ap.auroc; se = d / stats.norm.isf(gb.delong_p_vs_original_apache / 2)
    out.append(dict(analysis=f'repeated draw seed {seed}', test_set='external_test', comparison='gradient_boosting / dwac_m - APACHE IV', delta=d, lo=d - 1.96 * se, hi=d + 1.96 * se))
O = pd.DataFrame(out)
O.to_csv(os.path.join(T, 'table13_delta_auroc_delong.csv'), index=False)
print(O.round(4).to_string(index=False))
