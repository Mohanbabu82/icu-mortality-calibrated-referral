# Analyses added in v1.1.0

These scripts reproduce results added to the manuscript after release v1.0.0. They read only the aggregate tables in `outputs/tables/` and the calibrator code in `wids_calibration_study.ipynb`; no patient data are needed. Run them from the repository root.

| Script | Reproduces | Runtime |
| --- | --- | --- |
| `derived_analyses.py` | Table VI (system-level referral sensitivity, break-even clinician sensitivity); AUROC differences with DeLong 95% CIs (Table III, Section IV-B, five further hospital draws) | seconds |
| `fig_hospital_performance.py` | Fig. 5 (per-hospital AUROC and calibration intercept) | seconds |
| `dwac_simulation.py` | Section IV-D, Fig. 4 and Supplementary Appendix A.3 (DWAC vs DWAC-M on synthetic data, size of the monotone projection, study-sized check) | 5-10 min |

DeLong intervals are recovered exactly from the stored two-sided DeLong p-values: z = Φ⁻¹(1 − p/2), SE = Δ/z, 95% CI = Δ ± 1.96·SE.
