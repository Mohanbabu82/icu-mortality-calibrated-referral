# Calibrated ICU mortality prediction with decision-anchored clinician referral

Analysis code for the article *"Calibrated machine learning with decision-anchored clinician referral for in-hospital mortality prediction in intensive care: a multicentre external validation against APACHE IV"* (submitted to the International Journal of Medical Informatics).

## Contents

| Path | What it is |
| --- | --- |
| `wids_calibration_study.ipynb` | End-to-end pipeline, sections 0–20: cohort, hospital-held-out split, models, calibration (including DWAC-M), referral, APACHE IV benchmark, equity audit, sensitivity analyses, hospital-cluster bootstrap, per-hospital heterogeneity |
| `wids_calibration_study.py` | Same code as a plain script (cells separated by `# %%`) |
| `cluster_addon.ipynb` | Short companion (about 10–15 min) that refits only the primary model with the published hyperparameters, checks it reproduces the published external AUROC exactly, then runs the hospital-cluster bootstrap and per-hospital analysis (Supplementary Tables S1–S2) |
| `tests/test_dwac_monotonicity.py` | Simulation stress test of DWAC vs DWAC-M monotonicity (Appendix A.3) |
| `outputs/tables/*.csv` | Aggregate result tables from the published run (no patient-level data) |
| `outputs/run_metadata.json` | Package versions, seed, held-out hospital IDs |
| `requirements.txt`, `CITATION.cff`, `.zenodo.json`, `LICENSE` | Environment, citation and licence |

## Data

The data are **not** included. Download `training_v2.csv` from the WiDS Datathon 2020 competition on Kaggle or from PhysioNet (https://doi.org/10.13026/vc0e-th79) after accepting the data use terms. Do not redistribute it.

## Running

**On Kaggle:** create a notebook, click *Add Input* → *Competitions* → *WiDS Datathon 2020*, upload `wids_calibration_study.ipynb`, choose a CPU session, then *Save Version* → *Save & Run All (Commit)*. A full run takes about 64 minutes (sections 0–18), plus about 30 minutes for sections 19–20. Outputs appear in `/kaggle/working/paper_outputs` and `paper_outputs.zip`.

**Locally:**

```bash
pip install -r requirements.txt
WIDS_CSV=/path/to/training_v2.csv OUT_DIR=./paper_outputs python wids_calibration_study.py
```

Environment variables: `WIDS_CSV` (data path), `OUT_DIR`, `N_BOOT` (default 1000), `N_BOOT_SMALL` (200), `N_ENSEMBLE` (10), `FAST_MODE=1` (smoke test only, not for reported results).

**Test:** `python tests/test_dwac_monotonicity.py` (about 5–8 minutes; prints PASS).

## Hospital-cluster analysis (Supplementary Tables S1–S2)

Open a new Kaggle notebook, add the WiDS Datathon 2020 competition data, upload `cluster_addon.ipynb`, and run *Save Version → Save & Run All (Commit)*. Download `cluster_outputs.zip` from the Output tab. The run stops with an error if the refitted model does not reproduce the published external AUROC (0.896955), so results can only come from the published model.

## Design notes

- External test hospitals (seed 42, 20% of hospitals): 15, 17, 23, 33, 70, 72, 79, 84, 90, 92, 104, 105, 117, 125, 130, 132, 136, 142, 145, 151, 159, 168, 171, 176, 179, 184, 194, 196, 199.
- APACHE IV predicted probabilities are excluded from predictors and used only as the benchmark.
- All preprocessing is learned on the training set; calibrators are fitted on the separate calibration set; test labels are used once.
- Section 20 writes Supplementary Table S1 (`tableS1_per_hospital_external_validation.csv`), Table S2 (`tableS2_bootstrap_patient_vs_hospital_cluster.csv`), a heterogeneity summary, a per-hospital forest plot and referral-threshold sensitivity. It also saves test-set predictions (`predictions_*.csv.gz`). These are derived from the licensed data, so keep them out of public repositories (they are listed in `.gitignore`).

## Citation

See `CITATION.cff`. Please also cite the WiDS Datathon 2020 dataset and the eICU Collaborative Research Database.

## Licence

MIT for the code. The data remain subject to their original use terms.
