# %% [markdown]
# # Add-on: hospital-cluster bootstrap and per-hospital performance (Supplementary Tables S1–S2)
# Short companion to `wids_calibration_study.ipynb` (about 10–15 minutes on Kaggle CPU instead of about 2 hours).
# It re-runs only what section 20 needs: cohort, hospital split, preprocessing, the primary gradient-boosting model with the
# **published hyperparameters fixed** (no re-tuning), and the calibrators. A reproducibility gate then checks that the external
# AUROC, Brier score and ECE equal the published values before any new analysis is run. Attach the WiDS Datathon 2020
# competition data, then Save Version → Save & Run All (Commit). Outputs: /kaggle/working/cluster_outputs and cluster_outputs.zip.

# %% [markdown]
# ## 0. Setup

# %%
import os, glob, json, time, math, platform, warnings, shutil
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
N_BOOT = int(os.environ.get("N_BOOT", 1000))            # bootstrap resamples for main CIs
N_BOOT_SMALL = int(os.environ.get("N_BOOT_SMALL", 200))  # for subgroups / ablation / AURC
N_ENSEMBLE = int(os.environ.get("N_ENSEMBLE", 10))       # bagged models for epistemic uncertainty
EXTERNAL_HOSPITAL_FRACTION = 0.20
DECISION_THRESHOLD = 0.20      # pre-specified clinical alert threshold (mortality prevalence ~9%)
REPORT_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]
FAST = os.environ.get("FAST_MODE", "0") == "1"           # only for smoke tests

OUT = os.environ.get("OUT_DIR", "/kaggle/working/cluster_outputs")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
for d in (TAB, FIG):
    os.makedirs(d, exist_ok=True)

rng_global = np.random.default_rng(SEED)
np.random.seed(SEED)
T0 = time.time()

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
PALETTE = {"uncalibrated": "#7f7f7f", "platt": "#1f77b4", "beta": "#9467bd", "isotonic": "#2ca02c",
           "dwac": "#ff7f0e", "dwac_m": "#d62728", "apache_iv": "#000000"}
METHOD_LABEL = {"uncalibrated": "Uncalibrated", "platt": "Platt", "beta": "Beta", "isotonic": "Isotonic",
                "dwac": "DWAC", "dwac_m": "DWAC-M (monotone)", "apache_iv": "APACHE IV"}

def save_table(df, name, index=False):
    df.to_csv(os.path.join(TAB, f"{name}.csv"), index=index)
    try:
        with open(os.path.join(TAB, f"{name}.md"), "w") as f:
            f.write(df.to_markdown(index=index, floatfmt=".4f"))
    except Exception:
        pass  # tabulate missing; CSV is the canonical output

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)

def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
try:
    import shap
    HAVE_SHAP = True
except Exception:
    HAVE_SHAP = False
log(f"numpy {np.__version__}, pandas {pd.__version__}, sklearn {sklearn.__version__}, lightgbm={HAVE_LGB}, shap={HAVE_SHAP}")

# %% [markdown]
# ## 1. Load data and build the cohort

# %%
def find_data(root="/kaggle/input"):
    """Find the WiDS 2020 training table anywhere under /kaggle/input: any CSV whose header has
    'hospital_death' (labelled file, not the unlabelled test file). Picks the largest match."""
    if os.environ.get("WIDS_CSV"):
        return os.environ["WIDS_CSV"]
    found = []
    for path in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
        try:
            header = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "hospital_death" in header and "hospital_id" in header:
            found.append((os.path.getsize(path), path))
    if found:
        return sorted(found)[-1][1]
    print("Files currently attached under /kaggle/input:")
    for dirpath, _, files in os.walk(root):
        for f in files[:20]:
            print("  ", os.path.join(dirpath, f))
    raise FileNotFoundError(
        "No CSV with a 'hospital_death' column was found under /kaggle/input. In the right-hand panel click "
        "'Add Input' → 'Competitions' → search 'WiDS Datathon 2020' → click '+'. If Kaggle asks, open the "
        "competition page, click 'Join Competition' / accept the rules, then add it again and re-run this cell.")

DATA_PATH = find_data()
raw = pd.read_csv(DATA_PATH)
raw = raw.loc[:, [c for c in raw.columns if not c.startswith("Unnamed")]]
log(f"Loaded {DATA_PATH}: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

TARGET = "hospital_death"
ID_COLS = [c for c in ["encounter_id", "patient_id", "hospital_id", "icu_id"] if c in raw.columns]
APACHE_PROB = "apache_4a_hospital_death_prob"
LEAK_COLS = [c for c in raw.columns if c.startswith("apache_4a_")]

flow = [("ICU encounters in source file", len(raw))]
df = raw.loc[raw[TARGET].notna()].copy()
flow.append(("With recorded in-hospital outcome", len(df)))
if "age" in df.columns:
    df = df.loc[df["age"].isna() | (df["age"] >= 18)].copy()
    flow.append(("Adults (age ≥18 or age missing)", len(df)))
if "patient_id" in df.columns:
    df = df.drop_duplicates("patient_id", keep="first").copy()
    flow.append(("First encounter per patient", len(df)))
df[TARGET] = df[TARGET].astype(int)
if "hospital_id" not in df.columns:
    raise KeyError("hospital_id is required for the hospital-held-out external validation split.")
log(f"Analysis cohort: {len(df):,} encounters, mortality {df[TARGET].mean():.2%}, {df['hospital_id'].nunique()} hospitals")

# %% [markdown]
# ## 2. Splits: held-out hospitals (external) + stratified development split

# %%
hospitals = np.array(sorted(df["hospital_id"].unique()))
rng_split = np.random.default_rng(SEED)
ext_hosp = set(rng_split.choice(hospitals, size=max(1, int(round(EXTERNAL_HOSPITAL_FRACTION * len(hospitals)))), replace=False))
is_ext = df["hospital_id"].isin(ext_hosp)
dev, ext = df.loc[~is_ext].copy(), df.loc[is_ext].copy()

train, rest = train_test_split(dev, test_size=0.30, stratify=dev[TARGET], random_state=SEED)
cal, itest = train_test_split(rest, test_size=0.50, stratify=rest[TARGET], random_state=SEED)
SPLITS = {"train": train, "calibration": cal, "internal_test": itest, "external_test": ext}
for k, v in SPLITS.items():
    log(f"{k:14s} n={len(v):6,d}  deaths={v[TARGET].sum():5,d} ({v[TARGET].mean():.2%})  hospitals={v['hospital_id'].nunique()}")
assert not set(train["hospital_id"]) & ext_hosp and not set(cal["hospital_id"]) & ext_hosp
if "patient_id" in df.columns:
    ids = [set(v["patient_id"]) for v in SPLITS.values()]
    assert all(not (ids[i] & ids[j]) for i in range(4) for j in range(i + 1, 4)), "patient overlap across splits"

flow += [("Development hospitals", len(dev)), ("  Training", len(train)), ("  Calibration", len(cal)),
         ("  Internal test", len(itest)), ("External test (held-out hospitals)", len(ext))]
flow_df = pd.DataFrame(flow, columns=["step", "n"])
flow_df["n_hospitals_external"] = ""
flow_df.loc[flow_df["step"].str.startswith("External"), "n_hospitals_external"] = str(len(ext_hosp))
save_table(flow_df, "table_S1_cohort_flow")

# %% [markdown]
# ## 3. Features and preprocessing (all statistics learned on the training set only)

# %%
COMORB = [c for c in ["aids", "cirrhosis", "diabetes_mellitus", "hepatic_failure", "immunosuppression",
                      "leukemia", "lymphoma", "solid_tumor_with_metastasis"] if c in df.columns]
exclude = set(ID_COLS + LEAK_COLS + [TARGET])
candidate = [c for c in df.columns if c not in exclude]
constant = [c for c in candidate if train[c].nunique(dropna=True) <= 1]
candidate = [c for c in candidate if c not in constant]
CODE_AS_CAT = [c for c in ["apache_2_diagnosis", "apache_3j_diagnosis"] if c in candidate]
CAT_COLS = [c for c in candidate if (train[c].dtype == object or str(train[c].dtype) in ("category", "string", "str"))] + CODE_AS_CAT
CAT_COLS = list(dict.fromkeys(CAT_COLS))
NUM_COLS = [c for c in candidate if c not in CAT_COLS]
log(f"{len(NUM_COLS)} numeric + {len(CAT_COLS)} categorical predictors; dropped constant: {constant}; leakage-excluded: {LEAK_COLS}")

def feature_group(c):
    if c in COMORB: return "Comorbidities"
    if c.startswith("h1_"): return "First-hour vitals/labs (h1)"
    if c.startswith("d1_"):
        vit = ["diasbp", "sysbp", "mbp", "heartrate", "resprate", "spo2", "temp"]
        return "First-day vitals (d1)" if any(v in c for v in vit) else "First-day labs (d1)"
    if c.endswith("_apache") or c.startswith("apache_") or c in ("gcs_unable_apache",):
        return "APACHE physiology & diagnosis"
    return "Demographics & admission"
GROUPS = {}
for c in NUM_COLS + CAT_COLS:
    GROUPS.setdefault(feature_group(c), []).append(c)

# Category maps (top levels with >=30 training cases, rest -> "other")
CAT_MAPS = {}
for c in CAT_COLS:
    vc = train[c].dropna().astype(str).value_counts()
    keep = list(vc[vc >= 30].index[:60])
    CAT_MAPS[c] = keep + ["other"]

def encode_cats(frame):
    out = {}
    for c in CAT_COLS:
        s = frame[c]
        s_str = s.astype(str).where(s.notna(), None)
        levels = CAT_MAPS[c]
        mapped = s_str.where(s_str.isin(levels[:-1]) | s_str.isna(), "other")
        code = mapped.map({lvl: i for i, lvl in enumerate(levels)})
        out[c] = code.astype(float)       # NaN = missing
    return pd.DataFrame(out, index=frame.index)

def tree_matrix(frame, cols_num=None, cols_cat=None):
    cols_num = NUM_COLS if cols_num is None else cols_num
    cols_cat = CAT_COLS if cols_cat is None else cols_cat
    X = pd.concat([frame[cols_num].astype(float), encode_cats(frame)[cols_cat]], axis=1)
    return X, np.array([False] * len(cols_num) + [True] * len(cols_cat))

MED = train[NUM_COLS].median()
MISS_IND = [c for c in NUM_COLS if train[c].isna().any()]
X_tr_num = train[NUM_COLS].astype(float)
MU = X_tr_num.fillna(MED).mean()
SD = X_tr_num.fillna(MED).std().replace(0, 1)

def linear_matrix(frame):
    num = frame[NUM_COLS].astype(float)
    ind = num[MISS_IND].isna().astype(float).add_suffix("_missing")
    num = ((num.fillna(MED) - MU) / SD).clip(-6, 6)
    codes = encode_cats(frame)
    dummies = []
    for c in CAT_COLS:
        levels = CAT_MAPS[c] + ["missing"]
        s = codes[c].map(lambda v: levels[int(v)] if v == v else "missing")
        dummies.append(pd.get_dummies(pd.Categorical(s, categories=levels), prefix=c, dtype=float).iloc[:, 1:].set_index(frame.index))
    return pd.concat([num, ind] + dummies, axis=1).to_numpy(dtype=float)

Y = {k: v[TARGET].to_numpy() for k, v in SPLITS.items()}
XL = {k: linear_matrix(v) for k, v in SPLITS.items()}
XT = {}
for k, v in SPLITS.items():
    XT[k], CAT_MASK = tree_matrix(v)
log(f"Linear design: {XL['train'].shape[1]} columns; tree design: {XT['train'].shape[1]} columns")

# Missingness table (supplement)
miss = pd.DataFrame({"feature": NUM_COLS + CAT_COLS,
                     "group": [feature_group(c) for c in NUM_COLS + CAT_COLS],
                     "missing_pct_train": [train[c].isna().mean() * 100 for c in NUM_COLS + CAT_COLS],
                     "missing_pct_external": [ext[c].isna().mean() * 100 for c in NUM_COLS + CAT_COLS]})
save_table(miss.sort_values("missing_pct_train", ascending=False), "table_S2_missingness")
save_table(pd.DataFrame([(g, len(c), ", ".join(c[:12]) + (" …" if len(c) > 12 else "")) for g, c in GROUPS.items()],
                        columns=["feature_group", "n_features", "examples"]), "table_S3_feature_groups")

# %% [markdown]
# ## 4. Table 1: cohort characteristics

# %%
def med_iqr(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]" if len(s) else "NA"
def n_pct(mask):
    return f"{int(mask.sum()):,} ({100 * mask.mean():.1f}%)"

rows = []
def add_row(label, fn):
    rows.append([label] + [fn(v) for v in SPLITS.values()] + [fn(df)])
add_row("N", lambda d: f"{len(d):,}")
add_row("Hospitals", lambda d: f"{d['hospital_id'].nunique()}")
add_row("In-hospital deaths", lambda d: n_pct(d[TARGET] == 1))
for col, lab in [("age", "Age, years"), ("bmi", "BMI"), ("pre_icu_los_days", "Pre-ICU length of stay, days"),
                 (APACHE_PROB, "APACHE IV predicted mortality")]:
    if col in df.columns:
        add_row(f"{lab}, median [IQR]", lambda d, c=col: med_iqr(d[c].where(d[c] >= 0) if c == APACHE_PROB else d[c]))
if "gender" in df.columns:
    add_row("Female", lambda d: n_pct(d["gender"].astype(str) == "F"))
if "elective_surgery" in df.columns:
    add_row("Elective surgery", lambda d: n_pct(d["elective_surgery"] == 1))
for c in ["ventilated_apache"] + COMORB:
    if c in df.columns:
        add_row(c.replace("_", " ").capitalize(), lambda d, c=c: n_pct(d[c] == 1))
for cat in ["icu_type", "ethnicity", "apache_3j_bodysystem"]:
    if cat in df.columns:
        for lvl in df[cat].value_counts().index[:8]:
            add_row(f"{cat}: {lvl}", lambda d, cat=cat, lvl=lvl: n_pct(d[cat] == lvl))
table1 = pd.DataFrame(rows, columns=["characteristic"] + list(SPLITS.keys()) + ["overall"])
save_table(table1, "table1_cohort_characteristics")
table1.head(25)

# %% [markdown]
# ## 5. Metrics library (fast, version-independent, verified against scikit-learn in the smoke test)

# %%
EPS = 1e-6
def clip(p): return np.clip(np.asarray(p, float), EPS, 1 - EPS)
def logit(p): p = clip(p); return np.log(p / (1 - p))
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

def newton_logistic(X, y, offset=None, iters=50, ridge=1e-8):
    """Unregularised logistic MLE with optional offset (Newton–Raphson)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    w = np.zeros(X.shape[1]); off = 0 if offset is None else offset
    for _ in range(iters):
        p = sigmoid(X @ w + off)
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w

def auroc(y, p):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def auprc(y, p):
    return average_precision_score(y, p) if 0 < np.sum(y) < len(y) else np.nan

def brier(y, p): return float(np.mean((np.asarray(p) - y) ** 2))
def scaled_brier(y, p):
    pb = np.mean(y); return 1 - brier(y, p) / (pb * (1 - pb)) if 0 < pb < 1 else np.nan
def nll(y, p): p = clip(p); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def ece(y, p, n_bins=10, adaptive=False):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1))) if adaptive else np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2: return 0.0
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    tot = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any(): tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)

def cal_intercept_slope(y, p):
    """Calibration-in-the-large (slope fixed at 1, logit offset) and calibration slope."""
    lp = logit(p); y = np.asarray(y, float)
    if y.min() == y.max(): return np.nan, np.nan
    a = newton_logistic(np.ones((len(y), 1)), y, offset=lp)[0]
    b = newton_logistic(np.column_stack([np.ones(len(y)), lp]), y)[1]
    return float(a), float(b)

def rcs_basis(x, knots):
    k = np.asarray(knots); K = len(k)
    def pos3(v): return np.maximum(v, 0) ** 3
    cols = [x]
    for j in range(K - 2):
        cols.append(pos3(x - k[j]) - pos3(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                    + pos3(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2]))
    B = np.column_stack(cols)
    scale = (k[-1] - k[0]) ** 2
    B[:, 1:] /= max(scale, 1e-12)
    return B

def smoothed_calibration(y, p, grid=None):
    """Flexible calibration curve: logistic regression of outcome on a 4-knot restricted cubic spline of logit(p)."""
    lp = logit(p); knots = np.quantile(lp, [0.05, 0.35, 0.65, 0.95])
    if len(np.unique(knots)) < 4:
        return None
    w = newton_logistic(np.column_stack([np.ones(len(lp)), rcs_basis(lp, knots)]), y)
    f = lambda q: sigmoid(np.column_stack([np.ones(len(q)), rcs_basis(logit(q), knots)]) @ w)
    return f

def ici_e50_e90(y, p):
    f = smoothed_calibration(y, p)
    if f is None: return np.nan, np.nan, np.nan
    d = np.abs(f(p) - np.asarray(p))
    return float(d.mean()), float(np.median(d)), float(np.quantile(d, 0.9))

def all_metrics(y, p, ici=True):
    a, b = cal_intercept_slope(y, p)
    out = {"auroc": auroc(y, p), "auprc": auprc(y, p), "brier": brier(y, p), "scaled_brier": scaled_brier(y, p),
           "nll": nll(y, p), "ece_10": ece(y, p, 10), "ece_adaptive_10": ece(y, p, 10, adaptive=True),
           "cal_intercept": a, "cal_slope": b, "mean_pred": float(np.mean(p)), "obs_rate": float(np.mean(y))}
    if ici:
        out["ici"], out["e50"], out["e90"] = ici_e50_e90(y, p)
    return out

BOOT_METRICS = ["auroc", "auprc", "brier", "scaled_brier", "ece_10", "cal_intercept", "cal_slope", "ici"]
def boot_metric_fns():
    return {"auroc": auroc, "auprc": auprc, "brier": brier, "scaled_brier": scaled_brier,
            "ece_10": lambda y, p: ece(y, p, 10), "cal_intercept": lambda y, p: cal_intercept_slope(y, p)[0],
            "cal_slope": lambda y, p: cal_intercept_slope(y, p)[1], "ici": lambda y, p: ici_e50_e90(y, p)[0]}

def stratified_boot_indices(y, n_boot, seed):
    r = np.random.default_rng(seed); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for _ in range(n_boot):
        yield np.concatenate([r.choice(pos, len(pos)), r.choice(neg, len(neg))])

def delong_test(y, p1, p2):
    """Fast DeLong (Sun & Xu 2014) for two correlated AUROCs. Returns auc1, auc2, z, p."""
    y = np.asarray(y); order = np.argsort(-y, kind="stable"); y = y[order]
    preds = np.vstack([np.asarray(p1)[order], np.asarray(p2)[order]])
    m = int(y.sum()); n = len(y) - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.array([stats.rankdata(r) for r in pos]); ty = np.array([stats.rankdata(r) for r in neg])
    tz = np.array([stats.rankdata(r) for r in preds])
    aucs = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))

# %% [markdown]
# ## 6. Calibrators: Platt, beta, isotonic, DWAC (original) and DWAC-M (monotone projection)

# %%
class Identity:
    def fit(self, p, y): return self
    def transform(self, p): return clip(p)

class Platt:
    def fit(self, p, y):
        self.w = newton_logistic(np.column_stack([np.ones(len(p)), logit(p)]), y); return self
    def transform(self, p): return sigmoid(self.w[0] + self.w[1] * logit(p))

class Beta:
    """Beta calibration (Kull et al., 2017): logistic on [ln p, -ln(1-p)]; negative coefficients dropped."""
    def fit(self, p, y):
        p = clip(p); F = np.column_stack([np.log(p), -np.log(1 - p)])
        keep = [0, 1]
        for _ in range(3):
            w = newton_logistic(np.column_stack([np.ones(len(p)), F[:, keep]]), y)
            neg = [keep[i] for i, wi in enumerate(w[1:]) if wi < 0]
            if not neg: break
            keep = [k for k in keep if k not in neg]
        self.keep, self.w = keep, w; return self
    def transform(self, p):
        p = clip(p); F = np.column_stack([np.log(p), -np.log(1 - p)])
        return sigmoid(self.w[0] + (F[:, self.keep] @ self.w[1:] if self.keep else 0))

class Isotonic:
    def fit(self, p, y):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.asarray(p), y); return self
    def transform(self, p): return clip(self.iso.predict(np.asarray(p)))

class DWAC:
    """Density-Weighted Adaptive Calibration, identical to the repository implementation
    (src/calibration.py): w(p) = clipped, floor-adjusted Gaussian KDE density of calibration logits
    (Silverman bandwidth), p = w * isotonic + (1 - w) * Platt."""
    def __init__(self, density_floor=0.05): self.floor = density_floor
    def fit(self, p, y):
        self.platt, self.iso = Platt().fit(p, y), Isotonic().fit(p, y)
        self.lv = logit(p); n = len(self.lv)
        sd = np.std(self.lv, ddof=1); iqr = np.subtract(*np.percentile(self.lv, [75, 25]))
        sig = min(sd, iqr / 1.349) if iqr > 0 else sd
        self.bw = 0.9 * max(sig, 1e-6) * n ** (-0.2)
        self.maxd = self._density(self.lv).max(); return self
    def _density(self, q, chunk=2000):
        out = np.empty(len(q))
        for i in range(0, len(q), chunk):
            d = (q[i:i + chunk, None] - self.lv[None, :]) / self.bw
            out[i:i + chunk] = np.exp(-0.5 * d ** 2).mean(1)
        return out
    def weight(self, p):
        d = self._density(logit(p)) / self.maxd
        return np.clip((d - self.floor) / (1 - self.floor), 0, 1)
    def transform(self, p):
        w = self.weight(p); return clip(w * self.iso.transform(p) + (1 - w) * self.platt.transform(p))

class DWACMonotone(DWAC):
    """DWAC-M: the DWAC map evaluated on a dense logit grid and projected onto the set of
    non-decreasing functions (pool-adjacent-violators, L2), then interpolated. Guarantees that a
    higher raw risk never receives a lower calibrated risk."""
    def fit(self, p, y):
        super().fit(p, y)
        lo, hi = np.min(self.lv) - 1, np.max(self.lv) + 1
        self.grid = np.linspace(lo, hi, 4000)
        raw = DWAC.transform(self, sigmoid(self.grid))
        self.mono = IsotonicRegression(increasing=True).fit_transform(self.grid, raw); return self
    def transform(self, p):
        return clip(np.interp(logit(p), self.grid, self.mono))

CALIBRATORS = {"uncalibrated": Identity, "platt": Platt, "beta": Beta, "isotonic": Isotonic,
               "dwac": DWAC, "dwac_m": DWACMonotone}

def monotonicity_violations(cal, n=20000):
    g = sigmoid(np.linspace(-9, 9, n)); out = cal.transform(g); dd = np.diff(out)
    return int((dd < -1e-9).sum()), float(-dd.min()) if (dd < 0).any() else 0.0

# %% [markdown]
# ## 7. Model training (tuning by cross-validation inside the training set only)

# %%
def cv_select(make_model, grid, X, y, n_splits=3, cat_mask=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    res = []
    for params in grid:
        losses = []
        for tr, va in skf.split(X, y):
            m = make_model(params)
            Xtr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
            Xva = X.iloc[va] if hasattr(X, "iloc") else X[va]
            m.fit(Xtr, y[tr]); losses.append(log_loss(y[va], clip(m.predict_proba(Xva)[:, 1])))
        res.append({**params, "cv_log_loss": float(np.mean(losses)), "cv_log_loss_sd": float(np.std(losses))})
        log(f"   {params} -> CV log loss {np.mean(losses):.4f}")
    res = pd.DataFrame(res).sort_values("cv_log_loss")
    best = {k: v for k, v in res.iloc[0].items() if k not in ("cv_log_loss", "cv_log_loss_sd")}
    return best, res

def make_lr(params, class_weight=None):
    return LogisticRegression(C=params["C"], penalty="l2", solver="lbfgs", max_iter=3000, class_weight=class_weight)

def make_hgb(params, class_weight=None, seed=SEED):
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"], max_leaf_nodes=int(params["max_leaf_nodes"]),
        min_samples_leaf=int(params["min_samples_leaf"]), l2_regularization=params["l2_regularization"],
        max_iter=300 if FAST else 2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        categorical_features=CAT_MASK, class_weight=class_weight, random_state=seed)

# Published selections (table2_hyperparameters.csv); no re-tuning in this add-on.
lr_grid = [{"C": 0.1}]
hgb_grid = [{"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 0.0}]

log("Tuning logistic regression")
lr_best, lr_cv = cv_select(make_lr, lr_grid, XL["train"], Y["train"])
log("Tuning gradient boosting (HistGradientBoosting)")
hgb_best, hgb_cv = cv_select(make_hgb, hgb_grid, XT["train"], Y["train"])

MODELS, PRED_RAW = {}, {}
t = time.time(); MODELS["logistic_regression"] = make_lr(lr_best).fit(XL["train"], Y["train"]); lr_time = time.time() - t
t = time.time(); MODELS["gradient_boosting"] = make_hgb(hgb_best).fit(XT["train"], Y["train"]); hgb_time = time.time() - t
DESIGN = {"logistic_regression": XL, "gradient_boosting": XT}

hp_rows = [{"model": "logistic_regression", "selected": json.dumps(lr_best), "search_grid": json.dumps([g["C"] for g in lr_grid]),
            "selection": "3-fold CV log loss on training set", "fit_seconds": round(lr_time, 1),
            "notes": "L2, lbfgs, unweighted; standardized numerics + missingness indicators + one-hot categoricals"},
           {"model": "gradient_boosting", "selected": json.dumps(hgb_best), "search_grid": f"{len(hgb_grid)} configurations",
            "selection": "3-fold CV log loss on training set", "fit_seconds": round(hgb_time, 1),
            "notes": f"sklearn HistGradientBoosting, native categoricals, early stopping on 10% inner split, "
                     f"iterations used = {MODELS['gradient_boosting'].n_iter_}"}]

if False:  # LightGBM not needed for the add-on
    try:
        log("Fitting LightGBM (additional model, same tuned depth/leaf settings)")
        Xl_tr, Xl_es, yl_tr, yl_es = train_test_split(XT["train"], Y["train"], test_size=0.1, stratify=Y["train"], random_state=SEED)
        lgbm = lgb.LGBMClassifier(n_estimators=300 if FAST else 5000, learning_rate=0.03, num_leaves=int(hgb_best["max_leaf_nodes"]),
                                  min_child_samples=int(hgb_best["min_samples_leaf"]), reg_lambda=float(hgb_best["l2_regularization"]),
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
        cat_idx = [i for i, m in enumerate(CAT_MASK) if m]
        t = time.time()
        lgbm.fit(Xl_tr, yl_tr, eval_set=[(Xl_es, yl_es)], categorical_feature=cat_idx,
                 callbacks=[lgb.early_stopping(100, verbose=False)])
        MODELS["lightgbm"] = lgbm; DESIGN["lightgbm"] = XT
        hp_rows.append({"model": "lightgbm", "selected": json.dumps({"learning_rate": 0.03, "num_leaves": int(hgb_best["max_leaf_nodes"]),
                        "best_iteration": int(lgbm.best_iteration_ or 0)}), "search_grid": "inherits HGB tree-size settings",
                        "selection": "early stopping on 10% inner split of training set", "fit_seconds": round(time.time() - t, 1),
                        "notes": "subsample 0.8, colsample 0.8"})
    except Exception as e:
        log(f"LightGBM skipped: {e}")

save_table(pd.DataFrame(hp_rows), "table2_hyperparameters")
save_table(pd.concat([lr_cv.assign(model="logistic_regression"), hgb_cv.assign(model="gradient_boosting")]), "table_S4_cv_tuning_results")

for name, m in MODELS.items():
    PRED_RAW[name] = {k: m.predict_proba(DESIGN[name][k])[:, 1] for k in SPLITS}
log("Models trained: " + ", ".join(MODELS))

# %% [markdown]
# ## 8. Fit calibrators on the calibration set; predict on both test sets

# %%
CAL_OBJ, PRED = {}, {}
TESTS = ["internal_test", "external_test"]
for name in MODELS:
    for meth, C in CALIBRATORS.items():
        cobj = C().fit(PRED_RAW[name]["calibration"], Y["calibration"])
        CAL_OBJ[(name, meth)] = cobj
        for k in TESTS:
            PRED[(name, meth, k)] = cobj.transform(PRED_RAW[name][k])

mono_rows = []
for (name, meth), cobj in CAL_OBJ.items():
    v, mx = monotonicity_violations(cobj)
    rank_changes = {k: int(np.sum(np.diff(PRED[(name, meth, k)][np.argsort(PRED_RAW[name][k], kind="stable")]) < -1e-12)) for k in TESTS}
    mono_rows.append({"model": name, "method": meth, "grid_decreasing_steps": v, "largest_decrease": mx,
                      "rank_reversals_internal_test": rank_changes["internal_test"],
                      "rank_reversals_external_test": rank_changes["external_test"]})
mono_df = pd.DataFrame(mono_rows)
save_table(mono_df, "table_S5_monotonicity_check")
mono_df

# %% [markdown]

# %%
def entropy(p): p = clip(p); return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
def referral_scores(p, k):
    return {"entropy": entropy(p), "threshold_distance": -np.abs(logit(p) - logit(DECISION_THRESHOLD)),
            "ensemble_sd": ENS_SD[k]}   # higher = more uncertain = referred first

def risk_coverage(y, p, unc, coverages, t=DECISION_THRESHOLD):
    order = np.argsort(unc, kind="stable")  # least uncertain first
    rows = []
    for c in coverages:
        n_acc = max(1, int(round(c * len(y)))); acc = order[:n_acc]; ref = order[n_acc:]
        ya, pa = y[acc], p[acc]; pred = (pa >= t).astype(int)
        tp = int(((pred == 1) & (ya == 1)).sum()); fn = int(((pred == 0) & (ya == 1)).sum())
        tn = int(((pred == 0) & (ya == 0)).sum()); fp = int(((pred == 1) & (ya == 0)).sum())
        rows.append({"coverage": c, "n_accepted": n_acc, "n_referred": len(ref),
                     "selective_brier": brier(ya, pa), "selective_error": float((pred != ya).mean()),
                     "sensitivity": tp / (tp + fn) if tp + fn else np.nan, "specificity": tn / (tn + fp) if tn + fp else np.nan,
                     "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
                     "deaths_referred": int(y[ref].sum()), "deaths_referred_pct": float(y[ref].sum() / max(1, y.sum()) * 100),
                     "ece_accepted": ece(ya, pa, 10), "auroc_accepted": auroc(ya, pa)})
    return pd.DataFrame(rows)


# %%
PRIMARY_MODEL, PRIMARY_METHOD = "gradient_boosting", "dwac_m"

# %% [markdown]
# ## Reproducibility gate: the refitted model must match the published run

# %%
_ye = Y["external_test"]; _pe = PRED[("gradient_boosting", "dwac_m", "external_test")]
_check = {"n_external": (len(_ye), 19161), "deaths_external": (int(_ye.sum()), 1738),
          "auroc": (round(auroc(_ye, _pe), 6), 0.896955), "brier": (round(brier(_ye, _pe), 6), None), "ece_10": (round(ece(_ye, _pe, 10), 6), None)}
print(_check)
if len(df) == 91543:
    assert _check["n_external"][0] == 19161 and _check["deaths_external"][0] == 1738, "cohort differs from the published run"
    assert abs(_check["auroc"][0] - 0.896955) < 2e-6, "external AUROC differs from the published run: do not use these results"
    log("Reproducibility gate passed: identical cohort and model to the published run")
else:
    log("Gate skipped: not the full WiDS cohort (smoke test only)")

# ## 20. Hospital clustering: cluster bootstrap and per-hospital performance (Supplementary Tables S1-S2)
# Patients are clustered within hospitals. Section 9 resampled patients; this section resamples whole external
# hospitals (all patients of each drawn hospital kept), recomputes the principal metrics in every resample, and
# compares the resulting percentile 95% CIs with the patient-level, outcome-stratified bootstrap. Resamples that
# contain only one outcome class are counted as invalid and excluded. Per-hospital estimates are reported where
# estimable. Also re-evaluates threshold-distance referral at alert thresholds 0.10-0.30 and saves predictions.

# %%
T_NB = DECISION_THRESHOLD
def nb_at(y, p, t=T_NB):
    pr = p >= t; n = len(y)
    return float(np.sum(pr & (y == 1)) / n - np.sum(pr & (y == 0)) / n * t / (1 - t))
def metric_set(y, p):
    a, s = cal_intercept_slope(y, p)
    return {"auroc": auroc(y, p), "brier": brier(y, p), "ece_10": ece(y, p, 10),
            "cal_intercept": a, "cal_slope": s, "ici": ici_e50_e90(y, p)[0], "net_benefit_0.20": nb_at(y, p)}
CLUSTER_METRICS = ["auroc", "brier", "ece_10", "ici", "cal_intercept", "cal_slope", "net_benefit_0.20"]

yE = Y["external_test"]; pE = PRED[(PRIMARY_MODEL, PRIMARY_METHOD, "external_test")]
pE_raw = PRED[(PRIMARY_MODEL, "uncalibrated", "external_test")]
hE = SPLITS["external_test"]["hospital_id"].to_numpy()
uh = np.unique(hE); idx_by_h = {h: np.where(hE == h)[0] for h in uh}
point = metric_set(yE, pE)

# (a) patient-level, outcome-stratified bootstrap (same scheme and seed as section 9)
pl = {m: [] for m in CLUSTER_METRICS}
for ii in stratified_boot_indices(yE, N_BOOT, SEED):
    for m, v in metric_set(yE[ii], pE[ii]).items(): pl[m].append(v)
# (b) hospital-cluster bootstrap
N_CLUSTER_BOOT = N_BOOT
rng_c = np.random.default_rng(SEED + 20)
cl = {m: [] for m in CLUSTER_METRICS}; n_valid = n_invalid = 0
for b in range(N_CLUSTER_BOOT):
    ii = np.concatenate([idx_by_h[h] for h in rng_c.choice(uh, len(uh), replace=True)])
    if yE[ii].min() == yE[ii].max(): n_invalid += 1; continue
    n_valid += 1
    for m, v in metric_set(yE[ii], pE[ii]).items(): cl[m].append(v)
rows = []
for m in CLUSTER_METRICS:
    plo, phi = np.nanpercentile(pl[m], [2.5, 97.5]); clo, chi = np.nanpercentile(cl[m], [2.5, 97.5])
    rows.append({"metric": m, "estimate": point[m], "patient_lo": plo, "patient_hi": phi, "patient_width": phi - plo,
                 "cluster_lo": clo, "cluster_hi": chi, "cluster_width": chi - clo,
                 "width_ratio_cluster_vs_patient": (chi - clo) / (phi - plo) if phi > plo else np.nan,
                 "patient_resamples": len(pl[m]), "cluster_valid_resamples": n_valid, "cluster_invalid_resamples": n_invalid,
                 "n_hospitals": len(uh), "n_patients": len(yE), "n_deaths": int(yE.sum())})
s2 = pd.DataFrame(rows)
save_table(s2, "tableS2_bootstrap_patient_vs_hospital_cluster")

# (c) per-hospital performance. AUROC needs >=1 death and >=1 survivor; calibration intercept/slope need
#     >=10 deaths and >=10 survivors (otherwise reported as NA, not estimable).
hrows = []
for h in uh:
    ii = idx_by_h[h]; yh, ph = yE[ii], pE[ii]; d = int(yh.sum()); n = len(ii)
    r = {"hospital_id": int(h), "n_patients": n, "n_deaths": d, "mortality": yh.mean(), "mean_predicted": ph.mean(),
         "o_e_ratio": d / ph.sum() if ph.sum() > 0 else np.nan, "brier": brier(yh, ph), "ece_10": ece(yh, ph, 10),
         "auroc": np.nan, "auroc_lo": np.nan, "auroc_hi": np.nan, "cal_intercept": np.nan, "cal_slope": np.nan, "note": ""}
    if 0 < d < n:
        r["auroc"] = auroc(yh, ph)
        if d >= 10 and n - d >= 10:
            bs = [auroc(yh[j], ph[j]) for j in stratified_boot_indices(yh, N_BOOT_SMALL, SEED + int(h))]
            r["auroc_lo"], r["auroc_hi"] = np.percentile(bs, [2.5, 97.5])
            r["cal_intercept"], r["cal_slope"] = cal_intercept_slope(yh, ph)
        else:
            r["note"] = "fewer than 10 deaths or survivors: AUROC CI, intercept and slope not estimated"
    else:
        r["note"] = "single outcome class: AUROC not estimable"
    hrows.append(r)
s1 = pd.DataFrame(hrows).sort_values("hospital_id")
save_table(s1, "tableS1_per_hospital_external_validation")

# (d) random-effects (DerSimonian-Laird) summary of per-hospital logit-AUROC with 95% prediction interval
hv = s1.dropna(subset=["auroc_lo"]); hv = hv[(hv.auroc > 0) & (hv.auroc < 1) & (hv.auroc_lo > 0) & (hv.auroc_hi < 1)]
het = {"hospitals_in_meta_analysis": len(hv)}
if len(hv) >= 3:
    L = np.log(hv.auroc / (1 - hv.auroc)).to_numpy()
    se = ((np.log(hv.auroc_hi / (1 - hv.auroc_hi)) - np.log(hv.auroc_lo / (1 - hv.auroc_lo))) / (2 * 1.96)).to_numpy()
    se = np.maximum(se, 1e-3); w = 1 / se**2; Lf = np.sum(w * L) / w.sum(); Q = float(np.sum(w * (L - Lf) ** 2)); dfq = len(L) - 1
    tau2 = max(0.0, (Q - dfq) / (w.sum() - np.sum(w**2) / w.sum())); ws = 1 / (se**2 + tau2); Lr = np.sum(ws * L) / ws.sum()
    ser = np.sqrt(1 / ws.sum()); half = stats.t.ppf(0.975, len(L) - 2) * np.sqrt(tau2 + ser**2)
    het.update(pooled_auroc=float(sigmoid(Lr)), pooled_lo=float(sigmoid(Lr - 1.96 * ser)), pooled_hi=float(sigmoid(Lr + 1.96 * ser)),
               prediction_interval_lo=float(sigmoid(Lr - half)), prediction_interval_hi=float(sigmoid(Lr + half)),
               tau2_logit=float(tau2), I2=float(max(0.0, (Q - dfq) / Q)) if Q > 0 else 0.0)
het.update(o_e_median=float(s1.o_e_ratio.median()), o_e_q1=float(s1.o_e_ratio.quantile(.25)), o_e_q3=float(s1.o_e_ratio.quantile(.75)),
           o_e_min=float(s1.o_e_ratio.min()), o_e_max=float(s1.o_e_ratio.max()))
save_table(pd.DataFrame([het]), "tableS1b_hospital_heterogeneity_summary")

hv2 = s1.dropna(subset=["auroc"]).reset_index(drop=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 0.3 * len(hv2) + 1.6), sharey=True)
yy = np.arange(len(hv2))
lo = np.where(hv2.auroc_lo.notna(), hv2.auroc - hv2.auroc_lo, 0); hi = np.where(hv2.auroc_hi.notna(), hv2.auroc_hi - hv2.auroc, 0)
ax[0].errorbar(hv2.auroc, yy, xerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="o", color="#1f77b4", ms=3)
if "pooled_auroc" in het: ax[0].axvline(het["pooled_auroc"], ls="--", color="k", lw=1)
ax[0].set_xlabel("AUROC (95% CI)"); ax[0].set_yticks(yy)
ax[0].set_yticklabels([f"Hospital {int(h)} (n={n:,}; {d} deaths)" for h, n, d in zip(hv2.hospital_id, hv2.n_patients, hv2.n_deaths)], fontsize=7)
ax[1].scatter(hv2.o_e_ratio, yy, s=12, color="#d62728"); ax[1].axvline(1, ls="--", color="k", lw=1); ax[1].set_xlabel("Observed / expected deaths")
fig.suptitle("Per-hospital performance in the 29 held-out hospitals (gradient boosting + DWAC-M)", fontsize=10); fig.tight_layout()
save_fig(fig, "figure_S4_per_hospital_forest")

# (e) threshold-distance referral re-evaluated at other alert thresholds (20% referral fixed)
thr_rows = []
for t in [0.10, 0.15, 0.20, 0.30]:
    for nm, u in [("threshold_distance", -np.abs(logit(pE) - logit(t))), ("entropy", entropy(pE))] + ([("ensemble_sd", ENS_SD["external_test"])] if "ENS_SD" in globals() else []):
        for c in [1.0, 0.9, 0.8]:
            r = risk_coverage(yE, pE, u, [c], t=t).iloc[0].to_dict(); r.update(threshold=t, score=nm); thr_rows.append(r)
save_table(pd.DataFrame(thr_rows), "table14_referral_threshold_sensitivity")

# (f) predictions for re-analysis (derived from licensed data: do not publish)
for k in TESTS:
    po = pd.DataFrame({"y": Y[k], "hospital_id": SPLITS[k]["hospital_id"].to_numpy()})
    for (mn, me, kk), v in PRED.items():
        if kk == k: po[f"{mn}__{me}"] = v
    po.to_csv(os.path.join(OUT, f"predictions_{k}.csv.gz"), index=False, compression="gzip")

lines3 = ["", "## Hospital clustering (section 20)", f"- Cluster bootstrap: {n_valid} valid, {n_invalid} invalid resamples of {len(uh)} hospitals"]
for _, r in s2.iterrows():
    lines3.append(f"- {r.metric}: {r.estimate:.4f}; patient-level [{r.patient_lo:.4f}, {r.patient_hi:.4f}]; "
                  f"hospital-cluster [{r.cluster_lo:.4f}, {r.cluster_hi:.4f}] (width ratio {r.width_ratio_cluster_vs_patient:.2f})")
lines3.append("- Heterogeneity: " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in het.items()))
with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "a") as f: f.write("\n".join(lines3))
shutil.make_archive(os.path.join(os.path.dirname(OUT), "cluster_outputs"), "zip", OUT)
print("\n".join(lines3)); print(s1.to_string())

