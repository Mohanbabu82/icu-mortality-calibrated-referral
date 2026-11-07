# %% [markdown]
# # Confidence-Calibrated Machine Learning for Reliable In-Hospital Mortality Prediction
# ### Full study pipeline: WiDS Datathon 2020 (GOSSIS) ICU cohort
#
# This notebook runs the entire study end to end and writes every table and figure needed to draft the manuscript into
# `/kaggle/working/paper_outputs/` (also zipped as `paper_outputs.zip`).
#
# **Before you run**
# 1. On the Kaggle notebook page click **Add Input → Competitions → "WiDS Datathon 2020"** (accept the competition rules once).
#    The file `training_v2.csv` will appear under `/kaggle/input/`. The notebook finds it automatically.
# 2. For the paper, also register on PhysioNet and sign the data-use agreement for
#    *WiDS Datathon 2020: ICU Mortality Prediction* (physionet.org/content/widsdatathon2020/) and cite that DOI.
#    Never re-upload the data as a public Kaggle dataset.
# 3. **Run All**. CPU is enough; runtime is roughly 25–45 minutes with 1,000 bootstrap resamples.
#
# **Study design (pre-specified, nothing is tuned on test data)**
# | Element | Choice |
# |---|---|
# | Outcome | `hospital_death` (in-hospital mortality) |
# | Predictors | Demographics, admission info, comorbidities, APACHE physiology, first-hour (`h1_`) and first-day (`d1_`) vitals/labs |
# | Excluded as leakage | `apache_4a_hospital_death_prob`, `apache_4a_icu_death_prob` (used only as the clinical benchmark), all IDs |
# | External validation | 20% of hospitals held out entirely (hospital-level split) |
# | Development hospitals | 70% train / 15% calibration / 15% internal test (stratified) |
# | Tuning | Cross-validation inside the training set only (log loss) |
# | Calibrators | Fitted on the calibration set only: Platt, beta, isotonic, DWAC, DWAC-M (monotone) |
# | Primary analysis | Gradient boosting + DWAC-M, external test set |

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

OUT = os.environ.get("OUT_DIR", "/kaggle/working/paper_outputs")
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

lr_grid = [{"C": c} for c in ([0.01, 0.1, 1.0] if FAST else [0.001, 0.01, 0.1, 1.0, 10.0])]
hgb_grid = [{"learning_rate": lr, "max_leaf_nodes": ml, "min_samples_leaf": ms, "l2_regularization": l2}
            for lr in [0.05] for ml in ([15, 31] if FAST else [15, 31, 63]) for ms in ([50] if FAST else [20, 100])
            for l2 in ([1.0] if FAST else [0.0, 1.0])]

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

if HAVE_LGB:
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
# ## 9. Main results with stratified bootstrap 95% CIs (Table 3)

# %%
fns = boot_metric_fns()
def boot_ci(y, p, n_boot, seed, metrics=BOOT_METRICS):
    vals = {m: [] for m in metrics}
    for idx in stratified_boot_indices(y, n_boot, seed):
        for m in metrics:
            vals[m].append(fns[m](y[idx], p[idx]))
    return {m: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for m, v in vals.items()}

main_rows = []
APACHE = {k: SPLITS[k][APACHE_PROB].to_numpy(float) if APACHE_PROB in df.columns else np.full(len(SPLITS[k]), np.nan) for k in SPLITS}
for k in TESTS:
    y = Y[k]
    for name in MODELS:
        for meth in CALIBRATORS:
            p = PRED[(name, meth, k)]
            pt = all_metrics(y, p); ci = boot_ci(y, p, N_BOOT, SEED)
            row = {"test_set": k, "model": name, "method": meth, "n": len(y), "events": int(y.sum())}
            for m, v in pt.items():
                row[m] = v
                if m in ci: row[f"{m}_lo"], row[f"{m}_hi"] = ci[m]
            main_rows.append(row)
        log(f"{k} / {name}: done")
main = pd.DataFrame(main_rows)
save_table(main, "table3_main_results_full")

def fmt(r, m, d=3):
    return f"{r[m]:.{d}f} [{r[m + '_lo']:.{d}f}, {r[m + '_hi']:.{d}f}]"
compact = main.assign(**{
    "AUROC": main.apply(lambda r: fmt(r, "auroc"), axis=1), "AUPRC": main.apply(lambda r: fmt(r, "auprc"), axis=1),
    "Brier": main.apply(lambda r: fmt(r, "brier", 4), axis=1), "ECE": main.apply(lambda r: fmt(r, "ece_10", 4), axis=1),
    "ICI": main.apply(lambda r: fmt(r, "ici", 4), axis=1),
    "Intercept": main.apply(lambda r: fmt(r, "cal_intercept"), axis=1), "Slope": main.apply(lambda r: fmt(r, "cal_slope"), axis=1),
})[["test_set", "model", "method", "AUROC", "AUPRC", "Brier", "ECE", "ICI", "Intercept", "Slope"]]
save_table(compact, "table3_main_results_formatted")
compact

# %% [markdown]
# ## 10. Paired bootstrap comparisons between calibration methods (Table 4)

# %%
PRIMARY_MODEL = "gradient_boosting"
PRIMARY_METHOD = "dwac_m"
pair_list = [("dwac_m", m) for m in ["uncalibrated", "platt", "beta", "isotonic", "dwac"]] + [("isotonic", "platt"), ("beta", "platt")]
cmp_metrics = ["auroc", "brier", "ece_10", "ici", "cal_slope"]
pair_rows = []
for k in TESTS:
    y = Y[k]
    for name in MODELS:
        boots = list(stratified_boot_indices(y, N_BOOT, SEED + 7))
        for a, b in pair_list:
            pa, pb = PRED[(name, a, k)], PRED[(name, b, k)]
            row = {"test_set": k, "model": name, "comparison": f"{a} − {b}"}
            for m in cmp_metrics:
                f = fns[m]; diffs = np.array([f(y[i], pa[i]) - f(y[i], pb[i]) for i in boots])
                lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
                row.update({f"{m}_diff": f(y, pa) - f(y, pb), f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_sig": bool(lo > 0 or hi < 0)})
            pair_rows.append(row)
pairs = pd.DataFrame(pair_rows)
save_table(pairs, "table4_paired_method_comparisons")

# model comparison (DeLong) on uncalibrated scores
dl_rows = []
names = list(MODELS)
for k in TESTS:
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a1, a2, z, pv = delong_test(Y[k], PRED_RAW[names[i]][k], PRED_RAW[names[j]][k])
            dl_rows.append({"test_set": k, "model_a": names[i], "model_b": names[j], "auroc_a": a1, "auroc_b": a2, "z": z, "p_value": pv})
save_table(pd.DataFrame(dl_rows), "table4b_delong_model_comparison")
pairs.loc[pairs["model"] == PRIMARY_MODEL, ["test_set", "comparison"] + [c for c in pairs.columns if c.endswith("_diff") or c.endswith("_sig")]]

# %% [markdown]
# ## 11. Clinical benchmark: ML vs APACHE IV (Table 5)

# %%
bench_rows = []
for k in TESTS:
    ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0) & (ap <= 1)
    y = Y[k][ok]
    if ok.sum() < 100 or y.sum() < 10:
        continue
    cands = {"APACHE IV": clip(ap[ok])}
    for name in MODELS:
        cands[f"{name} / uncalibrated"] = PRED[(name, "uncalibrated", k)][ok]
        cands[f"{name} / {PRIMARY_METHOD}"] = PRED[(name, PRIMARY_METHOD, k)][ok]
    for lab, p in cands.items():
        pt = all_metrics(y, p); ci = boot_ci(y, p, N_BOOT_SMALL, SEED + 3, ["auroc", "brier", "ece_10", "ici"])
        row = {"test_set": k, "predictor": lab, "n_with_apache": int(ok.sum()), "events": int(y.sum()),
               **{m: pt[m] for m in ["auroc", "auprc", "brier", "scaled_brier", "ece_10", "ici", "cal_intercept", "cal_slope"]}}
        for m, (lo, hi) in ci.items(): row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        if lab != "APACHE IV":
            _, _, z, pv = delong_test(y, p, cands["APACHE IV"]); row["delong_p_vs_apache"] = pv
        bench_rows.append(row)
bench = pd.DataFrame(bench_rows)
save_table(bench, "table5_apache_iv_benchmark")
bench[[c for c in bench.columns if not c.endswith("_lo") and not c.endswith("_hi")]] if len(bench) else "APACHE IV column unavailable"

# %% [markdown]
# ## 12. Selective prediction / clinician referral (Table 6)
# Three uncertainty scores decide which patients are referred for full clinician review:
# * **Entropy** of the calibrated probability (original repository approach; equivalent to distance from 0.5)
# * **Threshold distance**: |logit(p) − logit(t)|, distance from the pre-specified alert threshold t = 0.20
# * **Ensemble disagreement**: SD across 10 bagged gradient-boosting models (epistemic uncertainty)

# %%
log(f"Training {N_ENSEMBLE} bagged models for epistemic uncertainty")
ens_preds = {k: [] for k in TESTS}
r_ens = np.random.default_rng(SEED + 11)
for e in range(N_ENSEMBLE):
    idx = r_ens.choice(len(Y["train"]), len(Y["train"]), replace=True)
    m = make_hgb(hgb_best, seed=SEED + e).fit(XT["train"].iloc[idx], Y["train"][idx])
    for k in TESTS:
        ens_preds[k].append(m.predict_proba(XT[k])[:, 1])
ENS_SD = {k: np.std(np.vstack(v), axis=0) for k, v in ens_preds.items()}

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

def aurc(y, p, unc, metric="selective_brier"):
    cov = np.linspace(0.1, 1.0, 46)
    rc = risk_coverage(y, p, unc, cov)
    return float(np.trapezoid(rc[metric], cov) / 0.9) if hasattr(np, "trapezoid") else float(np.trapz(rc[metric], cov) / 0.9)

ref_rows, aurc_rows, curves = [], [], {}
for k in TESTS:
    y = Y[k]
    for meth in ["uncalibrated", "platt", "isotonic", PRIMARY_METHOD]:
        p = PRED[(PRIMARY_MODEL, meth, k)]
        for sname, unc in referral_scores(p, k).items():
            rc = risk_coverage(y, p, unc, [1.0, 0.95, 0.90, 0.80, 0.70])
            ref_rows.append(rc.assign(test_set=k, model=PRIMARY_MODEL, method=meth, referral_score=sname))
            curves[(k, meth, sname)] = risk_coverage(y, p, unc, np.linspace(0.1, 1.0, 46))
            boots = [aurc(y[i], p[i], unc[i]) for i in stratified_boot_indices(y, N_BOOT_SMALL, SEED + 5)]
            aurc_rows.append({"test_set": k, "model": PRIMARY_MODEL, "method": meth, "referral_score": sname,
                              "aurc_brier": aurc(y, p, unc), "aurc_lo": np.percentile(boots, 2.5), "aurc_hi": np.percentile(boots, 97.5),
                              "aurc_error": aurc(y, p, unc, "selective_error")})
ref_df = pd.concat(ref_rows, ignore_index=True)
cols = ["test_set", "model", "method", "referral_score"] + [c for c in ref_df.columns if c not in ("test_set", "model", "method", "referral_score")]
save_table(ref_df[cols], "table6_referral_performance")
aurc_df = pd.DataFrame(aurc_rows).sort_values(["test_set", "aurc_brier"])
save_table(aurc_df, "table6b_aurc")
aurc_df

# %% [markdown]
# ## 13. Operating points (Table 7) and decision-curve analysis

# %%
thr_rows = []
for k in TESTS:
    y = Y[k]
    for meth in ["uncalibrated", PRIMARY_METHOD]:
        p = PRED[(PRIMARY_MODEL, meth, k)]
        for t in REPORT_THRESHOLDS:
            pr = (p >= t).astype(int)
            tp = ((pr == 1) & (y == 1)).sum(); fn = ((pr == 0) & (y == 1)).sum()
            tn = ((pr == 0) & (y == 0)).sum(); fp = ((pr == 1) & (y == 0)).sum()
            thr_rows.append({"test_set": k, "method": meth, "threshold": t, "flagged_pct": pr.mean() * 100,
                             "sensitivity": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp),
                             "ppv": tp / max(1, tp + fp), "npv": tn / max(1, tn + fn),
                             "net_benefit": tp / len(y) - fp / len(y) * t / (1 - t)})
save_table(pd.DataFrame(thr_rows), "table7_operating_points")

def net_benefit(y, p, ts):
    n = len(y); out = []
    for t in ts:
        pr = p >= t; tp = np.sum(pr & (y == 1)); fp = np.sum(pr & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)
DCA_T = np.linspace(0.01, 0.6, 60)
dca_rows = []
for k in TESTS:
    y = Y[k]; prev = y.mean()
    dca_rows.append(pd.DataFrame({"test_set": k, "strategy": "treat_all", "threshold": DCA_T, "net_benefit": prev - (1 - prev) * DCA_T / (1 - DCA_T)}))
    dca_rows.append(pd.DataFrame({"test_set": k, "strategy": "treat_none", "threshold": DCA_T, "net_benefit": 0.0}))
    for meth in ["uncalibrated", PRIMARY_METHOD]:
        dca_rows.append(pd.DataFrame({"test_set": k, "strategy": f"{PRIMARY_MODEL}/{meth}", "threshold": DCA_T,
                                      "net_benefit": net_benefit(y, PRED[(PRIMARY_MODEL, meth, k)], DCA_T)}))
    ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0) & (ap <= 1)
    if ok.mean() > 0.5:
        # restricted to patients with an APACHE score; ML curve recomputed on the same patients for fairness
        dca_rows.append(pd.DataFrame({"test_set": k, "strategy": "APACHE IV (subset)", "threshold": DCA_T, "net_benefit": net_benefit(y[ok], ap[ok], DCA_T)}))
        dca_rows.append(pd.DataFrame({"test_set": k, "strategy": f"{PRIMARY_MODEL}/{PRIMARY_METHOD} (APACHE subset)", "threshold": DCA_T,
                                      "net_benefit": net_benefit(y[ok], PRED[(PRIMARY_MODEL, PRIMARY_METHOD, k)][ok], DCA_T)}))
dca_df = pd.concat(dca_rows, ignore_index=True)
save_table(dca_df, "table_S6_decision_curve")

# %% [markdown]
# ## 14. Subgroup analysis (Table 8): discrimination and calibration by sex, age, ethnicity, ICU type, admission

# %%
def age_band(a):
    return pd.cut(a, [17, 44, 64, 79, 200], labels=["18-44", "45-64", "65-79", "80+"]).astype(str)
SUBGROUP_COLS = {"sex": "gender", "age_band": "age", "ethnicity": "ethnicity", "icu_type": "icu_type",
                 "elective_surgery": "elective_surgery", "diagnosis_system": "apache_3j_bodysystem"}
sub_rows = []
for k in TESTS:
    fr = SPLITS[k]; y = Y[k]; p = PRED[(PRIMARY_MODEL, PRIMARY_METHOD, k)]; pu = PRED[(PRIMARY_MODEL, "uncalibrated", k)]
    for lab, col in SUBGROUP_COLS.items():
        if col not in fr.columns: continue
        g = age_band(fr[col]) if lab == "age_band" else fr[col].astype(str)
        for lvl in g.unique():
            m = (g == lvl).to_numpy()
            if m.sum() < 100 or y[m].sum() < 10 or lvl in ("nan", "None"):
                sub_rows.append({"test_set": k, "subgroup": lab, "level": lvl, "n": int(m.sum()), "events": int(y[m].sum()), "reported": False}); continue
            ci = boot_ci(y[m], p[m], N_BOOT_SMALL, SEED + 9, ["auroc", "ece_10"])
            a, b = cal_intercept_slope(y[m], p[m])
            sub_rows.append({"test_set": k, "subgroup": lab, "level": lvl, "n": int(m.sum()), "events": int(y[m].sum()), "reported": True,
                             "observed_mortality": y[m].mean(), "mean_predicted": p[m].mean(),
                             "auroc": auroc(y[m], p[m]), "auroc_lo": ci["auroc"][0], "auroc_hi": ci["auroc"][1],
                             "ece_10": ece(y[m], p[m]), "ece_lo": ci["ece_10"][0], "ece_hi": ci["ece_10"][1],
                             "cal_intercept": a, "cal_slope": b, "ece_10_uncalibrated": ece(y[m], pu[m]), "brier": brier(y[m], p[m])})
sub_df = pd.DataFrame(sub_rows)
save_table(sub_df, "table8_subgroups")
sub_df.loc[sub_df["reported"]].head(20)

# %% [markdown]
# ## 15. Feature-group ablation (Table 9) and class-weighting sensitivity analysis (Table 10)

# %%
abl_rows = []
y = Y["internal_test"]
full_p = PRED[(PRIMARY_MODEL, "uncalibrated", "internal_test")]
boots = list(stratified_boot_indices(y, N_BOOT_SMALL, SEED + 13))
configs = [("Full model", [])] + [(f"Without {g}", cols_) for g, cols_ in GROUPS.items()] + \
          [(f"Only {g}", [c for c in NUM_COLS + CAT_COLS if c not in cols_]) for g, cols_ in GROUPS.items()]
for lab, drop in configs:
    if lab == "Full model":
        p = full_p
    else:
        keep_num = [c for c in NUM_COLS if c not in drop]; keep_cat = [c for c in CAT_COLS if c not in drop]
        if not keep_num + keep_cat: continue
        Xtr, cm = tree_matrix(SPLITS["train"], keep_num, keep_cat); Xte, _ = tree_matrix(SPLITS["internal_test"], keep_num, keep_cat)
        m = make_hgb(hgb_best).set_params(categorical_features=cm if cm.any() else None).fit(Xtr, Y["train"])
        p = m.predict_proba(Xte)[:, 1]
    diffs = [auroc(y[i], p[i]) - auroc(y[i], full_p[i]) for i in boots]
    abl_rows.append({"configuration": lab, "n_features": len(NUM_COLS + CAT_COLS) - len(drop), "auroc": auroc(y, p),
                     "brier": brier(y, p), "auroc_diff_vs_full": auroc(y, p) - auroc(y, full_p),
                     "diff_lo": np.percentile(diffs, 2.5), "diff_hi": np.percentile(diffs, 97.5)})
    log(f"ablation: {lab}")
abl = pd.DataFrame(abl_rows)
abl["diff_significant"] = (abl["diff_lo"] > 0) | (abl["diff_hi"] < 0)
save_table(abl, "table9_ablation_feature_groups")

# Class-weighted training (as in the original repository) vs unweighted
cw_rows = []
weighted = {"logistic_regression": make_lr(lr_best, class_weight="balanced").fit(XL["train"], Y["train"]),
            "gradient_boosting": make_hgb(hgb_best, class_weight="balanced").fit(XT["train"], Y["train"])}
WEIGHTED_PRED = {}
for name, m in weighted.items():
    pc = m.predict_proba(DESIGN[name]["calibration"])[:, 1]
    for k in TESTS:
        praw = m.predict_proba(DESIGN[name][k])[:, 1]
        for meth in ["uncalibrated", "platt", "dwac_m"]:
            pp = CALIBRATORS[meth]().fit(pc, Y["calibration"]).transform(praw)
            WEIGHTED_PRED[(name, meth, k)] = pp
            for train_w, pr_ in [("balanced", pp), ("unweighted", PRED[(name, meth, k)])]:
                mm = all_metrics(Y[k], pr_, ici=True)
                cw_rows.append({"test_set": k, "model": name, "training_weights": train_w, "method": meth,
                                **{x: mm[x] for x in ["auroc", "brier", "ece_10", "ici", "cal_intercept", "cal_slope", "mean_pred", "obs_rate"]}})
cw = pd.DataFrame(cw_rows).drop_duplicates()
save_table(cw, "table10_class_weighting_sensitivity")
abl

# %% [markdown]
# ## 16. Feature importance (permutation importance; SHAP if available)

# %%
Xi = XT["internal_test"]; yi = Y["internal_test"]
sub = np.random.default_rng(SEED).choice(len(yi), min(len(yi), 6000), replace=False)
pi = permutation_importance(MODELS[PRIMARY_MODEL], Xi.iloc[sub], yi[sub], scoring="neg_log_loss",
                            n_repeats=3 if FAST else 5, random_state=SEED)
imp = pd.DataFrame({"feature": Xi.columns, "group": [feature_group(c) for c in Xi.columns],
                    "importance_mean": pi.importances_mean, "importance_sd": pi.importances_std}).sort_values("importance_mean", ascending=False)
save_table(imp, "table_S7_permutation_importance")

if HAVE_SHAP and "lightgbm" in MODELS:
    try:
        expl = shap.TreeExplainer(MODELS["lightgbm"])
        sv = expl.shap_values(Xi.iloc[sub[:3000]])
        sv = sv[1] if isinstance(sv, list) else sv
        shap.summary_plot(sv, Xi.iloc[sub[:3000]], max_display=20, show=False)
        plt.gcf().set_size_inches(8, 8); save_fig(plt.gcf(), "figure_S2_shap_summary_lightgbm")
    except Exception as e:
        log(f"SHAP skipped: {e}")
imp.head(20)

# %% [markdown]
# ## 17. Figures

# %%
# Figure 1: study flow
fig, ax = plt.subplots(figsize=(8.5, 7)); ax.axis("off")
def box(x, y, text, w=0.42, h=0.07, fc="#f3f6fa"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec="#33475b", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, wrap=True)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color="#33475b"))
fd = dict(flow)
box(0.5, 0.95, f"WiDS Datathon 2020 ICU encounters\nn = {fd['ICU encounters in source file']:,}")
box(0.5, 0.82, f"Analysis cohort (outcome recorded, adults,\nfirst encounter per patient)\nn = {len(df):,}; deaths {df[TARGET].mean():.1%}; {len(hospitals)} hospitals", h=0.09)
arrow(0.5, 0.915, 0.5, 0.865)
box(0.27, 0.65, f"Development hospitals\nn = {len(dev):,} ({len(hospitals) - len(ext_hosp)} hospitals)", w=0.4)
box(0.75, 0.65, f"External test: held-out hospitals\nn = {len(ext):,} ({len(ext_hosp)} hospitals)", w=0.42, fc="#fdf1e6")
arrow(0.5, 0.775, 0.27, 0.69); arrow(0.5, 0.775, 0.75, 0.69)
for x, lab, n_, fc in [(0.1, "Training", len(train), "#eef7ee"), (0.32, "Calibration", len(cal), "#eef1fb"), (0.54, "Internal test", len(itest), "#fdf1e6")]:
    box(x, 0.47, f"{lab}\nn = {n_:,}", w=0.19); arrow(0.27, 0.615, x, 0.51)
box(0.1, 0.30, "Model fitting\n3-fold CV tuning", w=0.19); arrow(0.1, 0.435, 0.1, 0.34)
box(0.32, 0.30, "Calibrator fitting\nPlatt · Beta · Isotonic\nDWAC · DWAC-M", w=0.21, h=0.1); arrow(0.32, 0.435, 0.32, 0.35)
box(0.76, 0.30, "Evaluation (test labels touched once)\nDiscrimination · calibration · referral\nAPACHE IV benchmark · subgroups", w=0.44, h=0.1, fc="#fdf1e6")
arrow(0.54, 0.435, 0.66, 0.35); arrow(0.75, 0.615, 0.80, 0.35)
ax.set_xlim(0, 1); ax.set_ylim(0.22, 1.0)
save_fig(fig, "figure1_study_flow")

# Figure 2: ROC + PR curves
for k in TESTS:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    y = Y[k]
    for name in MODELS:
        p = PRED[(name, PRIMARY_METHOD, k)]
        f_, t_, _ = roc_curve(y, p); axes[0].plot(f_, t_, lw=1.6, label=f"{name} (AUROC {auroc(y, p):.3f})")
        pr_, rc_, _ = precision_recall_curve(y, p); axes[1].plot(rc_, pr_, lw=1.6, label=f"{name} (AUPRC {auprc(y, p):.3f})")
    ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0)
    if ok.sum() > 100:
        f_, t_, _ = roc_curve(y[ok], ap[ok]); axes[0].plot(f_, t_, "k--", lw=1.3, label=f"APACHE IV (AUROC {auroc(y[ok], ap[ok]):.3f}, n={ok.sum():,})")
        pr_, rc_, _ = precision_recall_curve(y[ok], ap[ok]); axes[1].plot(rc_, pr_, "k--", lw=1.3, label=f"APACHE IV (AUPRC {auprc(y[ok], ap[ok]):.3f})")
    axes[0].plot([0, 1], [0, 1], ":", c="grey"); axes[0].set(xlabel="1 − specificity", ylabel="Sensitivity", title=f"ROC — {k.replace('_', ' ')}")
    axes[1].axhline(y.mean(), ls=":", c="grey"); axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision–recall — {k.replace('_', ' ')}")
    for a in axes: a.legend(fontsize=8, loc="lower right" if a is axes[0] else "upper right")
    save_fig(fig, f"figure2_roc_pr_{k}")

# Figure 3: reliability diagrams (all methods, primary model) with smoothed curves + histograms
def reliability_panel(ax, y, p, label, color, bins=10):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    mp = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]; ob = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    ax.plot(mp, ob, "o", color=color, ms=4)
    f = smoothed_calibration(y, p)
    if f is not None:
        g = np.linspace(np.quantile(p, 0.005), np.quantile(p, 0.995), 200); ax.plot(g, f(g), color=color, lw=1.8, label=label)
for k in TESTS:
    meths = list(CALIBRATORS)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8)); y = Y[k]
    for ax, meth in zip(axes.ravel(), meths):
        p = PRED[(PRIMARY_MODEL, meth, k)]
        reliability_panel(ax, y, p, METHOD_LABEL[meth], PALETTE[meth])
        ax.plot([0, 1], [0, 1], ":", c="grey"); lim = max(0.6, np.quantile(p, 0.995))
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted risk", ylabel="Observed mortality",
               title=f"{METHOD_LABEL[meth]}\nECE {ece(y, p):.4f} · ICI {ici_e50_e90(y, p)[0]:.4f} · slope {cal_intercept_slope(y, p)[1]:.2f}")
        ax2 = ax.twinx(); ax2.hist(p, bins=40, range=(0, lim), color=PALETTE[meth], alpha=0.15); ax2.set_yticks([]); ax2.grid(False)
    fig.suptitle(f"Calibration of {PRIMARY_MODEL.replace('_', ' ')} — {k.replace('_', ' ')} (n={len(y):,})", y=1.01)
    save_fig(fig, f"figure3_reliability_{PRIMARY_MODEL}_{k}")

# Figure 4: ML vs APACHE IV calibration
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, k in zip(axes, TESTS):
    y = Y[k]; ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0) & (ap <= 1)
    if ok.sum() > 100:
        reliability_panel(ax, y[ok], clip(ap[ok]), "APACHE IV", PALETTE["apache_iv"])
    for meth in ["uncalibrated", PRIMARY_METHOD]:
        sel = ok if ok.sum() > 100 else np.ones(len(y), bool)
        reliability_panel(ax, y[sel], PRED[(PRIMARY_MODEL, meth, k)][sel], f"GBM / {METHOD_LABEL[meth]}", PALETTE[meth])
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="Predicted risk", ylabel="Observed mortality",
                                                   title=k.replace("_", " ")); ax.legend(fontsize=8)
save_fig(fig, "figure4_calibration_vs_apache")

# Figure 5: calibration maps (method illustration) + DWAC weight
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
g = np.linspace(0.001, 0.95, 2000)
for meth in ["platt", "beta", "isotonic", "dwac", "dwac_m"]:
    axes[0].plot(g, CAL_OBJ[(PRIMARY_MODEL, meth)].transform(g), color=PALETTE[meth], lw=1.4, label=METHOD_LABEL[meth])
axes[0].plot([0, 1], [0, 1], ":", c="grey"); axes[0].set(xlim=(0, 0.95), ylim=(0, 1), xlabel="Uncalibrated model probability", ylabel="Calibrated probability", title="Learned calibration maps")
axes[0].legend(fontsize=8)
dw = CAL_OBJ[(PRIMARY_MODEL, "dwac")]
axes[1].plot(g, dw.weight(g), color=PALETTE["dwac"], lw=1.6, label="DWAC isotonic weight w(p)")
axes[1].hist(PRED_RAW[PRIMARY_MODEL]["calibration"], bins=60, range=(0, 0.95), density=True, alpha=0.25, color="grey", label="Calibration-set density")
axes[1].set(xlabel="Uncalibrated model probability", ylabel="Weight / density", title="DWAC density weight"); axes[1].legend(fontsize=8)
save_fig(fig, "figure5_calibration_maps_dwac")

# Figure 6: risk–coverage curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, k in zip(axes, TESTS):
    for sname, ls in [("entropy", "-"), ("threshold_distance", "--"), ("ensemble_sd", ":")]:
        for meth in ["uncalibrated", PRIMARY_METHOD]:
            c = curves[(k, meth, sname)]
            ax.plot(c["coverage"], c["selective_brier"], ls, color=PALETTE[meth], lw=1.5, label=f"{METHOD_LABEL[meth]} · {sname}")
    ax.invert_xaxis(); ax.set(xlabel="Coverage (fraction not referred)", ylabel="Brier score on accepted patients", title=k.replace("_", " "))
    ax.legend(fontsize=7)
save_fig(fig, "figure6_risk_coverage")

# Figure 7: decision curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, k in zip(axes, TESTS):
    d = dca_df[dca_df["test_set"] == k]
    for s, grp in d.groupby("strategy"):
        ax.plot(grp["threshold"], grp["net_benefit"], lw=1.4, label=s,
                color="grey" if s.startswith("treat") else None, ls=":" if s.startswith("treat") else "-")
    ax.set(ylim=(-0.01, max(0.02, d["net_benefit"].max() * 1.1)), xlabel="Threshold probability", ylabel="Net benefit", title=k.replace("_", " "))
    ax.legend(fontsize=7)
save_fig(fig, "figure7_decision_curves")

# Figure 8: subgroup forest plot (external test)
sd = sub_df[(sub_df["reported"]) & (sub_df["test_set"] == "external_test")].reset_index(drop=True)
if len(sd):
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(sd) + 1.5), sharey=True)
    yy = np.arange(len(sd))[::-1]; lbl = sd["subgroup"] + ": " + sd["level"].astype(str) + "  (n=" + sd["n"].astype(str) + ")"
    axes[0].errorbar(sd["auroc"], yy, xerr=[(sd["auroc"] - sd["auroc_lo"]).clip(lower=0), (sd["auroc_hi"] - sd["auroc"]).clip(lower=0)], fmt="o", c="#1f77b4", ms=4)
    axes[0].axvline(auroc(Y["external_test"], PRED[(PRIMARY_MODEL, PRIMARY_METHOD, "external_test")]), ls="--", c="grey")
    axes[0].set(yticks=yy, yticklabels=lbl, xlabel="AUROC (95% CI)")
    axes[1].errorbar(sd["ece_10"], yy, xerr=[(sd["ece_10"] - sd["ece_lo"]).clip(lower=0), (sd["ece_hi"] - sd["ece_10"]).clip(lower=0)], fmt="o", c=PALETTE["dwac_m"], ms=4, label="DWAC-M")
    axes[1].plot(sd["ece_10_uncalibrated"], yy, "x", c=PALETTE["uncalibrated"], label="Uncalibrated"); axes[1].set(xlabel="ECE (95% CI)"); axes[1].legend(fontsize=8)
    fig.suptitle("Subgroup performance — external test (held-out hospitals)")
    save_fig(fig, "figure8_subgroups_external")

# Figure 9: feature importance
top = imp.head(20).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.5, 6.5))
grp_col = {g: c for g, c in zip(GROUPS, plt.cm.tab10.colors)}
ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_sd"], color=[grp_col.get(g, "grey") for g in top["group"]])
ax.set(xlabel="Increase in log loss when permuted", title="Top 20 predictors (permutation importance, internal test)")
for g, c in grp_col.items():
    ax.barh([], [], color=c, label=g)
ax.legend(fontsize=7, loc="lower right"); save_fig(fig, "figure9_feature_importance")

# Figure S1: class weighting effect
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, name in zip(axes, ["logistic_regression", "gradient_boosting"]):
    y = Y["external_test"]
    reliability_panel(ax, y, PRED[(name, "uncalibrated", "external_test")], "Unweighted training", "#1f77b4")
    reliability_panel(ax, y, WEIGHTED_PRED[(name, "uncalibrated", "external_test")], "Class-balanced training", "#d62728")
    reliability_panel(ax, y, WEIGHTED_PRED[(name, "dwac_m", "external_test")], "Class-balanced + DWAC-M", "#ff7f0e")
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted risk", ylabel="Observed mortality", title=name.replace("_", " "))
    ax.legend(fontsize=8)
save_fig(fig, "figure_S1_class_weighting_calibration")
log("Figures written")

# %% [markdown]
# ## 18. Auto-generated results summary for drafting + run metadata

# %%
def g(k, meth, metric, model=PRIMARY_MODEL):
    r = main[(main.test_set == k) & (main.model == model) & (main.method == meth)].iloc[0]
    return r[metric], r.get(f"{metric}_lo", np.nan), r.get(f"{metric}_hi", np.nan)
def s(t, d=3): return f"{t[0]:.{d}f} (95% CI {t[1]:.{d}f}–{t[2]:.{d}f})"

lines = [f"# Results summary (auto-generated {time.strftime('%Y-%m-%d %H:%M')})", "",
         "Every number below is computed in this run from real WiDS Datathon 2020 data. Copy into the manuscript and re-check against the CSV tables.", "",
         "## Cohort",
         f"- Source file: {raw.shape[0]:,} ICU encounters; analysis cohort {len(df):,} patients from {len(hospitals)} hospitals; in-hospital mortality {df[TARGET].mean():.2%}.",
         f"- Training {len(train):,}; calibration {len(cal):,}; internal test {len(itest):,} (development hospitals); external test {len(ext):,} from {len(ext_hosp)} held-out hospitals (mortality {ext[TARGET].mean():.2%}).",
         f"- Predictors: {len(NUM_COLS)} numeric and {len(CAT_COLS)} categorical; APACHE IV predicted-mortality columns excluded from predictors (used only as benchmark).", "",
         "## Discrimination and calibration (primary model: gradient boosting)"]
for k in TESTS:
    lines.append(f"- **{k.replace('_', ' ')}**: AUROC {s(g(k, PRIMARY_METHOD, 'auroc'))}; AUPRC {s(g(k, PRIMARY_METHOD, 'auprc'))}.")
    for meth in CALIBRATORS:
        lines.append(f"  - {METHOD_LABEL[meth]}: Brier {s(g(k, meth, 'brier'), 4)}, ECE {s(g(k, meth, 'ece_10'), 4)}, ICI {s(g(k, meth, 'ici'), 4)}, "
                     f"intercept {g(k, meth, 'cal_intercept')[0]:.3f}, slope {g(k, meth, 'cal_slope')[0]:.3f}.")
lines += ["", "## DWAC-M vs alternatives (paired bootstrap, external test, gradient boosting)"]
for _, r in pairs[(pairs.test_set == "external_test") & (pairs.model == PRIMARY_MODEL)].iterrows():
    lines.append(f"- {r['comparison']}: ΔBrier {r['brier_diff']:+.4f} [{r['brier_lo']:+.4f}, {r['brier_hi']:+.4f}]{' *' if r['brier_sig'] else ''}; "
                 f"ΔECE {r['ece_10_diff']:+.4f} [{r['ece_10_lo']:+.4f}, {r['ece_10_hi']:+.4f}]{' *' if r['ece_10_sig'] else ''}; "
                 f"ΔICI {r['ici_diff']:+.4f} [{r['ici_lo']:+.4f}, {r['ici_hi']:+.4f}]{' *' if r['ici_sig'] else ''}  (* = CI excludes 0)")
mv = mono_df[(mono_df.model == PRIMARY_MODEL)].set_index("method")
lines += ["", "## Monotonicity", f"- DWAC (original) decreasing steps on a 20,000-point grid: {mv.loc['dwac', 'grid_decreasing_steps']}; "
          f"rank reversals on external test: {mv.loc['dwac', 'rank_reversals_external_test']}. DWAC-M: {mv.loc['dwac_m', 'grid_decreasing_steps']} and {mv.loc['dwac_m', 'rank_reversals_external_test']}."]
if len(bench):
    lines += ["", "## APACHE IV benchmark"]
    for _, r in bench.iterrows():
        lines.append(f"- {r['test_set']} · {r['predictor']} (n={r['n_with_apache']:,}): AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}], "
                     f"Brier {r['brier']:.4f}, ECE {r['ece_10']:.4f}, slope {r['cal_slope']:.2f}" + (f", DeLong p vs APACHE {r['delong_p_vs_apache']:.2g}" if pd.notna(r.get('delong_p_vs_apache', np.nan)) else ""))
lines += ["", "## Referral (AURC on Brier; lower is better)"]
for _, r in aurc_df.iterrows():
    lines.append(f"- {r['test_set']} · {r['method']} · {r['referral_score']}: {r['aurc_brier']:.4f} [{r['aurc_lo']:.4f}, {r['aurc_hi']:.4f}]")
r80 = ref_df[(ref_df.test_set == "external_test") & (ref_df.method == PRIMARY_METHOD) & (np.isclose(ref_df.coverage, 0.8))]
for _, r in r80.iterrows():
    lines.append(f"- External, 20% referred by {r['referral_score']}: Brier on accepted {r['selective_brier']:.4f}; {r['deaths_referred_pct']:.1f}% of deaths routed to clinician review.")
cwx = cw[(cw.test_set == "external_test") & (cw.method == "uncalibrated")]
lines += ["", "## Class-weighting sensitivity (external test, uncalibrated)"] + \
         [f"- {r['model']} · {r['training_weights']}: intercept {r['cal_intercept']:.3f}, slope {r['cal_slope']:.3f}, mean predicted {r['mean_pred']:.3f} vs observed {r['obs_rate']:.3f}"
          for _, r in cwx.iterrows()]
lines += ["", "## Ablation (internal test, AUROC change vs full model)"] + \
         [f"- {r['configuration']}: {r['auroc']:.3f} (Δ {r['auroc_diff_vs_full']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}])" for _, r in abl.iterrows()]
lines += ["", "## Top predictors", ", ".join(imp.head(10)["feature"])]
with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w") as f:
    f.write("\n".join(lines))

meta = {"run_finished": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_minutes": round((time.time() - T0) / 60, 1), "data_file": DATA_PATH,
        "seed": SEED, "n_boot": N_BOOT, "n_boot_small": N_BOOT_SMALL, "n_ensemble": N_ENSEMBLE, "fast_mode": FAST,
        "decision_threshold": DECISION_THRESHOLD, "external_hospital_fraction": EXTERNAL_HOSPITAL_FRACTION,
        "external_hospital_ids": sorted(int(h) if str(h).isdigit() else str(h) for h in ext_hosp),
        "excluded_leakage_columns": LEAK_COLS, "dropped_constant_columns": constant, "id_columns": ID_COLS,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                     "lightgbm": lgb.__version__ if HAVE_LGB else None}}
with open(os.path.join(OUT, "run_metadata.json"), "w") as f:
    json.dump(meta, f, indent=2, default=str)

shutil.make_archive(os.path.join(os.path.dirname(OUT), "paper_outputs"), "zip", OUT)
log(f"Done. {len(os.listdir(TAB))} table files, {len(os.listdir(FIG))} figure files → {OUT} and paper_outputs.zip")
print("\n".join(lines[:40]))

# %% [markdown]
# ## 19. Reviewer analyses: (a) recalibrated APACHE IV benchmark, (b) repeated hospital-held-out splits
# (a) APACHE IV is recalibrated with Platt scaling fitted on the same calibration set as the ML calibrators, so the
#     ML-vs-APACHE comparison is not driven only by local recalibration.
# (b) The whole external-validation design is repeated with 5 further random draws of held-out hospitals. Hyperparameters
#     are fixed to the values selected in the primary analysis; category maps and calibrators are re-learned per split.

# %%
# (a) Recalibrated APACHE IV
ap_cal = APACHE["calibration"]; ok_cal = np.isfinite(ap_cal) & (ap_cal >= 0) & (ap_cal <= 1)
apache_recal = Platt().fit(clip(ap_cal[ok_cal]), Y["calibration"][ok_cal])
rec_rows = []
for k in TESTS:
    ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0) & (ap <= 1); y = Y[k][ok]
    cands = {"APACHE IV (original)": clip(ap[ok]), "APACHE IV (recalibrated, Platt)": apache_recal.transform(clip(ap[ok])),
             f"{PRIMARY_MODEL} / {PRIMARY_METHOD}": PRED[(PRIMARY_MODEL, PRIMARY_METHOD, k)][ok]}
    if "lightgbm" in MODELS:
        cands[f"lightgbm / {PRIMARY_METHOD}"] = PRED[("lightgbm", PRIMARY_METHOD, k)][ok]
    ref = cands["APACHE IV (recalibrated, Platt)"]
    boots = list(stratified_boot_indices(y, N_BOOT_SMALL, SEED + 21))
    for lab, p in cands.items():
        pt = all_metrics(y, p); ci = boot_ci(y, p, N_BOOT_SMALL, SEED + 21, ["auroc", "brier", "ece_10", "ici"])
        row = {"test_set": k, "predictor": lab, "n": int(ok.sum()), "events": int(y.sum()),
               **{m: pt[m] for m in ["auroc", "auprc", "brier", "scaled_brier", "ece_10", "ici", "cal_intercept", "cal_slope"]}}
        for m, (lo, hi) in ci.items(): row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        if "APACHE" not in lab:
            row["delong_p_vs_recal_apache"] = delong_test(y, p, ref)[3]
            for m in ["brier", "ece_10", "ici"]:
                f = fns[m]; d = np.array([f(y[i], p[i]) - f(y[i], ref[i]) for i in boots])
                row[f"{m}_diff_vs_recal_apache"] = f(y, p) - f(y, ref)
                row[f"{m}_diff_lo"], row[f"{m}_diff_hi"] = np.percentile(d, [2.5, 97.5])
        rec_rows.append(row)
rec_df = pd.DataFrame(rec_rows)
save_table(rec_df, "table5b_apache_recalibrated_benchmark")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, k in zip(axes, TESTS):
    ap = APACHE[k]; ok = np.isfinite(ap) & (ap >= 0) & (ap <= 1); y = Y[k][ok]
    reliability_panel(ax, y, clip(ap[ok]), "APACHE IV (original)", "#000000")
    reliability_panel(ax, y, apache_recal.transform(clip(ap[ok])), "APACHE IV (recalibrated)", "#8c564b")
    reliability_panel(ax, y, PRED[(PRIMARY_MODEL, PRIMARY_METHOD, k)][ok], "GBM / DWAC-M", PALETTE["dwac_m"])
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="Predicted risk", ylabel="Observed mortality", title=k.replace("_", " "))
    ax.legend(fontsize=8)
save_fig(fig, "figure_S3_apache_recalibrated")

# (b) Repeated hospital-held-out splits
def make_cat_maps(tr):
    maps = {}
    for c in CAT_COLS:
        vc = tr[c].dropna().astype(str).value_counts(); maps[c] = list(vc[vc >= 30].index[:60]) + ["other"]
    return maps
def tree_matrix_maps(frame, maps):
    out = {}
    for c in CAT_COLS:
        s = frame[c]; s_str = s.astype(str).where(s.notna(), None); lv = maps[c]
        mapped = s_str.where(s_str.isin(lv[:-1]) | s_str.isna(), "other")
        out[c] = mapped.map({l: i for i, l in enumerate(lv)}).astype(float)
    return pd.concat([frame[NUM_COLS].astype(float), pd.DataFrame(out, index=frame.index)[CAT_COLS]], axis=1)

REPEAT_SEEDS = [1, 2, 3] if FAST else [101, 202, 303, 404, 505]
rep_rows = []
for sd_ in REPEAT_SEEDS:
    r_ = np.random.default_rng(sd_)
    eh = set(r_.choice(hospitals, size=max(1, int(round(EXTERNAL_HOSPITAL_FRACTION * len(hospitals)))), replace=False))
    m_ext = df["hospital_id"].isin(eh); dv, ex = df.loc[~m_ext], df.loc[m_ext]
    tr_, rs_ = train_test_split(dv, test_size=0.30, stratify=dv[TARGET], random_state=sd_)
    cl_, _ = train_test_split(rs_, test_size=0.50, stratify=rs_[TARGET], random_state=sd_)
    maps = make_cat_maps(tr_)
    Xtr_, Xcl_, Xex_ = (tree_matrix_maps(f, maps) for f in (tr_, cl_, ex))
    gbm = make_hgb(hgb_best, seed=sd_).fit(Xtr_, tr_[TARGET].to_numpy())
    pc, pe = gbm.predict_proba(Xcl_)[:, 1], gbm.predict_proba(Xex_)[:, 1]
    ye, yc = ex[TARGET].to_numpy(), cl_[TARGET].to_numpy()
    preds = {meth: CALIBRATORS[meth]().fit(pc, yc).transform(pe) for meth in ["uncalibrated", "platt", "beta", "isotonic", "dwac", "dwac_m"]}
    apx = ex[APACHE_PROB].to_numpy(float) if APACHE_PROB in ex.columns else np.full(len(ex), np.nan)
    okx = np.isfinite(apx) & (apx >= 0) & (apx <= 1)
    apc = cl_[APACHE_PROB].to_numpy(float) if APACHE_PROB in cl_.columns else np.full(len(cl_), np.nan)
    okc = np.isfinite(apc) & (apc >= 0) & (apc <= 1)
    if okx.sum() > 100 and okc.sum() > 100:
        preds["APACHE IV (original)"] = np.where(okx, clip(np.nan_to_num(apx, nan=0.5)), np.nan)
        preds["APACHE IV (recalibrated)"] = np.where(okx, Platt().fit(clip(apc[okc]), yc[okc]).transform(clip(np.nan_to_num(apx, nan=0.5))), np.nan)
    for meth, p in preds.items():
        sel = okx if meth.startswith("APACHE") else np.ones(len(ye), bool)
        mm = all_metrics(ye[sel], p[sel])
        row = {"split_seed": sd_, "n_external_hospitals": len(eh), "n_external": int(sel.sum()), "external_mortality": float(ye[sel].mean()),
               "predictor": meth if meth.startswith("APACHE") else f"gradient_boosting / {meth}",
               **{x: mm[x] for x in ["auroc", "auprc", "brier", "ece_10", "ici", "cal_intercept", "cal_slope"]}}
        if not meth.startswith("APACHE") and okx.sum() > 100:
            row["auroc_on_apache_subset"] = auroc(ye[okx], p[okx])
            row["delong_p_vs_original_apache"] = delong_test(ye[okx], p[okx], preds["APACHE IV (original)"][okx])[3]
        rep_rows.append(row)
    vio = monotonicity_violations(CALIBRATORS["dwac"]().fit(pc, yc))[0]
    rep_rows.append({"split_seed": sd_, "predictor": "DWAC monotonicity check", "dwac_grid_decreasing_steps": vio})
    log(f"repeated split seed {sd_}: external n={len(ex):,} from {len(eh)} hospitals, GBM AUROC {auroc(ye, preds['dwac_m']):.3f}")
rep_df = pd.DataFrame(rep_rows)
save_table(rep_df, "table11_repeated_hospital_splits")
num = ["auroc", "auprc", "brier", "ece_10", "ici", "cal_intercept", "cal_slope"]
rep_sum = rep_df[rep_df["predictor"] != "DWAC monotonicity check"].groupby("predictor")[num].agg(["mean", "std", "min", "max"])
rep_sum.columns = [f"{a}_{b}" for a, b in rep_sum.columns]
rep_sum = rep_sum.reset_index()
save_table(rep_sum, "table11b_repeated_splits_summary")

lines2 = ["", "## Reviewer analyses", "### APACHE IV recalibrated on the calibration set"]
for _, r in rec_df.iterrows():
    extra = (f"; ΔBrier vs recalibrated APACHE {r['brier_diff_vs_recal_apache']:+.4f} [{r['brier_diff_lo']:+.4f}, {r['brier_diff_hi']:+.4f}], "
             f"DeLong p {r['delong_p_vs_recal_apache']:.2g}") if pd.notna(r.get("delong_p_vs_recal_apache", np.nan)) else ""
    lines2.append(f"- {r['test_set']} · {r['predictor']} (n={r['n']:,}): AUROC {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}], "
                  f"Brier {r['brier']:.4f}, ECE {r['ece_10']:.4f}, ICI {r['ici']:.4f}, intercept {r['cal_intercept']:.3f}, slope {r['cal_slope']:.3f}{extra}")
lines2 += ["", f"### {len(REPEAT_SEEDS)} repeated hospital-held-out splits (mean ± SD across splits; external test)"]
for _, r in rep_sum.iterrows():
    lines2.append(f"- {r['predictor']}: AUROC {r['auroc_mean']:.3f} ± {r['auroc_std']:.3f} (range {r['auroc_min']:.3f}–{r['auroc_max']:.3f}); "
                  f"Brier {r['brier_mean']:.4f} ± {r['brier_std']:.4f}; ECE {r['ece_10_mean']:.4f} ± {r['ece_10_std']:.4f}; "
                  f"ICI {r['ici_mean']:.4f} ± {r['ici_std']:.4f}; slope {r['cal_slope_mean']:.3f} ± {r['cal_slope_std']:.3f}")
mono_rep = rep_df[rep_df["predictor"] == "DWAC monotonicity check"]["dwac_grid_decreasing_steps"]
lines2.append(f"- Original DWAC was non-monotone in {int((mono_rep > 0).sum())}/{len(mono_rep)} repeated splits.")
with open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "a") as f:
    f.write("\n".join(lines2))
shutil.make_archive(os.path.join(os.path.dirname(OUT), "paper_outputs"), "zip", OUT)
log("Reviewer analyses done; zip refreshed.")
print("\n".join(lines2))

# %% [markdown]
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
    for nm, u in [("threshold_distance", -np.abs(logit(pE) - logit(t))), ("entropy", entropy(pE)), ("ensemble_sd", ENS_SD["external_test"])]:
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
shutil.make_archive(os.path.join(os.path.dirname(OUT), "paper_outputs"), "zip", OUT)
print("\n".join(lines3)); print(s1.to_string())
