
import os
import re
import time
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    MinimalFCParameters,
    EfficientFCParameters,
)

# new: for chunked parquet reads
import pyarrow as pa
import pyarrow.parquet as pq
from itertools import islice

# ---------- Plot style (consistent across all figures) ----------
import seaborn as sns
sns.set(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 22,
    "axes.labelsize": 16,
    "legend.fontsize": 13,
})

# Core palette (cool, slide-friendly)
COL_MALE   = "#1f4e79"   # deep blue
COL_FEMALE = "#2ca6a4"   # teal
COL_PD     = "#2f6de1"   # vivid blue (PD)  ~ around your #3a7bd5
COL_HC     = "#b7371f"   # cool gray (HC)   ~ around your #9aa0a6 #brick red
COL_EDGE   = "#f6f8fb"   # soft white edge for dot outlines

PALETTE_LABEL = {0: COL_HC, 1: COL_PD}      # 0=Control/HC, 1=PD
MARKER_LABEL  = {0: "o",    1: "D"}         # circle for HC, diamond for PD



# ======================
# Config (env-overridable)
# ======================
class CFG:
    PROJECT_ROOT = Path(os.environ.get("GAIT_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
    DATA_RAW = Path(os.environ.get("GAIT_DATA_RAW", str(Path(__file__).resolve().parents[1] / "data" / "raw")))
    DATA_PROC = Path(os.environ.get("GAIT_DATA_PROC", str(Path(__file__).resolve().parents[1] / "data" / "processed")))
    REPORT_DIR = Path(os.environ.get("GAIT_REPORT_DIR", str(Path(__file__).resolve().parents[1] / "reports")))

    RAW_SUBFOLDER_NAME = os.environ.get("RAW_SUBFOLDER_NAME", "gait-in-parkinsons-disease-1.0.0")

    FS = float(os.environ.get("FS", "100.0"))  # Hz

    WIN_MS = int(os.environ.get("WIN_MS", "300"))   # can set 30 later with cache
    STEP_MS = int(os.environ.get("STEP_MS", "150"))

    FEATURE_SET = os.environ.get("FEATURE_SET", "comprehensive").lower()  # minimal|efficient|comprehensive
    N_JOBS = int(os.environ.get("N_JOBS", "8"))
    DIMRED = os.environ.get("DIMRED", "pca").lower()  # pca|tsne

    LEFT_COL_INDEX = int(os.environ.get("LEFT_COL_INDEX", "0"))
    RIGHT_COL_INDEX = int(os.environ.get("RIGHT_COL_INDEX", "1"))

    DEBUG_MAX_SUBJECTS = int(os.environ.get("DEBUG_MAX_SUBJECTS", "0"))

    BATCH_IDS = int(os.environ.get("BATCH_IDS", "5000"))  # tsfresh chunk size


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ms_to_samples(ms: int, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


# ======================
# Dataset discovery
# ======================
def find_dataset_root() -> Path:
    candidate = CFG.DATA_RAW / CFG.RAW_SUBFOLDER_NAME
    if candidate.exists():
        return candidate
    for p in CFG.DATA_RAW.iterdir():
        if p.is_dir() and "gait-in-parkinsons" in p.name:
            return p
    raise FileNotFoundError("Could not locate raw dataset. Set RAW_SUBFOLDER_NAME or check data/raw.")


# ======================
# Subject-ID normalization & label inference
# ======================
def normalize_sid(s: str) -> str:
    s = str(s).strip().replace(" ", "").replace("_", "")
    m = re.match(r'^(ju)c0?(\d+)$', s, flags=re.IGNORECASE)
    if m:
        s = f"JuCo{int(m.group(2))}"
    return s


def infer_group_from_sid(sid: str):
    s = normalize_sid(sid).lower()
    if re.search(r'^[a-z]{2}pt\d+$', s):
        return 1.0
    if re.search(r'^[a-z]{2}co\d+$', s):
        return 0.0
    return np.nan


def parse_subject_id_from_filename(fn: str) -> str:
    base = Path(fn).stem
    m = re.match(r'^([A-Za-z]{2}(?:Pt|Co)\d+)_\d+$', base)
    sid = m.group(1) if m else base
    return normalize_sid(sid)


# ======================
# Load demographics
# ======================
def load_demographics(raw_root: Path) -> pd.DataFrame:
    candidates = list(raw_root.glob("demographics*.*"))
    if not candidates:
        log("WARN: No demographics file found; relying on ID-based inference only.")
        return pd.DataFrame(columns=["subject_id", "group", "age", "sex", "weight"]).set_index("subject_id")

    xls = [c for c in candidates if c.suffix.lower() in (".xlsx", ".xls")]
    if xls:
        df = pd.read_excel(xls[0])
    else:
        txt = candidates[0]
        df = pd.read_csv(txt, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]
    sid_col   = next((c for c in df.columns if c.lower() in ("id", "subject", "subject id", "subject_id", "subnum")), df.columns[0])
    group_col = next((c for c in df.columns if c.lower() in ("group", "diagnosis", "diag")), None)
    age_col   = next((c for c in df.columns if "age" in c.lower()), None)
    sex_col   = next((c for c in df.columns if c.lower() in ("gender", "sex")), None)
    weight_col= next((c for c in df.columns if "weight" in c.lower()), None)

    df = df.rename(columns={sid_col: "subject_id"})
    df["subject_id"] = df["subject_id"].astype(str).map(normalize_sid)

    group = None
    if group_col is not None:
        g = df[group_col].astype(str).str.strip().str.lower()
        mapdict = {
            "pd": 1.0, "parkinson": 1.0, "parkinson's": 1.0, "1": 1.0,
            "co": 0.0, "ctrl": 0.0, "control": 0.0, "healthy": 0.0, "hc": 0.0, "0": 0.0
        }
        group = g.map(mapdict)

    if group is None:
        group = df["subject_id"].apply(infer_group_from_sid)
    else:
        miss = group.isna()
        if miss.any():
            group.loc[miss] = df.loc[miss, "subject_id"].apply(infer_group_from_sid)

    out = pd.DataFrame({"group": group})
    if age_col:    out["age"]    = pd.to_numeric(df[age_col], errors="coerce")
    if sex_col:    out["sex"]    = df[sex_col].astype(str)
    if weight_col: out["weight"] = pd.to_numeric(df[weight_col], errors="coerce")

    out["subject_id"] = df["subject_id"]
    out = out.drop_duplicates("subject_id").set_index("subject_id")
    return out


# ======================
# Load trials
# ======================
def load_trial_matrix(filepath: Path) -> np.ndarray:
    arr = np.loadtxt(str(filepath))
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def split_left_right(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if arr.shape[1] == 1:
        return arr[:, 0], arr[:, 0]
    L = arr[:, min(CFG.LEFT_COL_INDEX, arr.shape[1] - 1)]
    R = arr[:, min(CFG.RIGHT_COL_INDEX, arr.shape[1] - 1)]
    return L, R


def collect_subject_series(raw_root: Path, demog: pd.DataFrame):
    files = [p for p in raw_root.iterdir()
             if p.is_file() and re.match(r'^[A-Za-z]{2}(?:Pt|Co)\d+_\d+$', p.stem)]
    files = sorted(files, key=lambda p: p.name)

    by_subject: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for fp in tqdm(files, desc="Reading trials", unit="file"):
        sid = parse_subject_id_from_filename(fp.name)
        arr = load_trial_matrix(fp)
        L, R = split_left_right(arr)
        if sid not in by_subject:
            by_subject[sid] = {"left": [], "right": []}
        by_subject[sid]["left"].append(L)
        by_subject[sid]["right"].append(R)

    subjects = sorted(by_subject.keys())
    if CFG.DEBUG_MAX_SUBJECTS > 0:
        subjects = subjects[:CFG.DEBUG_MAX_SUBJECTS]

    rows = []
    left_series = {}
    right_series = {}
    for sid in tqdm(subjects, desc="Concatenating per subject", unit="subj"):
        left = np.concatenate(by_subject[sid]["left"]) if by_subject[sid]["left"] else np.array([])
        right = np.concatenate(by_subject[sid]["right"]) if by_subject[sid]["right"] else np.array([])
        left_series[sid] = left
        right_series[sid] = right

        g = np.nan
        age = np.nan; sex = np.nan; wt = np.nan
        if sid in demog.index:
            drow = demog.loc[sid]
            g = drow.get("group", np.nan)
            age = drow.get("age", np.nan)
            sex = drow.get("sex", np.nan)
            wt  = drow.get("weight", np.nan)
        if pd.isna(g):
            g = infer_group_from_sid(sid)

        rows.append({"subject_id": sid, "group": g, "age": age, "sex": sex, "weight": wt,
                     "n_left": left.size, "n_right": right.size})

    meta = pd.DataFrame(rows).set_index("subject_id")
    return meta, left_series, right_series


# ======================
# Windowing helpers
# ======================
def build_windows(signal: np.ndarray, fs: float, win_ms: int, step_ms: int):
    w = ms_to_samples(win_ms, fs); s = ms_to_samples(step_ms, fs)
    if len(signal) < w:
        return []
    starts = np.arange(0, len(signal) - w + 1, s, dtype=int)
    return [(signal[i:i + w], int(i)) for i in starts]


def windows_to_long(windows, series_prefix: str) -> pd.DataFrame:
    frames = []
    for win, start in windows:
        wid = f"{series_prefix}_{start}"
        df = pd.DataFrame({"id": wid, "time": np.arange(len(win)), "value": win})
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "time", "value"])


# ======================
# TSFresh feature sets (short-window safe)
# ======================
def get_fc_params(feature_set: str, win_samples: int):
    feature_set = feature_set.lower()
    if win_samples < 10:
        if feature_set in ("comprehensive", "efficient"):
            return EfficientFCParameters()
        if feature_set == "minimal":
            return MinimalFCParameters()
        return {
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "variance": None,
            "skewness": None,
            "kurtosis": None,
            "sum_values": None,
            "maximum": None,
            "minimum": None,
            "mean_abs_change": None,
            "absolute_sum_of_changes": None,
        }
    if feature_set == "minimal":
        return MinimalFCParameters()
    if feature_set == "efficient":
        return EfficientFCParameters()
    return ComprehensiveFCParameters()


# ======================
# Chunked TSFresh extraction
# ======================
def batched(iterable, n):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


def extract_tsfresh_chunked(
    parquet_path: Path,
    feature_set: str,
    n_jobs: int,
    fs: float,
    win_ms: int,
    out_final: Path,
    tmp_dir: Path,
    batch_ids: int = 5000,
):
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # unique ids (efficient, without loading whole parquet into pandas)
    table = pq.read_table(parquet_path, columns=["id"])
    all_ids = pa.compute.unique(table["id"]).to_pylist()
    all_ids = [str(x) for x in all_ids]
    total = len(all_ids)
    print(f"[TSFRESH] {parquet_path.name}: {total:,} window IDs; batch size={batch_ids}")

    win_samples = ms_to_samples(win_ms, fs)
    fc = get_fc_params(feature_set, win_samples)

    part_paths = []
    for bi, id_batch in enumerate(batched(all_ids, batch_ids), start=1):
        part_path = tmp_dir / f"part_{bi:04d}.parquet"
        if part_path.exists():
            print(f"[TSFRESH] Skip existing {part_path.name}")
            part_paths.append(part_path)
            continue

        # read only this batch
        df = pd.read_parquet(parquet_path, engine="pyarrow", filters=[("id", "in", id_batch)])

        X_part = extract_features(
            df, column_id="id", column_sort="time",
            default_fc_parameters=fc, n_jobs=n_jobs, disable_progressbar=False
        )
        X_part = X_part.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        X_part.to_parquet(part_path)
        part_paths.append(part_path)
        done = min(bi * batch_ids, total)
        print(f"[TSFRESH] Batch {bi}: ids {done:,}/{total:,} -> {part_path.name} shape={X_part.shape}")

    print("[TSFRESH] Concatenating parts...")
    X_all = pd.concat([pd.read_parquet(p) for p in part_paths], axis=0).sort_index()
    X_all.to_parquet(out_final)
    print(f"[TSFRESH] Saved {out_final}  shape={X_all.shape}")
    return X_all


# ======================
# Aggregation
# ======================
def aggregate_features(window_features: pd.DataFrame) -> pd.DataFrame:
    idx = window_features.index.to_series().astype(str)
    subj_ids = idx.apply(lambda s: s.rpartition('_')[0] if '_' in s else s)
    W = window_features.copy()
    W["__sid__"] = subj_ids.values
    num_cols = W.select_dtypes(include=[np.number]).columns
    grp = W.groupby("__sid__")
    agg_mean = grp[num_cols].mean().add_suffix("__mean")
    agg_std  = grp[num_cols].std(ddof=0).add_suffix("__std")

    def _sk(g):
        arr = g[num_cols].to_numpy()
        return pd.Series(skew(arr, axis=0, nan_policy="omit"), index=num_cols)

    def _ku(g):
        arr = g[num_cols].to_numpy()
        return pd.Series(kurtosis(arr, axis=0, nan_policy="omit"), index=num_cols)

    agg_skew = grp.apply(_sk).add_suffix("__skew")
    agg_kurt = grp.apply(_ku).add_suffix("__kurt")
    agg = pd.concat([agg_mean, agg_std, agg_skew, agg_kurt], axis=1)
    return agg


# ======================
# Dimensionality reduction & plots
# ======================
def zscore(X: pd.DataFrame) -> np.ndarray:
    sc = StandardScaler()
    return sc.fit_transform(X.values)


'''def scatter2d(Z2: np.ndarray, y: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(6,5))
    valid = ~pd.isna(y)
    yv = y[valid]; Z2v = Z2[valid]
    classes = np.unique(yv)
    for c in classes:
        m = (yv == c)
        name = "PD" if int(c) == 1 else "Control"
        plt.scatter(Z2v[m,0], Z2v[m,1], s=24, alpha=0.85, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()'''

def scatter2d(Z2: np.ndarray, y: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(7.2, 5.6))
    valid = ~pd.isna(y)
    Z2v, yv = Z2[valid], y[valid]

    # jitter a touch to avoid perfect overlaps in PCA
    jitter = np.random.default_rng(0).normal(scale=0.05, size=Z2v.shape)
    Z2v = Z2v + jitter

    for lab in np.unique(yv.astype(int)):
        m = (yv == lab)
        plt.scatter(
            Z2v[m, 0], Z2v[m, 1],
            s=64, alpha=0.9,
            c=PALETTE_LABEL.get(int(lab), "#444"),
            marker=MARKER_LABEL.get(int(lab), "o"),
            edgecolors=COL_EDGE, linewidths=0.6,
            label="PD" if int(lab) == 1 else "Control",
        )

    plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
    plt.title(title)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, borderpad=0.6)
    sns.despine()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=450)
    plt.close()



def sanitize_features_for_dimred(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    Xc = Xc.dropna(axis=1, how="all")
    nunique = Xc.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index
    if len(const_cols) > 0:
        Xc = Xc.drop(columns=list(const_cols), errors="ignore")
    Xc = Xc.select_dtypes(include=[np.number])
    if Xc.shape[1] > 0:
        med = Xc.median(numeric_only=True)
        Xc = Xc.fillna(med)
    return Xc


def strip_lr_prefix_index(X: pd.DataFrame) -> pd.DataFrame:
    X2 = X.copy()
    X2.index = X2.index.to_series().astype(str).str.replace(r'^[LR]_', '', regex=True)
    return X2


def dimred_and_plot_safe(X: pd.DataFrame, y: np.ndarray, name: str, out_dir: Path, method: str):
    if not isinstance(X, pd.DataFrame) or X.shape[0] == 0 or X.shape[1] == 0:
        print(f"[WARN] Skipping {name}: empty feature matrix after cleaning (rows={X.shape[0]}, cols={X.shape[1]}).")
        return
    Z = zscore(X)
    if method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=1000, verbose=1)
        Z2 = reducer.fit_transform(Z)
        out_png = out_dir / f"{name}_tsne.png"
        scatter2d(Z2, y, f"{name} — t-SNE", out_png)
    else:
        pca = PCA(n_components=2, random_state=0)
        Z2 = pca.fit_transform(Z)
        out_png = out_dir / f"{name}_pca.png"
        scatter2d(Z2, y, f"{name} — PCA", out_png)


# ======================
# Main
# ======================
def main():
    print("=== GAIT NOV7 TSFRESH PIPELINE ===")
    raw_root = find_dataset_root()
    print(f"Dataset root: {raw_root}")

    figs_dir = CFG.REPORT_DIR / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    CFG.DATA_PROC.mkdir(parents=True, exist_ok=True)

    print("Loading demographics...")
    demog = load_demographics(raw_root)
    demog.to_csv(CFG.DATA_PROC / "demographics_raw.csv")

    print("Collecting and concatenating trials per subject...")
    meta, left_series, right_series = collect_subject_series(raw_root, demog)
    meta.index = meta.index.map(normalize_sid)
    meta["group"] = meta["group"].astype(float)
    meta.loc[meta["group"].isna(), "group"] = [infer_group_from_sid(s) for s in meta.index]

    study_prefix = meta.index.to_series().str[:2]
    print("Label distribution (0=Control, 1=PD):")
    print(meta["group"].value_counts(dropna=False).to_string())
    print("\nCross-tab Study x Group:")
    print(pd.crosstab(study_prefix, meta["group"]).to_string())
    print()

    meta.to_csv(CFG.DATA_PROC / "subject_overview.csv")
    print(f"Subjects prepared: {len(meta)}")

    print(f"Windowing with WIN={CFG.WIN_MS} ms, STEP={CFG.STEP_MS} ms at fs={CFG.FS} Hz ...")
    left_long_parts = []
    right_long_parts = []
    for sid in tqdm(meta.index, desc="Windowing subjects", unit="subj"):
        L = left_series[sid]; R = right_series[sid]
        lw = build_windows(L, CFG.FS, CFG.WIN_MS, CFG.STEP_MS)
        rw = build_windows(R, CFG.FS, CFG.WIN_MS, CFG.STEP_MS)
        left_long_parts.append(windows_to_long(lw, f"L_{sid}"))
        right_long_parts.append(windows_to_long(rw, f"R_{sid}"))

    left_long = pd.concat(left_long_parts, ignore_index=True)
    right_long = pd.concat(right_long_parts, ignore_index=True)
    left_long_path = CFG.DATA_PROC / "left_long.parquet"
    right_long_path = CFG.DATA_PROC / "right_long.parquet"
    left_long.to_parquet(left_long_path); right_long.to_parquet(right_long_path)
    print(f"Cached long frames -> {left_long_path.name} ({len(left_long):,} rows), {right_long_path.name} ({len(right_long):,} rows)")

    X_left_path = CFG.DATA_PROC / f"X_left_{CFG.FEATURE_SET}.parquet"
    X_right_path = CFG.DATA_PROC / f"X_right_{CFG.FEATURE_SET}.parquet"

    if X_left_path.exists() and X_right_path.exists():
        print("Found cached TSFresh window features; loading...")
        X_left = pd.read_parquet(X_left_path)
        X_right = pd.read_parquet(X_right_path)
    else:
        print("Extracting TSFresh features (LEFT) in batches...")
        X_left = extract_tsfresh_chunked(
            left_long_path, CFG.FEATURE_SET, CFG.N_JOBS, CFG.FS, CFG.WIN_MS,
            out_final=X_left_path,
            tmp_dir=CFG.DATA_PROC / "tsfresh_tmp_left",
            batch_ids=CFG.BATCH_IDS,
        )

        print("Extracting TSFresh features (RIGHT) in batches...")
        X_right = extract_tsfresh_chunked(
            right_long_path, CFG.FEATURE_SET, CFG.N_JOBS, CFG.FS, CFG.WIN_MS,
            out_final=X_right_path,
            tmp_dir=CFG.DATA_PROC / "tsfresh_tmp_right",
            batch_ids=CFG.BATCH_IDS,
        )

    agg_left_path = CFG.DATA_PROC / "features_left_agg.parquet"
    agg_right_path = CFG.DATA_PROC / "features_right_agg.parquet"
    if agg_left_path.exists() and agg_right_path.exists():
        print("Found cached aggregated features; loading...")
        agg_left = pd.read_parquet(agg_left_path)
        agg_right = pd.read_parquet(agg_right_path)
    else:
        print("Aggregating per-subject features (mean/std/skew/kurt)...")
        agg_left = aggregate_features(X_left)
        agg_right = aggregate_features(X_right)
        agg_left.to_parquet(agg_left_path)
        agg_right.to_parquet(agg_right_path)
        print(f"Saved aggregates: {agg_left_path.name}, {agg_right_path.name}")

    y_all = meta["group"].astype(float)

    print("Sanitizing features for PCA/t-SNE ...")
    agg_right_c = sanitize_features_for_dimred(agg_right)
    agg_left_c  = sanitize_features_for_dimred(agg_left)

    agg_right_c = strip_lr_prefix_index(agg_right_c)
    agg_left_c  = strip_lr_prefix_index(agg_left_c)

    subjects_right = agg_right_c.index.intersection(y_all.index)
    Xr = agg_right_c.loc[subjects_right]
    yr = y_all.loc[subjects_right].values
    r_mask = np.isfinite(yr); Xr = Xr.iloc[r_mask]; yr = yr[r_mask]

    subjects_left = agg_left_c.index.intersection(y_all.index)
    Xl = agg_left_c.loc[subjects_left]
    yl = y_all.loc[subjects_left].values
    l_mask = np.isfinite(yl); Xl = Xl.iloc[l_mask]; yl = yl[l_mask]

    subjects_both = Xl.index.intersection(Xr.index)
    Xc = pd.concat([Xl.loc[subjects_both].add_prefix("L_"),
                    Xr.loc[subjects_both].add_prefix("R_")], axis=1)
    yc = y_all.loc[subjects_both].values
    c_mask = np.isfinite(yc); Xc = Xc.iloc[c_mask]; yc = yc[c_mask]

    if Xc.shape[0] > 0 and Xc.shape[1] > 0:
        Xc.to_parquet(CFG.DATA_PROC / "features_combined_agg.parquet")
    else:
        print("[WARN] Combined features empty after cleaning; not saving.")

    print(f"[INFO] Right rows: {Xr.shape[0]}, Left rows: {Xl.shape[0]}, Combined rows: {Xc.shape[0]}")

    dimred_and_plot_safe(Xr, yr, "Right_Foot",  CFG.REPORT_DIR / "figs", CFG.DIMRED)
    dimred_and_plot_safe(Xl, yl, "Left_Foot",   CFG.REPORT_DIR / "figs", CFG.DIMRED)
    dimred_and_plot_safe(Xc, yc, "Combined_LR", CFG.REPORT_DIR / "figs", CFG.DIMRED)

    meta.to_csv(CFG.DATA_PROC / "demographics_for_slides.csv")
    print("DONE. Artifacts in data/processed and reports/figs")


if __name__ == "__main__":
    main()
