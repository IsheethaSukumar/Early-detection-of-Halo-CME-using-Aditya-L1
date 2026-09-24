"""
Isolation Forest -- Anomaly Detection Layer
===========================================

Role in the pipeline
--------------------
  Unsupervised pre-filter that learns "normal" solar-wind behaviour
  from the raw feature space and assigns a continuous anomaly_score
  to every observation.  The score is then appended to the feature
  matrix before the supervised models (RF / XGBoost / BiLSTM) are
  trained, giving them an extra signal for novel / out-of-distribution
  solar-wind conditions.

Outputs (ML_Model/artifacts/)
------------------------------
  iforest_model.pkl           -- fitted IsolationForest
  iforest_scaler.pkl          -- StandardScaler used before fitting
  iforest_augmented.csv       -- full dataset with anomaly_score column
  iforest_threshold.txt       -- chosen anomaly_score cut-off

Outputs (ML_Model/RESULTS/)
-----------------------------
  iforest_res.txt             -- run log (mirrored from stdout)

Plots (ML_Model/)
------------------
  iforest_score_dist.png      -- score distribution: CME vs No-CME
  iforest_roc.png             -- anomaly_score treated as CME predictor
  iforest_top_features.png    -- |Pearson r| of each feature with score

Usage
-----
  python ML_Model/iforest_model.py
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")           # headless -- no display required
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pickle
import sys
import warnings
import time
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent / "Dataset"
PROC_DIR     = BASE_DIR / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_DIR  = Path(__file__).resolve().parent / "RESULTS"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Hyper-parameters ----------------------------------------------------------
N_ESTIMATORS     = 200          # number of isolation trees
MAX_SAMPLES      = "auto"       # sub-sample per tree (sklearn default = min(256, n))
CONTAMINATION    = "auto"       # let sklearn estimate the anomaly boundary
RANDOM_STATE     = 42

# Columns that are NOT solar-wind features (labels / meta / leakage)
DROP_COLS = [
    "cme_label", "is_halo", "cme_speed_kmps",
    "cme_angular_w", "Unnamed: 0",
]

HEADER = "=" * 64


# -- Helpers -------------------------------------------------------------------
class _Tee:
    """Mirror all print() output to a log file simultaneously."""

    def __init__(self, real_stdout, log_file):
        self._real = real_stdout
        self._file = log_file

    def write(self, text):
        try:
            self._real.write(text)
        except (UnicodeEncodeError, AttributeError):
            self._real.buffer.write(text.encode("utf-8", errors="replace"))
        self._file.write(text)
        self._file.flush()

    def flush(self):
        try:
            self._real.flush()
        except Exception:
            pass
        self._file.flush()


def banner(title: str):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# -- 1. Load data --------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("\n[1/6] Loading cleaned dataset ...")
    csv = PROC_DIR / "full_merged_dataset.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv}.\n"
            "Run the data preprocessing pipeline first."
        )

    df = pd.read_csv(csv)

    # Keep only rows that have a valid label (needed for diagnostic overlap)
    df_labeled = df.dropna(subset=["cme_label"]).copy()
    df_labeled["cme_label"] = df_labeled["cme_label"].astype(int)

    n_total = len(df_labeled)
    n_cme   = int(df_labeled["cme_label"].sum())
    n_nocme = n_total - n_cme

    print(f"      Rows        : {n_total:,}")
    print(f"      CME  (1)    : {n_cme:,}  ({100 * n_cme / n_total:.1f}%)")
    print(f"      No-CME (0)  : {n_nocme:,}  ({100 * n_nocme / n_total:.1f}%)")
    print(f"      Columns     : {df_labeled.columns.tolist()}")
    return df_labeled


# -- 2. Build feature matrix ---------------------------------------------------
def build_features(df: pd.DataFrame):
    """
    Return (X_raw, y, feature_cols).
    Note: Isolation Forest is *unsupervised*, so y is only used for
    post-hoc diagnostics -- it is NEVER seen during fitting.
    """
    print("\n[2/6] Building feature matrix (unsupervised -- labels withheld) ...")

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()

    # Impute missing values with column median (robust to outliers)
    X.fillna(X.median(numeric_only=True), inplace=True)

    y = df["cme_label"].values

    print(f"      Feature columns : {len(feat_cols)}")
    print(f"      Missing values  : {X.isnull().sum().sum()} (after imputation)")
    return X, y, feat_cols


# -- 3. Scale ------------------------------------------------------------------
def scale(X: pd.DataFrame):
    """
    StandardScaler improves IF performance on mixed-magnitude features
    (e.g. plasma_temperature_k vs bz_gse).
    Fit on the FULL dataset -- IF is unsupervised, no leakage risk.
    """
    print("\n[3/6] Standardising features ...")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    path = ARTIFACT_DIR / "iforest_scaler.pkl"
    with open(path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"      Scaler saved -> {path}")
    return X_s, scaler


# -- 4. Fit Isolation Forest ---------------------------------------------------
def fit_isolation_forest(X_s: np.ndarray):
    """
    IsolationForest.score_samples() returns negative anomaly scores
    (more negative = more anomalous).  We negate them so that
    anomaly_score > 0  means "unusual".
    """
    print(f"\n[4/6] Fitting Isolation Forest  "
          f"(n_estimators={N_ESTIMATORS}, contamination={CONTAMINATION}) ...")
    t0  = time.time()

    iforest = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        max_samples   = MAX_SAMPLES,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
        warm_start    = False,
    )
    iforest.fit(X_s)

    elapsed = time.time() - t0
    print(f"      Fitted in {elapsed:.1f}s")

    # Raw sklearn scores: negative_of_average_path_length (lower = more anomalous)
    raw_scores = iforest.score_samples(X_s)   # shape (n,)

    # Negate -> anomaly_score: higher value = more anomalous
    anomaly_score = -raw_scores

    # Binary flag using the sklearn threshold (contamination-aware)
    is_anomaly = (iforest.predict(X_s) == -1).astype(int)  # 1 = anomaly

    n_anom = int(is_anomaly.sum())
    print(f"      Anomalies flagged : {n_anom:,}  "
          f"({100 * n_anom / len(is_anomaly):.1f}%)")

    return iforest, anomaly_score, is_anomaly


# -- 5. Evaluate (diagnostic -- labels used for assessment only) ---------------
def evaluate_overlap(y: np.ndarray, anomaly_score: np.ndarray,
                     is_anomaly: np.ndarray, feat_cols: list):
    """
    Isolation Forest is NOT a CME detector; this section only checks
    whether the anomaly score is positively correlated with CME events
    (which we expect but do not rely on).
    """
    print("\n[5/6] Diagnostic: anomaly <-> CME overlap ...")

    roc_auc = roc_auc_score(y, anomaly_score)
    pr_auc  = average_precision_score(y, anomaly_score)

    print(f"      Anomaly->CME  ROC-AUC : {roc_auc:.4f}  "
          "(>0.5 means anomalies lean toward CME events)")
    print(f"      Anomaly->CME  PR-AUC  : {pr_auc:.4f}")

    # Overlap statistics
    cme_mask   = y == 1
    nocme_mask = y == 0
    pct_cme_anomalous   = is_anomaly[cme_mask].mean()   * 100
    pct_nocme_anomalous = is_anomaly[nocme_mask].mean() * 100

    print(f"\n      CME rows flagged as anomaly    : {pct_cme_anomalous:.1f}%")
    print(f"      No-CME rows flagged as anomaly : {pct_nocme_anomalous:.1f}%")

    # Threshold that maximises F1  (diagnostic, not operational)
    prec_arr, rec_arr, thresholds = precision_recall_curve(y, anomaly_score)
    f1_arr = (2 * prec_arr[:-1] * rec_arr[:-1]
               / (prec_arr[:-1] + rec_arr[:-1] + 1e-9))
    best_idx       = int(np.argmax(f1_arr))
    best_thr       = float(thresholds[best_idx])
    best_f1        = float(f1_arr[best_idx])
    best_recall    = float(rec_arr[best_idx])
    best_precision = float(prec_arr[best_idx])

    print(f"\n      Best anomaly_score threshold   : {best_thr:.4f}")
    print(f"        F1        : {best_f1:.4f}")
    print(f"        Recall    : {best_recall:.4f}")
    print(f"        Precision : {best_precision:.4f}")

    # Save threshold
    thr_path = ARTIFACT_DIR / "iforest_threshold.txt"
    thr_path.write_text(str(best_thr))
    print(f"\n      Threshold saved -> {thr_path}")

    return roc_auc, pr_auc, best_thr


# -- 6. Augment dataset + save -------------------------------------------------
def augment_and_save(df: pd.DataFrame, anomaly_score: np.ndarray,
                     is_anomaly: np.ndarray):
    """
    Append anomaly_score and is_anomaly columns to the original dataframe
    and save as iforest_augmented.csv.  Downstream models load this file
    instead of full_merged_dataset.csv to gain the extra signal.
    """
    print("\n[6/6] Augmenting dataset with anomaly features ...")

    df_aug = df.copy()
    df_aug["anomaly_score"] = anomaly_score
    df_aug["is_anomaly"]    = is_anomaly

    out = ARTIFACT_DIR / "iforest_augmented.csv"
    df_aug.to_csv(out, index=False)
    size_kb = out.stat().st_size // 1024
    print(f"      Augmented CSV saved -> {out}  ({size_kb:,} KB)")
    print(f"      New columns : anomaly_score (float), is_anomaly (0/1)")
    return df_aug


# -- Plots ---------------------------------------------------------------------
def plot_diagnostics(df_aug: pd.DataFrame, y: np.ndarray,
                     anomaly_score: np.ndarray, roc_auc: float,
                     pr_auc: float, feat_cols: list):

    print("\n  Generating diagnostic plots ...")

    PLOT_DIR = Path(__file__).resolve().parent
    palette  = {"CME": "#e05c5c", "No CME": "#5c9ee0"}

    # -- Panel 1: Anomaly score distribution (CME vs No-CME) ------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    scores_cme   = anomaly_score[y == 1]
    scores_nocme = anomaly_score[y == 0]

    bins = np.linspace(anomaly_score.min(), anomaly_score.max(), 60)
    ax.hist(scores_nocme, bins=bins, alpha=0.65, color=palette["No CME"],
            label=f"No CME  (n={len(scores_nocme):,})", density=True)
    ax.hist(scores_cme,   bins=bins, alpha=0.75, color=palette["CME"],
            label=f"Halo CME (n={len(scores_cme):,})",  density=True)

    ax.set_xlabel("Anomaly Score  (higher = more anomalous)", color="white")
    ax.set_ylabel("Density", color="white")
    ax.set_title("Isolation Forest -- Anomaly Score Distribution",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["top"].set_color("#30363d")
    ax.spines["right"].set_color("#30363d")
    ax.spines["left"].set_color("#30363d")
    ax.spines["bottom"].set_color("#30363d")
    ax.legend(framealpha=0.3)
    plt.tight_layout()
    p1 = PLOT_DIR / "iforest_score_dist.png"
    plt.savefig(p1, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved -> {p1}")

    # -- Panel 2: ROC + PR curves ----------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
    fig.suptitle("Isolation Forest Anomaly Score as CME Signal (Diagnostic)",
                 fontsize=12, fontweight="bold", color="white")

    fpr, tpr, _ = roc_curve(y, anomaly_score)
    axes[0].plot(fpr, tpr, lw=2.5, color="#e05c5c",
                 label=f"ROC-AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False Positive Rate", color="white")
    axes[0].set_ylabel("True Positive Rate", color="white")
    axes[0].set_title("ROC Curve", color="white")
    axes[0].tick_params(colors="white")
    for sp in axes[0].spines.values():
        sp.set_color("#30363d")
    axes[0].legend(framealpha=0.3)

    prec, rec, _ = precision_recall_curve(y, anomaly_score)
    axes[1].plot(rec, prec, lw=2.5, color="#5ce08a",
                 label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall", color="white")
    axes[1].set_ylabel("Precision", color="white")
    axes[1].set_title("Precision-Recall Curve", color="white")
    axes[1].tick_params(colors="white")
    for sp in axes[1].spines.values():
        sp.set_color("#30363d")
    axes[1].legend(framealpha=0.3)

    plt.tight_layout()
    p2 = PLOT_DIR / "iforest_roc.png"
    plt.savefig(p2, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved -> {p2}")

    # -- Panel 3: Per-feature |Pearson r| with anomaly score -------------------
    feat_data = df_aug[[c for c in feat_cols if c in df_aug.columns]].copy()
    feat_data.fillna(feat_data.median(numeric_only=True), inplace=True)
    corrs = feat_data.corrwith(
        pd.Series(anomaly_score, index=feat_data.index)
    ).abs().dropna().sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    ax.barh(corrs.index[::-1], corrs.values[::-1],
            color="#e0b05c", edgecolor="#30363d")
    ax.set_xlabel("|Pearson r| with anomaly_score", color="white")
    ax.set_title("Top 20 Features Driving Anomaly Score",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    plt.tight_layout()
    p3 = PLOT_DIR / "iforest_top_features.png"
    plt.savefig(p3, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"    Saved -> {p3}")


# -- Save model ----------------------------------------------------------------
def save_model(iforest: IsolationForest) -> Path:
    print("\n[Save] Persisting Isolation Forest model ...")
    out = ARTIFACT_DIR / "iforest_model.pkl"
    with open(out, "wb") as fh:
        pickle.dump(iforest, fh)
    size_kb = out.stat().st_size // 1024
    print(f"  Model saved -> {out}  ({size_kb:,} KB)")
    return out


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":

    # Redirect stdout to both console and RESULTS/iforest_res.txt
    res_path = RESULTS_DIR / "iforest_res.txt"
    log_fh   = open(res_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)

    banner("SOLAR HALO CME  --  Isolation Forest (Anomaly Detection)")

    print(
        "\n  Role   : Unsupervised pre-filter for unusual solar-wind conditions"
        "\n  Input  : full_merged_dataset.csv  (labels withheld during fit)"
        "\n  Output : iforest_augmented.csv  <- use as input for RF / XGBoost / LSTM"
    )

    t_start = time.time()

    df                              = load_data()
    X, y, feat_cols                 = build_features(df)
    X_s, scaler                     = scale(X)
    iforest, anomaly_score, is_anom = fit_isolation_forest(X_s)
    roc_auc, pr_auc, best_thr       = evaluate_overlap(y, anomaly_score,
                                                        is_anom, feat_cols)
    df_aug                          = augment_and_save(df, anomaly_score, is_anom)
    plot_diagnostics(df_aug, y, anomaly_score, roc_auc, pr_auc, feat_cols)
    model_path                      = save_model(iforest)

    elapsed_total = time.time() - t_start

    print(f"\n{HEADER}")
    print("  DONE -- ML_Model/artifacts/ now contains:")
    print(f"    iforest_model.pkl         (fitted IsolationForest)")
    print(f"    iforest_scaler.pkl        (StandardScaler)")
    print(f"    iforest_augmented.csv     (original data + anomaly_score, is_anomaly)")
    print(f"    iforest_threshold.txt     (anomaly_score cut-off = {best_thr:.4f})")
    print(f"\n  ML_Model/RESULTS/")
    print(f"    iforest_res.txt           (this log)")
    print(f"\n  Diagnostic plots saved to ML_Model/:")
    print(f"    iforest_score_dist.png")
    print(f"    iforest_roc.png")
    print(f"    iforest_top_features.png")
    print(f"\n  -- Integration guide ------------------------------------------")
    print(f"  In rf_model.py / xgb_model.py / lstm_model.py, replace:")
    print(f"    df = pd.read_csv(PROC_DIR / 'full_merged_dataset.csv')")
    print(f"  with:")
    print(f"    df = pd.read_csv(ARTIFACT_DIR / 'iforest_augmented.csv')")
    print(f"  The 'anomaly_score' and 'is_anomaly' columns will automatically")
    print(f"  be picked up as extra features (they are NOT in DROP_COLS).")
    print(f"\n  Total run time : {elapsed_total:.1f}s")
    print(HEADER)

    log_fh.close()
    sys.stdout = sys.__stdout__
    print(f"\n  Results log saved -> {res_path}")
