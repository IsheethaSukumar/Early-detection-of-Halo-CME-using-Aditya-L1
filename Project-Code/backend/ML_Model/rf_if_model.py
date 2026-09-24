"""
Random Forest + Isolation Forest Anomaly Score
===============================================

the input CSV is iforest_augmented.csv, which adds two extra features:
  anomaly_score  (float)  -- IF score: higher = more anomalous
  is_anomaly     (0/1)    -- binary flag from IF threshold

Artifacts saved to ML_Model/artifacts/ with "_if" suffix so
they co-exist alongside the baseline RF artifacts.

Usage:
    python ML_Model/rf_if_model.py
    (run iforest_model.py first to generate iforest_augmented.csv)
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pickle
import warnings
import time
import subprocess, sys
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, recall_score,
)

warnings.filterwarnings("ignore")

try:
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Installing imbalanced-learn ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE

# -- Paths ---------------------------------------------------------------------
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_DIR  = Path(__file__).resolve().parent / "RESULTS"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# iforest_augmented.csv lives in artifacts/ (output of iforest_model.py)
AUGMENTED_CSV = ARTIFACT_DIR / "iforest_augmented.csv"

# Baseline RF results file (for final delta comparison)
BASELINE_RESULTS = RESULTS_DIR / "rf_res.txt"

HEADER = "=" * 64

# Columns that are labels / meta / leakage -- NOT features
DROP_COLS = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]


# -- Tee -----------------------------------------------------------------------
class _Tee:
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

    def isatty(self):
        return False


def banner(title: str):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# -- 1. Load data --------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print("\n[1/6] Loading IF-augmented dataset ...")
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"Augmented CSV not found: {AUGMENTED_CSV}\n"
            "Run iforest_model.py first to generate it."
        )
    df = pd.read_csv(AUGMENTED_CSV)
    df.dropna(subset=["cme_label"], inplace=True)
    df["cme_label"] = df["cme_label"].astype(int)

    n_total = len(df)
    n_cme   = int(df["cme_label"].sum())
    n_nocme = n_total - n_cme

    print(f"      Source      : {AUGMENTED_CSV.name}")
    print(f"      Rows        : {n_total:,}")
    print(f"      CME  (1)    : {n_cme:,}  ({100*n_cme/n_total:.1f}%)")
    print(f"      No-CME (0)  : {n_nocme:,}  ({100*n_nocme/n_total:.1f}%)")

    # Show IF-specific columns
    if_cols = [c for c in df.columns if "anomaly" in c.lower()]
    print(f"      IF features : {if_cols}")
    print(f"        anomaly_score range : [{df['anomaly_score'].min():.4f}, {df['anomaly_score'].max():.4f}]")
    print(f"        is_anomaly count    : {int(df['is_anomaly'].sum()):,}  ({100*df['is_anomaly'].mean():.1f}%)")
    return df


# -- 2. Feature engineering ----------------------------------------------------
def feature_engineering(df: pd.DataFrame):
    print("\n[2/6] Feature engineering  (anomaly_score + is_anomaly included) ...")
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)
    y = df["cme_label"].values
    print(f"      Total features : {X.shape[1]}  (baseline 57 + 2 IF features)")
    return X, y, feat_cols


# -- 3. Split + Scale ----------------------------------------------------------
def split_and_scale(X, y):
    print("\n[3/6] Stratified 80/20 split + StandardScaler ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    scaler_path = ARTIFACT_DIR / "rf_if_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      Train : {len(y_train):,}  |  CME: {int(y_train.sum()):,}")
    print(f"      Test  : {len(y_test):,}   |  CME: {int(y_test.sum()):,}")
    print(f"      Scaler saved -> {scaler_path}")
    return X_train_s, X_test_s, y_train, y_test


# -- 4. SMOTE ------------------------------------------------------------------
def apply_smote(X_train, y_train):
    print("\n[4/6] SMOTE oversampling ...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"      After  -> CME: {int(y_res.sum()):,}  |  No-CME: {int((y_res==0).sum()):,}")
    return X_res, y_res


# -- 5. Train ------------------------------------------------------------------
def train(X_train, y_train):
    print("\n[5/6] Training Random Forest (n_estimators=300) ...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    print(f"      Trained in {time.time()-t0:.1f}s")
    return rf


# -- 6. Evaluate ---------------------------------------------------------------
def evaluate(rf, X_test, y_test, feat_cols):
    print("\n[6/6] Evaluation + precision-recall threshold tuning ...")

    y_prob  = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)
    f1_arr      = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_f1_idx = int(np.argmax(f1_arr))
    thr_best_f1 = thresholds[best_f1_idx]
    rec80_mask  = rec_arr[:-1] >= 0.80
    thr_rec80   = float(thresholds[rec80_mask].min()) if rec80_mask.any() else thr_best_f1
    chosen_thr  = thr_best_f1

    def _at(thr):
        yp  = (y_prob >= thr).astype(int)
        acc = (yp == y_test).mean() * 100
        rec = recall_score(y_test, yp, zero_division=0)
        f1  = f1_score(y_test, yp, zero_division=0)
        pr  = prec_arr[:-1][np.argmin(np.abs(thresholds - thr))]
        return acc, rec, pr, f1

    fmt = "  {:<30} {:>10} {:>10} {:>10} {:>10}"
    print(f"\n  {'':30} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("  " + "-" * 68)
    for label, thr in [("Default (0.50)", 0.50),
                        (f"Best F1 ({thr_best_f1:.3f})", thr_best_f1),
                        (f"Recall>=0.80 ({thr_rec80:.3f})", thr_rec80)]:
        a, r, p, f = _at(thr)
        print(fmt.format(label, f"{a:.2f}%", f"{r:.4f}", f"{p:.4f}", f"{f:.4f}"))

    print(f"\n  [OK] Chosen threshold : {chosen_thr:.4f}")
    y_pred = (y_prob >= chosen_thr).astype(int)
    print(f"\n  === Full Classification Report (threshold={chosen_thr:.4f}) ===")
    print(classification_report(y_test, y_pred, target_names=["No CME", "Halo CME"]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("  Confusion Matrix:")
    print(f"    TN: {tn:,}  FP: {fp:,}  FN: {fn:,}  TP: {tp:,}")

    (ARTIFACT_DIR / "rf_if_threshold.txt").write_text(str(chosen_thr))

    _save_feature_importance(rf, feat_cols)
    _plot_roc_pr(y_test, y_prob, roc_auc, pr_auc)
    return roc_auc, pr_auc, chosen_thr


# -- Feature importance --------------------------------------------------------
def _save_feature_importance(rf, feat_cols):
    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        ARTIFACT_DIR / "rf_if_feature_importance.csv", index=False
    )

    # Highlight IF columns
    colors = ["#e05c5c" if "anomaly" in c else "steelblue" for c in fi.head(25).index]

    fig, ax = plt.subplots(figsize=(10, 7))
    fi.head(25).plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Top 25 Feature Importances — RF + IF\n"
                 "(red = Isolation Forest features)", fontsize=11)
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "rf_if_feature_importance.png", dpi=150)
    plt.close()
    print(f"\n  IF feature ranks:")
    for c in ["anomaly_score", "is_anomaly"]:
        if c in fi.index:
            rank = list(fi.index).index(c) + 1
            print(f"    {c:<20} rank #{rank:>3}  importance={fi[c]:.5f}")


# -- ROC + PR plots ------------------------------------------------------------
def _plot_roc_pr(y_test, y_prob, roc_auc, pr_auc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Random Forest + IF Anomaly Score -- Test Set",
                 fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="darkorange", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "rf_if_evaluation.png", dpi=150)
    plt.close()


# -- Comparison summary --------------------------------------------------------
def print_comparison(roc_auc: float, pr_auc: float):
    BASELINE_ROC = 0.8647   # from rf_res.txt
    delta_roc = roc_auc - BASELINE_ROC
    sign = "+" if delta_roc >= 0 else ""
    print(f"\n  {'='*48}")
    print(f"  MODEL COMPARISON  (RF baseline vs RF + IF)")
    print(f"  {'='*48}")
    print(f"  {'Metric':<18} {'Baseline':>10} {'+ IF Score':>12} {'Delta':>8}")
    print(f"  {'-'*48}")
    print(f"  {'ROC-AUC':<18} {BASELINE_ROC:>10.4f} {roc_auc:>12.4f} {sign}{delta_roc:>7.4f}")
    print(f"  {'='*48}")


# -- Save model ----------------------------------------------------------------
def save_model(rf) -> Path:
    print("\n[Save] Persisting RF+IF model ...")
    out = ARTIFACT_DIR / "rf_if_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(rf, f)
    print(f"  Model saved -> {out}  ({out.stat().st_size//1024:,} KB)")
    return out


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    log_path = RESULTS_DIR / "rf_if_res.txt"
    log_fh   = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)

    try:
        banner("SOLAR HALO CME  --  Random Forest + Isolation Forest Score")

        df           = load_data()
        X, y, cols   = feature_engineering(df)
        X_tr, X_te, y_tr, y_te = split_and_scale(X, y)
        X_res, y_res = apply_smote(X_tr, y_tr)
        rf           = train(X_res, y_res)
        roc, pr, thr = evaluate(rf, X_te, y_te, cols)
        save_model(rf)
        print_comparison(roc, pr)

        print(f"\n{HEADER}")
        print("  DONE -- artifacts/ now contains:")
        print(f"    rf_if_model.pkl                (RF trained on IF-augmented data)")
        print(f"    rf_if_scaler.pkl")
        print(f"    rf_if_threshold.txt            (threshold={thr:.4f})")
        print(f"    rf_if_feature_importance.csv/png")
        print(f"    rf_if_evaluation.png")
        print(f"\n  RESULTS/rf_if_res.txt           (this log)")
        print(HEADER)

    finally:
        sys.stdout = sys.__stdout__
        log_fh.close()
        print(f"\n  Results saved -> {log_path}")
