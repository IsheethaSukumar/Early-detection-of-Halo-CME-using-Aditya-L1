"""
Random Forest Model — Solar Halo CME Forecasting & Nowcasting
=============================================================
Usage:
    python ML_Model/rf_model.py
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
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

# ── install imbalanced-learn if missing ────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE
except ImportError:
    print("Installing imbalanced-learn ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent / "Dataset"
PROC_DIR     = BASE_DIR / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

HEADER = "=" * 64

# ── Feature columns (all except label/meta columns) ───────────────────────────
DROP_COLS = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]


def banner(title):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data():
    print("\n[1/6] Loading cleaned data ...")
    df = pd.read_csv(PROC_DIR / "full_merged_dataset.csv")
    df.dropna(subset=["cme_label"], inplace=True)
    df["cme_label"] = df["cme_label"].astype(int)

    n_total = len(df)
    n_cme   = int(df["cme_label"].sum())
    n_nocme = n_total - n_cme
    cols    = df.columns.tolist()

    print(f"      Rows        : {n_total:,}")
    print(f"      CME  (1)    : {n_cme:,}  ({100*n_cme/n_total:.1f}%)")
    print(f"      No-CME (0)  : {n_nocme:,}  ({100*n_nocme/n_total:.1f}%)")
    print(f"      Columns     : {cols}")
    return df


# ── 2. Feature engineering ─────────────────────────────────────────────────────
def feature_engineering(df):
    print("\n[2/6] Feature engineering ...")

    # Drop non-feature / leakage columns
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()

    # Fill missing values
    X.fillna(X.median(numeric_only=True), inplace=True)

    y = df["cme_label"].values

    print(f"      Total features : {X.shape[1]}")
    return X, y, feat_cols


# ── 3. Split + Scale ───────────────────────────────────────────────────────────
def split_and_scale(X, y):
    print("\n[3/6] Stratified 80/20 split + StandardScaler ...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    scaler_path = ARTIFACT_DIR / "rf_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"      Train : {len(y_train):,}  |  CME: {int(y_train.sum()):,}  ({100*y_train.mean():.1f}%)")
    print(f"      Test  : {len(y_test):,}   |  CME: {int(y_test.sum()):,}   ({100*y_test.mean():.1f}%)")
    print(f"      Scaler saved -> {scaler_path}")

    return X_train_s, X_test_s, y_train, y_test


# ── 4. SMOTE ───────────────────────────────────────────────────────────────────
def apply_smote(X_train, y_train):
    print("\n[4/6] SMOTE oversampling ...")
    n_cme_before   = int(y_train.sum())
    n_nocme_before = int((y_train == 0).sum())
    print(f"      Before -> CME: {n_cme_before:,}  |  No-CME: {n_nocme_before:,}")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    n_cme_after   = int(y_res.sum())
    n_nocme_after = int((y_res == 0).sum())
    print(f"      After  -> CME: {n_cme_after:,}  |  No-CME: {n_nocme_after:,}")
    return X_res, y_res


# ── 5. Train ───────────────────────────────────────────────────────────────────
def train(X_train, y_train):
    print("\n[5/6] Training Random Forest (n_estimators=300) ...")
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators     = 300,
        min_samples_leaf = 5,
        random_state     = 42,
        n_jobs           = -1,
    )
    rf.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"      Trained in {elapsed:.1f}s")
    return rf


# ── 6. Evaluate + threshold tuning ────────────────────────────────────────────
def evaluate(rf, X_test, y_test, feat_cols):
    print("\n[6/6] Evaluation + precision-recall threshold tuning ...")

    y_prob = rf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n  ROC-AUC : {roc_auc:.4f}")

    # -- Threshold sweep --
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)

    # Best F1
    f1_arr      = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_f1_idx = int(np.argmax(f1_arr))
    thr_best_f1 = thresholds[best_f1_idx]

    # Recall >= 0.80
    rec80_mask = rec_arr[:-1] >= 0.80
    thr_rec80  = float(thresholds[rec80_mask].min()) if rec80_mask.any() else thr_best_f1

    # chosen threshold
    chosen_thr  = thr_best_f1

    def metrics_at(thr):
        y_pred = (y_prob >= thr).astype(int)
        acc    = (y_pred == y_test).mean() * 100
        rec    = recall_score(y_test, y_pred, zero_division=0)
        prec   = prec_arr[:-1][np.searchsorted(thresholds, thr, side="right") - 1] if thr in thresholds else \
                 prec_arr[:-1][np.argmin(np.abs(thresholds - thr))]
        f1     = f1_score(y_test, y_pred, zero_division=0)
        return acc, rec, prec, f1

    acc0, rec0, pre0, f10 = metrics_at(0.50)
    acc1, rec1, pre1, f11 = metrics_at(thr_best_f1)
    acc2, rec2, pre2, f12 = metrics_at(thr_rec80)

    header_fmt = "  {:<30} {:>10} {:>10} {:>10} {:>10}"
    row_fmt    = "  {:<30} {:>10} {:>10} {:>10} {:>10}"

    print(f"\n  {'':<30} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("  " + "-" * 68)
    print(row_fmt.format(f"Default (0.50)",        f"{acc0:.2f}%", f"{rec0:.4f}", f"{pre0:.4f}", f"{f10:.4f}"))
    print(row_fmt.format(f"Best F1 ({thr_best_f1:.3f})",  f"{acc1:.2f}%", f"{rec1:.4f}", f"{pre1:.4f}", f"{f11:.4f}"))
    print(row_fmt.format(f"Recall>=0.80 ({thr_rec80:.3f})", f"{acc2:.2f}%", f"{rec2:.4f}", f"{pre2:.4f}", f"{f12:.4f}"))

    print(f"\n  [OK] Chosen threshold : {chosen_thr:.4f}")

    # -- Full report at chosen threshold --
    y_pred_final = (y_prob >= chosen_thr).astype(int)
    print(f"\n  === Full Classification Report (threshold={chosen_thr:.4f}) ===")
    print(classification_report(y_test, y_pred_final, target_names=["No CME", "Halo CME"]))

    cm = confusion_matrix(y_test, y_pred_final)
    tn, fp, fn, tp = cm.ravel()
    print("  Confusion Matrix:")
    print(f"    TN (correct No-CME) : {tn:,}")
    print(f"    FP (false alarm)     : {fp:,}")
    print(f"    FN (missed CME)      : {fn:,}")
    print(f"    TP (caught CME)      : {tp:,}")

    # -- Save threshold --
    thr_path = ARTIFACT_DIR / "rf_threshold.txt"
    thr_path.write_text(str(chosen_thr))
    print(f"\n  Threshold saved -> {thr_path}")

    # -- Feature importance --
    _save_feature_importance(rf, feat_cols)

    # -- Plots --
    _plot_roc_pr(y_test, y_prob, roc_auc)

    return roc_auc, chosen_thr


# ── Feature importance ─────────────────────────────────────────────────────────
def _save_feature_importance(rf, feat_cols):
    print("  Saving feature importance ...")
    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)

    csv_path = ARTIFACT_DIR / "rf_feature_importance.csv"
    fi.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(csv_path, index=False)
    print(f"  CSV saved -> {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    fi.head(20).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 20 Feature Importances — Random Forest")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    png_path = ARTIFACT_DIR / "rf_feature_importance.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {png_path}")


# ── ROC + PR plots ─────────────────────────────────────────────────────────────
def _plot_roc_pr(y_test, y_prob, roc_auc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Random Forest — Test Set", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve"); axes[0].legend()

    pr_auc = average_precision_score(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="darkorange", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve"); axes[1].legend()

    plt.tight_layout()
    out = ARTIFACT_DIR / "rf_evaluation.png"
    plt.savefig(out, dpi=150)
    plt.close()


# ── Save model ─────────────────────────────────────────────────────────────────
def save_model(rf):
    print("\n[Save] Persisting model ...")
    out = ARTIFACT_DIR / "random_forest.pkl"
    with open(out, "wb") as f:
        pickle.dump(rf, f)
    size_kb = out.stat().st_size // 1024
    print(f"  Model saved -> {out}  ({size_kb:,} KB)")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner("SOLAR HALO CME  --  Random Forest")

    df           = load_data()
    X, y, cols   = feature_engineering(df)
    X_tr, X_te, y_tr, y_te = split_and_scale(X, y)
    X_res, y_res = apply_smote(X_tr, y_tr)
    rf           = train(X_res, y_res)
    roc, thr     = evaluate(rf, X_te, y_te, cols)
    model_path   = save_model(rf)

    print(f"\n{HEADER}")
    print("  DONE -- ML_Model/artifacts/ now contains:")
    print(f"    random_forest.pkl")
    print(f"    rf_scaler.pkl")
    print(f"    rf_threshold.txt              (threshold={thr:.4f})")
    print(f"    rf_feature_importance.csv")
    print(f"    rf_feature_importance.png")
    print(f"    rf_evaluation.png")
    print(HEADER)
