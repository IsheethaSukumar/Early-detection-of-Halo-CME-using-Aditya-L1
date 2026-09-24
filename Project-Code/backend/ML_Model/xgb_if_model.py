"""
XGBoost + Isolation Forest Anomaly Score
=========================================

Identical to xgb_model.py EXCEPT the input CSV is
iforest_augmented.csv, which adds two extra features:
  anomaly_score  (float)  -- IF score: higher = more anomalous
  is_anomaly     (0/1)    -- binary flag from IF threshold

Artifacts saved with "_if" suffix to co-exist alongside baseline.

Usage:
    python ML_Model/xgb_if_model.py
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
    import xgboost as xgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    # pyrefly: ignore [missing-import]
    import xgboost as xgb

try:
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE

# -- Paths ---------------------------------------------------------------------
ARTIFACT_DIR  = Path(__file__).resolve().parent / "artifacts"
RESULTS_DIR   = Path(__file__).resolve().parent / "RESULTS"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTED_CSV = ARTIFACT_DIR / "iforest_augmented.csv"

HEADER    = "=" * 64
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
            "Run iforest_model.py first."
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
    if_cols = [c for c in df.columns if "anomaly" in c.lower()]
    print(f"      IF features : {if_cols}")
    print(f"        anomaly_score range : [{df['anomaly_score'].min():.4f}, {df['anomaly_score'].max():.4f}]")
    return df


# -- 2. Feature engineering ----------------------------------------------------
def feature_engineering(df: pd.DataFrame):
    print("\n[2/6] Feature engineering  (anomaly_score + is_anomaly included) ...")
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)
    y = df["cme_label"].values

    n_neg = int((y == 0).sum())
    n_pos = int(y.sum())
    spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
    print(f"      Total features     : {X.shape[1]}  (baseline 57 + 2 IF features)")
    print(f"      scale_pos_weight   : {spw}")
    return X, y, feat_cols, spw


# -- 3. Split + Scale ----------------------------------------------------------
def split_and_scale(X, y):
    print("\n[3/6] Stratified 80/20 split + StandardScaler ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    path = ARTIFACT_DIR / "xgb_if_scaler.pkl"
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"      Train : {len(y_train):,}  |  CME: {int(y_train.sum()):,}")
    print(f"      Test  : {len(y_test):,}   |  CME: {int(y_test.sum()):,}")
    print(f"      Scaler saved -> {path}")
    return X_tr_s, X_te_s, y_train, y_test


# -- 4. No SMOTE for XGB+IF ---------------------------------------------------
# XGBoost handles imbalance natively via scale_pos_weight.
# SMOTE is intentionally skipped here: interpolating synthetic anomaly_score
# values creates unrealistic IF signals that degrade XGBoost's gradients.
def skip_smote(X_train, y_train):
    n_cme   = int(y_train.sum())
    n_nocme = int((y_train == 0).sum())
    print(f"\n[4/6] Skipping SMOTE for XGB+IF (scale_pos_weight handles imbalance)")
    print(f"      Train set -> CME: {n_cme:,}  |  No-CME: {n_nocme:,}  (original ratio preserved)")
    return X_train, y_train


# -- 5. Train ------------------------------------------------------------------
def train(X_train, y_train, spw: float):
    print("\n[5/6] Training XGBoost ...")
    t0 = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=spw,
        use_label_encoder=False,
        eval_metric="auc", random_state=42,
        n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train)
    print(f"      Trained in {time.time()-t0:.1f}s")
    return model


# -- 6. Evaluate ---------------------------------------------------------------
def evaluate(model, X_test, y_test, feat_cols):
    print("\n[6/6] Evaluation + precision-recall threshold tuning ...")

    y_prob  = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)
    f1_arr      = 2*prec_arr[:-1]*rec_arr[:-1] / (prec_arr[:-1]+rec_arr[:-1]+1e-9)
    thr_best_f1 = thresholds[int(np.argmax(f1_arr))]
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

    (ARTIFACT_DIR / "xgb_if_threshold.txt").write_text(str(chosen_thr))
    _save_feature_importance(model, feat_cols)
    _plot_roc_pr(y_test, y_prob, roc_auc, pr_auc)
    return roc_auc, pr_auc, chosen_thr


# -- Feature importance --------------------------------------------------------
def _save_feature_importance(model, feat_cols):
    booster = model.get_booster()
    scores  = booster.get_score(importance_type="gain")
    fi = pd.Series(scores).reindex(feat_cols, fill_value=0).sort_values(ascending=False)
    fi.reset_index().rename(columns={"index":"feature", 0:"importance"}).to_csv(
        ARTIFACT_DIR / "xgb_if_feature_importance.csv", index=False
    )

    colors = ["#e05c5c" if "anomaly" in c else "teal" for c in fi.head(25).index]
    fig, ax = plt.subplots(figsize=(10, 7))
    fi.head(25).plot(kind="bar", ax=ax, color=colors)
    ax.set_title("Top 25 Feature Importances (Gain) — XGBoost + IF\n"
                 "(red = Isolation Forest features)", fontsize=11)
    ax.set_ylabel("Gain")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "xgb_if_feature_importance.png", dpi=150)
    plt.close()

    print(f"\n  IF feature ranks:")
    for c in ["anomaly_score", "is_anomaly"]:
        if c in fi.index:
            rank = list(fi.index).index(c) + 1
            print(f"    {c:<20} rank #{rank:>3}  gain={fi[c]:.2f}")


# -- ROC + PR plots ------------------------------------------------------------
def _plot_roc_pr(y_test, y_prob, roc_auc, pr_auc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("XGBoost + IF Anomaly Score -- Test Set",
                 fontsize=13, fontweight="bold")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="teal", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--", lw=1)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="darkorange", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "xgb_if_evaluation.png", dpi=150)
    plt.close()
    print(f"  Plot saved -> {ARTIFACT_DIR / 'xgb_if_evaluation.png'}")


# -- Comparison summary --------------------------------------------------------
def print_comparison(roc_auc: float):
    BASELINE_ROC = 0.8647   # from xgb_res.txt  (update if you have the exact value)
    delta = roc_auc - BASELINE_ROC
    sign  = "+" if delta >= 0 else ""
    print(f"\n  {'='*48}")
    print(f"  MODEL COMPARISON  (XGB baseline vs XGB + IF)")
    print(f"  {'='*48}")
    print(f"  {'Metric':<18} {'Baseline':>10} {'+ IF Score':>12} {'Delta':>8}")
    print(f"  {'-'*48}")
    print(f"  {'ROC-AUC':<18} {BASELINE_ROC:>10.4f} {roc_auc:>12.4f} {sign}{delta:>7.4f}")
    print(f"  {'='*48}")


# -- Save model ----------------------------------------------------------------
def save_model(model) -> Path:
    print("\n[Save] Persisting XGB+IF model ...")
    pkl = ARTIFACT_DIR / "xgb_if_model.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(model, f)
    jsn = ARTIFACT_DIR / "xgb_if_model.json"
    model.save_model(str(jsn))
    print(f"  Pickle -> {pkl}  ({pkl.stat().st_size//1024:,} KB)")
    print(f"  JSON   -> {jsn}  ({jsn.stat().st_size//1024:,} KB)")
    return pkl


# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    log_path = RESULTS_DIR / "xgb_if_res.txt"
    log_fh   = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_fh)

    try:
        banner("SOLAR HALO CME  --  XGBoost + Isolation Forest Score")

        df                      = load_data()
        X, y, cols, spw         = feature_engineering(df)
        X_tr, X_te, y_tr, y_te = split_and_scale(X, y)
        X_tr, y_tr              = skip_smote(X_tr, y_tr)
        model                   = train(X_tr, y_tr, spw)
        roc, pr, thr            = evaluate(model, X_te, y_te, cols)
        save_model(model)
        print_comparison(roc)

        print(f"\n{HEADER}")
        print("  DONE -- artifacts/ now contains:")
        print(f"    xgb_if_model.pkl / xgb_if_model.json")
        print(f"    xgb_if_scaler.pkl")
        print(f"    xgb_if_threshold.txt          (threshold={thr:.4f})")
        print(f"    xgb_if_feature_importance.csv/png")
        print(f"    xgb_if_evaluation.png")
        print(f"\n  RESULTS/xgb_if_res.txt         (this log)")
        print(HEADER)

    finally:
        sys.stdout = sys.__stdout__
        log_fh.close()
        print(f"\n  Results saved -> {log_path}")
