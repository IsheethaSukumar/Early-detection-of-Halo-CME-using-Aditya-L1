"""
Random Forest Hyperparameter Tuning — Optuna
=============================================
Usage:
    python ML_Model/rf_tune.py
"""

# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import pickle
import warnings
import time
from pathlib import Path

# pyrefly: ignore [missing-import]
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, average_precision_score,
    precision_recall_curve, roc_curve,
)
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

try:
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE
    # pyrefly: ignore [missing-import]
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"])
    # pyrefly: ignore [missing-import]
    from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent / "Dataset"
PROC_DIR     = BASE_DIR / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

N_TRIALS     = 30        # Optuna trials (more = better, but slower)
CV_FOLDS     = 3         # cross-val folds inside each trial
HEADER       = "=" * 64
DROP_COLS    = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]


def banner(title):
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# ── 1. Load + prepare data ────────────────────────────────────────────────────
def load_and_prepare():
    print("\n[1/5] Loading & preparing data ...")
    df = pd.read_csv(PROC_DIR / "full_merged_dataset.csv")
    df.dropna(subset=["cme_label"], inplace=True)
    df["cme_label"] = df["cme_label"].astype(int)

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True))
    y = df["cme_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # SMOTE on train only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)

    print(f"      Train (after SMOTE) : {X_res.shape[0]:,}  |  features: {X_res.shape[1]}")
    print(f"      Test                : {X_test_s.shape[0]:,}")
    return X_res, X_test_s, y_res, y_test, feat_cols, scaler


# ── 2. Optuna objective ────────────────────────────────────────────────────────
def make_objective(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators",      100, 600, step=50),
            "max_depth"        : trial.suggest_int("max_depth",            3,  30),
            "min_samples_split": trial.suggest_int("min_samples_split",    2,  20),
            "min_samples_leaf" : trial.suggest_int("min_samples_leaf",     1,  20),
            "max_features"     : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "random_state"     : 42,
            "n_jobs"           : -1,
        }
        rf = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        return scores.mean()
    return objective


# ── 3. Run Optuna ─────────────────────────────────────────────────────────────
def run_optuna(X_train, y_train):
    print(f"\n[2/5] Running Optuna ({N_TRIALS} trials, {CV_FOLDS}-fold CV) ...")
    study = optuna.create_study(direction="maximize")
    t0 = time.time()
    study.optimize(make_objective(X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)
    elapsed = time.time() - t0

    print(f"\n      Finished {N_TRIALS} trials in {elapsed:.1f}s")
    print(f"      Best ROC-AUC (CV) : {study.best_value:.4f}")
    print(f"      Best params       :")
    for k, v in study.best_params.items():
        print(f"        {k:<25} = {v}")
    return study.best_params


# ── 4. Retrain with best params ────────────────────────────────────────────────
def retrain(X_train, y_train, best_params):
    print("\n[3/5] Retraining with best params ...")
    t0 = time.time()
    rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"      Trained in {time.time()-t0:.1f}s")
    return rf


# ── 5. Evaluate ───────────────────────────────────────────────────────────────
def evaluate(rf, X_test, y_test, feat_cols):
    print("\n[4/5] Evaluating on test set ...")
    y_prob = rf.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)

    # Best F1 threshold
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)
    f1_arr      = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + 1e-9)
    best_f1_idx = int(np.argmax(f1_arr))
    chosen_thr  = float(thresholds[best_f1_idx])
    y_pred      = (y_prob >= chosen_thr).astype(int)

    print(f"\n  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  Chosen threshold (Best F1) : {chosen_thr:.4f}")

    print(f"\n  === Full Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No CME", "Halo CME"]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("  Confusion Matrix:")
    print(f"    TN (correct No-CME) : {tn:,}")
    print(f"    FP (false alarm)     : {fp:,}")
    print(f"    FN (missed CME)      : {fn:,}")
    print(f"    TP (caught CME)      : {tp:,}")

    # Save threshold
    thr_path = ARTIFACT_DIR / "rf_tuned_threshold.txt"
    thr_path.write_text(str(chosen_thr))
    print(f"\n  Threshold saved -> {thr_path}")

    _plot(y_test, y_prob, roc_auc, pr_auc, rf, feat_cols)
    return roc_auc, chosen_thr


def _plot(y_test, y_prob, roc_auc, pr_auc, rf, feat_cols):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RF Tuned — Test Set", fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0,1],[0,1],"k--"); axes[0].set_title("ROC Curve"); axes[0].legend()
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="darkorange", label=f"PR-AUC={pr_auc:.3f}")
    axes[1].set_title("PR Curve"); axes[1].legend()
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")

    fi = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
    fi.head(15).plot(kind="bar", ax=axes[2], color="seagreen")
    axes[2].set_title("Top 15 Features"); axes[2].set_xlabel("")

    plt.tight_layout()
    out = ARTIFACT_DIR / "rf_tuned_evaluation.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Plot saved -> {out}")


# ── 6. Save ───────────────────────────────────────────────────────────────────
def save(rf, best_params):
    print("\n[5/5] Saving tuned model ...")
    out = ARTIFACT_DIR / "random_forest_tuned.pkl"
    with open(out, "wb") as f:
        pickle.dump(rf, f)
    size_kb = out.stat().st_size // 1024
    print(f"  Model saved -> {out}  ({size_kb:,} KB)")

    params_df = pd.DataFrame([best_params])
    params_df.to_csv(ARTIFACT_DIR / "rf_best_params.csv", index=False)
    print(f"  Params saved -> {ARTIFACT_DIR / 'rf_best_params.csv'}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    banner("SOLAR HALO CME  --  RF Hyperparameter Tuning (Optuna)")

    X_tr, X_te, y_tr, y_te, cols, scaler = load_and_prepare()
    best_params = run_optuna(X_tr, y_tr)
    rf_tuned    = retrain(X_tr, y_tr, best_params)
    roc, thr    = evaluate(rf_tuned, X_te, y_te, cols)
    save(rf_tuned, best_params)

    print(f"\n{HEADER}")
    print("  DONE -- ML_Model/artifacts/ now contains:")
    print("    random_forest_tuned.pkl")
    print("    rf_tuned_threshold.txt")
    print("    rf_tuned_evaluation.png")
    print("    rf_best_params.csv")
    print(f"  Final ROC-AUC : {roc:.4f}")
    print(HEADER)
