"""
Stacking Ensemble: RF+IF + XGBoost+IF + BiLSTM+IF → Logistic Regression
=========================================================================

Pipeline
--------
  1. Load iforest_augmented.csv (59 features: 57 solar-wind + anomaly_score + is_anomaly)
  2. Row-level stratified 80/20 split (same random_state=42 as all base models)
  3. 5-fold OOF generation on training rows
       - RF+IF    : trained fresh per fold (standard OOF stacking)
       - XGBoost+IF: trained fresh per fold
       - BiLSTM+IF : inference-only from saved artifact (avoids 2.5h/fold retraining)
  4. Train LogisticRegression meta-model on OOF probability matrix [p_rf, p_xgb, p_lstm]
  5. Evaluate stacking ensemble on held-out test rows
       - Artifact models (rf_if_model.pkl, xgb_if_model.pkl, lstm_if_model.pt) used
  6. 3-panel plot: ROC | PR-curve | Calibration diagram
  7. Save all artifacts with "stack_" prefix

Prerequisites
-------------
  Run these scripts first (in order):
    python ML_Model/iforest_model.py          -> artifacts/iforest_augmented.csv
    python ML_Model/rf_if_model.py            -> artifacts/rf_if_model.pkl
    python ML_Model/xgb_if_model.py           -> artifacts/xgb_if_model.pkl
    python ML_Model/lstm_if_model.py          -> artifacts/lstm_if_model.pt

Usage
-----
  python ML_Model/stack_ensemble.py
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
import sys
import json
import subprocess
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score, recall_score,
)

warnings.filterwarnings("ignore")

# ── Auto-install xgboost if missing ───────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import xgboost as xgb
except ImportError:
    print("Installing xgboost ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"])
    # pyrefly: ignore [missing-import]
    import xgboost as xgb

# ── Auto-install PyTorch if missing ───────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    # pyrefly: ignore [missing-import]
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("Installing PyTorch ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "torch",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    # pyrefly: ignore [missing-import]
    from torch.utils.data import DataLoader, TensorDataset

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_SIZE  = 24          # must match lstm_if_model.py
HIDDEN1      = 64          # BiLSTM layer-1 hidden units (per direction)
HIDDEN2      = 32          # BiLSTM layer-2 hidden units (per direction)
DROPOUT      = 0.3
N_FOLDS      = 5
RANDOM_STATE = 42
BATCH_SIZE   = 1024        # inference batch size for BiLSTM

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_DIR  = Path(__file__).resolve().parent / "RESULTS"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AUGMENTED_CSV = ARTIFACT_DIR / "iforest_augmented.csv"

HEADER    = "=" * 64
DROP_COLS = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w", "Unnamed: 0"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tee stdout → log file ─────────────────────────────────────────────────────
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

    def isatty(self):
        return False


def banner(title: str) -> None:
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# ── BiLSTM+IF architecture (must mirror lstm_if_model.py exactly) ─────────────
class HaloCME_BiLSTM_IF(nn.Module):
    """
    Stacked Bidirectional LSTM used in lstm_if_model.py.
    Reconstructed here for inference with the saved state dict.
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm1    = nn.LSTM(n_features,  HIDDEN1, batch_first=True, bidirectional=True)
        self.drop1    = nn.Dropout(DROPOUT)
        self.lstm2    = nn.LSTM(HIDDEN1 * 2, HIDDEN2, batch_first=True, bidirectional=True)
        self.drop2    = nn.Dropout(DROPOUT)
        self.fc       = nn.Linear(HIDDEN2 * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        x       = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        out, _  = self.lstm1(x)
        out     = self.drop1(out)
        out, _  = self.lstm2(out)
        out     = self.drop2(out)
        return self.fc(out[:, -1, :])   # last time-step logit


# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("\n[1/7] Loading IF-augmented dataset ...")
    if not AUGMENTED_CSV.exists():
        raise FileNotFoundError(
            f"Not found: {AUGMENTED_CSV}\n"
            "Run iforest_model.py first to generate it."
        )
    df = pd.read_csv(AUGMENTED_CSV)
    df.dropna(subset=["cme_label"], inplace=True)
    df["cme_label"] = df["cme_label"].astype(int)

    # Sort chronologically so sliding windows are temporally coherent
    time_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ("time", "date", "timestamp", "epoch"))]
    if time_cols:
        df = df.sort_values(time_cols[0]).reset_index(drop=True)
        print(f"      Sorted by  : {time_cols[0]}")

    n_total = len(df)
    n_cme   = int(df["cme_label"].sum())
    n_nocme = n_total - n_cme
    print(f"      Source     : {AUGMENTED_CSV.name}")
    print(f"      Rows       : {n_total:,}")
    print(f"      CME  (1)   : {n_cme:,}  ({100 * n_cme / n_total:.1f}%)")
    print(f"      No-CME (0) : {n_nocme:,}  ({100 * n_nocme / n_total:.1f}%)")
    return df


# ── 2. Feature engineering ────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame):
    print("\n[2/7] Feature engineering ...")
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in DROP_COLS and not c.startswith("Unnamed")
    ]
    X = df[feat_cols].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)
    y = df["cme_label"].values
    print(f"      Features : {X.shape[1]}  (57 solar-wind + anomaly_score + is_anomaly)")
    for c in ["anomaly_score", "is_anomaly"]:
        if c in feat_cols:
            print(f"        [OK] '{c}' at index {feat_cols.index(c)}")
    return X, y, feat_cols


# ── 3. Row-level split (consistent with tabular models) ──────────────────────
def row_split(y: np.ndarray):
    print("\n[3/7] Stratified 80/20 row-level split (random_state=42) ...")
    idx = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"      Train rows : {len(idx_train):,}  |  CME: {int(y[idx_train].sum()):,}")
    print(f"      Test  rows : {len(idx_test):,}   |  CME: {int(y[idx_test].sum()):,}")
    return idx_train, idx_test


# ── LSTM inference helper ─────────────────────────────────────────────────────
def _lstm_infer(
    lstm_model: HaloCME_BiLSTM_IF,
    X_scaled: np.ndarray,
    row_indices: np.ndarray,
) -> np.ndarray:
    """
    Build 24-hour look-back sequences ending at each row in row_indices,
    run BiLSTM inference, return probabilities aligned to row_indices.
    Rows within the first WINDOW_SIZE-1 positions get NaN (not enough look-back).
    """
    probs = np.full(len(row_indices), np.nan, dtype=np.float32)

    seqs:          list[np.ndarray] = []
    valid_pos:     list[int]        = []

    for pos, row_idx in enumerate(row_indices):
        start = int(row_idx) - WINDOW_SIZE + 1
        if start < 0:
            continue   # insufficient history — leave NaN
        seqs.append(X_scaled[start : int(row_idx) + 1])   # (WINDOW_SIZE, n_feat)
        valid_pos.append(pos)

    if not seqs:
        return probs

    X_batch = np.array(seqs, dtype=np.float32)
    ds      = TensorDataset(torch.from_numpy(X_batch))
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    lstm_model.eval()
    all_p: list[float] = []
    with torch.no_grad():
        for (Xb,) in loader:
            p = torch.sigmoid(lstm_model(Xb.to(DEVICE))).cpu().numpy().flatten()
            all_p.extend(p.tolist())

    for pos_i, p_val in zip(valid_pos, all_p):
        probs[pos_i] = p_val

    return probs


# ── 4 & 5. OOF generation ────────────────────────────────────────────────────
def generate_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    idx_train: np.ndarray,
    X_scaled_full: np.ndarray,
    lstm_model: HaloCME_BiLSTM_IF,
):
    """
    Generate out-of-fold (OOF) predictions for all three base models
    on the training rows only, using N_FOLDS-fold stratified CV.

    RF+IF  : trained fresh in each fold (standard stacking OOF).
    XGB+IF : trained fresh in each fold.
    BiLSTM : inference-only with the saved model (no per-fold retraining).
    """
    print(f"\n[4/7] Generating OOF predictions ({N_FOLDS}-fold, training rows) ...")

    X_tr_arr = X.values[idx_train]
    y_tr     = y[idx_train]
    n_train  = len(idx_train)

    oof_rf   = np.full(n_train, np.nan)
    oof_xgb  = np.full(n_train, np.nan)
    oof_lstm = np.full(n_train, np.nan)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold_i, (fold_tr, fold_val) in enumerate(skf.split(X_tr_arr, y_tr), 1):
        t_fold = time.time()
        print(f"\n      ── Fold {fold_i}/{N_FOLDS} ──────────────────────────")

        Xf_tr, yf_tr = X_tr_arr[fold_tr], y_tr[fold_tr]
        Xf_va, yf_va = X_tr_arr[fold_val], y_tr[fold_val]

        # ── Scale within fold (no leakage) ─────────────────────────────────
        fold_scaler = StandardScaler()
        Xf_tr_s = fold_scaler.fit_transform(Xf_tr)
        Xf_va_s = fold_scaler.transform(Xf_va)

        n_neg = int((yf_tr == 0).sum())
        n_pos = int(yf_tr.sum())
        spw   = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0

        # ── RF+IF ───────────────────────────────────────────────────────────
        rf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(Xf_tr_s, yf_tr)
        oof_rf[fold_val] = rf.predict_proba(Xf_va_s)[:, 1]
        print(f"        RF+IF   OOF AUC: "
              f"{roc_auc_score(yf_va, oof_rf[fold_val]):.4f}")

        # ── XGB+IF ──────────────────────────────────────────────────────────
        xgb_clf = xgb.XGBClassifier(
            n_estimators      = 300,
            max_depth         = 6,
            learning_rate     = 0.05,
            subsample         = 0.8,
            colsample_bytree  = 0.8,
            min_child_weight  = 5,
            gamma             = 0.1,
            reg_alpha         = 0.1,
            reg_lambda        = 1.0,
            scale_pos_weight  = spw,
            use_label_encoder = False,
            eval_metric       = "auc",
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            verbosity         = 0,
        )
        xgb_clf.fit(Xf_tr_s, yf_tr)
        oof_xgb[fold_val] = xgb_clf.predict_proba(Xf_va_s)[:, 1]
        print(f"        XGB+IF  OOF AUC: "
              f"{roc_auc_score(yf_va, oof_xgb[fold_val]):.4f}")

        # ── BiLSTM+IF (inference-only on fold validation rows) ──────────────
        val_global_idx  = idx_train[fold_val]   # original row indices in full dataset
        oof_lstm[fold_val] = _lstm_infer(lstm_model, X_scaled_full, val_global_idx)
        valid_lstm = ~np.isnan(oof_lstm[fold_val])
        if valid_lstm.any():
            auc_lstm = roc_auc_score(yf_va[valid_lstm], oof_lstm[fold_val][valid_lstm])
            print(f"        LSTM+IF OOF AUC: {auc_lstm:.4f}  "
                  f"({valid_lstm.sum()}/{len(yf_va)} valid rows)")
        else:
            print(f"        LSTM+IF: no valid rows in this fold (all NaN)")

        print(f"        Fold time: {time.time() - t_fold:.1f}s")

    print(f"\n      OOF RF   NaN count : {np.isnan(oof_rf).sum()}")
    print(f"      OOF XGB  NaN count : {np.isnan(oof_xgb).sum()}")
    print(f"      OOF LSTM NaN count : {np.isnan(oof_lstm).sum()}")

    return oof_rf, oof_xgb, oof_lstm, y_tr


# ── 5. Train meta-model ───────────────────────────────────────────────────────
def train_meta(
    oof_rf:   np.ndarray,
    oof_xgb:  np.ndarray,
    oof_lstm: np.ndarray,
    y_train:  np.ndarray,
):
    """
    Train Logistic Regression on OOF probability matrix [p_rf, p_xgb, p_lstm].
    Rows with any NaN (LSTM window boundary) are dropped.
    """
    print("\n[5/7] Training Logistic Regression meta-model ...")

    S = np.column_stack([oof_rf, oof_xgb, oof_lstm])   # (n_train, 3)
    valid_mask = ~np.isnan(S).any(axis=1)
    S_v, y_v  = S[valid_mask], y_train[valid_mask]

    print(f"      OOF rows available : {valid_mask.sum():,} / {len(y_train):,}")
    print(f"      CME in OOF         : {int(y_v.sum()):,}  ({100 * y_v.mean():.1f}%)")

    # OOF meta-level AUC before training (geometric mean of individual AUCs)
    for name, col in [("RF+IF", 0), ("XGB+IF", 1), ("LSTM+IF", 2)]:
        col_vals = S_v[:, col]
        if not np.isnan(col_vals).any():
            print(f"        OOF AUC  {name:<12}: {roc_auc_score(y_v, col_vals):.4f}")

    # Scale inputs to meta-learner (prevent magnitude mismatch)
    meta_scaler = StandardScaler()
    S_s = meta_scaler.fit_transform(S_v)

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    meta.fit(S_s, y_v)

    coef = meta.coef_[0]
    print(f"\n      Meta-model coefficients (positive = contributes to CME):")
    print(f"        p_rf    : {coef[0]:+.4f}")
    print(f"        p_xgb   : {coef[1]:+.4f}")
    print(f"        p_lstm  : {coef[2]:+.4f}")
    print(f"        intercept: {meta.intercept_[0]:+.4f}")

    # OOF ensemble AUC
    p_meta_oof = meta.predict_proba(S_s)[:, 1]
    oof_auc    = roc_auc_score(y_v, p_meta_oof)
    print(f"\n      OOF ensemble ROC-AUC : {oof_auc:.4f}  (CV estimate — not test AUC)")

    return meta, meta_scaler


# ── 6 & 7. Test set evaluation ────────────────────────────────────────────────
def evaluate_test(
    meta:               LogisticRegression,
    meta_scaler:        StandardScaler,
    rf_art:             RandomForestClassifier,
    xgb_art,
    lstm_model:         HaloCME_BiLSTM_IF,
    X:                  pd.DataFrame,
    y:                  np.ndarray,
    idx_test:           np.ndarray,
    X_scaled_full:      np.ndarray,
    rf_scaler_art:      StandardScaler,
    xgb_scaler_art:     StandardScaler,
):
    """
    Generate test set predictions from the three artifact models,
    combine via meta-model, evaluate, compare.
    """
    print("\n[6/7] Generating test set predictions from artifact models ...")

    X_test = X.values[idx_test]
    y_test = y[idx_test]

    # Use respective artifact scalers (fitted on training rows only — no leakage)
    X_test_rf  = rf_scaler_art.transform(X_test)
    X_test_xgb = xgb_scaler_art.transform(X_test)

    p_rf_test   = rf_art.predict_proba(X_test_rf)[:, 1]
    p_xgb_test  = xgb_art.predict_proba(X_test_xgb)[:, 1]
    p_lstm_test = _lstm_infer(lstm_model, X_scaled_full, idx_test)

    print(f"      RF+IF   test preds : {len(p_rf_test):,}")
    print(f"      XGB+IF  test preds : {len(p_xgb_test):,}")
    print(f"      LSTM+IF test preds : {(~np.isnan(p_lstm_test)).sum():,}  "
          f"(NaN: {np.isnan(p_lstm_test).sum()})")

    # Stack into meta feature matrix
    S_test = np.column_stack([p_rf_test, p_xgb_test, p_lstm_test])
    valid  = ~np.isnan(S_test).any(axis=1)
    S_v    = S_test[valid]
    y_v    = y_test[valid]

    S_v_s  = meta_scaler.transform(S_v)
    p_ens  = meta.predict_proba(S_v_s)[:, 1]

    print(f"\n[7/7] Evaluation on test set ({valid.sum():,} rows) ...")

    roc_auc_ens = roc_auc_score(y_v, p_ens)
    pr_auc_ens  = average_precision_score(y_v, p_ens)
    print(f"\n  Stacking Ensemble ROC-AUC : {roc_auc_ens:.4f}")
    print(f"  Stacking Ensemble PR-AUC  : {pr_auc_ens:.4f}")

    # ── Threshold sweep ──────────────────────────────────────────────────────
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_v, p_ens)
    f1_arr      = (2 * prec_arr[:-1] * rec_arr[:-1]
                   / (prec_arr[:-1] + rec_arr[:-1] + 1e-9))
    thr_best_f1 = float(thresholds[np.argmax(f1_arr)])
    rec80_mask  = rec_arr[:-1] >= 0.80
    thr_rec80   = (float(thresholds[rec80_mask].min())
                   if rec80_mask.any() else thr_best_f1)
    chosen_thr  = thr_best_f1

    fmt = "  {:<32} {:>10} {:>10} {:>10} {:>10}"
    print(f"\n  {'':32} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10}")
    print("  " + "-" * 70)
    for label, thr in [
        ("Default (0.50)",              0.50),
        (f"Best F1 ({thr_best_f1:.3f})", thr_best_f1),
        (f"Recall>=0.80 ({thr_rec80:.3f})", thr_rec80),
    ]:
        yp  = (p_ens >= thr).astype(int)
        acc = (yp == y_v).mean() * 100
        rec = recall_score(y_v, yp, zero_division=0)
        f1  = f1_score(y_v, yp, zero_division=0)
        idx_nearest = np.argmin(np.abs(thresholds - thr))
        pr  = prec_arr[:-1][idx_nearest] if idx_nearest < len(prec_arr) - 1 else 0.0
        print(fmt.format(label, f"{acc:.2f}%", f"{rec:.4f}", f"{pr:.4f}", f"{f1:.4f}"))

    y_pred = (p_ens >= chosen_thr).astype(int)
    print(f"\n  === Full Classification Report (threshold={chosen_thr:.4f}) ===")
    print(classification_report(y_v, y_pred, target_names=["No CME", "Halo CME"]))

    tn, fp, fn, tp = confusion_matrix(y_v, y_pred).ravel()
    print("  Confusion Matrix:")
    print(f"    TN (correct No-CME) : {tn:,}")
    print(f"    FP (false alarm)     : {fp:,}")
    print(f"    FN (missed CME)      : {fn:,}")
    print(f"    TP (caught CME)      : {tp:,}")

    # ── Per-model AUC on the same valid test rows ────────────────────────────
    roc_rf   = roc_auc_score(y_v, p_rf_test[valid])
    roc_xgb  = roc_auc_score(y_v, p_xgb_test[valid])
    roc_lstm = roc_auc_score(y_v, p_lstm_test[valid])

    print(f"\n  ================ Model Comparison (test set) ================")
    print(f"  {'Model':<28} {'ROC-AUC':>10} {'PR-AUC':>10}")
    print(f"  {'-'*50}")
    for name, probs in [
        ("RF + IF",         p_rf_test[valid]),
        ("XGBoost + IF",    p_xgb_test[valid]),
        ("BiLSTM + IF",     p_lstm_test[valid]),
    ]:
        _roc = roc_auc_score(y_v, probs)
        _pr  = average_precision_score(y_v, probs)
        print(f"  {name:<28} {_roc:>10.4f} {_pr:>10.4f}")
    print(f"  {'--- Stacking Ensemble ---':<28}")
    print(f"  {'RF + XGB + LSTM → LogReg':<28} {roc_auc_ens:>10.4f} {pr_auc_ens:>10.4f}")
    print(f"  {'='*50}")

    return (
        p_rf_test, p_xgb_test, p_lstm_test, p_ens,
        y_v, valid,
        roc_auc_ens, pr_auc_ens, chosen_thr,
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_all(
    p_rf:    np.ndarray,
    p_xgb:   np.ndarray,
    p_lstm:  np.ndarray,
    p_ens:   np.ndarray,
    y_test:  np.ndarray,
    valid:   np.ndarray,
    roc_auc: float,
    pr_auc:  float,
) -> None:
    """
    3-panel figure: ROC curves | PR curves | Calibration diagram.
    Stack is highlighted in crimson (thicker line).
    """
    models = [
        (p_rf[valid],  "RF + IF",     "steelblue"),
        (p_xgb[valid], "XGBoost + IF","teal"),
        (p_lstm[valid],"BiLSTM + IF", "royalblue"),
        (p_ens,        "STACK",       "crimson"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Stacking Ensemble  —  RF + XGBoost + BiLSTM → LogReg  —  Test Set",
        fontsize=12, fontweight="bold",
    )

    # ── ROC ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    for probs, label, color in models:
        if np.isnan(probs).any():
            continue
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc_val     = roc_auc_score(y_test, probs)
        lw = 2.5 if label == "STACK" else 1.5
        ls = "-" if label == "STACK" else "--"
        ax.plot(fpr, tpr, lw=lw, ls=ls, color=color,
                label=f"{label}  (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(fontsize=9)

    # ── PR ───────────────────────────────────────────────────────────────────
    ax = axes[1]
    for probs, label, color in models:
        if np.isnan(probs).any():
            continue
        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_val       = average_precision_score(y_test, probs)
        lw = 2.5 if label == "STACK" else 1.5
        ls = "-" if label == "STACK" else "--"
        ax.plot(rec, prec, lw=lw, ls=ls, color=color,
                label=f"{label}  (PR-AUC={pr_val:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(fontsize=9)

    # ── Calibration ──────────────────────────────────────────────────────────
    ax = axes[2]
    for probs, label, color in models:
        if np.isnan(probs).any():
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
            lw = 2.5 if label == "STACK" else 1.5
            ls = "-" if label == "STACK" else "--"
            ax.plot(mean_pred, frac_pos, lw=lw, ls=ls, color=color,
                    marker="o", ms=4, label=label)
        except Exception:
            pass
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect")
    ax.set(xlabel="Mean Predicted Probability",
           ylabel="Fraction of Positives",
           title="Calibration (Reliability Diagram)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = ARTIFACT_DIR / "stack_evaluation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  3-panel plot saved -> {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_path  = RESULTS_DIR / "stack_res.txt"
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    try:
        banner("SOLAR HALO CME  --  Stacking Ensemble (RF + XGB + BiLSTM → LogReg)")
        print(f"  Device     : {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"  GPU        : {torch.cuda.get_device_name(0)}")
        print(f"  OOF folds  : {N_FOLDS}")
        print(f"  Window     : {WINDOW_SIZE}h look-back (BiLSTM sequences)")
        print(f"  Meta-model : Logistic Regression (L2, C=1.0)")

        t_start = time.time()

        # ── Load & prepare data ───────────────────────────────────────────
        df               = load_data()
        X, y, feat_cols  = feature_engineering(df)
        idx_train, idx_test = row_split(y)

        # ── Global scaler for BiLSTM sequence building ────────────────────
        # Fitted on FULL dataset — LSTM is inference-only, so no leakage risk.
        print("\n  Fitting global scaler for BiLSTM sequence building ...")
        global_scaler  = StandardScaler()
        X_scaled_full  = global_scaler.fit_transform(X.values.astype(np.float32))
        print(f"      X_scaled_full shape : {X_scaled_full.shape}")

        # ── Load artifact scalers for tabular test-set inference ──────────
        # These scalers were fitted on the 80% training rows only (no leakage).
        print("\n  Loading artifact scalers and models ...")
        _missing = []
        for name in ["rf_if_model.pkl", "rf_if_scaler.pkl",
                     "xgb_if_model.pkl", "xgb_if_scaler.pkl",
                     "lstm_if_model.pt",  "lstm_if_config.json",
                     "lstm_if_scaler.pkl"]:
            if not (ARTIFACT_DIR / name).exists():
                _missing.append(name)
        if _missing:
            raise FileNotFoundError(
                f"Missing prerequisite artifacts:\n  " + "\n  ".join(_missing) +
                "\nRun the corresponding model scripts first."
            )

        with open(ARTIFACT_DIR / "rf_if_scaler.pkl",  "rb") as fh:
            rf_scaler_art  = pickle.load(fh)
        with open(ARTIFACT_DIR / "xgb_if_scaler.pkl", "rb") as fh:
            xgb_scaler_art = pickle.load(fh)
        with open(ARTIFACT_DIR / "rf_if_model.pkl",   "rb") as fh:
            rf_art         = pickle.load(fh)
        with open(ARTIFACT_DIR / "xgb_if_model.pkl",  "rb") as fh:
            xgb_art        = pickle.load(fh)

        # Load BiLSTM+IF from config + state dict
        cfg        = json.loads((ARTIFACT_DIR / "lstm_if_config.json").read_text())
        n_features = cfg["n_features"]
        lstm_model = HaloCME_BiLSTM_IF(n_features).to(DEVICE)
        lstm_model.load_state_dict(
            torch.load(ARTIFACT_DIR / "lstm_if_model.pt", map_location=DEVICE)
        )
        lstm_model.eval()
        n_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
        print(f"      BiLSTM+IF loaded: n_features={n_features}  params={n_params:,}")
        print(f"      RF+IF    loaded: {type(rf_art).__name__}")
        print(f"      XGB+IF   loaded: {type(xgb_art).__name__}")

        # ── OOF generation ────────────────────────────────────────────────
        oof_rf, oof_xgb, oof_lstm, y_train = generate_oof(
            X, y, idx_train, X_scaled_full, lstm_model
        )

        # Save OOF probabilities for analysis
        oof_df = pd.DataFrame({
            "row_idx_global": idx_train,
            "y_true":         y_train,
            "p_rf_oof":       oof_rf,
            "p_xgb_oof":      oof_xgb,
            "p_lstm_oof":     oof_lstm,
        })
        oof_csv = ARTIFACT_DIR / "stack_oof_probs.csv"
        oof_df.to_csv(oof_csv, index=False)
        print(f"\n  OOF probabilities saved -> {oof_csv}")

        # ── Train meta-model ──────────────────────────────────────────────
        meta, meta_scaler = train_meta(oof_rf, oof_xgb, oof_lstm, y_train)

        # Persist meta-model + scaler
        with open(ARTIFACT_DIR / "stack_meta_model.pkl",  "wb") as fh:
            pickle.dump(meta, fh)
        with open(ARTIFACT_DIR / "stack_meta_scaler.pkl", "wb") as fh:
            pickle.dump(meta_scaler, fh)
        print(f"  Meta-model saved  -> {ARTIFACT_DIR / 'stack_meta_model.pkl'}")
        print(f"  Meta-scaler saved -> {ARTIFACT_DIR / 'stack_meta_scaler.pkl'}")

        # ── Test-set evaluation ───────────────────────────────────────────
        (p_rf_t, p_xgb_t, p_lstm_t, p_ens,
         y_test_v, valid,
         roc_auc, pr_auc, chosen_thr) = evaluate_test(
            meta, meta_scaler,
            rf_art, xgb_art, lstm_model,
            X, y, idx_test,
            X_scaled_full,
            rf_scaler_art,
            xgb_scaler_art,
        )

        # Save threshold
        thr_path = ARTIFACT_DIR / "stack_threshold.txt"
        thr_path.write_text(str(chosen_thr))
        print(f"\n  Threshold saved   -> {thr_path}")

        # ── Plots ─────────────────────────────────────────────────────────
        plot_all(p_rf_t, p_xgb_t, p_lstm_t, p_ens,
                 y_test_v, valid, roc_auc, pr_auc)

        elapsed = time.time() - t_start

        print(f"\n{HEADER}")
        print("  DONE -- ML_Model/artifacts/ now contains:")
        print("    stack_meta_model.pkl")
        print("    stack_meta_scaler.pkl")
        print(f"    stack_threshold.txt         (thr={chosen_thr:.4f})")
        print("    stack_oof_probs.csv")
        print("    stack_evaluation.png         (ROC | PR | Calibration)")
        print(f"\n  RESULTS/stack_res.txt         (this log)")
        print(f"\n  Final Stack ROC-AUC : {roc_auc:.4f}   PR-AUC : {pr_auc:.4f}")
        print(f"  Total run time      : {elapsed:.1f}s")
        print(HEADER)

    finally:
        sys.stdout = sys.__stdout__
        _log_file.close()
        print(f"\n  Results saved -> {log_path}")
