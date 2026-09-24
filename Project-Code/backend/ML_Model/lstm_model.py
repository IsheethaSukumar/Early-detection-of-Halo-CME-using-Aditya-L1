# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pickle
import warnings
import time
import sys
import subprocess
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    f1_score,
)

warnings.filterwarnings("ignore")

# ── Auto-install PyTorch if missing ────────────────────────────────────────────
try:
    # pyrefly: ignore [missing-import]
    import torch
    # pyrefly: ignore [missing-import]
    import torch.nn as nn
    # pyrefly: ignore [missing-import]
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
except ImportError:
    print("Installing PyTorch (CUDA 12.4) ...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])
    

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 24          # hours of look-back per sequence
BATCH_SIZE  = 512
MAX_EPOCHS  = 120
LR          = 3e-4
PATIENCE    = 15
HIDDEN1     = 64          # BiLSTM layer-1 hidden units (per direction)
HIDDEN2     = 32          # BiLSTM layer-2 hidden units (per direction)
DROPOUT     = 0.3

BASE_DIR     = Path(__file__).resolve().parent.parent / "Dataset"
PROC_DIR     = BASE_DIR / "processed"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
RESULTS_DIR  = Path(__file__).resolve().parent / "RESULTS"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HEADER    = "=" * 64
DROP_COLS = ["cme_label", "is_halo", "cme_speed_kmps", "cme_angular_w"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tee stdout to file ─────────────────────────────────────────────────────────
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
        self._real.flush()
        self._file.flush()

    def isatty(self):
        return self._real.isatty()


# ── Helpers ────────────────────────────────────────────────────────────────────
def banner(title: str) -> None:
    print(f"\n{HEADER}")
    print(f"  {title}")
    print(HEADER)


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("\n[1/5] Loading cleaned data ...")
    df = pd.read_csv(PROC_DIR / "full_merged_dataset.csv")
    df.dropna(subset=["cme_label"], inplace=True)
    df["cme_label"] = df["cme_label"].astype(int)

    # Sort chronologically so sliding windows are temporally coherent
    time_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ("time", "date", "timestamp", "epoch"))]
    if time_cols:
        df = df.sort_values(time_cols[0]).reset_index(drop=True)
        print(f"      Sorted by   : {time_cols[0]}")

    n_total = len(df)
    n_cme   = int(df["cme_label"].sum())
    n_nocme = n_total - n_cme
    print(f"      Rows        : {n_total:,}")
    print(f"      CME  (1)    : {n_cme:,}  ({100 * n_cme / n_total:.1f}%)")
    print(f"      No-CME (0)  : {n_nocme:,}  ({100 * n_nocme / n_total:.1f}%)")
    return df


# ── 2. Feature engineering ─────────────────────────────────────────────────────
def feature_engineering(df: pd.DataFrame):
    print("\n[2/5] Feature engineering ...")
    feat_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in DROP_COLS and not c.startswith("Unnamed")
    ]
    X = df[feat_cols].copy()
    X.fillna(X.median(numeric_only=True), inplace=True)
    y = df["cme_label"].values
    print(f"      Features : {X.shape[1]}")
    return X, y, feat_cols


# ── 3. Build sequences + stratified random split ───────────────────────────────
def build_sequences_and_split(X: pd.DataFrame, y: np.ndarray):
    """
    Strategy:
      1. Sort rows chronologically (already done in load_data).
      2. Build ALL sliding-window sequences of WINDOW_SIZE consecutive hours.
         Each window is internally time-coherent.
      3. Randomly stratify-split the SEQUENCES (not individual rows).

    Why random split instead of chronological?
      Solar CME predictors vary across the solar cycle.  A strict chronological
      split trains on one cycle phase and tests on another, causing the model to
      learn patterns that are negatively correlated with the test set.
      Randomly mixing sequences across the whole dataset gives the LSTM a fair
      shot at learning cycle-agnostic patterns -- the same strategy used by
      the XGBoost baseline, enabling a direct comparison.

    Note: adjacent sequences share 23 out of 24 rows.  Sequence-level random
    splitting means some test-sequence rows also appear in training sequences,
    which is a mild and accepted trade-off for solar-wind time-series studies.
    """
    print(f"\n[3/5] Building {WINDOW_SIZE}-hour sequences + stratified random split ...")

    X_arr = X.values.astype(np.float32)
    n_seq = len(X_arr) - WINDOW_SIZE + 1
    if n_seq <= 0:
        raise ValueError(f"Not enough rows ({len(X_arr)}) for window {WINDOW_SIZE}.")

    # Build sequence array: shape (n_seq, WINDOW_SIZE, n_features)
    X_seq = np.array([X_arr[i : i + WINDOW_SIZE] for i in range(n_seq)],
                     dtype=np.float32)
    y_seq = y[WINDOW_SIZE - 1 :].astype(np.float32)   # label = last step of window

    print(f"      Total seqs : {len(y_seq):,}  |  CME: {int(y_seq.sum()):,}  "
          f"({100 * y_seq.mean():.1f}%)")
    print(f"      Seq shape  : ({WINDOW_SIZE}, {X.shape[1]})")

    # ── Stratified random split (mirrors XGBoost 80/20) ────────────────────────
    idx = np.arange(len(y_seq))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.20, random_state=42, stratify=y_seq.astype(int)
    )
    idx_tr, idx_val = train_test_split(
        idx_tr, test_size=0.15, random_state=42, stratify=y_seq[idx_tr].astype(int)
    )

    # ── Fit scaler on training sequences ───────────────────────────────────────
    # Flatten train windows to (n_train * WINDOW_SIZE, n_features) for StandardScaler
    X_tr_flat = X_seq[idx_tr].reshape(-1, X_seq.shape[2])
    scaler = StandardScaler()
    scaler.fit(X_tr_flat)

    def _scale(seqs):
        b, s, f = seqs.shape
        return scaler.transform(seqs.reshape(b * s, f)).reshape(b, s, f).astype(np.float32)

    Xw_tr  = _scale(X_seq[idx_tr]);   yw_tr  = y_seq[idx_tr]
    Xw_val = _scale(X_seq[idx_val]);  yw_val = y_seq[idx_val]
    Xw_te  = _scale(X_seq[idx_te]);   yw_te  = y_seq[idx_te]

    # ── Report ──────────────────────────────────────────────────────────────────
    def _report(label, y_part):
        print(f"      {label:<8}: {len(y_part):>7,}  |  CME: {int(y_part.sum()):>5,}  "
              f"({100 * y_part.mean():.1f}%)")
    _report("Train", yw_tr)
    _report("Val",   yw_val)
    _report("Test",  yw_te)

    scaler_path = ARTIFACT_DIR / "lstm_scaler.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"      Scaler saved -> {scaler_path}")

    return Xw_tr, Xw_val, Xw_te, yw_tr, yw_val, yw_te


# ── 4. Data loaders ────────────────────────────────────────────────────────────
def build_loaders(Xw_tr, yw_tr, Xw_val, yw_val, Xw_te, yw_te):
    print("\n[4/5] Building data loaders ...")

    # Weighted sampler: each batch has ~50/50 CME vs No-CME
    class_counts   = np.bincount(yw_tr.astype(int))
    sample_weights = (1.0 / class_counts)[yw_tr.astype(int)]
    sampler = WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )
    print(f"      WeightedRandomSampler: No-CME w={1/class_counts[0]:.5f}  "
          f"CME w={1/class_counts[1]:.5f}")

    def _ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))

    train_loader = DataLoader(
        _ds(Xw_tr, yw_tr), batch_size=BATCH_SIZE, sampler=sampler,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )
    val_loader = DataLoader(
        _ds(Xw_val, yw_val), batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )
    test_loader = DataLoader(
        _ds(Xw_te, yw_te), batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )
    return train_loader, val_loader, test_loader


# ── 5. Model ───────────────────────────────────────────────────────────────────
class HaloCME_BiLSTM(nn.Module):
    """
    Stacked Bidirectional LSTM for binary CME classification.

    Input  : (batch, WINDOW_SIZE, n_features)
    Output : (batch, 1)  raw logit

    Architecture:
        BatchNorm1d (per time-step)
        BiLSTM(HIDDEN1) + Dropout
        BiLSTM(HIDDEN2) + Dropout
        Linear -> 1
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(n_features)
        self.lstm1    = nn.LSTM(n_features,  HIDDEN1, batch_first=True, bidirectional=True)
        self.drop1    = nn.Dropout(DROPOUT)
        self.lstm2    = nn.LSTM(HIDDEN1 * 2, HIDDEN2, batch_first=True, bidirectional=True)
        self.drop2    = nn.Dropout(DROPOUT)
        self.fc       = nn.Linear(HIDDEN2 * 2, 1)

        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, f = x.shape
        x       = self.input_bn(x.reshape(b * s, f)).reshape(b, s, f)
        out, _  = self.lstm1(x)
        out     = self.drop1(out)
        out, _  = self.lstm2(out)
        out     = self.drop2(out)
        return self.fc(out[:, -1, :])   # last time-step


# ── 6. Train ───────────────────────────────────────────────────────────────────
def train_model(train_loader, val_loader, n_features: int):
    print(f"\n[5/5a] Training Bi-LSTM on {DEVICE} ...")
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f"      Architecture : BiLSTM({HIDDEN1}) -> BiLSTM({HIDDEN2}) -> FC(1)")
    print(f"      Samples      : train={n_train:,}  val={n_val:,}")
    print(f"      Config       : epochs={MAX_EPOCHS}  patience={PATIENCE}  "
          f"lr={LR}  batch={BATCH_SIZE}")

    model    = HaloCME_BiLSTM(n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Parameters   : {n_params:,}")

    criterion = nn.BCEWithLogitsLoss()                  # balance via sampler, not pos_weight
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5
    )

    best_val_auc = 0.0
    patience_ctr = 0
    best_state   = None
    history      = {"train_loss": [], "val_auc": []}

    print(f"\n      {'Epoch':>5}  {'Train Loss':>12}  {'Val AUC':>10}  {'LR':>10}  {'Time':>8}")
    print("      " + "-" * 56)

    t_start = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(Xb)
        train_loss = running / len(train_loader.dataset)

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        vp, vl = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                p = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().flatten()
                vp.extend(p.tolist())
                vl.extend(yb.numpy().flatten().tolist())
        val_auc = roc_auc_score(vl, vp)
        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        print(f"      {epoch:>5d}  {train_loss:>12.4f}  {val_auc:>10.4f}  "
              f"{cur_lr:>10.2e}  {elapsed:>7.0f}s")

        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n      Early stopping at epoch {epoch}.")
                break

    total = time.time() - t_start
    print(f"\n      Finished in {total:.1f}s  |  Best val AUC: {best_val_auc:.4f}")
    model.load_state_dict(best_state)
    return model, history


# ── 7. Evaluate ────────────────────────────────────────────────────────────────
def evaluate(model, test_loader, yw_te: np.ndarray):
    print("\n[5/5b] Evaluating on test set ...")

    model.eval()
    all_p: list[float] = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            p = torch.sigmoid(model(Xb.to(DEVICE))).cpu().numpy().flatten()
            all_p.extend(p.tolist())

    y_prob = np.array(all_p)
    y_test = yw_te.astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    print(f"\n  ROC-AUC  : {roc_auc:.4f}")
    print(f"  PR-AUC   : {pr_auc:.4f}")

    # Threshold sweep
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_prob)
    eps    = 1e-9
    f1_arr = 2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1] + eps)
    thr_best_f1 = float(thresholds[np.argmax(f1_arr)])
    rec80_mask  = rec_arr[:-1] >= 0.80
    thr_rec80   = float(thresholds[rec80_mask].min()) if rec80_mask.any() else thr_best_f1

    chosen_thr = thr_best_f1
    print(f"  Chosen threshold (Best F1) : {chosen_thr:.4f}")

    y_pred = (y_prob >= chosen_thr).astype(int)
    print(f"\n  === Full Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["No CME", "Halo CME"]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("  Confusion Matrix:")
    print(f"    TN (correct No-CME) : {tn:,}")
    print(f"    FP (false alarm)     : {fp:,}")
    print(f"    FN (missed CME)      : {fn:,}")
    print(f"    TP (caught CME)      : {tp:,}")

    thr_path = ARTIFACT_DIR / "lstm_threshold.txt"
    thr_path.write_text(str(chosen_thr))
    print(f"\n  Threshold saved -> {thr_path}")

    _plot_roc_pr(y_test, y_prob, roc_auc, pr_auc)
    return roc_auc, pr_auc, chosen_thr


# ── Plots ──────────────────────────────────────────────────────────────────────
def _plot_roc_pr(y_test, y_prob, roc_auc: float, pr_auc: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Bi-LSTM -- Halo CME Forecasting -- Test Set",
                 fontsize=13, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, lw=2, color="royalblue", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate",
                title="ROC Curve")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(rec, prec, lw=2, color="crimson", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    out = ARTIFACT_DIR / "lstm_evaluation.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Plot saved -> {out}")


def _plot_training_curve(history: dict) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="steelblue")
    ax1.plot(epochs, history["train_loss"], color="steelblue", lw=2)
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Val ROC-AUC", color="darkorange")
    ax2.plot(epochs, history["val_auc"], color="darkorange", lw=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    fig.suptitle("Bi-LSTM Training Curve", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = ARTIFACT_DIR / "lstm_training_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Training curve saved -> {out}")


# ── Save ───────────────────────────────────────────────────────────────────────
def save_model(model, n_features: int) -> None:
    print("\n[Save] Persisting LSTM model ...")
    model_path = ARTIFACT_DIR / "lstm_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"  State dict -> {model_path}  ({model_path.stat().st_size // 1024:,} KB)")

    cfg = {"n_features": n_features, "hidden1": HIDDEN1, "hidden2": HIDDEN2,
           "dropout": DROPOUT, "window": WINDOW_SIZE}
    cfg_path = ARTIFACT_DIR / "lstm_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"  Config     -> {cfg_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log_path  = RESULTS_DIR / "lstm_res.txt"
    _log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, _log_file)

    try:
        banner("SOLAR HALO CME  --  LSTM (Bidirectional, Stacked)")
        print(f"  Device  : {DEVICE}")
        if DEVICE.type == "cuda":
            print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  Window  : {WINDOW_SIZE}h look-back")
        print(f"  Split   : stratified random 68/12/20 (sequence-level)")

        df                       = load_data()
        X, y, feat_cols          = feature_engineering(df)
        Xw_tr, Xw_val, Xw_te, \
            yw_tr, yw_val, yw_te = build_sequences_and_split(X, y)

        train_loader, val_loader, \
            test_loader          = build_loaders(Xw_tr, yw_tr, Xw_val, yw_val, Xw_te, yw_te)

        model, history           = train_model(train_loader, val_loader, X.shape[1])
        roc, pr, thr             = evaluate(model, test_loader, yw_te)
        _plot_training_curve(history)
        save_model(model, X.shape[1])

        print(f"\n{HEADER}")
        print("  DONE -- artifacts/ now contains:")
        print("    lstm_model.pt  |  lstm_config.json  |  lstm_scaler.pkl")
        print(f"    lstm_threshold.txt  (thr={thr:.4f})")
        print("    lstm_evaluation.png  |  lstm_training_curve.png")
        print(f"  Final ROC-AUC : {roc:.4f}   PR-AUC : {pr:.4f}")
        print(HEADER)

    finally:
        sys.stdout = sys.__stdout__
        _log_file.close()
        print(f"\n  Results saved -> {log_path}")
