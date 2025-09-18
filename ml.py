#!/usr/bin/env python3
"""
LSTM-only spoof vs truth classifier for IMU/GRU time series with strict time-aware validation.

- Uses ONLY IMU/GRU columns (e.g., drone1_IMUx, drone1_IMUy, drone1_GRUx, drone1_GRUy).
- Drops the first N rows from each file (to ignore startup transients).
- Segments into NON-OVERLAPPING sequence windows; each window is one sample.
- Validation: blocked time-series CV with a purge gap (no temporal bleed).
- Trains for EXACTLY 5 epochs (as requested).

Prints per-fold Accuracy, F1, AUC, confusion matrices, and overall means.

Requires: torch, numpy, pandas, scikit-learn.
"""

# ========================== CONFIG (edit here) ==========================
SPOOF_CSV   = "imu_spoof.csv"   # path to spoof data
TRUTH_CSV   = "imu_truth.csv"   # path to truth data

DROP_FIRST  = 600               # rows to drop from the start of EACH file
WINDOW      = 10               # samples per window (sequence length)
STEP        = 10          # step size; keep == WINDOW for non-overlap
FOLDS       = 5                 # number of blocked folds
PURGE       = 0                 # purge gap (#windows) around test block

EPOCHS      = 5                 # EXACT training epochs (do not change per request)
BATCH       = 16
LR          = 1e-3
HIDDEN      = 64                # LSTM hidden size
BIDIR       = True              # bidirectional LSTM
SEED        = 42                # for minor reproducibility
# =======================================================================

import sys
import numpy as np
import pandas as pd

# Torch (required for this LSTM-only version)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:
    raise SystemExit(
        "PyTorch is required for this LSTM-only script.\n"
        f"Import error: {e}\nInstall with: pip install torch --extra-index-url https://download.pytorch.org/whl/cpu"
    )

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# ------------------------- Utilities -------------------------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def select_seq_cols(df: pd.DataFrame):
    """Pick IMU/GRU channels only (exclude GPS/lat/lon/absolute time)."""
    return sorted([c for c in df.columns if c.startswith("drone") and ("IMU" in c or "GRU" in c)])

def make_windows(X: np.ndarray, y: np.ndarray, window: int, step: int):
    """Create (N,T,F) windows and window labels by majority vote."""
    xs, ys = [], []
    for s in range(0, len(X) - window + 1, step):
        e = s + window
        xs.append(X[s:e])
        ys.append(int(np.round(np.mean(y[s:e]))))
    return np.array(xs), np.array(ys)

def blocked_folds(n_windows: int, n_folds: int, purge: int):
    """
    Contiguous test blocks with a purge gap around the test block to avoid temporal bleed.
    Returns: list of (train_idx, test_idx) index arrays.
    """
    fold_sizes = [n_windows // n_folds + (1 if i < n_windows % n_folds else 0) for i in range(n_folds)]
    spans, start = [], 0
    for fs in fold_sizes:
        spans.append((start, start + fs))
        start += fs

    folds = []
    for a, b in spans:
        test = np.arange(a, b)
        purge_start, purge_end = max(0, a - purge), min(n_windows, b + purge)
        train = np.array([k for k in range(n_windows) if not (purge_start <= k < purge_end)])
        if len(test) and len(train):
            folds.append((train, test))
    return folds

def fit_standardizer(X: np.ndarray):
    """Fit mean/std over (B,T,F) on TRAIN ONLY."""
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0
    return mean, std

def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std

# ------------------------- Model -------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, bidir: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidir,
        )
        out_dim = hidden * (2 if bidir else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: [B, T, F]
        out, _ = self.lstm(x)             # out: [B, T, H*dir]
        last = out[:, -1, :]              # last time step: [B, H*dir]
        return self.head(last).squeeze(-1)  # [B]

# ------------------------- Pipeline -------------------------
def main():
    set_seeds(SEED)

    # Load CSVs
    spoof = pd.read_csv(SPOOF_CSV)
    truth = pd.read_csv(TRUTH_CSV)

    # Drop first N rows
    spoof = spoof.iloc[DROP_FIRST:].reset_index(drop=True)
    truth = truth.iloc[DROP_FIRST:].reset_index(drop=True)

    # Basic sanity checks
    if len(spoof) < WINDOW or len(truth) < WINDOW:
        raise SystemExit("Not enough rows after dropping to form at least one window. Reduce DROP_FIRST or WINDOW.")

    # Select IMU/GRU columns
    seq_cols = select_seq_cols(spoof)
    if not seq_cols:
        raise SystemExit("No IMU/GRU columns found (expected names like 'drone1_IMUx', 'drone1_GRUy', etc.).")

    # Build arrays
    Xs = spoof[seq_cols].to_numpy(); ys = np.ones(len(spoof), dtype=int)
    Xt = truth[seq_cols].to_numpy(); yt = np.zeros(len(truth), dtype=int)

    # Windowing (non-overlapping by default)
    Xs_w, ys_w = make_windows(Xs, ys, WINDOW, STEP)
    Xt_w, yt_w = make_windows(Xt, yt, WINDOW, STEP)

    # Blocked folds with purge, per class
    folds_s = blocked_folds(len(Xs_w), FOLDS, PURGE)
    folds_t = blocked_folds(len(Xt_w), FOLDS, PURGE)
    n_folds = min(len(folds_s), len(folds_t))
    if n_folds == 0:
        raise SystemExit("Not enough windows for the requested number of folds. Adjust WINDOW/STEP/FOLDS.")

    device = torch.device("cpu")
    results = []

    print(f"\nConfig: DROP_FIRST={DROP_FIRST}, WINDOW={WINDOW}, STEP={STEP}, FOLDS={FOLDS}, PURGE={PURGE}, "
          f"EPOCHS={EPOCHS}, BATCH={BATCH}, LR={LR}, HIDDEN={HIDDEN}, BIDIR={BIDIR}")
    print(f"Columns used (IMU/GRU only): {len(seq_cols)} features")

    for f in range(n_folds):
        trS, teS = folds_s[f]
        trT, teT = folds_t[f]

        # Build fold-specific train/test
        X_train = np.concatenate([Xs_w[trS], Xt_w[trT]], axis=0)
        y_train = np.concatenate([ys_w[trS], yt_w[trT]], axis=0)
        X_test  = np.concatenate([Xs_w[teS], Xt_w[teT]], axis=0)
        y_test  = np.concatenate([ys_w[teS], yt_w[teT]], axis=0)

        # Standardize on TRAIN only
        mean, std = fit_standardizer(X_train)
        X_train_n = apply_standardizer(X_train, mean, std)
        X_test_n  = apply_standardizer(X_test,  mean, std)

        # Torch tensors & loader
        ds_tr = TensorDataset(torch.tensor(X_train_n, dtype=torch.float32),
                              torch.tensor(y_train,   dtype=torch.float32))
        dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)

        model = LSTMClassifier(n_features=X_train.shape[-1], hidden=HIDDEN, bidir=BIDIR).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.BCEWithLogitsLoss()

        # ---- Train (EXACTLY EPOCHS=5) ----
        for epoch in range(EPOCHS):
            model.train()
            running = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += float(loss.item())
            print(f"[Fold {f+1}/{n_folds}] Epoch {epoch+1}/{EPOCHS} - loss: {running/len(dl_tr):.4f}")

        # ---- Evaluate ----
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_test_n, dtype=torch.float32))
            probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        acc = float((preds == y_test).mean())
        # F1
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = float("nan")
        cm = confusion_matrix(y_test, preds)

        print(f"\n[LSTM] Fold {f+1}/{n_folds}")
        print(f"  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  (n_test={len(y_test)})")
        print("  Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
        print(cm)

        results.append({"fold": f + 1, "acc": acc, "f1": f1, "auc": auc, "n_test": len(y_test)})

    # ---- Summary ----
    res_df = pd.DataFrame(results)
    print("\n[LSTM] Per-fold metrics:")
    print(res_df.to_string(index=False))
    print("\n[LSTM] Means:", res_df[["acc", "f1", "auc"]].mean().to_dict())


if __name__ == "__main__":
    main()
