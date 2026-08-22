"""
train_multimodal.py
====================
Complete training pipeline for the multimodal angina prediction model.

USAGE:
  # Phase 1 – Tabular only (XGBoost baseline, already done)
  # Phase 2 – Tabular + ECG multimodal (run this script)

  python train_multimodal.py

WHAT THIS DOES:
  1. Loads tabular features (with NaN preserved)
  2. Loads ECG arrays from preprocessing (if available)
  3. Builds proper train/val/test splits (no leakage)
  4. Trains tabular-only baseline (for ablation)
  5. Trains full multimodal model
  6. Evaluates both and reports improvement
  7. Saves models and results
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

from multimodal_model import (
    AnginaMultimodalModel,
    AnginaTabularOnlyModel,
    FocalLoss,
    count_parameters,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TABULAR_PATH   = "data/processed/tabular_features.csv"
ECG_NPZ_PATH   = "data/ecg/processed/ecg_arrays.npz"
ECG_INDEX_PATH = "data/ecg/processed/ecg_index.csv"
OUTPUT_DIR     = Path("results/multimodal")

BATCH_SIZE     = 64    # best found by hyperparameter search
EPOCHS         = 50
LR             = 5e-4  # best found by hyperparameter search
WEIGHT_DECAY   = 1e-4
PATIENCE       = 10          # early stopping patience
N_FOLDS        = 5
RANDOM_SEED    = 42

# Mac M3 / CPU / CUDA auto-detect
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)
print(f"Device: {DEVICE}")
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AnginaDataset(Dataset):
    """
    Multimodal dataset that handles missing ECG gracefully.

    Tabular preprocessing:
      - For each NaN value, set feature = 0 and add a corresponding
        missingness indicator column (1 = was NaN, 0 = had real value).
      - This doubles the feature count but preserves missingness signal.

    ECG:
      - If available for this hadm_id: load from npz
      - If not: return zeros tensor + ecg_available=False flag
    """

    def __init__(
        self,
        tabular_df: pd.DataFrame,
        feature_cols: list[str],
        scaler: StandardScaler,
        ecg_data: dict | None = None,
        fit_scaler: bool = False,
    ):
        self.hadm_ids = tabular_df["hadm_id"].values.astype(int)
        self.labels   = tabular_df["label"].values.astype(np.float32)
        self.ecg_data = ecg_data or {}
        self.feature_cols = feature_cols

        raw = tabular_df[feature_cols].values.astype(np.float32)

        # ── Missingness indicators ────────────────────────────────────────────
        missing_mask = np.isnan(raw).astype(np.float32)   # 1 where NaN
        raw_filled   = np.nan_to_num(raw, nan=0.0)        # 0 where NaN

        # Scale filled values
        if fit_scaler:
            scaler.fit(raw_filled)
        raw_scaled = scaler.transform(raw_filled)

        # Concatenate: [scaled_features | missingness_indicators]
        self.tabular = np.concatenate([raw_scaled, missing_mask], axis=1)
        self.tabular = self.tabular.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hadm_id = self.hadm_ids[idx]
        tabular  = torch.tensor(self.tabular[idx])
        label    = torch.tensor(self.labels[idx])

        if hadm_id in self.ecg_data:
            ecg           = torch.tensor(self.ecg_data[hadm_id])
            ecg_available = torch.tensor(True)
        else:
            ecg           = torch.zeros(12, 5000, dtype=torch.float32)
            ecg_available = torch.tensor(False)

        return tabular, ecg, ecg_available, label


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_tabular(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    assert "label"   in df.columns, "Missing 'label' column"
    assert "hadm_id" in df.columns, "Missing 'hadm_id' column"

    # Drop non-feature columns
    drop = [c for c in ["subject_id", "admittime", "dischtime"] if c in df.columns]
    df = df.drop(columns=drop)

    feature_cols = [c for c in df.columns if c not in ("label", "hadm_id")]

    print(f"Tabular: {len(df)} samples, {len(feature_cols)} features")
    print(f"  Missing values: {df[feature_cols].isna().sum().sum()} cells")
    print(f"  Label balance: {df['label'].value_counts().to_dict()}")
    return df, feature_cols


def load_ecg(npz_path: str, index_path: str) -> dict | None:
    if not Path(npz_path).exists():
        print("ECG arrays not found — running tabular-only mode")
        print(f"  Expected: {npz_path}")
        print("  Run ecg_preprocessor.py first after downloading ECG data")
        return None

    npz = np.load(npz_path)
    ecg_data = {int(k): npz[k] for k in npz.files}
    print(f"ECG: loaded {len(ecg_data)} arrays from {npz_path}")
    return ecg_data


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
        "avg_prec":  average_precision_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def forward_pass(model, tabular, ecg, ecg_avail):
    """Call model correctly regardless of whether it's multimodal or tabular-only."""
    from multimodal_model import AnginaMultimodalModel
    if isinstance(model, AnginaMultimodalModel):
        return model(tabular, ecg, ecg_avail)
    else:
        return model(tabular)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for tabular, ecg, ecg_avail, labels in loader:
        tabular, ecg   = tabular.to(device), ecg.to(device)
        ecg_avail      = ecg_avail.to(device)
        labels         = labels.to(device)

        optimizer.zero_grad()
        logits = forward_pass(model, tabular, ecg, ecg_avail)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for tabular, ecg, ecg_avail, labels in loader:
        tabular, ecg  = tabular.to(device), ecg.to(device)
        ecg_avail     = ecg_avail.to(device)
        labels        = labels.to(device)

        logits = forward_pass(model, tabular, ecg, ecg_avail)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs  = torch.sigmoid(logits).numpy()

    metrics = compute_metrics(labels.numpy(), probs)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    model_name: str,
    output_dir: Path,
):
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR / 10
    )

    best_auc      = 0.0
    patience_cnt  = 0
    history       = []
    best_path     = output_dir / f"{model_name}_best.pt"

    print(f"\n── Training {model_name} ({'{'}{count_parameters(model):,} params{'}'}) ──")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Val Loss':>9} {'Val AUC':>9} {'Val F1':>8}")
    print("─" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        val_auc = val_metrics["roc_auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>5}  {train_loss:>10.4f}  {val_metrics['loss']:>9.4f}  "
                  f"{val_auc:>9.4f}  {val_metrics['f1']:>8.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
                break

    # Restore best weights
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"  Best val AUC: {best_auc:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("MULTIMODAL ANGINA PREDICTION — Training Pipeline")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df, feature_cols = load_tabular(TABULAR_PATH)
    ecg_data = load_ecg(ECG_NPZ_PATH, ECG_INDEX_PATH)

    has_ecg = ecg_data is not None
    if has_ecg:
        ecg_coverage = sum(1 for h in df["hadm_id"] if int(h) in ecg_data)
        print(f"  ECG coverage: {ecg_coverage}/{len(df)} "
              f"({ecg_coverage/len(df):.1%}) admissions")

    # ── Train/Test split (stratified, fixed seed) ─────────────────────────────
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.10, stratify=train_val_df["label"],
        random_state=RANDOM_SEED
    )

    print(f"\nSplits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ── Build datasets ────────────────────────────────────────────────────────
    scaler = StandardScaler()

    train_ds = AnginaDataset(train_df, feature_cols, scaler, ecg_data, fit_scaler=True)
    val_ds   = AnginaDataset(val_df,   feature_cols, scaler, ecg_data, fit_scaler=False)
    test_ds  = AnginaDataset(test_df,  feature_cols, scaler, ecg_data, fit_scaler=False)

    tabular_input_dim = train_ds.tabular.shape[1]  # features + missingness indicators
    print(f"Tabular input dim (features + missingness indicators): {tabular_input_dim}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model 1: Tabular-only (ablation baseline) ─────────────────────────────
    tabular_model = AnginaTabularOnlyModel(tabular_input_dim).to(DEVICE)
    tabular_model, tab_history = train_model(
        tabular_model, train_loader, val_loader,
        "tabular_only", OUTPUT_DIR
    )
    criterion = FocalLoss()
    tab_test_metrics = evaluate(tabular_model, test_loader, criterion, DEVICE)

    # ── Model 2: Multimodal (tabular + ECG) ───────────────────────────────────
    multi_model = AnginaMultimodalModel(tabular_input_dim).to(DEVICE)
    multi_model, multi_history = train_model(
        multi_model, train_loader, val_loader,
        "multimodal", OUTPUT_DIR
    )
    multi_test_metrics = evaluate(multi_model, test_loader, criterion, DEVICE)

    # ── Print comparison ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    header = f"{'Metric':<12} {'Tabular-Only':>13} {'Multimodal':>13} {'Δ':>8}"
    print(header)
    print("─" * 50)

    for metric in ["roc_auc", "f1", "accuracy", "precision", "recall", "avg_prec"]:
        t_val = tab_test_metrics[metric]
        m_val = multi_test_metrics[metric]
        delta = m_val - t_val
        flag  = "▲" if delta > 0 else "▼" if delta < 0 else "="
        print(f"{metric:<12} {t_val:>13.4f} {m_val:>13.4f} {flag}{abs(delta):>6.4f}")

    print("─" * 50)
    print(f"\nMultimodal ROC-AUC: {multi_test_metrics['roc_auc']:.4f}")
    print("Hypothesis target:  0.85")
    if multi_test_metrics["roc_auc"] >= 0.85:
        print("✓ TARGET ACHIEVED — Multimodal model surpasses 0.85 ROC-AUC!")
    else:
        gap = 0.85 - multi_test_metrics["roc_auc"]
        print(f"  Gap to target: {gap:.4f} (ECG coverage may be limiting factor)")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "tabular_only": tab_test_metrics,
        "multimodal":   multi_test_metrics,
        "ecg_coverage": ecg_coverage / len(df) if has_ecg else 0.0,
        "tabular_input_dim": tabular_input_dim,
        "n_train": len(train_df),
        "n_val":   len(val_df),
        "n_test":  len(test_df),
    }
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(tab_history).to_csv(OUTPUT_DIR / "tabular_history.csv", index=False)
    pd.DataFrame(multi_history).to_csv(OUTPUT_DIR / "multimodal_history.csv", index=False)

    # Save scaler
    import joblib
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(feature_cols, OUTPUT_DIR / "feature_cols.pkl")

    print(f"\n✓ All results saved to {OUTPUT_DIR}/")
    print("  results.json, *_history.csv, *_best.pt, scaler.pkl")


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
