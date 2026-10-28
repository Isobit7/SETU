"""
SETU — Phase 3.4: ISL Bimanual 126-D Alphabet PyTorch MLP Training

Trains a 26-class PyTorch MLP on 126-D dual-hand ISL landmarks:
- Architecture: Input(126) -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(64) -> BatchNorm1d -> ReLU -> Linear(26)
- Input representation:
    * Left Hand normalized 3D coords: slots [0:63]
    * Right Hand normalized 3D coords: slots [63:126]
    * Automatically handles both two-handed ISL signs (A, B, D, E, G, K, M, N, P, T, X, Y, Z)
      and one-handed ISL signs (C, H, I, J, L, O, Q, R, S, U, V, W).
- Outputs:
    * Model checkpoint: 'models/weights/realsign_26class_mlp.pth'
    * Evaluation report: 'reports/realsign_26class_evaluation_report.json'
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "isl_dualhand_26class.npz"
DEFAULT_MODEL_SAVE_PATH = ROOT_DIR / "models" / "weights" / "realsign_26class_mlp.pth"
DEFAULT_REPORT_SAVE_PATH = ROOT_DIR / "reports" / "realsign_26class_evaluation_report.json"
CLASSES_26 = [chr(ord("A") + i) for i in range(26)]


class RealSignMLP(nn.Module):
    """PyTorch MLP for 26-class ISL Hand Alphabet Recognition."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 26,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.BatchNorm1d(hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.BatchNorm1d(hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_classes),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden1, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ISLDualHandDataset(Dataset):
    """PyTorch Dataset wrapper with coordinate jitter augmentation for 126-D vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, noise_std: float = 0.006):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            # Add small Gaussian noise to non-zero coordinates
            mask = (x != 0.0).float()
            noise = torch.randn_like(x) * self.noise_std * mask
            x = x + noise
        return x, self.y[idx]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    num_classes = len(classes)
    total_samples = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    acc = float(correct / total_samples) if total_samples > 0 else 0.0

    per_class = {}
    f1s = []
    for c_idx, c_name in enumerate(classes):
        tp = int(np.sum((y_true == c_idx) & (y_pred == c_idx)))
        fp = int(np.sum((y_true != c_idx) & (y_pred == c_idx)))
        fn = int(np.sum((y_true == c_idx) & (y_pred != c_idx)))
        support = int(np.sum(y_true == c_idx))

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        per_class[c_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f1), 4),
            "support": support,
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    conf_pairs = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf_pairs[(classes[t], classes[p])] += 1

    top_conf = [
        {"true_class": k[0], "pred_class": k[1], "count": v}
        for k, v in conf_pairs.most_common(10)
    ]

    return {
        "overall_accuracy": round(acc, 4),
        "overall_accuracy_pct": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "top_confusion_pairs": top_conf,
    }


def train_model(
    features_path: Path = DEFAULT_FEATURES_PATH,
    model_save_path: Path = DEFAULT_MODEL_SAVE_PATH,
    report_save_path: Path = DEFAULT_REPORT_SAVE_PATH,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("SETU — Training ISL Bimanual 126-D MLP Classifier")
    print(f"Loading features from: {features_path}")
    print("=" * 70)

    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")

    with np.load(features_path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"]
        splits = data["splits"] if "splits" in data else None
        classes = [str(c) for c in data["classes"]] if "classes" in data else CLASSES_26

    print(f"Loaded {len(X):,} samples of dimension {X.shape[1]} across {len(classes)} classes.")

    if splits is None or len(splits) != len(y):
        indices = np.arange(len(y))
        train_idx, rest_idx = train_test_split(indices, test_size=0.20, stratify=y, random_state=seed)
        val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed)
        splits = np.empty(len(y), dtype="<U10")
        splits[train_idx] = "train"
        splits[val_idx] = "val"
        splits[test_idx] = "test"

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Splits: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

    train_ds = ISLDualHandDataset(X_train, y_train, augment=True)
    val_ds = ISLDualHandDataset(X_val, y_val, augment=False)
    test_ds = ISLDualHandDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RealSignMLP(
        input_dim=126,
        hidden1=128,
        hidden2=64,
        num_classes=26,
        dropout=0.2,
        use_batchnorm=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_val_acc = 0.0
    best_weights = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=-1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        train_loss = total_loss / total_train if total_train > 0 else 0.0
        train_acc = correct_train / total_train if total_train > 0 else 0.0

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds = out.argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += bx.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or val_acc > 0.98:
            print(f"Epoch {epoch:02d}/{epochs:02d} — Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Evaluate on held-out test split
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            preds = out.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_metrics = compute_metrics(all_targets, all_preds, classes)

    print("=" * 70)
    print(f"Test Accuracy: {test_metrics['overall_accuracy_pct']}% | Macro F1: {test_metrics['macro_f1']}")
    print("=" * 70)

    # Save model checkpoint
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "input_dim": 126,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": 26,
            "dropout": 0.2,
            "use_batchnorm": True,
        },
        "classes": classes,
        "test_accuracy": test_metrics["overall_accuracy"],
        "macro_f1": test_metrics["macro_f1"],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    torch.save(checkpoint, model_save_path)
    print(f"Saved trained ISL MLP model to: {model_save_path}")

    # Save report
    report_save_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model_name": "SETU-ISL-DualHand-126D-MLP",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "architecture": checkpoint["architecture"],
        "test_evaluation": test_metrics,
        "training_time_seconds": round(time.time() - t0, 2),
    }
    with open(report_save_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {report_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        features_path=args.features,
        model_save_path=args.model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
