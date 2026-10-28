"""
SETU — Training PyTorch MLP on Hugging Face ISL Dual-Hand 126-D Dataset
Dataset source: akritRihal/Indian_Sign_Language_dataset
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NPZ_PATH = PROJECT_ROOT / "data" / "processed" / "hf_isl_dualhand_26class.npz"
SAVE_PATH = PROJECT_ROOT / "models" / "weights" / "hf_isl_26class_mlp.pth"
REPORT_PATH = PROJECT_ROOT / "reports" / "hf_isl_evaluation_report.json"
CLASSES_26 = [chr(ord("A") + i) for i in range(26)]


class RealSignMLP(nn.Module):
    def __init__(self, input_dim=126, hidden1=128, hidden2=64, num_classes=26, dropout=0.2):
        super().__init__()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ISLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            mask = (x != 0.0).float()
            x = x + torch.randn_like(x) * 0.006 * mask
        return x, self.y[idx]


def train_model(epochs: int = 40, batch_size: int = 64, lr: float = 1e-3) -> None:
    if not NPZ_PATH.exists():
        print(f"Error: {NPZ_PATH} does not exist. Run ingest_hf_isl_dataset.py first.")
        return

    data = np.load(NPZ_PATH, allow_pickle=True)
    X, y, splits = data["X"], data["y"], data["splits"]
    classes = [str(c) for c in data["classes"]]

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Loaded {len(X)} samples across {len(classes)} classes.")
    print(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_ds = ISLDataset(X_train, y_train, augment=True)
    val_ds = ISLDataset(X_val, y_val, augment=False)
    test_ds = ISLDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = RealSignMLP(input_dim=126, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            train_total += len(by)

        scheduler.step()
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Load best weights for final test evaluation
    if best_weights is not None:
        model.load_state_dict(best_weights)
    model.eval()

    all_preds, all_trues = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(by.numpy())

    all_preds = np.array(all_preds)
    all_trues = np.array(all_trues)

    test_acc = float(np.mean(all_preds == all_trues))
    macro_f1 = float(f1_score(all_trues, all_preds, average="macro", zero_division=0))
    print(f"\nFINAL TEST RESULTS: Accuracy: {test_acc*100:.2f}%, Macro F1: {macro_f1:.4f}")

    # Save model weights
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 126,
        "classes": classes,
        "num_classes": len(classes),
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "architecture": {
            "input_dim": 126,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": len(classes),
            "use_batchnorm": True,
            "dropout": 0.2,
        },
        "dataset_source": "huggingface:akritRihal/Indian_Sign_Language_dataset",
        "created_at": datetime.datetime.now().isoformat(),
    }
    torch.save(checkpoint, SAVE_PATH)
    print(f"Checkpoint saved to: {SAVE_PATH}")

    # Save evaluation report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "dataset_name": "akritRihal/Indian_Sign_Language_dataset",
        "total_samples": len(X),
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "classes": classes,
        "classification_report": classification_report(all_trues, all_preds, target_names=[classes[i] for i in np.unique(all_trues)], output_dict=True, zero_division=0),
        "created_at": datetime.datetime.now().isoformat(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Evaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    train_model(epochs=40)
