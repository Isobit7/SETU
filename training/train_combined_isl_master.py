"""
SETU — Master Combined ISL Training Pipeline
Combines:
1. Local Indian Sign Language Dataset (42K authentic ISL images -> 9,754 samples)
2. Hugging Face Dataset (akritRihal/Indian_Sign_Language_dataset -> 4,984 samples)
Total: 14,738 Dual-Hand 126-D ISL Samples
Trains and deploys directly to models/weights/realsign_26class_mlp.pth
"""

import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCAL_NPZ = PROJECT_ROOT / "data" / "processed" / "isl_dualhand_26class.npz"
HF_NPZ = PROJECT_ROOT / "data" / "processed" / "hf_isl_dualhand_26class.npz"
COMBINED_NPZ = PROJECT_ROOT / "data" / "processed" / "isl_combined_master_26class.npz"
DEPLOY_PATH = PROJECT_ROOT / "models" / "weights" / "realsign_26class_mlp.pth"
REPORT_PATH = PROJECT_ROOT / "reports" / "realsign_26class_evaluation_report.json"
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
            x = x + torch.randn_like(x) * 0.005 * mask
        return x, self.y[idx]


def run_master_training(epochs: int = 45, batch_size: int = 64, lr: float = 1e-3) -> None:
    data_local = np.load(LOCAL_NPZ, allow_pickle=True)
    data_hf = np.load(HF_NPZ, allow_pickle=True)

    X_all = np.concatenate([data_local["X"], data_hf["X"]], axis=0)
    y_all = np.concatenate([data_local["y"], data_hf["y"]], axis=0)

    print(f"\n=======================================================")
    print(f"MASTER ISL TRAINING: Local ({len(data_local['X'])}) + Hugging Face ({len(data_hf['X'])})")
    print(f"Total Combined Training Vectors: {len(X_all)} across 26 letters (A-Z)")
    print(f"=======================================================\n")

    # Stratified 80/10/10 split
    indices = np.arange(len(y_all))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    splits = np.empty(len(y_all), dtype=object)
    splits[idx_train] = "train"
    splits[idx_val] = "val"
    splits[idx_test] = "test"

    np.savez_compressed(
        COMBINED_NPZ,
        X=X_all,
        y=y_all,
        splits=splits,
        classes=np.array(CLASSES_26),
    )
    print(f"Saved master combined dataset to {COMBINED_NPZ}")

    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_val, y_val = X_all[idx_val], y_all[idx_val]
    X_test, y_test = X_all[idx_test], y_all[idx_test]

    train_loader = DataLoader(ISLDataset(X_train, y_train, augment=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ISLDataset(X_val, y_val, augment=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ISLDataset(X_test, y_test, augment=False), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = RealSignMLP(input_dim=126, num_classes=26).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state = None

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

        val_acc = val_correct / val_total

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}%)")

    # Load best weights for test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
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

    print(f"\n=======================================================")
    print(f"MASTER ISL MODEL EVALUATION:")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    print(f"=======================================================\n")

    # Save to active model weights path
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 126,
        "classes": CLASSES_26,
        "num_classes": 26,
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "architecture": {
            "input_dim": 126,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": 26,
            "use_batchnorm": True,
            "dropout": 0.2,
        },
        "dataset_source": "combined:ISL_42K_Dataset+HuggingFace_akritRihal",
        "created_at": datetime.datetime.now().isoformat(),
    }
    torch.save(checkpoint, DEPLOY_PATH)
    print(f"Deployed master weights to: {DEPLOY_PATH}")

    # Save report
    report = {
        "dataset_name": "ISL_42K_Dataset + HuggingFace_akritRihal",
        "total_samples": len(X_all),
        "test_accuracy": test_acc,
        "macro_f1": macro_f1,
        "classes": CLASSES_26,
        "created_at": datetime.datetime.now().isoformat(),
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved evaluation report to: {REPORT_PATH}")


if __name__ == "__main__":
    run_master_training(epochs=45)
