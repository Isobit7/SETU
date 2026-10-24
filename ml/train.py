"""
SETU — ML Training Pipeline: RealSign 26-Class Alphabet PyTorch MLP Classifier

Trains and evaluates a 26-class PyTorch MLP on clean MediaPipe hand landmarks:
- Architecture: Input(63) -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(64) -> ReLU -> Linear(26)
- Optimization: CrossEntropyLoss, Adam(lr=0.001), 30 epochs
- Data Splits: 80% Train, 10% Val, 10% Test from 'data/processed/realsign_landmarks_26class.npz'
- Outputs:
    * Model checkpoints: 'models/realsign_26class_mlp.pt' and 'models/realsign_26class_mlp.pth'
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
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.feature_extractor import normalize_landmarks_to_63d
from backend.app.services.classifier import RealSignMLP

DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"
DEFAULT_MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "realsign_26class_mlp.pt"
DEFAULT_ALT_MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "realsign_26class_mlp.pth"
DEFAULT_REPORT_SAVE_PATH = PROJECT_ROOT / "reports" / "realsign_26class_evaluation_report.json"

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


def generate_stratified_splits(y: np.ndarray, seed: int = 42) -> np.ndarray:
    """Auto-generates a stratified 80% Train / 10% Validation / 10% Test split across y."""
    n = len(y)
    indices = np.arange(n)
    train_idx, rest_idx = train_test_split(
        indices, test_size=0.20, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=seed
    )

    splits = np.empty(n, dtype="<U10")
    splits[train_idx] = "train"
    splits[val_idx] = "val"
    splits[test_idx] = "test"
    return splits


def load_dataset(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads dataset from .npz file, guaranteeing that splits and classes metadata
    are present and auto-generating them if missing.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with np.load(dataset_path, allow_pickle=True) as data:
        X = np.array(data["X"], dtype=np.float32)
        y_raw = np.array(data["y"])

        # Label encoding
        if y_raw.dtype.kind in ("U", "S", "O"):
            classes_in_y = sorted(list(set(str(c).strip().upper() for c in y_raw)))
            c_to_idx = {c: i for i, c in enumerate(classes_in_y)}
            y = np.array([c_to_idx[str(c).strip().upper()] for c in y_raw], dtype=np.int64)
            detected_classes = classes_in_y
        else:
            y = np.array(y_raw, dtype=np.int64)
            detected_classes = CLASSES_26

        # Check splits
        splits = None
        if "splits" in data:
            splits_raw = data["splits"]
            if hasattr(splits_raw, "__len__") and len(splits_raw) == len(y):
                splits = np.array(splits_raw, dtype=str)

        if splits is None or len(set(splits)) < 2:
            splits = generate_stratified_splits(y)
            np.savez_compressed(
                dataset_path,
                X=X,
                y=y,
                splits=splits,
                classes=np.array(detected_classes, dtype=str),
            )

        classes = detected_classes
        if "classes" in data:
            classes = [str(c) for c in data["classes"]]

    return X, y, splits, classes


class RealSignDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for RealSign landmark feature vectors."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_model(
    dataset_path: Path = DEFAULT_FEATURES_PATH,
    model_save_path: Path = DEFAULT_MODEL_SAVE_PATH,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> Dict[str, Any]:
    """Trains RealSignMLP on the dataset and saves checkpoints."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Loading dataset from {dataset_path}...")
    X, y, splits, classes = load_dataset(dataset_path)

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    print(f"[+] Dataset splits: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_mask]), torch.from_numpy(y[train_mask])),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_mask]), torch.from_numpy(y[val_mask])),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[test_mask]), torch.from_numpy(y[test_mask])),
        batch_size=batch_size,
        shuffle=False,
    )

    model = RealSignMLP(input_dim=63, hidden1=128, hidden2=64, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    best_state = None

    print(f"[+] Training RealSignMLP for {epochs} epochs on device: {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            preds = out.argmax(dim=-1)
            correct += (preds == by).sum().item()
            total += len(by)

        train_acc = correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                val_total += len(by)

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "architecture": {"input_dim": 63, "hidden1": 128, "hidden2": 64, "num_classes": len(classes)},
                "classes": classes,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    # Evaluate on held-out test split
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(dim=-1)
            test_correct += (preds == by).sum().item()
            test_total += len(by)

    test_acc = test_correct / test_total
    print(f"[+] Held-out Test Accuracy: {test_acc*100:.2f}%")

    # Save checkpoints to both .pt and .pth
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(model_save_path))
    torch.save(best_state, str(DEFAULT_ALT_MODEL_SAVE_PATH))
    print(f"[+] Model checkpoint saved to: {model_save_path} and {DEFAULT_ALT_MODEL_SAVE_PATH}")

    return {
        "test_accuracy": test_acc,
        "best_val_accuracy": best_val_acc,
        "classes": classes,
        "model_path": str(model_save_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RealSign 26-class alphabet model")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    train_model(
        dataset_path=args.dataset,
        model_save_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
