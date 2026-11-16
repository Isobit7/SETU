"""
SETU — Master Multi-Dataset Training Pipeline (80 / 10 / 10 Split)
==================================================================

Trains all active Indian Sign Language (ISL) datasets with a strict 80% Train,
10% Validation, and 10% Test split on GPU (CUDA), while explicitly excluding the
ISL-Fingerspelling continuous dataset per user requirement.

Datasets Trained:
1. Static ISL Alphabet (26 classes, A–Z) — 126-D Dual-Hand Wrist-Normalized
2. ISL Numerals (10 classes, 0–9) — 126-D Dual-Hand Wrist-Normalized
3. ISL Hindi Vowels (8 classes, अ–ऐ) — 126-D Dual-Hand Wrist-Normalized
4. Dynamic ISL Sequence BiLSTM (53 classes, conversational phrases + greetings) — 316-D Spatio-Temporal
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SETU.TrainAll")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# Model Architectures
# ==============================================================================

class StaticSignMLP(nn.Module):
    """Canonical 3-Layer BatchNorm MLP for 126-D Static Signs (RealSignMLP compatible)."""
    def __init__(self, input_dim: int = 126, hidden1: int = 128, hidden2: int = 64, num_classes: int = 26, dropout: float = 0.2):
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


class SequenceBiLSTM(nn.Module):
    """2-Layer BiLSTM with Temporal Attention for Spatio-Temporal Sequences."""
    def __init__(self, input_size: int = 316, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 53, dropout: float = 0.3):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)  # (B, T, 2H)
        scores = self.attn(out)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(out * weights, dim=1)  # (B, 2H)
        return self.classifier(context)


# ==============================================================================
# Generic PyTorch Datasets
# ==============================================================================

class VectorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.is_train and np.random.rand() > 0.4:
            # Subtle Gaussian landmark jitter on active landmarks
            mask = (x != 0.0).float()
            x = x + torch.randn_like(x) * 0.006 * mask
        return x, self.y[idx]


# ==============================================================================
# Helper: Export ONNX
# ==============================================================================

def export_onnx_model(model: nn.Module, input_shape: Tuple[int, ...], output_path: Path):
    model.eval()
    cpu_model = model.to("cpu")
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    try:
        torch.onnx.export(
            cpu_model,
            dummy,
            str(output_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=14,
        )
        logger.info("  ✓ Successfully exported ONNX: %s", output_path)
    except Exception as e:
        logger.warning("  ⚠ ONNX export warning: %s", e)


# ==============================================================================
# Training Engine for Static MLP Models
# ==============================================================================

def train_static_task(
    task_name: str,
    npz_path: Path,
    pth_save_path: Path,
    onnx_save_path: Optional[Path],
    extra_pth_paths: List[Path] = None,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    logger.info("==================================================================")
    logger.info("TRAINING TASK: %s", task_name.upper())
    logger.info("Dataset: %s", npz_path)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    splits = data["splits"]
    classes = [str(c) for c in data["classes"]]
    num_classes = len(classes)

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    logger.info(
        "Split breakdown: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Test=%d (%.1f%%)",
        np.sum(train_mask), np.mean(train_mask) * 100,
        np.sum(val_mask), np.mean(val_mask) * 100,
        np.sum(test_mask), np.mean(test_mask) * 100,
    )

    train_loader = DataLoader(VectorDataset(X[train_mask], y[train_mask], is_train=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(VectorDataset(X[val_mask], y[val_mask], is_train=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(VectorDataset(X[test_mask], y[test_mask], is_train=False), batch_size=batch_size, shuffle=False)

    model = StaticSignMLP(input_dim=126, hidden1=128, hidden2=64, num_classes=num_classes, dropout=0.2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == by).sum().item()
                total_val += len(bx)

        val_loss /= max(1, total_val)
        val_acc = val_correct / max(1, total_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("  Epoch %2d/%2d: Train Loss=%.4f | Val Loss=%.4f | Val Acc=%.2f%%", epoch, epochs, total_loss / len(train_loader.dataset), val_loss, val_acc * 100)

    # Load best model for evaluation on Test set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    test_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    logger.info("  >> %s FINAL 10%% TEST EVALUATION: Acc=%.2f%% | Prec=%.2f%% | Rec=%.2f%% | F1=%.4f", task_name.upper(), test_acc * 100, test_prec * 100, test_rec * 100, test_f1)

    # Save Checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": 126,
        "num_classes": num_classes,
        "classes": classes,
        "test_accuracy": float(test_acc),
        "macro_precision": float(test_prec),
        "macro_recall": float(test_rec),
        "macro_f1": float(test_f1),
        "architecture": {
            "input_dim": 126,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": num_classes,
            "use_batchnorm": True,
            "dropout": 0.2,
        },
        "trained_at": datetime.datetime.now().isoformat(),
        "device": str(DEVICE),
    }

    pth_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(pth_save_path))
    logger.info("  ✓ Saved PyTorch weights to %s", pth_save_path)

    if extra_pth_paths:
        for ep in extra_pth_paths:
            ep.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, str(ep))
            logger.info("  ✓ Synchronized weights to %s", ep)

    if onnx_save_path:
        export_onnx_model(model, (1, 126), onnx_save_path)

    return {
        "task": task_name,
        "num_classes": num_classes,
        "test_accuracy": test_acc,
        "macro_precision": test_prec,
        "macro_recall": test_rec,
        "macro_f1": test_f1,
    }


# ==============================================================================
# Training Engine for Dynamic Sequence BiLSTM
# ==============================================================================

def train_dynamic_sequence_task(
    npz_path: Path,
    pth_save_path: Path,
    extra_pth_paths: List[Path] = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    task_name = "Dynamic ISL Sequence BiLSTM (53 Classes)"
    logger.info("==================================================================")
    logger.info("TRAINING TASK: %s", task_name.upper())
    logger.info("Dataset: %s", npz_path)

    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]  # Shape: (N, 20, 316)
    y = data["y"]
    splits = data["splits"]
    classes = [str(c) for c in data["classes"]]
    num_classes = len(classes)

    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    logger.info(
        "Split breakdown: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Test=%d (%.1f%%)",
        np.sum(train_mask), np.mean(train_mask) * 100,
        np.sum(val_mask), np.mean(val_mask) * 100,
        np.sum(test_mask), np.mean(test_mask) * 100,
    )

    train_loader = DataLoader(VectorDataset(X[train_mask], y[train_mask], is_train=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(VectorDataset(X[val_mask], y[val_mask], is_train=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(VectorDataset(X[test_mask], y[test_mask], is_train=False), batch_size=batch_size, shuffle=False)

    model = SequenceBiLSTM(input_size=316, hidden_size=128, num_layers=2, num_classes=num_classes, dropout=0.3).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Any]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
                preds = torch.argmax(out, dim=1)
                val_correct += (preds == by).sum().item()
                total_val += len(bx)

        val_loss /= max(1, total_val)
        val_acc = val_correct / max(1, total_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == epochs:
            logger.info("  Epoch %2d/%2d: Train Loss=%.4f | Val Loss=%.4f | Val Acc=%.2f%%", epoch, epochs, total_loss / len(train_loader.dataset), val_loss, val_acc * 100)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())

    test_acc = accuracy_score(all_targets, all_preds)
    test_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    test_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    logger.info("  >> DYNAMIC SEQUENCE FINAL 10%% TEST EVALUATION: Acc=%.2f%% | Prec=%.2f%% | Rec=%.2f%% | F1=%.4f", test_acc * 100, test_prec * 100, test_rec * 100, test_f1)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": 316,
        "hidden_size": 128,
        "num_layers": 2,
        "num_classes": num_classes,
        "classes": classes,
        "test_accuracy": float(test_acc),
        "macro_precision": float(test_prec),
        "macro_recall": float(test_rec),
        "macro_f1": float(test_f1),
        "trained_at": datetime.datetime.now().isoformat(),
        "device": str(DEVICE),
    }

    pth_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(pth_save_path))
    logger.info("  ✓ Saved PyTorch weights to %s", pth_save_path)

    if extra_pth_paths:
        for ep in extra_pth_paths:
            ep.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, str(ep))
            logger.info("  ✓ Synchronized weights to %s", ep)

    return {
        "task": task_name,
        "num_classes": num_classes,
        "test_accuracy": test_acc,
        "macro_precision": test_prec,
        "macro_recall": test_rec,
        "macro_f1": test_f1,
    }


# ==============================================================================
# Master Execution Flow
# ==============================================================================

def main():
    logger.info("##################################################################")
    logger.info("STARTING SETU MULTI-DATASET TRAINING (80 / 10 / 10 SPLIT)")
    logger.info("Compute Hardware: %s (%s)", DEVICE, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    logger.info("Excluding: data/raw/NOT TRAINED/ISL-Fingerspelling (per instruction)")
    logger.info("##################################################################")

    start_time = time.time()
    results = []

    # 1. Static ISL Alphabet (26 classes, A–Z)
    res_alpha = train_static_task(
        task_name="ISL Alphabet (A–Z)",
        npz_path=PROJECT_ROOT / "data" / "processed" / "static_isl_alphabet_126d.npz",
        pth_save_path=PROJECT_ROOT / "models" / "weights" / "realsign_26class_mlp.pth",
        onnx_save_path=PROJECT_ROOT / "models" / "weights" / "realsign_multisigner.onnx",
        extra_pth_paths=[PROJECT_ROOT / "models" / "weights" / "realsign_multisigner_mlp.pth"],
        epochs=40,
        batch_size=64,
        lr=1e-3,
    )
    results.append(res_alpha)

    # 2. ISL Numerals (10 classes, 0–9)
    res_num = train_static_task(
        task_name="ISL Numerals (0–9)",
        npz_path=PROJECT_ROOT / "data" / "processed" / "static_isl_numerals_126d.npz",
        pth_save_path=PROJECT_ROOT / "models" / "weights" / "isl_numerals_mlp.pth",
        onnx_save_path=PROJECT_ROOT / "models" / "weights" / "isl_numerals.onnx",
        epochs=35,
        batch_size=64,
        lr=1e-3,
    )
    results.append(res_num)

    # 3. ISL Hindi Vowels (8 classes, अ–ऐ)
    res_vow = train_static_task(
        task_name="ISL Hindi Vowels (अ–ऐ)",
        npz_path=PROJECT_ROOT / "data" / "processed" / "static_isl_hindi_vowels_126d.npz",
        pth_save_path=PROJECT_ROOT / "models" / "weights" / "isl_hindi_vowels_mlp.pth",
        onnx_save_path=PROJECT_ROOT / "models" / "weights" / "isl_hindi_vowels.onnx",
        epochs=35,
        batch_size=64,
        lr=1e-3,
    )
    results.append(res_vow)

    # 4. Dynamic ISL Sequence BiLSTM (53 classes)
    res_dyn = train_dynamic_sequence_task(
        npz_path=PROJECT_ROOT / "data" / "processed" / "isl_dynamic_master_53class.npz",
        pth_save_path=PROJECT_ROOT / "models" / "weights" / "isl_dynamic_master_bilstm.pth",
        extra_pth_paths=[
            PROJECT_ROOT / "models" / "weights" / "amrita_isl_phrase_bilstm.pth",
        ],
        epochs=40,
        batch_size=32,
        lr=1e-3,
    )
    results.append(res_dyn)

    total_time = time.time() - start_time
    logger.info("##################################################################")
    logger.info("ALL TRAINING TASKS COMPLETED IN %.2f SECONDS!", total_time)
    logger.info("##################################################################")

    # Save summary report
    report_file = PROJECT_ROOT / "reports" / "multi_dataset_80_10_10_training_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump({
            "generated_at": datetime.datetime.now().isoformat(),
            "device": str(DEVICE),
            "total_training_time_seconds": total_time,
            "results": results,
        }, f, indent=2)
    logger.info("Saved full benchmark report to: %s", report_file)


if __name__ == "__main__":
    main()
