"""
SETU — Unified Master 53-Class Dynamic Sequence BiLSTM Training Pipeline
========================================================================
Merges all dynamic ISL sequence tasks into a single high-performance model:
- 49 Amrita conversational phrases
- 'namaste' (custom recordings)
- 'i' (INCLUDE recordings)
- 'you' (INCLUDE recordings)
- 'no_sign' (idle / resting background class)

Architecture:
- Input: (Batch, Seq_len=20, Input_dim=316: 158 Holistic + 158 Velocity)
- Backbone: 2-Layer BiLSTM with inter-layer dropout
- Temporal Attention: Salient frame weighting over 20 temporal steps
- Head: Multi-layer LayerNorm + SiLU + Dropout + Linear(53)
- Regularization: Label smoothing (0.08), Gradient clipping (1.0), Cosine Annealing
- Augmentation: Spatial scale, temporal speed warping, velocity recalculation, class balancing

Outputs:
- models/weights/isl_dynamic_master_bilstm.pth
- models/weights/amrita_isl_phrase_bilstm.pth (drop-in compatibility)
- models/weights/isl_sequence_lstm.pth (drop-in compatibility)
- reports/isl_dynamic_master_evaluation_report.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("SETU.TrainDynamicMaster")

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_NPZ = ROOT_DIR / "data" / "processed" / "isl_dynamic_master_53class.npz"
MASTER_MODEL_PATH = ROOT_DIR / "models" / "weights" / "isl_dynamic_master_bilstm.pth"
AMRITA_COMPAT_PATH = ROOT_DIR / "models" / "weights" / "amrita_isl_phrase_bilstm.pth"
SEQ_COMPAT_PATH = ROOT_DIR / "models" / "weights" / "isl_sequence_lstm.pth"
REPORT_SAVE_PATH = ROOT_DIR / "reports" / "isl_dynamic_master_evaluation_report.json"
LABEL_MAP_PATH = ROOT_DIR / "models" / "metadata" / "dynamic_master_label_map.json"

NUM_FRAMES = 20
FEATURE_DIM = 316  # 158 position + 158 velocity


# ==============================================================================
# 1. Physics-Consistent Sequence Augmentation
# ==============================================================================

class SequenceAugmenter:
    """Applies realistic jitter, spatial scaling, temporal warping and consistent velocity updates."""

    @staticmethod
    def augment(seq: np.ndarray, jitter_std: float = 0.004) -> np.ndarray:
        out = seq.copy()
        T = out.shape[0]
        pos = out[:, :158].copy()

        # 1. Spatial scale jitter (0.94 to 1.06)
        scale = np.random.uniform(0.94, 1.06)
        pos_mask = (pos != 0.0).astype(np.float32)
        pos = pos * scale

        # 2. Add small Gaussian noise to positions
        noise = np.random.normal(0.0, jitter_std, size=pos.shape).astype(np.float32)
        pos += noise * pos_mask

        # 3. Temporal speed warping / resampling
        speed = np.random.uniform(0.88, 1.12)
        warped_len = max(6, int(round(T * speed)))
        orig_t = np.linspace(0.0, 1.0, T)
        warped_t = np.linspace(0.0, 1.0, warped_len)

        warped_pos = np.zeros((warped_len, 158), dtype=np.float32)
        for d in range(158):
            warped_pos[:, d] = np.interp(warped_t, orig_t, pos[:, d])

        resampled_pos = np.zeros((T, 158), dtype=np.float32)
        for d in range(158):
            resampled_pos[:, d] = np.interp(orig_t, warped_t, warped_pos[:, d])

        # 4. Consistent velocity recomputation
        dt = 1.0 / 30.0
        vel = np.zeros_like(resampled_pos)
        if T > 1:
            vel[1:] = (resampled_pos[1:] - resampled_pos[:-1]) / dt
            vel[0] = vel[1]
        vel[:, 146:148] = 0.0  # Zero presence flag velocity

        out[:, :158] = resampled_pos
        out[:, 158:] = vel
        return out.astype(np.float32)


class ClassBalancedDynamicDataset(Dataset):
    """PyTorch Dataset that dynamically balances minority classes (namaste, i, you, no_sign)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        is_train: bool = True,
        target_min_samples: int = 35,
        multiplier: int = 2,
    ) -> None:
        self.is_train = is_train
        self.samples: List[Tuple[np.ndarray, int]] = []

        if not is_train:
            for i in range(len(y)):
                self.samples.append((X[i], int(y[i])))
            return

        # Group by class
        by_class: Dict[int, List[np.ndarray]] = {}
        for i in range(len(y)):
            by_class.setdefault(int(y[i]), []).append(X[i])

        # Balance minority classes and multiply dataset
        for c_idx, items in by_class.items():
            # Add all original items
            for it in items:
                self.samples.append((it, c_idx))

            # Augment to reach at least target_min_samples
            n_orig = len(items)
            target = max(n_orig * multiplier, target_min_samples)
            needed = target - n_orig

            for i in range(needed):
                src = items[i % n_orig]
                aug = SequenceAugmenter.augment(src)
                self.samples.append((aug, c_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, label = self.samples[idx]
        if self.is_train and np.random.rand() > 0.35:
            feat = SequenceAugmenter.augment(feat)
        return torch.tensor(feat, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 2. Master Dynamic Sequence BiLSTM Architecture
# ==============================================================================

class DynamicMasterBiLSTM(nn.Module):
    """
    2-Layer Bidirectional PyTorch LSTM with Temporal Attention mechanism.
    Input Shape: (Batch, Seq_len=20, Input_dim=316)
    Output Shape: (Batch, Num_classes=53)
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 53,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_size * 2  # Bidirectional -> 256

        # Temporal Attention Mechanism over time steps
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # Multi-layer classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)  # (Batch, T, 256)

        # Compute temporal attention weights across time steps
        scores = self.attention(lstm_out)  # (Batch, T, 1)
        weights = torch.softmax(scores, dim=1)  # (Batch, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (Batch, 256)

        return self.classifier(context)


# ==============================================================================
# 3. Training & Evaluation Engine
# ==============================================================================

def train_master_model(
    epochs: int = 70,
    batch_size: int = 32,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Starting Master 53-Class Dynamic Training on device: %s", device)

    # 1. Load Dataset
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    splits = data["splits"]
    class_names = [str(c) for c in data["classes"]]
    num_classes = len(class_names)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Dataset Splitting: Base Train=%d | Val=%d | Test=%d", len(X_train), len(X_val), len(X_test))

    train_ds = ClassBalancedDynamicDataset(X_train, y_train, is_train=True, target_min_samples=35, multiplier=3)
    val_ds = ClassBalancedDynamicDataset(X_val, y_val, is_train=False)
    test_ds = ClassBalancedDynamicDataset(X_test, y_test, is_train=False)

    logger.info("Balanced Augmented Training Set size: %d samples per epoch.", len(train_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer, Scheduler
    model = DynamicMasterBiLSTM(input_size=FEATURE_DIM, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state: Optional[Dict[str, Any]] = None

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = out.argmax(dim=-1)
            train_correct += (preds == by).sum().item()
            total_train += len(by)

        scheduler.step()
        train_acc = train_correct / max(total_train, 1)

        # Validation
        model.eval()
        val_correct, total_val = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                preds = out.argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                total_val += len(by)

        val_acc = val_correct / max(total_val, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            logger.info(
                "Epoch [%2d/%2d] | Train Acc: %5.2f%% | Val Acc: %5.2f%% (Peak: %5.2f%%) | LR: %.6f",
                epoch, epochs, train_acc * 100, val_acc * 100, best_val_acc * 100, scheduler.get_last_lr()[0]
            )

    logger.info("Training complete! Best Validation Accuracy: %.2f%%", best_val_acc * 100)

    # 4. Evaluation on Held-Out Test Set
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    top3_correct = 0
    top5_correct = 0

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            preds = out.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(by.tolist())

            top3 = out.topk(min(3, num_classes), dim=-1).indices.cpu().numpy()
            top5 = out.topk(min(5, num_classes), dim=-1).indices.cpu().numpy()
            by_np = by.numpy()
            for i in range(len(by_np)):
                if by_np[i] in top3[i]:
                    top3_correct += 1
                if by_np[i] in top5[i]:
                    top5_correct += 1

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_acc = float(np.mean(y_true == y_pred))
    top3_acc = float(top3_correct / len(y_true))
    top5_acc = float(top5_correct / len(y_true))

    # Per-class metrics
    per_class = {}
    f1s = []
    for c_idx, c_name in enumerate(class_names):
        tp = int(np.sum((y_pred == c_idx) & (y_true == c_idx)))
        fp = int(np.sum((y_pred == c_idx) & (y_true != c_idx)))
        fn = int(np.sum((y_pred != c_idx) & (y_true == c_idx)))
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

    macro_f1 = float(np.mean(f1s))
    logger.info("Test Evaluation: Top-1 Acc: %.2f%% | Top-3 Acc: %.2f%% | Top-5 Acc: %.2f%% | Macro-F1: %.4f",
                test_acc * 100, top3_acc * 100, top5_acc * 100, macro_f1)

    # 5. Export Unified Checkpoint
    checkpoint = {
        "model_state_dict": best_state if best_state else model.state_dict(),
        "architecture": {
            "name": "DynamicMasterBiLSTM",
            "input_size": FEATURE_DIM,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": num_classes,
            "dropout": 0.30,
            "bidirectional": True,
            "temporal_attention": True,
            "sequence_length": NUM_FRAMES,
        },
        "class_names": class_names,
        "num_classes": num_classes,
        "test_accuracy": round(test_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "split_ratio": {"train": 0.80, "val": 0.10, "test": 0.10},
        "dataset_name": "SETU Unified Master Dynamic Dataset (Amrita 49 + Namaste + I + You + NoSign)",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Save to primary and backward-compatible paths
    MASTER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, MASTER_MODEL_PATH)
    logger.info("Saved Master Model to: %s", MASTER_MODEL_PATH.resolve())

    torch.save(checkpoint, AMRITA_COMPAT_PATH)
    logger.info("Synced to Amrita path for instant compatibility: %s", AMRITA_COMPAT_PATH.resolve())

    torch.save(checkpoint, SEQ_COMPAT_PATH)
    logger.info("Synced to Sequence LSTM path for instant compatibility: %s", SEQ_COMPAT_PATH.resolve())

    # 6. Save Evaluation Report
    REPORT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_name": "SETU Unified Master Dynamic Dataset",
        "total_samples": len(X),
        "num_classes": num_classes,
        "class_names": class_names,
        "splits": {
            "train": int(np.sum(train_mask)),
            "validation": int(np.sum(val_mask)),
            "test": int(np.sum(test_mask)),
        },
        "test_metrics": {
            "overall_accuracy": round(test_acc, 4),
            "overall_accuracy_pct": round(test_acc * 100, 2),
            "top3_accuracy_pct": round(top3_acc * 100, 2),
            "top5_accuracy_pct": round(top5_acc * 100, 2),
            "macro_f1": round(macro_f1, 4),
        },
        "per_class": per_class,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved Evaluation Report to: %s", REPORT_SAVE_PATH.resolve())

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SETU Master Dynamic BiLSTM")
    parser.add_argument("--epochs", type=int, default=70, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate")
    args = parser.parse_args()

    train_master_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

