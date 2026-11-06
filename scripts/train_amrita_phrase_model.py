"""
SETU — Amrita SLR Dynamic Phrase Dataset Feature Extraction & Training Pipeline
================================================================================
Processes 978 authentic ISL dynamic phrase videos across 49 categories:
1. Extracts 316-D spatio-temporal features (158-D Holistic + 158-D Velocity) over 20 temporal steps.
2. Partitions dataset using Stratified 80% Train / 10% Validation / 10% Test split.
3. Trains a 2-layer Bidirectional LSTM with Temporal Attention on GPU (CUDA).
4. Evaluates performance on held-out test split (Overall Accuracy, Top-3, Top-5, Macro-F1).
5. Exports model checkpoint to 'models/weights/amrita_isl_phrase_bilstm.pth' and report to 'reports/'.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.vision.landmarks import HolisticLandmarkerWrapper
from training.features.extractor import FeatureExtractor158

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SETU.AmritaTrain")

DATA_RAW_DIR = ROOT_DIR / "data" / "raw" / "amrita_slr" / "sign project"
PROCESSED_NPZ_PATH = ROOT_DIR / "data" / "processed" / "amrita_slr_sequences.npz"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "weights" / "amrita_isl_phrase_bilstm.pth"
REPORT_SAVE_PATH = ROOT_DIR / "reports" / "amrita_slr_evaluation_report.json"
TASK_PATH = ROOT_DIR / "models" / "tasks" / "holistic_landmarker.task"

NUM_FRAMES = 20
FEATURE_DIM_BASE = 158
FEATURE_DIM_VEL = 316  # 158 position + 158 velocity


# ==============================================================================
# 1. Feature Extraction & Temporal Resampling
# ==============================================================================

def clean_class_name(raw_name: str) -> str:
    """Standardizes folder names like '10.youareperfect' -> 'you_are_perfect'."""
    clean = re.sub(r"^\d+\.", "", raw_name).strip().lower()
    mapping = {
        "youareperfect": "you_are_perfect",
        "goodmorning": "good_morning",
        "thankyou": "thank_you",
        "umberlla": "umbrella",
        "tommorow": "tomorrow",
    }
    return mapping.get(clean, clean)


def compute_velocity_316(seq_tx158: np.ndarray, dt: float = 1.0 / 30.0) -> np.ndarray:
    """Computes frame-to-frame velocity derivatives and concatenates (T, 316)."""
    T, D = seq_tx158.shape
    vel = np.zeros_like(seq_tx158)
    if T > 1:
        vel[1:] = (seq_tx158[1:] - seq_tx158[:-1]) / dt
        vel[0] = vel[1]  # Replicate forward velocity for first step
    # Velocity of binary presence flags is not meaningful -> zero out
    vel[:, 146:148] = 0.0
    return np.concatenate([seq_tx158, vel], axis=-1).astype(np.float32)


def extract_features_from_video(
    video_path: Path,
    landmarker: HolisticLandmarkerWrapper,
    target_frames: int = NUM_FRAMES,
) -> Optional[np.ndarray]:
    """Uniformly samples frames, extracts 158-D holistic landmarks, and adds velocity."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 2:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    extracted = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            # Fallback zero vector
            extracted.append(np.zeros(FEATURE_DIM_BASE, dtype=np.float32))
            continue

        h, w = frame.shape[:2]
        if max(h, w) > 480:
            scale = 480.0 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        try:
            res = landmarker.detect(frame)
            vec = FeatureExtractor158.extract_from_mediapipe_result(res)
            extracted.append(vec)
        except Exception:
            extracted.append(np.zeros(FEATURE_DIM_BASE, dtype=np.float32))

    cap.release()

    if len(extracted) != target_frames:
        return None

    raw_seq = np.array(extracted, dtype=np.float32)
    seq_316 = compute_velocity_316(raw_seq)
    return seq_316


def prepare_amrita_dataset(
    raw_dir: Path = DATA_RAW_DIR,
    output_npz: Path = PROCESSED_NPZ_PATH,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Discovers videos, extracts landmarks, performs 80/10/10 stratified split, and saves NPZ."""
    if output_npz.exists() and not force_rebuild:
        logger.info("Loading existing preprocessed dataset from: %s", output_npz.resolve())
        data = np.load(output_npz, allow_pickle=True)
        return data["X"], data["y"], data["splits"], list(data["classes"])

    logger.info("Initializing MediaPipe HolisticLandmarker...")
    landmarker = HolisticLandmarkerWrapper(task_model_path=TASK_PATH)
    landmarker.initialize()

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    class_names = [clean_class_name(d.name) for d in class_dirs]
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_samples: List[Tuple[np.ndarray, int, str]] = []
    logger.info("Found %d classes. Beginning sequence feature extraction across videos...", len(class_names))

    total_vids = 0
    t_start = time.perf_counter()

    for d_idx, c_dir in enumerate(class_dirs, 1):
        c_label = class_names[d_idx - 1]
        c_idx = label_to_idx[c_label]
        vids = sorted([f for f in c_dir.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mov")])

        c_extracted = 0
        for v in vids:
            total_vids += 1
            feat_316 = extract_features_from_video(v, landmarker, target_frames=NUM_FRAMES)
            if feat_316 is not None:
                all_samples.append((feat_316, c_idx, v.name))
                c_extracted += 1

        if d_idx % 5 == 0 or d_idx == len(class_dirs):
            elapsed = time.perf_counter() - t_start
            logger.info("Processed [%2d/%2d] classes (%3d samples so far) in %.1fs", d_idx, len(class_dirs), len(all_samples), elapsed)

    logger.info("Feature extraction complete! Successfully extracted %d valid sequences.", len(all_samples))

    X = np.stack([s[0] for s in all_samples], axis=0).astype(np.float32)
    y = np.array([s[1] for s in all_samples], dtype=np.int64)

    # Stratified 80% Train, 10% Validation, 10% Test split
    indices = np.arange(len(y))
    idx_train, idx_temp, y_train, y_temp = train_test_split(
        indices, y, test_size=0.20, random_state=42, stratify=y
    )
    idx_val, idx_test, y_val, y_test = train_test_split(
        idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    splits = np.empty(len(y), dtype="<U10")
    splits[idx_train] = "train"
    splits[idx_val] = "val"
    splits[idx_test] = "test"

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        splits=splits,
        classes=np.array(class_names),
        sample_ids=np.array([s[2] for s in all_samples]),
    )
    logger.info("Saved preprocessed dataset to: %s (Splits: Train=%d, Val=%d, Test=%d)", output_npz.resolve(), len(idx_train), len(idx_val), len(idx_test))
    return X, y, splits, class_names


# ==============================================================================
# 2. PyTorch Dataset with Dynamic Temporal Data Augmentation
# ==============================================================================

class SequenceAugmenter:
    """Applies realistic jitter and speed perturbations to (30, 316) sequences."""

    @staticmethod
    def augment(seq: np.ndarray, jitter_std: float = 0.005) -> np.ndarray:
        out = seq.copy()
        # Add small Gaussian noise to positions [0..157]
        pos_mask = (out[:, :158] != 0.0).astype(np.float32)
        noise = np.random.normal(0.0, jitter_std, size=out[:, :158].shape).astype(np.float32)
        out[:, :158] += noise * pos_mask
        return out


class AmritaSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, is_train: bool = True) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.X[idx].clone().numpy()
        if self.is_train and np.random.rand() > 0.4:
            feat = SequenceAugmenter.augment(feat)
        return torch.tensor(feat, dtype=torch.float32), self.y[idx]


# ==============================================================================
# 3. Dynamic Phrase Bidirectional LSTM with Temporal Attention
# ==============================================================================

class AmritaPhraseBiLSTM(nn.Module):
    """
    2-Layer Bidirectional PyTorch LSTM with Temporal Attention mechanism.
    Input Shape: (Batch, Seq_len=20, Input_dim=316)
    Output Shape: (Batch, Num_classes=49)
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM_VEL,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 49,
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

        # Dense classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)  # (B, T, 256)

        # Compute temporal attention weights across time steps
        scores = self.attention(lstm_out)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        context = torch.sum(weights * lstm_out, dim=1)  # (B, 256)

        return self.classifier(context)


# ==============================================================================
# 4. Training, Scoring & Comprehensive Evaluation Engine
# ==============================================================================

def train_and_evaluate(
    epochs: int = 55,
    batch_size: int = 32,
    lr: float = 1.2e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Beginning Amrita SLR Phrase Training on device: %s", device)

    # 1. Dataset Preparation
    X, y, splits, class_names = prepare_amrita_dataset()
    num_classes = len(class_names)

    train_mask = splits == "train"
    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info("Dataset Splitting: Train=%d (80%%) | Val=%d (10%%) | Test=%d (10%%)", len(X_train), len(X_val), len(X_test))

    train_loader = DataLoader(AmritaSequenceDataset(X_train, y_train, is_train=True), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(AmritaSequenceDataset(X_val, y_val, is_train=False), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(AmritaSequenceDataset(X_test, y_test, is_train=False), batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer, Scheduler
    model = AmritaPhraseBiLSTM(input_size=FEATURE_DIM_VEL, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_state: Optional[Dict[str, Any]] = None

    # 3. Main Epoch Loop
    t_start = time.perf_counter()
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
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(by)
                preds = out.argmax(dim=-1)
                val_correct += (preds == by).sum().item()
                total_val += len(by)

        val_acc = val_correct / max(total_val, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs or epoch == 1:
            logger.info(
                "Epoch [%2d/%2d] | Train Acc: %5.2f%% | Val Acc: %5.2f%% (Best: %5.2f%%) | LR: %.6f",
                epoch, epochs, train_acc * 100, val_acc * 100, best_val_acc * 100, scheduler.get_last_lr()[0]
            )

    train_duration = time.perf_counter() - t_start
    logger.info("Training complete in %.1f seconds! Peak Validation Accuracy: %.2f%%", train_duration, best_val_acc * 100)

    # 4. Evaluation on Held-Out 10% Test Split
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

            # Top-3 and Top-5 accuracy
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

    # Per-class precision, recall, F1
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

    # Top confusion pairs
    conf_pairs = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            conf_pairs[(class_names[t], class_names[p])] += 1

    top_confusion = [
        {"true_sign": k[0], "predicted_sign": k[1], "count": v}
        for k, v in conf_pairs.most_common(10)
    ]

    # 5. Save Trained Checkpoint
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "architecture": {
            "input_size": FEATURE_DIM_VEL,
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
        "dataset_name": "Amrita SLR Dataset",
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    logger.info("Successfully exported trained Amrita phrase model to: %s", MODEL_SAVE_PATH.resolve())

    # 6. Save Evaluation Report
    REPORT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_name": "Amrita SLR Dataset",
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
        "top_confusion_pairs": top_confusion,
        "per_class": per_class,
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved evaluation report to: %s", REPORT_SAVE_PATH.resolve())

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SETU Phrase BiLSTM on Amrita SLR Dataset")
    parser.add_argument("--epochs", type=int, default=55, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1.2e-3, help="Learning rate")
    args = parser.parse_args()

    train_and_evaluate(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

