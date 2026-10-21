"""
SETU — Dynamic ISL Bidirectional LSTM Classifier with Temporal Attention
Builds, trains, and evaluates a 2-layer Bidirectional LSTM sequence classifier on GPU (CUDA)
for dynamic Indian Sign Language (ISL) recognition over 316-D spatio-temporal features
(158-D Holistic coordinates + 158-D frame-to-frame velocity vectors).

Model Architecture:
- Backbone: 2-layer Bidirectional LSTM (input_size=316, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
- Temporal Attention Mechanism: Attention weights computed over 30 time steps to weight salient gesture frames over idle frames
- Classifier Head: Linear(256) -> LayerNorm(256) -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Linear(N_CLASSES=6)
- Input Shape: (Batch, 30, 316)
- Optimization: AdamW(lr=0.001, weight_decay=1e-3), CosineAnnealingLR, 60 epochs on CUDA

Target Classes (6):
["hello", "thank_you", "good_morning", "i", "you", "no_sign"]
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
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SETU.LSTM")

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DEFAULT_MANIFEST_PATH = ROOT_DIR / "data" / "manifests" / "setu_isl_mvp_manifest.json"
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DEFAULT_MODEL_SAVE_PATH = ROOT_DIR / "models" / "isl_sequence_lstm.pth"
DEFAULT_REPORT_SAVE_PATH = ROOT_DIR / "reports" / "isl_sequence_lstm_evaluation.json"
DEFAULT_LABEL_MAP_PATH = ROOT_DIR / "models" / "label_map.json"

BASE_CLASSES = ["hello", "thank_you", "good_morning", "i", "you", "no_sign"]
CLASS_NAMES = list(BASE_CLASSES)
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}


from setu.features.extractor import compute_velocity_features


class ISLSequenceLSTM(nn.Module):
    """
    2-Layer Bidirectional PyTorch LSTM Sequence Classifier with Temporal Attention
    for 328-D ISL Landmark + Facial Ratios + Velocity Sequences.
    Input Shape: (Batch, 30, 328)
    Output Shape: (Batch, num_classes)
    """

    def __init__(
        self,
        input_size: int = 328,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        # Layer normalization to stabilize varying landmark scales and velocity magnitudes
        self.input_norm = nn.LayerNorm(input_size)

        # 2-layer Bidirectional LSTM with inter-layer dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size

        # Temporal Attention Mechanism over the 30 time steps to weight salient gesture frames
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # Classifier head: Linear(256) -> LayerNorm -> ReLU -> Dropout(0.3) -> Linear(64) -> ReLU -> Linear(N_CLASSES)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, Seq_len=30, Input_dim=316)
        Returns:
            logits: Tensor of shape (Batch, Num_classes=6)
        """
        x_norm = self.input_norm(x)
        # lstm_out: (Batch, Seq_len=30, Hidden_dim=256)
        lstm_out, _ = self.lstm(x_norm)

        # Compute temporal attention weights across the 30 time steps
        attn_scores = self.attention(lstm_out)  # (Batch, 30, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (Batch, 30, 1)

        # Salient feature aggregation via attention-weighted sum
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (Batch, 256)

        logits = self.fc(context)
        return logits

    def forward_with_attention(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns both logits and temporal attention weights for diagnostics/interpretability."""
        x_norm = self.input_norm(x)
        lstm_out, _ = self.lstm(x_norm)
        attn_scores = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.fc(context)
        return logits, attn_weights.squeeze(-1)


def temporal_resample_np(seq: np.ndarray, target_len: int = 15) -> np.ndarray:
    """Resamples a (T, D) array along the temporal dimension using 1D linear interpolation."""
    T, D = seq.shape
    if T == target_len:
        return seq.copy()
    orig_idx = np.linspace(0, T - 1, T)
    target_idx = np.linspace(0, T - 1, target_len)
    res = np.zeros((target_len, D), dtype=np.float32)
    for d in range(D):
        res[:, d] = np.interp(target_idx, orig_idx, seq[:, d])
    return res


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Applies realistic sequence augmentations for ISL hand/body landmark sequences:
    1. Coordinate Gaussian jitter (std=0.008)
    2. Subtle scaling (0.96 to 1.04)
    3. Spatial translation (-0.02 to 0.02)
    4. Temporal speed warping (resampled between 0.85x and 1.15x duration back to 30)
    Preserves binary presence flags (indices 146, 147).
    """
    T, D = seq.shape
    aug = seq.copy().astype(np.float32)

    # 1. Coordinate Gaussian jitter
    noise = np.random.normal(0, 0.008, size=(T, D)).astype(np.float32)
    noise[:, 146:148] = 0.0
    aug += noise

    # 2. Subtle scaling
    scale = np.random.uniform(0.96, 1.04)
    aug[:, :146] *= scale
    if D >= 158:
        aug[:, 148:158] *= scale
    else:
        aug[:, 148:] *= scale

    # 3. Spatial translation
    shift = np.random.uniform(-0.02, 0.02, size=(1, D)).astype(np.float32)
    shift[:, 146:148] = 0.0
    if D >= 164:
        shift[:, 158:164] = 0.0
    aug += shift

    # 4. Temporal speed warping (0.85x to 1.15x)
    speed = np.random.uniform(0.85, 1.15)
    warped_len = max(8, int(round(T * speed)))
    warped_seq = temporal_resample_np(aug, warped_len)
    resampled = temporal_resample_np(warped_seq, T)

    # Preserve binary presence flags
    resampled[:, 146:148] = np.round(np.clip(resampled[:, 146:148], 0.0, 1.0))
    return resampled


class ISLSequenceDataset(Dataset):
    """
    PyTorch Dataset yielding (features: (30, 328), label: int, sample_id: str).
    Converts 158-D or 164-D features into 328-D spatio-temporal velocity vectors.
    Supports class-balanced offline sequence augmentation.
    """

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, int, str]],
        augment: bool = False,
        augment_factor: int = 1,
        balance_classes: bool = True,
        target_samples_per_class: Optional[int] = None,
        no_sign_class_idx: Optional[int] = None,
        dt: Optional[float] = None,
        fps: Optional[float] = 30.0,
    ) -> None:
        self.data: List[Tuple[np.ndarray, int, str]] = []
        if dt is not None:
            self.dt = dt
        elif fps is not None and fps > 0:
            self.dt = 1.0 / fps
        else:
            self.dt = 1.0

        if not augment:
            # Validation and test: exact samples with velocity features
            for feat, label, sid in samples:
                base_328 = compute_velocity_features(feat, target_dim=328, dt=self.dt)
                self.data.append((base_328, label, sid))
            return

        # Group training samples by class
        by_class: Dict[int, List[Tuple[np.ndarray, int, str]]] = {}
        for s in samples:
            by_class.setdefault(s[1], []).append(s)

        if balance_classes and by_class:
            # Compute target samples per class
            active_class_counts = [
                len(grp) * max(1, augment_factor)
                for c_idx, grp in by_class.items()
                if no_sign_class_idx is None or c_idx != no_sign_class_idx
            ]
            default_target = max(active_class_counts) if active_class_counts else 30
            target_count = target_samples_per_class or max(30, default_target)

            for c_idx, grp in by_class.items():
                is_no_sign = (no_sign_class_idx is not None and c_idx == no_sign_class_idx)
                c_target = max(len(grp), target_count // 2 if is_no_sign else target_count)

                # Add all base samples
                for feat, label, sid in grp:
                    base_328 = compute_velocity_features(feat, target_dim=328, dt=self.dt)
                    self.data.append((base_328, label, sid))

                # Augment to reach c_target
                needed = c_target - len(grp)
                for i in range(needed):
                    src_feat, label, sid = grp[i % len(grp)]
                    aug_feat = augment_sequence(src_feat)
                    aug_328 = compute_velocity_features(aug_feat, target_dim=328, dt=self.dt)
                    self.data.append((aug_328, label, f"{sid}_bal_aug_{i}"))
        else:
            for feat, label, sid in samples:
                base_328 = compute_velocity_features(feat, target_dim=328, dt=self.dt)
                self.data.append((base_328, label, sid))

                count = (
                    augment_factor
                    if (no_sign_class_idx is None or label != no_sign_class_idx)
                    else max(1, augment_factor // 4)
                )
                for i in range(count):
                    aug_feat = augment_sequence(feat)
                    aug_328 = compute_velocity_features(aug_feat, target_dim=328, dt=self.dt)
                    self.data.append((aug_328, label, f"{sid}_aug_{i}"))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        feat, label, sid = self.data[idx]
        return torch.from_numpy(feat), torch.tensor(label, dtype=torch.long), sid


def load_sequence_dataset(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    label_map_path: Path = DEFAULT_LABEL_MAP_PATH,
    custom_split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    update_label_map: bool = True,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, int, str]]], List[str], Dict[str, int]]:
    """
    Loads sequence dataset aggregating standard manifest samples and any custom recorded samples:
    1. Scans manifest_path for registered sequence samples.
    2. Scans processed_dir for all custom_*.npz files (or uncatalogued .npz sequence files).
    3. Dynamically extracts all unique labels, preserving base classes and keeping 'no_sign' at the end.
    4. Automatically writes/updates label_map.json.
    5. Partitions custom samples into train/val/test splits based on custom_split_ratios.
    6. Returns (splits, class_names, label_to_idx).
    """
    manifest_path = Path(manifest_path)
    processed_dir = Path(processed_dir)
    label_map_path = Path(label_map_path)

    seen_paths: set[Path] = set()
    manifest_samples: List[Tuple[np.ndarray, str, str, str]] = []  # (feat, label_str, sid, split)
    unassigned_samples: List[Tuple[np.ndarray, str, str]] = []     # (feat, label_str, sid)
    all_labels_set: set[str] = set(BASE_CLASSES)

    # 1. Load manifest if it exists
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for item in manifest.get("samples", []):
            split = item.get("split", "train")
            label_str = item.get("label", "").strip()
            if not label_str:
                continue
            sample_id = item.get("sample_id", "unknown")

            raw_feat_path = Path(item["features_path"])
            if raw_feat_path.exists():
                feat_path = raw_feat_path
            else:
                feat_path = processed_dir / raw_feat_path.name

            if not feat_path.exists():
                logger.warning(f"Manifest feature file missing: {feat_path}")
                continue

            try:
                with np.load(feat_path) as data:
                    features = data["features"].astype(np.float32)
                if features.ndim != 2 or features.shape[1] not in (158, 164):
                    logger.warning(f"Unexpected shape {features.shape} in manifest sample {feat_path.name}")
                    continue
                if features.shape[0] != 15:
                    features = temporal_resample_np(features, target_len=15)
                manifest_samples.append((features, label_str, sample_id, split))
                all_labels_set.add(label_str)
                seen_paths.add(feat_path.resolve())
            except Exception as e:
                logger.warning(f"Could not load manifest sample {feat_path.name}: {e}")

    # 2. Automatically scan processed_dir for custom_*.npz or other uncatalogued .npz sequence files
    if processed_dir.exists():
        for npz_path in sorted(processed_dir.glob("*.npz")):
            if npz_path.resolve() in seen_paths:
                continue
            try:
                with np.load(npz_path) as data:
                    if "features" not in data:
                        continue
                    feat = data["features"].astype(np.float32)
                    lbl = str(data.get("label", "")).strip()

                if feat.ndim != 2 or feat.shape[1] not in (158, 164):
                    continue

                if feat.shape[0] != 15:
                    feat = temporal_resample_np(feat, target_len=15)

                if not lbl and npz_path.name.startswith("custom_"):
                    parts = npz_path.stem.split("_")
                    if len(parts) >= 2:
                        lbl = parts[1]
                if not lbl:
                    continue

                sid = npz_path.stem
                unassigned_samples.append((feat, lbl, sid))
                all_labels_set.add(lbl)
                seen_paths.add(npz_path.resolve())
            except Exception as e:
                logger.warning(f"Skipping corrupted custom npz file {npz_path.name}: {e}")

    # 3. Build dynamic class names: base signs -> custom signs (alphabetical) -> no_sign at end
    base_signs = [c for c in BASE_CLASSES if c != "no_sign"]
    custom_signs = sorted([c for c in all_labels_set if c not in BASE_CLASSES])
    class_names: List[str] = base_signs + custom_signs + ["no_sign"]
    label_to_idx: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
    idx_to_label: Dict[int, str] = {idx: name for idx, name in enumerate(class_names)}

    # 4. Automatically update label_map.json
    if update_label_map:
        label_map_data = {
            "class_names": class_names,
            "label_to_idx": label_to_idx,
            "idx_to_label": {str(k): v for k, v in idx_to_label.items()},
            "num_classes": len(class_names),
            "base_classes": BASE_CLASSES,
            "custom_classes": custom_signs,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        label_map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map_data, f, indent=2)
        logger.info(f"Updated dynamic label map at {label_map_path} ({len(class_names)} classes: {class_names}).")

    # 5. Populate splits
    splits: Dict[str, List[Tuple[np.ndarray, int, str]]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    # Add manifest samples to splits
    for feat, lbl, sid, sp in manifest_samples:
        if lbl in label_to_idx:
            c_idx = label_to_idx[lbl]
            splits[sp].append((feat, c_idx, sid))

    # Partition unassigned custom samples
    custom_by_label: Dict[str, List[Tuple[np.ndarray, str]]] = {}
    for feat, lbl, sid in unassigned_samples:
        custom_by_label.setdefault(lbl, []).append((feat, sid))

    for lbl, sample_list in custom_by_label.items():
        c_idx = label_to_idx[lbl]
        total_lbl = len(sample_list)
        if total_lbl == 1:
            splits["train"].append((sample_list[0][0], c_idx, sample_list[0][1]))
        elif total_lbl == 2:
            splits["train"].append((sample_list[0][0], c_idx, sample_list[0][1]))
            splits["val"].append((sample_list[1][0], c_idx, sample_list[1][1]))
        else:
            n_train = max(1, int(round(total_lbl * custom_split_ratios[0])))
            n_val = max(1, int(round(total_lbl * custom_split_ratios[1])))
            if n_train + n_val >= total_lbl:
                n_train = total_lbl - 2
                n_val = 1
            n_test = total_lbl - n_train - n_val
            if n_test < 1 and total_lbl >= 3:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1

            for feat, sid in sample_list[:n_train]:
                splits["train"].append((feat, c_idx, sid))
            for feat, sid in sample_list[n_train:n_train + n_val]:
                splits["val"].append((feat, c_idx, sid))
            for feat, sid in sample_list[n_train + n_val:]:
                splits["test"].append((feat, c_idx, sid))

    for sp_name, sp_list in splits.items():
        logger.info(f"Loaded split '{sp_name}': {len(sp_list)} total samples.")

    return splits, class_names, label_to_idx


def load_manifest_data(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> Dict[str, List[Tuple[np.ndarray, int, str]]]:
    """Compatibility wrapper that calls load_sequence_dataset and returns splits."""
    splits, _, _ = load_sequence_dataset(
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        update_label_map=False,
    )
    return splits


def compute_metrics(
    y_true: Any,
    y_pred: Any,
    classes: List[str],
) -> Dict[str, Any]:
    """Computes overall accuracy, per-class precision/recall/F1, and confusion matrix."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    num_classes = len(classes)
    total = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    accuracy = float(correct / total) if total > 0 else 0.0

    conf_mat = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []

    for c_idx, c_name in enumerate(classes):
        tp = int(conf_mat[c_idx, c_idx])
        fp = int(np.sum(conf_mat[:, c_idx]) - tp)
        fn = int(np.sum(conf_mat[c_idx, :]) - tp)
        support = int(np.sum(conf_mat[c_idx, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        per_class[c_name] = {
            "class_id": c_idx,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "total_samples": total,
        "correct_predictions": correct,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "classes": classes,
    }


def train_sequence_lstm(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    model_save_path: Path = DEFAULT_MODEL_SAVE_PATH,
    report_save_path: Path = DEFAULT_REPORT_SAVE_PATH,
    epochs: int = 60,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-3,
    device_name: str = "cuda",
    seed: int = 123,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """Builds, trains, and evaluates the ISL Sequence LSTM on GPU."""
    start_time = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determine PyTorch execution device
    if device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        logger.info(f"Target device: CUDA GPU ({gpu_name}, {vram_gb} GB VRAM)")
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        vram_gb = 0.0
        logger.warning("CUDA unavailable or not requested. Falling back to CPU.")

    # Load sequence feature splits and dynamic class mapping
    splits, class_names, label_to_idx = load_sequence_dataset(
        manifest_path=manifest_path,
        processed_dir=processed_dir,
        label_map_path=DEFAULT_LABEL_MAP_PATH,
        update_label_map=True,
    )
    idx_to_label = {idx: name for idx, name in enumerate(class_names)}
    no_sign_idx = label_to_idx.get("no_sign", len(class_names) - 1)

    # Create PyTorch datasets
    # Augment training set with 30 variations per sign for robust sequence learning + class balancing
    train_dataset = ISLSequenceDataset(
        splits["train"],
        augment=True,
        augment_factor=30,
        balance_classes=True,
        no_sign_class_idx=no_sign_idx,
        fps=fps,
    )
    val_dataset = ISLSequenceDataset(splits["val"], augment=False, fps=fps)
    test_dataset = ISLSequenceDataset(splits["test"], augment=False, fps=fps)

    logger.info(
        f"Dataset counts: Train={len(train_dataset)} (with augmentations and class balancing), "
        f"Val={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ISLSequenceLSTM(
        input_size=328,
        hidden_size=128,
        num_layers=2,
        num_classes=len(class_names),
        dropout=0.3,
        bidirectional=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"ISLSequenceLSTM initialized on {device}. Total parameters: {total_params:,} ({trainable_params:,} trainable)")

    # Loss function and optimizer (AdamW + CosineAnnealingLR)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_score = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_weights = None

    history: List[Dict[str, Any]] = []

    logger.info(f"--- Starting Training on {device} ({epochs} epochs, batch_size={batch_size}, lr={learning_rate}) ---")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()
        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_preds_list: List[int] = []
        val_trues_list: List[int] = []

        with torch.no_grad():
            for val_x, val_y, _ in val_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)

                val_logits = model(val_x)
                v_loss = criterion(val_logits, val_y)
                running_val_loss += v_loss.item() * val_x.size(0)
                v_preds = torch.argmax(val_logits, dim=1)
                val_preds_list.extend(v_preds.cpu().numpy().tolist())
                val_trues_list.extend(val_y.cpu().numpy().tolist())

        epoch_val_loss = running_val_loss / len(val_trues_list)
        val_metrics = compute_metrics(val_trues_list, val_preds_list, class_names)
        epoch_val_acc = val_metrics["accuracy"]
        epoch_val_macro_f1 = val_metrics["macro_f1"]
        epoch_duration = time.time() - epoch_start

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4),
            "val_macro_f1": round(epoch_val_macro_f1, 4),
            "duration_sec": round(epoch_duration, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
        })

        # Save checkpoint based on highest validation accuracy and lowest validation loss
        is_best = False
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            is_best = True

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or is_best:
            best_flag = " [★ BEST]" if is_best else ""
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.1f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.1f}% | "
                f"Val F1: {epoch_val_macro_f1:.3f}{best_flag}"
            )

    total_training_duration = time.time() - start_time
    logger.info(f"Training completed in {total_training_duration:.2f}s. Best Epoch: {best_epoch} with Val Acc: {best_val_acc*100:.1f}%")

    # Load best checkpoint for test evaluation
    assert best_weights is not None
    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()

    # Held-out Test Evaluation
    test_y_true: List[int] = []
    test_y_pred: List[int] = []

    with torch.no_grad():
        for test_x, test_y, _ in test_loader:
            test_x = test_x.to(device)
            logits = model(test_x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_y_pred.extend(preds)
            test_y_true.extend(test_y.numpy())

    test_metrics = compute_metrics(
        np.array(test_y_true, dtype=int),
        np.array(test_y_pred, dtype=int),
        class_names,
    )

    logger.info(f"=== Held-Out Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['macro_f1']:.4f} ===")
    for c_name, c_res in test_metrics["per_class"].items():
        logger.info(
            f"  - {c_name:14s}: Precision={c_res['precision']:.3f}, Recall={c_res['recall']:.3f}, "
            f"F1={c_res['f1']:.3f} (support={c_res['support']})"
        )

    # Save checkpoint
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_weights,
        "model_architecture": {
            "class_name": "ISLSequenceLSTM",
            "input_size": 328,
            "hidden_size": 128,
            "num_layers": 2,
            "num_classes": len(class_names),
            "dropout": 0.3,
            "bidirectional": True,
            "has_attention": True,
        },
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "total_training_duration_sec": round(total_training_duration, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_trained_on": str(device),
        "gpu_name": gpu_name,
    }
    torch.save(checkpoint, model_save_path)
    model_size_mb = round(os.path.getsize(model_save_path) / (1024**2), 2)
    logger.info(f"Saved PyTorch model checkpoint to {model_save_path} ({model_size_mb} MB)")

    # Save evaluation report
    report_save_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "task": "PyTorch ISL Sequence LSTM Training and Evaluation",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device_info": {
            "device": str(device),
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
        },
        "model": {
            "class_name": "ISLSequenceLSTM",
            "input_size": 328,
            "sequence_length": 30,
            "hidden_size": 128,
            "num_layers": 2,
            "bidirectional": True,
            "has_attention": True,
            "num_classes": len(class_names),
            "dropout": 0.3,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "checkpoint_path": str(model_save_path),
            "checkpoint_size_mb": model_size_mb,
        },
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "loss_function": "CrossEntropyLoss",
            "scheduler": "CosineAnnealingLR",
        },
        "dataset": {
            "manifest_path": str(manifest_path),
            "classes": class_names,
            "num_classes": len(class_names),
            "train_samples_raw": len(splits["train"]),
            "train_samples_augmented": len(train_dataset),
            "val_samples": len(val_dataset),
            "test_samples": len(test_dataset),
        },
        "training_summary": {
            "best_epoch": best_epoch,
            "best_val_accuracy": round(best_val_acc, 4),
            "best_val_loss": round(best_val_loss, 4),
            "total_training_duration_sec": round(total_training_duration, 2),
            "final_train_loss": round(history[-1]["train_loss"], 4),
        },
        "test_evaluation": test_metrics,
        "epoch_trajectory": history,
    }

    with open(report_save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation report to {report_save_path}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PyTorch ISL Sequence LSTM on GPU")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Target device")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH), help="Path to dataset manifest")
    parser.add_argument("--model-out", type=str, default=str(DEFAULT_MODEL_SAVE_PATH), help="Path to save model checkpoint")
    parser.add_argument("--report-out", type=str, default=str(DEFAULT_REPORT_SAVE_PATH), help="Path to save evaluation report")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate for time-normalized velocity computation")
    args = parser.parse_args()

    train_sequence_lstm(
        manifest_path=Path(args.manifest),
        model_save_path=Path(args.model_out),
        report_save_path=Path(args.report_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
        seed=args.seed,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
