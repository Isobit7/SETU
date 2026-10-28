"""
SETU — Automated Retraining & Fine-Tuning CLI from Captured Sessions
Scans data/captured_sessions/ and data/processed/ for new user-confirmed samples,
validates feature geometry, applies temporal speed warping and jitter augmentation,
and fine-tunes the BiLSTM classifier weights.
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
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.features.extractor import compute_velocity_features
from training.features.schema import (
    FEATURE_DIM,
    EXPANDED_FEATURE_DIM,
    MULTIMODAL_FEATURE_DIM,
    VELOCITY_FEATURE_DIM_328,
    VELOCITY_FEATURE_DIM_352,
)
from training.train_isl_phrase_bilstm import (
    ISLSequenceLSTM,
    ISLSequenceDataset,
    temporal_resample_np,
    augment_sequence,
    compute_metrics,
    BASE_CLASSES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SETU.FineTune")


def scan_captured_sessions(
    sessions_root: Path = ROOT_DIR / "data" / "captured_sessions",
    processed_dir: Path = ROOT_DIR / "data" / "processed",
) -> List[Tuple[np.ndarray, str, str]]:
    """
    Gathers confirmed samples from captured sessions and processed npz files.
    Returns list of (features_array_30xDim, label_name, sample_id).
    """
    samples: List[Tuple[np.ndarray, str, str]] = []

    # 1. Scan captured_sessions directory
    if sessions_root.exists():
        for meta_file in sorted(sessions_root.glob("**/metadata.json")):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                label = meta.get("user_confirmed_label") or meta.get("label", "")
                if not label or label in ("no_sign", "unknown", "awaiting_sign"):
                    continue

                npz_file = meta_file.parent / "features.npz"
                if not npz_file.exists():
                    continue

                with np.load(npz_file) as d:
                    feat = d["features"].astype(np.float32)

                if feat.ndim == 2 and feat.shape[0] >= 10:
                    feat_30 = temporal_resample_np(feat, target_len=30)
                    sample_id = meta.get("sample_id", meta_file.parent.name)
                    samples.append((feat_30, label.lower().strip(), sample_id))
            except Exception as e:
                logger.warning("Could not read capture session %s: %s", meta_file, e)

    # 2. Scan custom_*.npz files in processed_dir
    if processed_dir.exists():
        for npz_file in sorted(processed_dir.glob("custom_*.npz")):
            try:
                with np.load(npz_file) as d:
                    feat = d["features"].astype(np.float32)
                    lbl = str(d.get("label", "")).strip()

                if not lbl:
                    parts = npz_file.stem.split("_")
                    if len(parts) >= 2:
                        lbl = parts[1]

                if lbl and feat.ndim == 2 and feat.shape[0] >= 10:
                    feat_30 = temporal_resample_np(feat, target_len=30)
                    samples.append((feat_30, lbl.lower().strip(), npz_file.stem))
            except Exception as e:
                logger.warning("Could not read npz %s: %s", npz_file, e)

    logger.info("Found %d total confirmed samples from captures & processed archives.", len(samples))
    return samples


def fine_tune_bilstm(
    epochs: int = 25,
    learning_rate: float = 0.0003,
    batch_size: int = 8,
    weights_path: Path = ROOT_DIR / "models" / "weights" / "isl_sequence_lstm.pth",
    device_name: str = "cuda",
) -> Dict[str, Any]:
    """Fine-tunes the dynamic BiLSTM sequence classifier on captured samples."""
    device = torch.device(device_name if (device_name == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info("Running fine-tuning on %s", device)

    if not weights_path.exists():
        fallback_path = ROOT_DIR / "models" / "isl_sequence_lstm.pth"
        if fallback_path.exists():
            weights_path = fallback_path
        else:
            raise FileNotFoundError(f"Model weights not found at {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location=device, weights_only=False)
    existing_classes: List[str] = checkpoint.get("class_names", list(BASE_CLASSES))
    arch = checkpoint.get("model_architecture", {})
    input_size = arch.get("input_size", 328)

    captured_samples = scan_captured_sessions()
    if not captured_samples:
        logger.warning("No captured samples found to fine-tune on. Exiting.")
        return {"status": "no_samples", "trained": False}

    # Discover any new classes
    all_classes_set = set(existing_classes)
    for _, lbl, _ in captured_samples:
        all_classes_set.add(lbl)

    updated_classes = [c for c in existing_classes if c != "no_sign"]
    new_classes = sorted([c for c in all_classes_set if c not in existing_classes])
    updated_classes.extend(new_classes)
    if "no_sign" not in updated_classes:
        updated_classes.append("no_sign")

    label_to_idx = {c: i for i, c in enumerate(updated_classes)}
    num_classes = len(updated_classes)

    # Initialize model
    model = ISLSequenceLSTM(
        input_size=input_size,
        hidden_size=arch.get("hidden_size", 128),
        num_layers=arch.get("num_layers", 2),
        num_classes=num_classes,
        dropout=0.2,
        bidirectional=arch.get("bidirectional", True),
    ).to(device)

    # If class count matches, load full state_dict; if classes grew, expand final linear layer
    state_dict = checkpoint["model_state_dict"]
    if len(existing_classes) == num_classes:
        model.load_state_dict(state_dict)
    else:
        logger.info("Vocabulary expanded from %d to %d classes. Adapting classifier head.", len(existing_classes), num_classes)
        # Load all layers except final fc layer
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.6")}
        model_dict.update(pretrained_dict)
        # Copy existing weights for original classes into new fc.6
        orig_fc_weight = state_dict.get("fc.6.weight")
        orig_fc_bias = state_dict.get("fc.6.bias")
        if orig_fc_weight is not None and orig_fc_bias is not None:
            for old_idx, old_c in enumerate(existing_classes):
                new_idx = label_to_idx[old_c]
                model_dict["fc.6.weight"][new_idx] = orig_fc_weight[old_idx]
                model_dict["fc.6.bias"][new_idx] = orig_fc_bias[old_idx]
        model.load_state_dict(model_dict)

    # Convert samples to dataset entries
    dataset_entries: List[Tuple[np.ndarray, int, str]] = []
    for feat, lbl, sid in captured_samples:
        c_idx = label_to_idx[lbl]
        dataset_entries.append((feat, c_idx, sid))

    train_dataset = ISLSequenceDataset(
        dataset_entries,
        augment=True,
        augment_factor=15,
        balance_classes=True,
        no_sign_class_idx=label_to_idx.get("no_sign"),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    model.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for bx, by, _ in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)

        ep_loss = running_loss / max(total, 1)
        ep_acc = correct / max(total, 1)
        if ep % 5 == 0 or ep == epochs:
            logger.info("Fine-tune Epoch %d/%d - Loss: %.4f | Acc: %.2f%%", ep, epochs, ep_loss, ep_acc * 100)

    # Save updated checkpoint
    updated_checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_architecture": {
            "input_size": input_size,
            "hidden_size": arch.get("hidden_size", 128),
            "num_layers": arch.get("num_layers", 2),
            "num_classes": num_classes,
            "dropout": 0.2,
            "bidirectional": arch.get("bidirectional", True),
        },
        "class_names": updated_classes,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fine_tune_samples": len(captured_samples),
    }

    backup_path = weights_path.with_suffix(".pth.bak")
    if weights_path.exists():
        import shutil
        shutil.copy2(weights_path, backup_path)

    torch.save(updated_checkpoint, str(weights_path))
    logger.info("Fine-tuned weights successfully saved to %s (backup at %s)", weights_path, backup_path)

    return {
        "status": "success",
        "trained": True,
        "classes": updated_classes,
        "samples_count": len(captured_samples),
        "final_loss": round(ep_loss, 4),
        "final_acc": round(ep_acc, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SETU Fine-Tune from Captured Sessions")
    parser.add_argument("--epochs", type=int, default=25, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    args = parser.parse_args()

    res = fine_tune_bilstm(epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size)
    print(json.dumps(res, indent=2))

