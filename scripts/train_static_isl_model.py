"""
SETU — High-Accuracy Indian Sign Language (ISL) Static Gesture Training Pipeline
Trains on authentic IEEE / Mendeley ISL static gesture dataset:
"Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals.zip"

Capabilities:
1. Multi-Signer & Multi-Demographic: Kids, Teenagers, Adults in Half and Full Sleeves.
2. Dual-Hand 126-D MediaPipe Feature Extraction with ambidextrous symmetry.
3. Stratified 80% Train / 10% Validation / 10% Held-Out Test Split.
4. PyTorch RealSignMLP training on CUDA GPU with Cosine Annealing and Label Smoothing.
5. Rigorous test evaluation: Top-1, Top-3, Top-5 Accuracy, Macro-F1, and Confusion Analysis.
6. Automated ONNX export for browser edge execution (synced to frontend/public/models).
7. Comprehensive evaluation report JSON export.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from training.features.geometry import extract_dual_hand_126d
from training.train_isl_dualhand_mlp import RealSignMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setu.static_isl_trainer")

DEFAULT_ZIP_PATH = ROOT_DIR / "data" / "raw" / "Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals.zip"
DEFAULT_TASK_PATH = ROOT_DIR / "models" / "tasks" / "hand_landmarker.task"

CLASS_NAMES_ALPHABET = [chr(ord("A") + i) for i in range(26)]
CLASS_NAMES_NUMERALS = [str(i) for i in range(10)]
CLASS_NAMES_HINDI_VOWELS = ["अ", "अं", "आ", "इ", "ई", "ऋ", "ए", "ऐ"]


# ---------------------------------------------------------------------------
# Multiprocessing Worker for MediaPipe Landmark Extraction
# ---------------------------------------------------------------------------

_worker_detector: Optional[vision.HandLandmarker] = None
_worker_zip: Optional[zipfile.ZipFile] = None


def init_extract_worker(model_path: str, zip_path: str) -> None:
    """Initializes per-worker MediaPipe HandLandmarker and open ZipFile handle."""
    global _worker_detector, _worker_zip
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.25,
    )
    _worker_detector = vision.HandLandmarker.create_from_options(options)
    _worker_zip = zipfile.ZipFile(zip_path, "r")


def extract_single_image(item: Tuple[str, int, str]) -> Optional[Tuple[np.ndarray, int, str]]:
    """
    Extracts 126-D dual-hand normalized landmarks from an image path inside the ZIP.
    item: (archive_path, label_idx, class_name)
    Returns: (features_126d, label_idx, class_name) or None if no hands detected.
    """
    global _worker_detector, _worker_zip
    path, label_idx, class_name = item
    try:
        raw_bytes = _worker_zip.read(path)
        img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        res = _worker_detector.detect(mp_img)

        if not res.hand_landmarks or len(res.hand_landmarks) == 0:
            return None

        lh_pts = None
        rh_pts = None

        if res.handedness:
            for idx, handedness_list in enumerate(res.handedness):
                label = handedness_list[0].category_name  # "Left" or "Right"
                lms = res.hand_landmarks[idx]
                pts = [[pt.x, pt.y, pt.z] for pt in lms]
                if label == "Left":
                    lh_pts = pts
                elif label == "Right":
                    rh_pts = pts
        else:
            # Fallback if handedness metadata missing
            lh_pts = [[pt.x, pt.y, pt.z] for pt in res.hand_landmarks[0]]
            if len(res.hand_landmarks) > 1:
                rh_pts = [[pt.x, pt.y, pt.z] for pt in res.hand_landmarks[1]]

        features_126d = extract_dual_hand_126d(lh_pts, rh_pts)
        if np.count_nonzero(features_126d) == 0:
            return None

        return features_126d, label_idx, class_name

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset Collector & Stratified Sampler
# ---------------------------------------------------------------------------

def collect_image_items(
    zip_path: Path,
    mode: str = "alphabet",
    max_samples_per_class: int = 400,
    random_seed: int = 42,
) -> Tuple[List[Tuple[str, int, str]], List[str]]:
    """
    Scans the ZIP archive and collects a balanced, stratified selection of images
    across demographics (Kids, Teenagers, Adults) and clothing (Half/Full sleeves).
    """
    random.seed(random_seed)
    items_by_class: Dict[int, List[str]] = defaultdict(list)

    if mode == "alphabet":
        classes = CLASS_NAMES_ALPHABET
        target_category = "English Alphabet"
    elif mode == "numerals":
        classes = CLASS_NAMES_NUMERALS
        target_category = "Numerals"
    elif mode == "hindi_vowels":
        classes = CLASS_NAMES_HINDI_VOWELS
        target_category = "Hindi Vowels"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    class_to_idx = {name: i for i, name in enumerate(classes)}

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            parts = name.split("/")
            if len(parts) >= 6 and parts[4] == target_category:
                raw_cls = parts[5]
                # Normalize class mapping
                if target_category == "English Alphabet":
                    if raw_cls in ("E1", "E2"):
                        cls_name = "E"
                    elif raw_cls in class_to_idx:
                        cls_name = raw_cls
                    else:
                        continue
                elif target_category == "Numerals":
                    if raw_cls in ("9a", "9b"):
                        cls_name = "9"
                    elif raw_cls in class_to_idx:
                        cls_name = raw_cls
                    else:
                        continue
                elif target_category == "Hindi Vowels":
                    if raw_cls in class_to_idx:
                        cls_name = raw_cls
                    else:
                        continue
                else:
                    continue

                idx = class_to_idx[cls_name]
                items_by_class[idx].append(name)

    logger.info("Found total items for category '%s':", target_category)
    final_items: List[Tuple[str, int, str]] = []

    for idx, name in enumerate(classes):
        available = items_by_class[idx]
        random.shuffle(available)
        selected = available[:max_samples_per_class]
        for p in selected:
            final_items.append((p, idx, name))
        logger.info("  Class %2d [%s]: %d available, %d selected", idx, name, len(available), len(selected))

    random.shuffle(final_items)
    logger.info("Total selected items across all %d classes: %d", len(classes), len(final_items))
    return final_items, classes


# ---------------------------------------------------------------------------
# Feature Extraction Runner
# ---------------------------------------------------------------------------

def extract_features(
    zip_path: Path,
    items: List[Tuple[str, int, str]],
    num_workers: int = 8,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Runs multiprocessing landmark extraction across all selected images."""
    model_path = str(DEFAULT_TASK_PATH)
    logger.info("Starting multi-process MediaPipe feature extraction with %d workers...", num_workers)
    t0 = time.time()

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []
    classes_list: List[str] = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_extract_worker,
        initargs=(model_path, str(zip_path)),
    ) as executor:
        for res in executor.map(extract_single_image, items, chunksize=16):
            if res is not None:
                feat, lbl, cname = res
                features_list.append(feat)
                labels_list.append(lbl)
                classes_list.append(cname)

    elapsed = time.time() - t0
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int64)

    logger.info(
        "Feature extraction completed in %.2f s (%.1f ms/img). Yielded %d valid 126-D samples (%.1f%% detection yield).",
        elapsed, (elapsed / max(1, len(items))) * 1000, len(X), (len(X) / max(1, len(items))) * 100
    )
    return X, y, classes_list


# ---------------------------------------------------------------------------
# Stratified 80 / 10 / 10 Dataset Splitter
# ---------------------------------------------------------------------------

class StaticISLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def split_80_10_10(
    X: np.ndarray,
    y: np.ndarray,
    random_seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Splits dataset into 80% Train, 10% Validation, 10% Held-Out Test. Returns splits mask array."""
    np.random.seed(random_seed)
    num_classes = len(np.unique(y))

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for c in range(num_classes):
        c_idx = np.where(y == c)[0]
        np.random.shuffle(c_idx)
        n = len(c_idx)
        n_train = int(round(0.80 * n))
        n_val = int(round(0.10 * n))

        train_indices.extend(c_idx[:n_train])
        val_indices.extend(c_idx[n_train : n_train + n_val])
        test_indices.extend(c_idx[n_train + n_val :])

    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)
    test_idx = np.array(test_indices)

    splits_array = np.empty(len(X), dtype=object)
    splits_array[train_idx] = "train"
    splits_array[val_idx] = "val"
    splits_array[test_idx] = "test"

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    logger.info(
        "Stratified 80/10/10 Split: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Test=%d (%.1f%%)",
        len(train_idx), (len(train_idx) / len(X)) * 100,
        len(val_idx), (len(val_idx) / len(X)) * 100,
        len(test_idx), (len(test_idx) / len(X)) * 100,
    )
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X[test_idx], y[test_idx]), splits_array


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_static_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    classes: List[str],
    epochs: int = 45,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Trains RealSignMLP with Cosine Annealing on CUDA/CPU."""
    X_train, y_train = train_data
    X_val, y_val = val_data
    num_classes = len(classes)

    train_ds = StaticISLDataset(X_train, y_train)
    val_ds = StaticISLDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = RealSignMLP(
        input_dim=126,
        hidden1=128,
        hidden2=64,
        num_classes=num_classes,
        dropout=0.2,
        use_batchnorm=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights: Optional[Dict[str, Any]] = None
    best_epoch = 0

    logger.info("Training %s on %s for %d epochs...", model.__class__.__name__, device, epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            total_train += len(by)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_top3_correct = 0
        total_val = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * len(by)

                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()

                if num_classes >= 3:
                    top3 = logits.topk(min(3, num_classes), dim=1).indices
                    for i in range(len(by)):
                        if by[i] in top3[i]:
                            val_top3_correct += 1

                total_val += len(by)

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val
        val_top3_acc = val_top3_correct / total_val if num_classes >= 3 else val_acc

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_val_acc:
            logger.info(
                "Epoch %02d/%d | Train Loss: %.4f | Train Acc: %.2f%% | Val Acc: %.2f%% | Top-3: %.2f%%",
                epoch, epochs, train_loss / total_train, train_acc * 100, val_acc * 100, val_top3_acc * 100,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)
        logger.info("Best validation checkpoint restored from Epoch %d (Val Acc: %.2f%%)", best_epoch, best_val_acc * 100)

    train_stats = {
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 4),
        "final_train_acc": round(train_acc, 4),
    }
    return model, train_stats


# ---------------------------------------------------------------------------
# Held-Out Test Evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model: nn.Module,
    test_data: Tuple[np.ndarray, np.ndarray],
    classes: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Computes Top-1, Top-3, Top-5, Macro-F1, Precision, and Recall on the 10% test split."""
    X_test, y_test = test_data
    num_classes = len(classes)
    model.eval()
    model.to(device)

    bx = torch.from_numpy(X_test).to(device)
    by = torch.from_numpy(y_test)

    with torch.no_grad():
        logits = model(bx).cpu()
        probs = torch.softmax(logits, dim=1)

    preds = logits.argmax(dim=1).numpy()
    targets = by.numpy()

    # Top-1 Accuracy
    top1_correct = (preds == targets).sum()
    top1_acc = float(top1_correct / len(targets))

    # Top-3 Accuracy
    top3_acc = top1_acc
    if num_classes >= 3:
        top3_indices = logits.topk(min(3, num_classes), dim=1).indices.numpy()
        top3_matches = sum(1 for i in range(len(targets)) if targets[i] in top3_indices[i])
        top3_acc = float(top3_matches / len(targets))

    # Top-5 Accuracy
    top5_acc = top3_acc
    if num_classes >= 5:
        top5_indices = logits.topk(min(5, num_classes), dim=1).indices.numpy()
        top5_matches = sum(1 for i in range(len(targets)) if targets[i] in top5_indices[i])
        top5_acc = float(top5_matches / len(targets))

    # Confusion matrix & Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for c in range(num_classes):
        cname = classes[c]
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class_metrics[cname] = {
            "samples": int((targets == c).sum()),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    # Top Confusions
    confusions: List[Dict[str, Any]] = []
    for i in range(len(targets)):
        if preds[i] != targets[i]:
            confusions.append({
                "true": classes[targets[i]],
                "pred": classes[preds[i]],
            })

    confusion_counts = Counter(f"{c['true']} -> {c['pred']}" for c in confusions)

    report = {
        "num_test_samples": len(targets),
        "num_classes": num_classes,
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "top_confusion_pairs": confusion_counts.most_common(5),
        "per_class": per_class_metrics,
    }

    logger.info("=== Held-Out 10%% Test Set Evaluation ===")
    logger.info("Top-1 Accuracy:    %.2f%%", top1_acc * 100)
    logger.info("Top-3 Accuracy:    %.2f%%", top3_acc * 100)
    logger.info("Top-5 Accuracy:    %.2f%%", top5_acc * 100)
    logger.info("Macro F1-Score:    %.4f", macro_f1)
    logger.info("Macro Precision:   %.4f", macro_precision)
    logger.info("Macro Recall:      %.4f", macro_recall)
    if confusion_counts:
        logger.info("Top Confusions:    %s", confusion_counts.most_common(3))

    return report


# ---------------------------------------------------------------------------
# ONNX Exporter
# ---------------------------------------------------------------------------

def export_onnx_model(
    model: nn.Module,
    output_path: Path,
    sync_to_frontend: bool = True,
) -> None:
    """Exports PyTorch model to ONNX with dynamic batching."""
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 126, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["features_126d"],
        output_names=["logits"],
        dynamic_axes={"features_126d": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=14,
    )
    logger.info("Exported ONNX model to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)

    if sync_to_frontend:
        fe_dir = ROOT_DIR / "frontend" / "public" / "models"
        if fe_dir.exists():
            shutil.copy2(output_path, fe_dir / output_path.name)
            logger.info("Synced ONNX model to frontend: %s", fe_dir / output_path.name)


# ---------------------------------------------------------------------------
# Master Pipeline Controller
# ---------------------------------------------------------------------------

def run_pipeline(
    zip_path: Path,
    mode: str = "alphabet",
    max_samples_per_class: int = 400,
    epochs: int = 45,
    batch_size: int = 64,
    num_workers: int = 8,
) -> Dict[str, Any]:
    """Executes the full end-to-end extraction, split, training, evaluation, and export."""
    logger.info("==================================================================")
    logger.info("Running SETU Static ISL High-Accuracy Pipeline [Mode: %s]", mode.upper())
    logger.info("==================================================================")

    # 1. Collect Items
    items, classes = collect_image_items(
        zip_path=zip_path,
        mode=mode,
        max_samples_per_class=max_samples_per_class,
    )

    # 2. Extract 126-D Landmarks
    X, y, _ = extract_features(zip_path, items, num_workers=num_workers)

    # 3. Stratified 80 / 10 / 10 Split
    train_data, val_data, test_data, splits_array = split_80_10_10(X, y, random_seed=42)

    # Cache extracted features with split annotations
    cache_file = ROOT_DIR / "data" / "processed" / f"static_isl_{mode}_126d.npz"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, y=y, splits=splits_array, classes=classes)
    logger.info("Saved processed dataset cache to %s (%.2f MB)", cache_file, cache_file.stat().st_size / (1024 * 1024))

    # 4. Train Model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, train_stats = train_static_model(
        train_data=train_data,
        val_data=val_data,
        classes=classes,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    # 5. Evaluate on 10% Held-Out Test Set
    eval_report = evaluate_test_set(model, test_data, classes, device="cpu")

    # 6. Checkpoint & ONNX Export
    weights_dir = ROOT_DIR / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    if mode == "alphabet":
        pth_main = weights_dir / "realsign_26class_mlp.pth"
        pth_copy = weights_dir / "realsign_multisigner_mlp.pth"
        onnx_file = weights_dir / "realsign_multisigner.onnx"
    else:
        pth_main = weights_dir / f"isl_{mode}_mlp.pth"
        pth_copy = None
        onnx_file = weights_dir / f"isl_{mode}.onnx"

    ckpt_payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": 126,
        "num_classes": len(classes),
        "classes": classes,
        "architecture": {
            "input_dim": 126,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": len(classes),
            "use_batchnorm": True,
            "dropout": 0.2,
        },
        "test_accuracy": eval_report["top1_accuracy"],
        "macro_f1": eval_report["macro_f1"],
        "train_samples": len(train_data[0]),
        "val_samples": len(val_data[0]),
        "test_samples": len(test_data[0]),
        "mode": mode,
    }

    torch.save(ckpt_payload, pth_main)
    logger.info("Saved PyTorch checkpoint to %s", pth_main)
    if pth_copy is not None:
        torch.save(ckpt_payload, pth_copy)
        logger.info("Saved checkpoint mirror to %s", pth_copy)

    export_onnx_model(model, onnx_file, sync_to_frontend=True)

    # 7. Write Comprehensive JSON Report
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"static_isl_{mode}_evaluation_report.json"

    full_report = {
        "dataset": "Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals",
        "mode": mode,
        "num_classes": len(classes),
        "classes": classes,
        "split": {
            "train": len(train_data[0]),
            "val": len(val_data[0]),
            "test": len(test_data[0]),
            "train_ratio": 0.80,
            "val_ratio": 0.10,
            "test_ratio": 0.10,
        },
        "training": train_stats,
        "test_evaluation": eval_report,
        "weights_path": str(pth_main),
        "onnx_path": str(onnx_file),
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    logger.info("Wrote full evaluation report to %s", report_file)

    return full_report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="SETU Static ISL High-Accuracy Training Pipeline")
    parser.add_argument("--zip-path", type=str, default=str(DEFAULT_ZIP_PATH), help="Path to raw dataset ZIP")
    parser.add_argument("--mode", type=str, default="alphabet", choices=["alphabet", "numerals", "hindi_vowels", "all"])
    parser.add_argument("--samples-per-class", type=int, default=400, help="Max image samples extracted per class")
    parser.add_argument("--epochs", type=int, default=45, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes for landmark extraction")
    args = parser.parse_args()

    zip_file = Path(args.zip_path)
    if not zip_file.exists():
        logger.error("Dataset ZIP file not found at: %s", zip_file)
        sys.exit(1)

    modes = ["alphabet", "numerals", "hindi_vowels"] if args.mode == "all" else [args.mode]

    for m in modes:
        run_pipeline(
            zip_path=zip_file,
            mode=m,
            max_samples_per_class=args.samples_per_class,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )


if __name__ == "__main__":
    main()

