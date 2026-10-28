"""
SETU — Phase 3.3: RealSign 26-Class Alphabet PyTorch MLP Classifier

Trains and evaluates a 26-class PyTorch MLP on clean MediaPipe hand landmarks:
- Architecture: Input(63) -> Linear(128) -> ReLU -> Dropout(0.2) -> Linear(64) -> ReLU -> Linear(26)
- Optimization: CrossEntropyLoss, Adam(lr=0.001), 30 epochs
- Data Splits: Clean Training, Validation, and held-out Testing splits from 'realsign_landmarks_26class.npz'
- Metrics: Overall Accuracy, Macro F1, Per-class Precision/Recall/F1, Top Confusion Pairs
- Outputs:
    * Model checkpoint: 'models/realsign_26class_mlp.pth'
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

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.feature_extractor import normalize_wrist_coordinates
from training.feature_extractor import extract_realsign_hand_features
from training.features.geometry import expand_63d_to_89d, ACTIVE_COORDINATE_INDICES_57

DEFAULT_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "realsign_landmarks_26class.npz"
DEFAULT_MODEL_SAVE_PATH = ROOT_DIR / "models" / "weights" / "realsign_26class_mlp.pth"
DEFAULT_REPORT_SAVE_PATH = ROOT_DIR / "reports" / "realsign_26class_evaluation_report.json"


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


def load_and_validate_dataset(
    features_path: Path,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    """
    Loads RealSign dataset from NPZ, ensuring X, y, splits, and classes exist and are valid.
    Auto-generates stratified splits or classes if missing, empty, or mismatched.
    Returns (X_raw, y, splits, classes, extra_metadata).
    """
    if not features_path.exists():
        raise FileNotFoundError(f"Extracted landmark features not found: {features_path}")

    print(f"Loading features from: {features_path}...")
    with np.load(features_path, allow_pickle=True) as npz_data:
        if "X" not in npz_data or "y" not in npz_data:
            raise KeyError(f"Expected keys 'X' and 'y' in {features_path}, found {npz_data.files}")

        X_raw = npz_data["X"]
        y = npz_data["y"]
        n_samples = len(y)

        # Classes validation & auto-generation
        if "classes" in npz_data and len(npz_data["classes"]) == 26:
            classes = [str(c) for c in npz_data["classes"]]
        else:
            print("Auto-generating classes (A-Z, 26 classes)...")
            classes = [chr(ord("A") + i) for i in range(26)]

        # Extra metadata preservation (e.g. stats_json)
        extra_meta = {
            k: npz_data[k]
            for k in npz_data.files
            if k not in ("X", "y", "splits", "classes")
        }

        # Splits validation & auto-generation
        splits = None
        if "splits" in npz_data and len(npz_data["splits"]) == n_samples:
            raw_splits = npz_data["splits"]
            cleaned = np.array([
                (s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, bytearray)) else str(s)).strip().lower()
                for s in raw_splits
            ], dtype=str)

            unique_splits = set(cleaned)
            has_train = bool(unique_splits & {"train", "training"})
            has_val = bool(unique_splits & {"val", "validation", "valid"})
            has_test = bool(unique_splits & {"test", "testing"})

            if has_train and has_val and has_test:
                splits = cleaned

        if splits is None:
            print(f"Splits missing, invalid, or mismatched for {n_samples:,} samples. Generating stratified 80/10/10 splits...")
            splits = generate_stratified_splits(y, seed=seed)

    return X_raw, y, splits, classes, extra_meta


def save_dataset_archive(
    save_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    classes: List[str],
    extra_meta: Optional[Dict[str, Any]] = None,
    sync_to_alt: bool = True,
) -> None:
    """
    Saves the dataset archive retaining X, y, splits, classes, and any extra metadata.
    Also synchronizes to the companion dataset path if requested.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_splits = np.array(splits, dtype=str)
    save_classes = np.array(classes, dtype=str)
    extra_kwargs = {
        k: v for k, v in (extra_meta or {}).items()
        if k not in ("X", "y", "splits", "classes")
    }

    np.savez_compressed(
        save_path,
        X=X,
        y=y,
        splits=save_splits,
        classes=save_classes,
        **extra_kwargs,
    )
    print(f"Saved transformed invariant dataset to: {save_path}")

    if sync_to_alt:
        alt_path = (
            ROOT_DIR / "data" / "processed" / "realsign_landmarks_26class.npz"
            if save_path.resolve() == (ROOT_DIR / "data" / "processed" / "realsign" / "realsign_landmarks_26class.npz").resolve()
            else ROOT_DIR / "data" / "processed" / "realsign" / "realsign_landmarks_26class.npz"
        )
        try:
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                alt_path,
                X=X,
                y=y,
                splits=save_splits,
                classes=save_classes,
                **extra_kwargs,
            )
            print(f"Synchronized transformed invariant dataset to: {alt_path}")
        except Exception as e:
            print(f"Note: Could not sync to companion path {alt_path}: {e}")


def transform_landmarks_to_canonical_invariants(X: np.ndarray) -> np.ndarray:
    """
    Transforms raw or wrist-centered 63-D hand landmark coordinates into the strictly
    canonical 3D rotation-aligned, palm-scale normalized invariant feature representation
    matching extract_realsign_hand_features and normalize_wrist_coordinates.

    Canonical landmarks:
    - Origin at wrist: pts[0] = (0, 0, 0)
    - Palm reference length = 1.0: ||pts[9] - pts[0]|| = 1.0
    - Middle MCP (index 9) aligned strictly with -Y axis: pts[9] = (0, -1, 0)
    """
    num_samples = len(X)
    X_canonical = np.empty_like(X, dtype=np.float32)
    print(f"Re-extracting 63-D rotation-invariant features across {num_samples:,} samples...")
    t0 = time.time()
    for i in range(num_samples):
        pts_21x3 = X[i].reshape(21, 3)
        feat = extract_realsign_hand_features(pts_21x3)
        if feat is not None and feat.shape == (63,):
            X_canonical[i] = feat
        else:
            norm_pts, _ = normalize_wrist_coordinates(pts_21x3)
            X_canonical[i] = norm_pts.ravel().astype(np.float32)
    t1 = time.time()
    print(f"Invariant feature transformation completed in {t1 - t0:.2f}s.")
    return X_canonical



class RealSignDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for RealSign 89-D invariant landmark feature vectors with coordinate jitter augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, noise_std: float = 0.008):
        self.X = torch.from_numpy(np.array(X, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y, dtype=np.int64))
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        if self.augment:
            # Add small Gaussian noise to the active 57 coordinate features
            noise = torch.randn_like(x[:57]) * self.noise_std
            x[:57] = x[:57] + noise
        return x, self.y[idx]


class RealSignMLP(nn.Module):
    """PyTorch MLP Classifier for 26-class ISL Hand Alphabet Recognition."""

    def __init__(
        self,
        input_dim: int = 89,
        hidden1: int = 128,
        hidden2: int = 64,
        num_classes: int = 26,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ):
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


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Dict[str, Any]:
    """Computes overall accuracy, per-class precision, recall, F1, and confusion pairs."""
    num_classes = len(classes)
    total_samples = len(y_true)
    correct_predictions = int(np.sum(y_true == y_pred))
    overall_accuracy = float(correct_predictions / total_samples) if total_samples > 0 else 0.0

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    f1_list: List[float] = []

    for c_idx, c_name in enumerate(classes):
        tp = int(np.sum((y_true == c_idx) & (y_pred == c_idx)))
        fp = int(np.sum((y_true != c_idx) & (y_pred == c_idx)))
        fn = int(np.sum((y_true == c_idx) & (y_pred != c_idx)))
        support = int(np.sum(y_true == c_idx))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[c_name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # Top confusion pairs (where true != pred)
    confusion_pairs_counter = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            confusion_pairs_counter[(classes[t], classes[p])] += 1

    top_confusion_pairs = [
        {"true_class": pair[0], "pred_class": pair[1], "count": count}
        for pair, count in confusion_pairs_counter.most_common(15)
    ]

    # Rank worst 3 performing letters by F1-score
    sorted_by_f1 = sorted(
        per_class_metrics.items(),
        key=lambda item: (item[1]["f1_score"], item[1]["recall"]),
    )
    worst_3_letters = [
        {"class": item[0], "f1_score": item[1]["f1_score"], "recall": item[1]["recall"], "support": item[1]["support"]}
        for item in sorted_by_f1[:3]
    ]

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "overall_accuracy_pct": round(overall_accuracy * 100.0, 2),
        "macro_f1": round(macro_f1, 4),
        "macro_f1_pct": round(macro_f1 * 100.0, 2),
        "total_evaluated_samples": total_samples,
        "correct_predictions": correct_predictions,
        "worst_3_letters": worst_3_letters,
        "per_class": per_class_metrics,
        "top_confusion_pairs": top_confusion_pairs,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates model over DataLoader, returning loss, accuracy, and true/pred vectors."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)

            all_y_true.append(batch_y.cpu().numpy())
            all_y_pred.append(preds.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    y_true = np.concatenate(all_y_true) if all_y_true else np.array([])
    y_pred = np.concatenate(all_y_pred) if all_y_pred else np.array([])
    return avg_loss, acc, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    scheduler: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs training loop for specified epochs and tracks epoch history."""
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None

    t0 = time.time()
    print(f"\nBeginning training for {epochs} epochs on device: {device}...")
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Time':<6}")
    print("-" * 68)

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        if scheduler is not None:
            scheduler.step()

        epoch_train_loss = train_loss / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_t0

        history.append({
            "epoch": epoch,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "time_sec": round(epoch_time, 2),
        })

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:<7} | {epoch_train_loss:<12.4f} | {epoch_train_acc*100:>8.2f}% | "
                f"{val_loss:<10.4f} | {val_acc*100:>8.2f}% | {epoch_time:>5.1f}s"
            )

    total_training_time = time.time() - t0
    print("-" * 68)
    print(f"Training completed in {total_training_time:.2f}s.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Loss: {best_val_loss:.4f})")

    # Load best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    summary = {
        "epochs": epochs,
        "total_training_time_sec": round(total_training_time, 2),
        "final_train_loss": round(history[-1]["train_loss"], 4),
        "final_train_acc": round(history[-1]["train_acc"], 4),
        "final_val_loss": round(history[-1]["val_loss"], 4),
        "final_val_acc": round(history[-1]["val_acc"], 4),
        "best_val_acc": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "epoch_history": history,
    }
    return summary, best_model_state


def main():
    parser = argparse.ArgumentParser(description="Train 26-class PyTorch MLP on RealSign landmarks")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH, help="Input NPZ features file")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_SAVE_PATH, help="Output .pth path")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_SAVE_PATH, help="Output report JSON path")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-transformed-npz", action="store_true", default=True, help="Save transformed invariant dataset back to NPZ")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 75)
    print("SETU — RealSign 26-Class Alphabet PyTorch MLP Training")
    print("=" * 75)

    if not args.features.exists():
        raise FileNotFoundError(f"Extracted landmark features not found: {args.features}")

    X_raw, y, splits, classes, extra_meta = load_and_validate_dataset(args.features, seed=args.seed)
    stats_json_str = str(extra_meta.get("stats_json", "{}"))
    try:
        stats_meta = json.loads(stats_json_str) if isinstance(stats_json_str, str) else {}
    except Exception:
        stats_meta = {}

    # Expand canonical coordinates to the 89-D invariant geometric representation
    if X_raw.shape[1] == 63:
        print(f"Expanding 63-D coordinates to 89-D invariant geometric features across {len(X_raw):,} samples...")
        X = expand_63d_to_89d(X_raw)
    elif X_raw.shape[1] == 89:
        X = X_raw
    else:
        X_canon_63 = transform_landmarks_to_canonical_invariants(X_raw)
        X = expand_63d_to_89d(X_canon_63)

    if args.save_transformed_npz:
        save_dataset_archive(
            save_path=args.features,
            X=X,
            y=y,
            splits=splits,
            classes=classes,
            extra_meta=extra_meta,
            sync_to_alt=True,
        )

    print(f"Total dataset vectors: {len(X):,}, Feature Dimension: {X.shape[1]}, Classes: {len(classes)}")

    # Robust String Matching for Splits (supports 'train', 'Training', b'train', etc.)
    splits_str = np.array([
        (s.decode("utf-8", errors="ignore") if isinstance(s, (bytes, bytearray)) else str(s)).strip().lower()
        for s in splits
    ], dtype=str)

    train_mask = np.isin(splits_str, ["train", "training"])
    val_mask = np.isin(splits_str, ["val", "validation", "valid"])
    test_mask = np.isin(splits_str, ["test", "testing"])

    if np.sum(train_mask) == 0 or np.sum(val_mask) == 0 or np.sum(test_mask) == 0:
        print("Warning: One or more split masks resolved to 0 samples. Re-generating stratified splits...")
        splits = generate_stratified_splits(y, seed=args.seed)
        splits_str = np.array([s.strip().lower() for s in splits], dtype=str)
        train_mask = np.isin(splits_str, ["train", "training"])
        val_mask = np.isin(splits_str, ["val", "validation", "valid"])
        test_mask = np.isin(splits_str, ["test", "testing"])
        if args.save_transformed_npz:
            save_dataset_archive(
                save_path=args.features,
                X=X,
                y=y,
                splits=splits,
                classes=classes,
                extra_meta=extra_meta,
                sync_to_alt=True,
            )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Training Split:   {len(X_train):,} samples")
    print(f"  Validation Split: {len(X_val):,} samples")
    print(f"  Testing Split:    {len(X_test):,} samples")

    # PyTorch Datasets & Loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_dataset = RealSignDataset(X_train, y_train, augment=True, noise_std=0.008)
    val_dataset = RealSignDataset(X_val, y_val, augment=False)
    test_dataset = RealSignDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Initialize model
    model = RealSignMLP(
        input_dim=89,
        hidden1=128,
        hidden2=64,
        num_classes=26,
        dropout=0.2,
        use_batchnorm=True,
    ).to(device)

    print("\nModel Architecture:")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Run training
    training_summary, best_state = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        scheduler=scheduler,
    )

    # Evaluate on held-out test split
    print("\nEvaluating on Held-Out Test Split...")
    test_loss, test_acc, y_test_true, y_test_pred = evaluate(model, test_loader, criterion, device)
    test_metrics = compute_classification_metrics(y_test_true, y_test_pred, classes)

    print("\n" + "=" * 75)
    print("HELD-OUT TEST SET EVALUATION SUMMARY")
    print("=" * 75)
    print(f"Overall Accuracy:  {test_metrics['overall_accuracy_pct']:.2f}% ({test_metrics['correct_predictions']}/{test_metrics['total_evaluated_samples']})")
    print(f"Macro F1-Score:    {test_metrics['macro_f1_pct']:.2f}%")
    print(f"Test Loss:         {test_loss:.4f}")

    print("\nWorst 3 Performing Letters:")
    for w in test_metrics["worst_3_letters"]:
        print(f"  - Letter '{w['class']}': F1 = {w['f1_score']*100:.2f}%, Recall = {w['recall']*100:.2f}% (Support: {w['support']})")

    print("\nTop 5 Confusion Pairs:")
    for p in test_metrics["top_confusion_pairs"][:5]:
        print(f"  - True '{p['true_class']}' predicted as '{p['pred_class']}': {p['count']} occurrences")

    # Save model weights
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "model_state_dict": best_state if best_state is not None else model.state_dict(),
        "architecture": {
            "input_dim": 89,
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
    torch.save(save_payload, args.model_out)
    alt_model_out = ROOT_DIR / "models" / "realsign_26class_mlp.pth"
    try:
        alt_model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_payload, alt_model_out)
    except Exception:
        pass
    model_size_kb = args.model_out.stat().st_size / 1024
    print(f"\nTrained model weights saved to: {args.model_out} and {alt_model_out} ({model_size_kb:.2f} KB)")

    # Save evaluation report
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model_name": "SETU-RealSign-26Class-MLP",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hyperparameters": {
            "input_dim": 63,
            "hidden1": 128,
            "hidden2": 64,
            "num_classes": 26,
            "dropout": 0.2,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "device": str(device),
        },
        "training_summary": training_summary,
        "test_evaluation": test_metrics,
        "extraction_metadata": stats_meta,
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    report_size_kb = args.report_out.stat().st_size / 1024
    print(f"Evaluation report saved to: {args.report_out} ({report_size_kb:.2f} KB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
