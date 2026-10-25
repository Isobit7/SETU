"""
Training and evaluation pipeline for SETU ISL MVP 5-sign (+ no-sign) model.

Produces a lightweight, pure NumPy regularized linear/softmax classifier
over pooled multi-resolution temporal features extracted from (30, 158) sequences.
No heavy dependencies (TensorFlow/Torch) needed, providing instant zero-latency inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["hello", "thank_you", "good_morning", "i", "you", "no_sign"]
LABEL_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def featurize_sequence(feat_seq: np.ndarray) -> np.ndarray:
    """
    Transform (T, 158) spatio-temporal feature matrix into a 1D feature vector.

    Captures:
    1. Temporal mean (158)
    2. Temporal standard deviation (158)
    3. Global min & max (316)
    4. Motion trajectory velocity / end-to-start delta (158)
    5. Early, Mid, Late phase means (158 * 3 = 474)

    Total pooled dimension: 158 * 8 = 1,264 features.
    """
    assert feat_seq.ndim == 2 and feat_seq.shape[1] == 158, f"Expected (T, 158), got {feat_seq.shape}"
    T = feat_seq.shape[0]

    mean_f = np.mean(feat_seq, axis=0)
    std_f = np.std(feat_seq, axis=0)
    min_f = np.min(feat_seq, axis=0)
    max_f = np.max(feat_seq, axis=0)
    delta_f = feat_seq[-1] - feat_seq[0]

    # Split into 3 temporal phases: early, mid, late
    t1 = max(1, T // 3)
    t2 = max(2, 2 * T // 3)
    early_f = np.mean(feat_seq[:t1], axis=0)
    mid_f = np.mean(feat_seq[t1:t2], axis=0)
    late_f = np.mean(feat_seq[t2:], axis=0)

    pooled = np.concatenate([
        mean_f, std_f, min_f, max_f, delta_f, early_f, mid_f, late_f
    ], axis=0).astype(np.float32)

    return pooled


def load_dataset(manifest_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Load and pool features from processed .npz files according to manifest."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits: Dict[str, Tuple[List[np.ndarray], List[int], List[str]]] = {
        "train": ([], [], []),
        "val": ([], [], []),
        "test": ([], [], []),
    }

    base_dir = manifest_path.parent.parent

    for item in manifest["samples"]:
        split = item["split"]
        label = item["label"]
        if label not in LABEL_TO_IDX:
            continue
        class_idx = LABEL_TO_IDX[label]
        feat_path = Path(item["features_path"])
        sample_id = item.get("sample_id", item.get("id", "unknown"))

        if not feat_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feat_path}")

        data = np.load(feat_path)
        raw_feats = data["features"]  # (30, 158)
        pooled = featurize_sequence(raw_feats)

        splits[split][0].append(pooled)
        splits[split][1].append(class_idx)
        splits[split][2].append(sample_id)

    res = {}
    for sp, (X_list, y_list, id_list) in splits.items():
        res[sp] = (np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), id_list)
        logger.info(f"Split {sp:5s}: X shape {res[sp][0].shape}, y count {len(res[sp][1])}")

    return res


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z_max = np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class PureNumpyISLClassifier:
    """
    Lightweight, highly reproducible regularized linear softmax classifier.
    Trains using vectorized gradient descent with L2 regularization and feature standardization.
    """

    def __init__(self, num_classes: int = 6, l2_reg: float = 0.05, lr: float = 0.02, epochs: int = 400):
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs

        self.mean: np.ndarray = np.empty(0)
        self.scale: np.ndarray = np.empty(0)
        self.weights: np.ndarray = np.empty((0, 0))  # (D, C)
        self.bias: np.ndarray = np.empty(0)  # (C,)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 42) -> None:
        rng = np.random.RandomState(seed)
        N, D = X.shape

        # Standardize features (with epsilon guard against zero variance)
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale < 1e-6] = 1.0

        X_norm = (X - self.mean) / self.scale

        # Small random initialization
        self.weights = rng.normal(0, 0.01, size=(D, self.num_classes)).astype(np.float32)
        self.bias = np.zeros(self.num_classes, dtype=np.float32)

        # One-hot encode targets
        Y_oh = np.zeros((N, self.num_classes), dtype=np.float32)
        Y_oh[np.arange(N), y] = 1.0

        # Gradient descent with gentle cosine learning rate decay
        for epoch in range(self.epochs):
            lr = self.lr * 0.5 * (1.0 + np.cos(np.pi * epoch / self.epochs))

            logits = X_norm @ self.weights + self.bias
            probs = softmax(logits)

            grad_logits = (probs - Y_oh) / N
            grad_w = X_norm.T @ grad_logits + self.l2_reg * self.weights
            grad_b = np.sum(grad_logits, axis=0)

            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[np.newaxis, :]
        X_norm = (X - self.mean) / self.scale
        logits = X_norm @ self.weights + self.bias
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            filepath,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            scale=self.scale,
            class_names=np.array(CLASS_NAMES),
            num_classes=self.num_classes,
        )
        logger.info(f"Saved trained ISL MVP model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> PureNumpyISLClassifier:
        data = np.load(filepath, allow_pickle=True)
        model = cls(num_classes=int(data["num_classes"]))
        model.weights = data["weights"]
        model.bias = data["bias"]
        model.mean = data["mean"]
        model.scale = data["scale"]
        return model


def evaluate_split(model: PureNumpyISLClassifier, X: np.ndarray, y: np.ndarray, split_name: str) -> float:
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=-1)
    acc = np.mean(preds == y) * 100.0

    logger.info(f"=== Results for {split_name.upper()} (Accuracy: {acc:.1f}%) ===")
    for idx, (target, pred, prob) in enumerate(zip(y, preds, probs)):
        status = "OK  " if target == pred else "FAIL"
        target_name = IDX_TO_LABEL[target]
        pred_name = IDX_TO_LABEL[pred]
        conf = prob[pred] * 100.0
        logger.info(f"  [{status}] sample #{idx:02d} | True: {target_name:12s} | Pred: {pred_name:12s} ({conf:.1f}%)")

    return acc


def main():
    manifest_path = Path("data/manifests/setu_isl_mvp_manifest.json")
    model_save_path = Path("models/isl_mvp_model.npz")

    logger.info("Loading dataset splits...")
    dataset = load_dataset(manifest_path)

    X_train, y_train, _ = dataset["train"]
    X_val, y_val, _ = dataset["val"]
    X_test, y_test, _ = dataset["test"]

    # Training with mild jitter augmentation on train split to prevent overfitting
    rng = np.random.RandomState(42)
    X_aug_list = [X_train]
    y_aug_list = [y_train]
    for _ in range(5):
        noise = rng.normal(0, 0.015, size=X_train.shape).astype(np.float32)
        X_aug_list.append(X_train + noise)
        y_aug_list.append(y_train)

    X_train_full = np.concatenate(X_aug_list, axis=0)
    y_train_full = np.concatenate(y_aug_list, axis=0)

    logger.info(f"Training regularized softmax classifier (Augmented train size: {len(y_train_full)})...")
    model = PureNumpyISLClassifier(num_classes=len(CLASS_NAMES), l2_reg=0.20, lr=0.03, epochs=400)
    model.fit(X_train_full, y_train_full)

    # Evaluate on all splits
    train_acc = evaluate_split(model, X_train, y_train, "train")
    val_acc = evaluate_split(model, X_val, y_val, "val")
    test_acc = evaluate_split(model, X_test, y_test, "test")

    logger.info(f"Final Summary: Train={train_acc:.1f}% | Val={val_acc:.1f}% | Test={test_acc:.1f}%")

    # Save model
    model.save(model_save_path)
    logger.info(f"MVP Model successfully written to {model_save_path}")


if __name__ == "__main__":
    main()
