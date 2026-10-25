"""
SETU — ML Evaluation Pipeline
Generates detailed test metrics, confusion matrix, and evaluation reports
for the 26-class RealSign sign language alphabet model.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.realsign_classifier_service import RealSignMLP

DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "realsign_26class_mlp.pt"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "realsign_26class_evaluation_report.json"

CLASSES_26: List[str] = [chr(ord("A") + i) for i in range(26)]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str] = CLASSES_26,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics and optionally exports JSON report."""
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # Identify top misclassifications
    confusion_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j]),
                })

    confusion_pairs.sort(key=lambda x: x["count"], reverse=True)
    top_5_confusions = confusion_pairs[:5]

    summary = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_test_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
        "top_confusions": top_5_confusions,
        "per_class_metrics": {
            cls: {
                "precision": round(float(report_dict[cls]["precision"]), 4),
                "recall": round(float(report_dict[cls]["recall"]), 4),
                "f1": round(float(report_dict[cls]["f1-score"]), 4),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in classes if cls in report_dict
        },
        "confusion_matrix": cm.tolist(),
    }

    if output_report_path:
        output_report_path = Path(output_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Evaluation report saved to: {output_report_path}")

    return summary


def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    report_path: Path = DEFAULT_REPORT,
) -> Dict[str, Any]:
    """Loads checkpoint, evaluates against test split, and generates report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        alt_path = checkpoint_path.with_suffix(".pth")
        if alt_path.exists():
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"[+] Evaluating checkpoint: {checkpoint_path}")
    print(f"[+] Dataset path: {dataset_path}")

    # Load dataset
    with np.load(dataset_path, allow_pickle=True) as data:
        X = np.array(data["X"], dtype=np.float32)
        y = np.array(data["y"], dtype=np.int64)
        splits = np.array(data["splits"], dtype=str) if "splits" in data else None
        classes = [str(c) for c in data["classes"]] if "classes" in data else CLASSES_26

    test_mask = (splits == "test") if splits is not None else np.ones(len(y), dtype=bool)
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"[+] Evaluating on {len(X_test):,} test samples across {len(classes)} classes...")

    # Load model
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    arch = ckpt.get("architecture", {}) if isinstance(ckpt, dict) else {}
    model = RealSignMLP(
        input_dim=arch.get("input_dim", 63),
        hidden1=arch.get("hidden1", 128),
        hidden2=arch.get("hidden2", 64),
        num_classes=len(classes),
    ).to(device)

    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Predict in batches
    preds = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.from_numpy(X_test[i : i + batch_size]).to(device)
            out = model(batch)
            p = out.argmax(dim=-1).cpu().numpy()
            preds.append(p)

    y_pred = np.concatenate(preds)
    results = evaluate_predictions(y_test, y_pred, classes=classes, output_report_path=report_path)

    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score  : {results['macro_f1'] * 100:.2f}%")
    print("Top Misclassifications:")
    for c in results["top_confusions"]:
        print(f"  {c['true_class']} -> {c['predicted_class']} ({c['count']} times)")
    print("=" * 60)

    return results


evaluate_model = run_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RealSign model checkpoint")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    run_evaluation(dataset_path=args.dataset, checkpoint_path=args.checkpoint, report_path=args.report)
