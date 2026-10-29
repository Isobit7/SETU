"""
SETU — Command-Line Model Evaluation Matrices & Parameter Generator

Scans models/weights/ (and registered subdirectories), inspects weights and biases
layer-by-layer, extracts parameter statistics, training hyperparameters (learning rate,
optimizer, loss function, epochs), and evaluation metrics (accuracy, F1, latency).

Outputs:
  - reports/evaluation_matrices/master_evaluation_matrix.json
  - reports/evaluation_matrices/README.md
  - reports/evaluation_matrices/<model_id>_matrix.json
  - reports/evaluation_matrices/<model_id>_matrix.md
  - Synchronized copy in reports/model_matrices/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.services.model_matrices import (
    scan_and_generate_model_matrices,
    MATRICES_DIR,
    ALIAS_MATRICES_DIR,
    has_weights_changed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SETU.ModelMatrices")


def main() -> None:
    parser = argparse.ArgumentParser(description="SETU Model Evaluation Matrices Generator")
    parser.add_argument("--force", action="store_true", help="Force regenerate even if weights unchanged")
    parser.add_argument("--watch", action="store_true", help="Continuously poll and regenerate on model changes")
    parser.add_argument("--interval", type=int, default=5, help="Watch polling interval in seconds")
    args = parser.parse_args()

    if args.watch:
        logger.info("Watching models/weights/ for model changes (interval: %ds)... Press Ctrl+C to exit.", args.interval)
        try:
            while True:
                if has_weights_changed():
                    logger.info("Detected change in models/weights/. Regenerating evaluation matrices...")
                    res = scan_and_generate_model_matrices()
                    logger.info("Regenerated matrices for %d models.", res.get("total_models", 0))
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Watcher terminated by user.")
            return

    logger.info("Generating model evaluation matrices in %s...", MATRICES_DIR)
    master = scan_and_generate_model_matrices()

    print("\n" + "=" * 80)
    print("SETU MODEL EVALUATION MATRICES & PARAMETER SUMMARY")
    print("=" * 80)
    print(f"Total Models Cataloged: {master.get('total_models', 0)}")
    print(f"Directory:              {MATRICES_DIR}")
    print(f"Synchronized Alias:     {ALIAS_MATRICES_DIR}\n")

    for m_id, m in master.get("models", {}).items():
        lr = m.get("hyperparameters", {}).get("learning_rate", "N/A")
        acc = m.get("evaluation_metrics", {}).get("test_accuracy", "N/A")
        if isinstance(acc, float):
            acc_str = f"{acc * 100:.2f}%"
        else:
            acc_str = str(acc)
        print(f" • [{m_id}] {m['name']}")
        print(f"   Architecture: {m['architecture_type']}")
        print(f"   Parameters:   {m['total_parameters']:,} (Weights: {m['weights_parameters']:,}, Biases: {m['biases_parameters']:,})")
        print(f"   Learning Rate: {lr} | Accuracy/F1: {acc_str}")
        print(f"   Matrix Docs:  {m_id}_matrix.md & {m_id}_matrix.json\n")

    print("=" * 80)
    print(f"Master Catalog: {MATRICES_DIR / 'README.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

