"""
SETU — Automated Test Suite for Model Evaluation Matrices & Parameter Inspection

Validates:
1. Complete extraction of model weights, biases, layers, and tensor statistics.
2. Training hyperparameters (learning rate, optimizer, epochs) and evaluation metrics.
3. Master JSON, Markdown summary, and individual model reports generation.
4. Dynamic auto-listing when a new model is added, and automatic purging when deleted.
5. FastAPI REST API endpoints: GET /v1/models/matrices, POST /v1/models/matrices/refresh.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.model_matrices import (
    scan_and_generate_model_matrices,
    get_model_matrices,
    MATRICES_DIR,
    ALIAS_MATRICES_DIR,
    WEIGHTS_DIR,
)

client = TestClient(app)


def test_generate_model_matrices_structure():
    """Verifies that scan_and_generate_model_matrices discovers models and writes files."""
    master = scan_and_generate_model_matrices()
    assert master is not None
    assert "total_models" in master
    assert master["total_models"] >= 5
    assert "models" in master

    # Check files on disk
    master_json = MATRICES_DIR / "master_evaluation_matrix.json"
    assert master_json.exists(), "master_evaluation_matrix.json should exist"

    readme_md = MATRICES_DIR / "README.md"
    assert readme_md.exists(), "README.md should exist in evaluation_matrices"

    readme_content = readme_md.read_text(encoding="utf-8")
    assert "Model Evaluation Matrices" in readme_content
    assert "Executive Model Comparison Matrix" in readme_content

    # Check alias directory synchronization
    alias_master = ALIAS_MATRICES_DIR / "master_evaluation_matrix.json"
    assert alias_master.exists(), "master_evaluation_matrix.json should be synced to alias dir"


def test_model_matrix_contents_and_layer_stats():
    """Verifies weights, biases, learning rate, and tensor statistics for core models."""
    matrices = get_model_matrices()
    models = matrices["models"]

    # 1. RealSign MLP
    assert "realsign_26class_mlp" in models
    realsign = models["realsign_26class_mlp"]
    assert realsign["total_parameters"] > 20000
    assert realsign["weights_parameters"] > 0
    assert realsign["biases_parameters"] > 0
    assert realsign["layers_count"] >= 8
    assert "learning_rate" in realsign["hyperparameters"]
    assert realsign["hyperparameters"]["learning_rate"] == 0.001

    # Check layer tensor statistics
    first_layer = realsign["layers"][0]
    assert "shape" in first_layer
    assert "mean" in first_layer
    assert "std" in first_layer
    assert "min" in first_layer
    assert "max" in first_layer
    assert "l2_norm" in first_layer
    assert "sparsity_pct" in first_layer

    # Check individual report files exist
    assert (MATRICES_DIR / "realsign_26class_mlp_matrix.json").exists()
    assert (MATRICES_DIR / "realsign_26class_mlp_matrix.md").exists()

    # 2. ISL Sequence LSTM
    assert "isl_sequence_lstm" in models
    lstm = models["isl_sequence_lstm"]
    assert lstm["total_parameters"] > 900000
    assert lstm["hyperparameters"]["learning_rate"] == 0.001
    assert (MATRICES_DIR / "isl_sequence_lstm_matrix.json").exists()
    assert (MATRICES_DIR / "isl_sequence_lstm_matrix.md").exists()

    # 3. YOLO-World v2
    assert "yolov8s_worldv2" in models
    yolo = models["yolov8s_worldv2"]
    assert yolo["total_parameters"] > 10000000
    assert (MATRICES_DIR / "yolov8s_worldv2_matrix.json").exists()


def test_dynamic_model_addition_and_deletion(tmp_path):
    """
    Verifies that saving a new model file automatically lists it in the matrix,
    and deleting that file automatically removes it.
    """
    dummy_model_path = WEIGHTS_DIR / "test_dynamic_dummy.pth"

    try:
        # Create a small dummy PyTorch model
        dummy_net = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        dummy_payload = {
            "model_state_dict": dummy_net.state_dict(),
            "learning_rate": 0.0005,
            "optimizer": "AdamW",
            "epochs": 15,
            "test_accuracy": 0.965,
        }
        torch.save(dummy_payload, dummy_model_path)

        # Regenerate matrix
        updated_master = scan_and_generate_model_matrices()
        assert "test_dynamic_dummy" in updated_master["models"], "New model should be dynamically listed"

        dummy_record = updated_master["models"]["test_dynamic_dummy"]
        assert dummy_record["hyperparameters"]["learning_rate"] == 0.0005
        assert dummy_record["evaluation_metrics"]["test_accuracy"] == 0.965
        assert (MATRICES_DIR / "test_dynamic_dummy_matrix.json").exists()
        assert (MATRICES_DIR / "test_dynamic_dummy_matrix.md").exists()

    finally:
        # Delete dummy model file
        if dummy_model_path.exists():
            dummy_model_path.unlink()

        # Rescan - model should now be automatically deleted
        cleaned_master = scan_and_generate_model_matrices()
        assert "test_dynamic_dummy" not in cleaned_master["models"], "Deleted model should be removed from catalog"
        assert not (MATRICES_DIR / "test_dynamic_dummy_matrix.json").exists(), "Deleted model report should be purged"
        assert not (MATRICES_DIR / "test_dynamic_dummy_matrix.md").exists(), "Deleted model markdown should be purged"


def test_api_endpoints_model_matrices():
    """Verifies the REST API endpoints for model evaluation matrices."""
    # 1. GET /api/v1/models/matrices
    res = client.get("/api/v1/models/matrices")
    assert res.status_code == 200
    data = res.json()
    assert "total_models" in data
    assert data["total_models"] >= 5
    assert "realsign_26class_mlp" in data["models"]

    # 2. GET /api/v1/models/matrices/realsign_26class_mlp
    res_single = client.get("/api/v1/models/matrices/realsign_26class_mlp")
    assert res_single.status_code == 200
    single_data = res_single.json()
    assert single_data["model_id"] == "realsign_26class_mlp"
    assert "layers" in single_data
    assert "hyperparameters" in single_data
    assert single_data["hyperparameters"]["learning_rate"] == 0.001

    # 3. POST /api/v1/models/matrices/refresh
    res_refresh = client.post("/api/v1/models/matrices/refresh")
    assert res_refresh.status_code == 200
    assert res_refresh.json()["total_models"] >= 5

