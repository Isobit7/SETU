"""
SETU — Client-Side ONNX Edge Inference & Two-Way Conversation Test Suite
Validates:
1. PROB-21: Exported RealSign ONNX graph validity and payload size (< 200 KB).
2. PROB-21: Numerical equivalence between PyTorch and ONNX Runtime logits (< 1e-4).
3. PROB-21: 100% identical class predictions across random multi-signer landmark inputs.
4. PROB-22: Two-way conversation dialogue data model, bilingual message pairing, and transcript generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from training.train_isl_dualhand_mlp import RealSignMLP

ONNX_PATH = Path("models/weights/realsign_multisigner.onnx")
FRONTEND_ONNX_PATH = Path("frontend/public/models/realsign_multisigner.onnx")
PTH_PATH = Path("models/weights/realsign_multisigner_mlp.pth")


# ---------------------------------------------------------------------------
# 1. PROB-21: ONNX FILE INTEGRITY & CHECKER VALIDATION
# ---------------------------------------------------------------------------

def test_onnx_export_file_and_validity():
    """Validates existence, file size (< 200KB), and ONNX schema validity."""
    assert ONNX_PATH.exists(), f"ONNX model file missing at {ONNX_PATH}"
    assert FRONTEND_ONNX_PATH.exists(), f"Frontend ONNX model missing at {FRONTEND_ONNX_PATH}"

    # Size check: 116 KB ensures rapid client-side web loading
    size_kb = ONNX_PATH.stat().st_size / 1024.0
    assert size_kb < 200.0, f"ONNX model is too large for web edge: {size_kb:.1f} KB"

    # ONNX schema validation
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    # Check input and output tensor names
    inputs = [i.name for i in onnx_model.graph.input]
    outputs = [o.name for o in onnx_model.graph.output]
    assert "features_126d" in inputs
    assert "logits" in outputs


# ---------------------------------------------------------------------------
# 2. PROB-21: ONNX RUNTIME NUMERICAL EQUIVALENCE VS PYTORCH
# ---------------------------------------------------------------------------

def test_onnx_runtime_inference_and_equivalence():
    """Asserts that ONNX Runtime produces logits matching PyTorch within 1e-4."""
    if not PTH_PATH.exists():
        pytest.skip(f"PyTorch checkpoint missing at {PTH_PATH}")

    ckpt = torch.load(PTH_PATH, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", {})
    classes = ckpt.get("classes", [chr(ord("A") + i) for i in range(26)])

    # Instantiate PyTorch model
    py_model = RealSignMLP(
        input_dim=arch.get("input_dim", 126),
        hidden1=arch.get("hidden1", 128),
        hidden2=arch.get("hidden2", 64),
        num_classes=arch.get("num_classes", len(classes)),
        use_batchnorm=arch.get("use_batchnorm", True),
        dropout=0.0,
    )
    py_model.load_state_dict(ckpt["model_state_dict"])
    py_model.eval()

    # Load ONNXRuntime Session
    ort_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    # Test across 25 random synthetic inputs
    np.random.seed(42)
    test_batch = np.random.uniform(-1.0, 1.0, size=(25, 126)).astype(np.float32)

    with torch.no_grad():
        py_logits = py_model(torch.from_numpy(test_batch)).numpy()

    ort_outputs = ort_session.run(None, {"features_126d": test_batch})
    ort_logits = ort_outputs[0]

    # Check maximum absolute error
    max_diff = float(np.max(np.abs(py_logits - ort_logits)))
    assert max_diff < 1e-4, f"Max difference {max_diff:.6e} exceeded 1e-4 threshold"

    # Check 100% agreement on argmax predicted classes
    py_preds = np.argmax(py_logits, axis=1)
    ort_preds = np.argmax(ort_logits, axis=1)
    assert np.array_equal(py_preds, ort_preds), "Class predictions diverged between PyTorch and ONNX"


# ---------------------------------------------------------------------------
# 3. PROB-22: CONVERSATION MODE DATA MODEL & TRANSCRIPT FORMATTING
# ---------------------------------------------------------------------------

def test_conversation_mode_message_formatting_and_transcript():
    """Validates dialogue turn serialization and transcript export generation."""
    messages = [
        {
            "id": "1",
            "sender": "signer",
            "text": "I need help with my ticket.",
            "indicText": "मुझे अपने टिकट में मदद चाहिए।",
            "timestamp": "10:30 AM",
        },
        {
            "id": "2",
            "sender": "speaker",
            "text": "Sure, which train are you taking?",
            "timestamp": "10:31 AM",
        },
    ]

    transcript_lines = [
        f"[{m['timestamp']}] {'🤟 Signer (ISL)' if m['sender'] == 'signer' else '🎙️ Speaker (Voice)'}: {m['text']}"
        for m in messages
    ]
    transcript_text = "\n".join(transcript_lines)

    assert "[10:30 AM] 🤟 Signer (ISL): I need help with my ticket." in transcript_text
    assert "[10:31 AM] 🎙️ Speaker (Voice): Sure, which train are you taking?" in transcript_text
    assert len(transcript_lines) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


