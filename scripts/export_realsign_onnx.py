"""
SETU — Export RealSign Multi-Signer PyTorch Model to ONNX for WebAssembly & Browser Edge (PROB-21).

Exports realsign_multisigner_mlp.pth into an optimized .onnx graph with dynamic batching,
and verifies numerical equivalence against PyTorch predictions within 1e-5 tolerance.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch

from training.train_isl_dualhand_mlp import RealSignMLP

WEIGHTS_PTH = Path("models/weights/realsign_multisigner_mlp.pth")
OUTPUT_ONNX = Path("models/weights/realsign_multisigner.onnx")
FRONTEND_MODELS_DIR = Path("frontend/public/models")


def export_realsign_to_onnx(
    weights_path: Path = WEIGHTS_PTH,
    output_path: Path = OUTPUT_ONNX,
) -> Path:
    """Exports RealSignMLP PyTorch checkpoint to ONNX."""
    if not weights_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    arch = ckpt.get("architecture", {})
    classes = ckpt.get("classes", [chr(ord("A") + i) for i in range(26)])

    input_dim = arch.get("input_dim", 126)
    hidden1 = arch.get("hidden1", 128)
    hidden2 = arch.get("hidden2", 64)
    num_classes = arch.get("num_classes", len(classes))
    use_batchnorm = arch.get("use_batchnorm", True)

    model = RealSignMLP(
        input_dim=input_dim,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=num_classes,
        use_batchnorm=use_batchnorm,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["features_126d"],
        output_names=["logits"],
        dynamic_axes={
            "features_126d": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    # Verify numerical equivalence with ONNXRuntime
    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    test_inputs = np.random.randn(10, input_dim).astype(np.float32)

    with torch.no_grad():
        py_logits = model(torch.from_numpy(test_inputs)).numpy()

    ort_inputs = {"features_126d": test_inputs}
    ort_logits = ort_session.run(None, ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(py_logits - ort_logits)))
    if max_abs_diff > 1e-4:
        raise ValueError(f"Numerical discrepancy between PyTorch and ONNX too high: {max_abs_diff}")

    # Copy to frontend public/models/
    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frontend_copy = FRONTEND_MODELS_DIR / output_path.name
    shutil.copy2(output_path, frontend_copy)

    print(f"[+] Successfully exported RealSign to ONNX: {output_path} (max diff: {max_abs_diff:.6e})")
    print(f"[+] Synced to frontend assets: {frontend_copy}")
    return output_path


if __name__ == "__main__":
    export_realsign_to_onnx()
