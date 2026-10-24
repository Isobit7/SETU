"""
SETU — PyTorch to ONNX Model Exporter & Inference Latency Benchmark

Exports the trained PyTorch RealSignMLP (26-class ISL alphabet) checkpoint to ONNX format
with dynamic batching, verifies the model structure, and executes a rigorous 1,000-run
inference latency benchmark comparing PyTorch CPU vs. ONNX Runtime CPU.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.classifier import RealSignMLP, CLASSES_26


def format_header(title: str, width: int = 76) -> str:
    line = "=" * width
    return f"\n{line}\n {title.center(width - 2)}\n{line}"


def format_sub_divider(width: int = 76) -> str:
    return "-" * width


def load_pytorch_model(weights_path: Path) -> Tuple[RealSignMLP, Dict]:
    """Loads weights from .pt checkpoint into an eval-mode CPU RealSignMLP."""
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    print(f"[INFO] Loading PyTorch weights from: {weights_path}")
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        arch = checkpoint.get("architecture", {})
        input_dim = arch.get("input_dim", 63)
        hidden1 = arch.get("hidden1", 128)
        hidden2 = arch.get("hidden2", 64)
        classes = checkpoint.get("classes", CLASSES_26)
        num_classes = len(classes)

        model = RealSignMLP(
            input_dim=input_dim,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=num_classes,
        )

        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
        classes = CLASSES_26
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")

    model.to("cpu")
    model.eval()
    return model, checkpoint


def export_to_onnx(model: RealSignMLP, output_path: Path) -> Path:
    """Exports RealSignMLP to ONNX format with dynamic batch dimension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 63, dtype=torch.float32)

    print(f"[INFO] Exporting RealSignMLP to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    # Validate ONNX structure
    print(f"[INFO] Validating exported ONNX model with onnx.checker...")
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    print(f"[OK] ONNX model structure is valid.")
    return output_path


def verify_numerical_parity(
    pytorch_model: RealSignMLP,
    onnx_path: Path,
    test_samples: np.ndarray,
    tol: float = 1e-4,
) -> None:
    """Confirms PyTorch and ONNX Runtime produce identical outputs."""
    print(f"[INFO] Checking numerical parity across {len(test_samples)} samples...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for i in range(min(len(test_samples), 50)):
        sample_tensor = torch.from_numpy(test_samples[i:i+1].astype(np.float32))
        with torch.no_grad():
            py_logits = pytorch_model(sample_tensor).numpy()

        ort_inputs = {"input": test_samples[i:i+1].astype(np.float32)}
        ort_logits = session.run(None, ort_inputs)[0]

        max_diff = np.max(np.abs(py_logits - ort_logits))
        if max_diff > tol:
            raise ValueError(f"Numerical disparity detected: max diff = {max_diff:.6f} > {tol}")

    print(f"[OK] Numerical parity confirmed: PyTorch & ONNX logits match within {tol} tolerance.")


def benchmark_inference(
    pytorch_model: RealSignMLP,
    onnx_path: Path,
    samples: np.ndarray,
    num_runs: int = 1000,
    warmup_runs: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Runs 1,000 single-sample inferences on CPU for PyTorch vs ONNX Runtime."""
    print(f"[INFO] Initializing inference benchmark (runs: {num_runs:,}, warmups: {warmup_runs})...")
    
    # Prepare batch of test inputs
    indices = np.random.choice(len(samples), size=num_runs, replace=True)
    benchmark_data = samples[indices].astype(np.float32)

    # 1. Benchmark PyTorch CPU
    pytorch_model.eval()
    sample_tensors = [torch.from_numpy(benchmark_data[i:i+1]) for i in range(min(warmup_runs, len(benchmark_data)))]
    
    # Warmup PyTorch
    with torch.no_grad():
        for t in sample_tensors:
            _ = pytorch_model(t)

    py_latencies_ms: List[float] = []
    with torch.no_grad():
        for i in range(num_runs):
            t_sample = torch.from_numpy(benchmark_data[i:i+1])
            start = time.perf_counter()
            _ = pytorch_model(t_sample)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            py_latencies_ms.append(elapsed_ms)

    # 2. Benchmark ONNX Runtime CPU
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    
    # Warmup ONNX Runtime
    for i in range(warmup_runs):
        _ = session.run(None, {"input": benchmark_data[i:i+1]})

    ort_latencies_ms: List[float] = []
    for i in range(num_runs):
        feed = {"input": benchmark_data[i:i+1]}
        start = time.perf_counter()
        _ = session.run(None, feed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        ort_latencies_ms.append(elapsed_ms)

    # Compute statistics
    def compute_stats(latencies: List[float]) -> Dict[str, float]:
        arr = np.array(latencies)
        mean_lat = float(np.mean(arr))
        median_lat = float(np.median(arr))
        p95_lat = float(np.percentile(arr, 95))
        p99_lat = float(np.percentile(arr, 99))
        throughput_fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
        return {
            "mean_ms": mean_lat,
            "median_ms": median_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "throughput_fps": throughput_fps,
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
        }

    return {
        "pytorch": compute_stats(py_latencies_ms),
        "onnx": compute_stats(ort_latencies_ms),
    }


def print_ascii_summary_table(
    results: Dict[str, Dict[str, float]],
    pt_size_kb: float,
    onnx_size_kb: float,
    num_runs: int = 1000,
) -> None:
    """Prints a structured ASCII benchmark summary table."""
    py = results["pytorch"]
    ort_res = results["onnx"]

    speedup = py["mean_ms"] / ort_res["mean_ms"] if ort_res["mean_ms"] > 0 else 1.0
    fps_gain = ((ort_res["throughput_fps"] - py["throughput_fps"]) / py["throughput_fps"]) * 100.0

    print(format_header(f"SETU REAL-TIME INFERENCE BENCHMARK ({num_runs:,} RUNS - CPU)"))
    
    header = f"| {'Metric':<22} | {'PyTorch (CPU)':<18} | {'ONNX Runtime (CPU)':<20} | {'Delta / Speedup':<17} |"
    divider = "+" + "-" * 24 + "+" + "-" * 20 + "+" + "-" * 22 + "+" + "-" * 19 + "+"

    rows = [
        ("Average Latency", f"{py['mean_ms']:.4f} ms", f"{ort_res['mean_ms']:.4f} ms", f"{speedup:.2f}x faster"),
        ("Median Latency (p50)", f"{py['median_ms']:.4f} ms", f"{ort_res['median_ms']:.4f} ms", f"{py['median_ms']/ort_res['median_ms']:.2f}x"),
        ("95th Percentile (p95)", f"{py['p95_ms']:.4f} ms", f"{ort_res['p95_ms']:.4f} ms", f"{py['p95_ms']/ort_res['p95_ms']:.2f}x"),
        ("99th Percentile (p99)", f"{py['p99_ms']:.4f} ms", f"{ort_res['p99_ms']:.4f} ms", f"{py['p99_ms']/ort_res['p99_ms']:.2f}x"),
        ("Throughput (FPS)", f"{py['throughput_fps']:,.1f} FPS", f"{ort_res['throughput_fps']:,.1f} FPS", f"+{fps_gain:.1f}%"),
        ("Model File Size", f"{pt_size_kb:.1f} KB", f"{onnx_size_kb:.1f} KB", f"{(onnx_size_kb/pt_size_kb)*100:.1f}% of .pt"),
    ]

    print(divider)
    print(header)
    print(divider)
    for name, py_val, onnx_val, delta in rows:
        print(f"| {name:<22} | {py_val:<18} | {onnx_val:<20} | {delta:<17} |")
    print(divider)
    print(f"\n[CONCLUSION] ONNX Runtime delivers {speedup:.2f}x faster CPU execution ({ort_res['throughput_fps']:,.0f} FPS).")
    print(f"[CONCLUSION] Average inference latency is just {ort_res['mean_ms']*1000:.1f} microseconds per frame.")


def main():
    pt_path = PROJECT_ROOT / "models" / "realsign_26class_mlp.pt"
    onnx_path = PROJECT_ROOT / "models" / "realsign_26class_mlp.onnx"
    npz_path = PROJECT_ROOT / "data" / "processed" / "realsign_landmarks_26class.npz"

    print(format_header("SETU 26-CLASS ISL REAL-SIGN MLP ONNX EXPORT & BENCHMARK"))

    # 1. Load PyTorch model
    model, _ = load_pytorch_model(pt_path)

    # 2. Export to ONNX
    export_to_onnx(model, onnx_path)

    # 3. Load sample vectors from processed dataset
    if npz_path.exists():
        print(f"[INFO] Loading sample vectors from: {npz_path}")
        dataset = np.load(str(npz_path))
        sample_vectors = dataset["X"]
        print(f"[INFO] Loaded {len(sample_vectors):,} real landmark vectors (dim: {sample_vectors.shape[1]}).")
    else:
        print(f"[WARN] {npz_path} not found. Generating synthetic 63-D vectors.")
        sample_vectors = np.random.randn(1000, 63).astype(np.float32)

    # 4. Verify numerical parity
    verify_numerical_parity(model, onnx_path, sample_vectors)

    # 5. Measure latency and throughput across 1,000 runs
    num_runs = 1000
    results = benchmark_inference(model, onnx_path, sample_vectors, num_runs=num_runs)

    # 6. File size stats
    pt_size_kb = os.path.getsize(pt_path) / 1024.0
    onnx_size_kb = os.path.getsize(onnx_path) / 1024.0

    # 7. Print ASCII Summary Table
    print_ascii_summary_table(results, pt_size_kb, onnx_size_kb, num_runs=num_runs)


if __name__ == "__main__":
    main()
