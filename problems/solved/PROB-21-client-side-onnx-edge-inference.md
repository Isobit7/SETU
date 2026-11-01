---
id: PROB-21
title: "Real-Time Client-Side Edge Inference via ONNX Web"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:30
severity: High
category: Performance & Edge Deployment
---

# PROB-21: Real-Time Client-Side Edge Inference via ONNX Web

## 1. Problem Description
Currently, all video frames captured by the browser webcam must be compressed into JPEGs and streamed across a WebSocket connection to the Python backend on every frame. When the network connection drops or fluctuates (common in hospital corridors, mobile data, or rural regions), inference freezes and framerates plunge. Furthermore, server-side processing for thousands of simultaneous streams would require prohibitive GPU/cloud infrastructure.

## 2. Root Cause Analysis
- The architecture is 100% server-dependent for inference.
- However, our PyTorch `RealSignMLP` weights file is only **116 KB**, and modern browsers natively support WebAssembly (Wasm) and WebGL-accelerated ONNX runtime.

## 3. Implementation Plan & Proposed Architecture
1. **Export PyTorch MLP to ONNX Format**:
   - Write `scripts/export_realsign_onnx.py`:
     ```python
     torch.onnx.export(
         model, dummy_input, "frontend/public/models/realsign_multisigner.onnx",
         input_names=["landmarks_126d"], output_names=["logits"],
         dynamic_axes={"landmarks_126d": {0: "batch_size"}}
     )
     ```
2. **In-Browser Landmark Extraction & Inference**:
   - Install `@mediapipe/camera_utils`, `@mediapipe/hands`, and `onnxruntime-web` in the frontend.
   - Run landmark extraction and ONNX model evaluation inside a dedicated WebWorker.
3. **Hybrid Fallback Mode**:
   - **Offline / Fast Mode (Client-Side):** 60 FPS Alphabet recognition with 0ms network latency.
   - **Cloud Mode (Server-Side):** Used for heavy 263-class INCLUDE video models or Qwen2.5-0.5B neural translations when online.

## 4. Acceptance Criteria & Verification
- [x] Exported `.onnx` model reproduces PyTorch logits within $\le 1e-4$ max absolute error.
- [x] Browser can run continuous sign recognition with Wi-Fi/Internet completely disabled.
- [x] Latency reduces from ~45ms (network WebSocket) to $< 8$ ms (local Wasm execution).
