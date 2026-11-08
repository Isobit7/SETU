# Models Architecture & Use-Case Directory (`models/`)
## Project SETU: Neural Networks, Model Adapters & Weights

This directory contains all neural network definitions, weights, metadata, and runtime inference adapters. Every folder and file is categorized below by its specific use case:

---

### Directory Layout & File Use Cases

```
models/
├── adapters/               # Use Case: Standardized inference wrappers connecting PyTorch/ONNX to backend
│   ├── realsign.py         # USE CASE: 26-Class Alphabet PyTorch MLP adapter (A–Z)
│   ├── amrita.py           # USE CASE: 100-Phrase Continuous BiLSTM adapter
│   ├── digits.py           # USE CASE: 10-Class Numerals ONNX adapter (0–9)
│   ├── include.py          # USE CASE: AI4Bharat INCLUDE 263-sign continuous gesture adapter
│   ├── sequence_lstm.py    # USE CASE: Temporal sequence LSTM adapter for prototype signs
│   ├── mvp.py              # USE CASE: Lightweight Ridge/Logistic regression baseline adapter
│   └── rf_uv11.py          # USE CASE: Random Forest / Keras legacy baseline adapter
├── weights/                # Use Case: Trained binary weight files (See WEIGHTS_REGISTRY.md)
│   ├── realsign_26class_mlp.pth     # ACTIVE: 26-class Alphabet MLP weights (Nov 6, 2026)
│   ├── amrita_isl_phrase_bilstm.pth # ACTIVE: 100-phrase BiLSTM weights (Nov 6, 2026)
│   ├── realsign_multisigner.onnx    # ACTIVE: Multi-signer invariant ONNX runtime (Nov 6, 2026)
│   ├── isl_numerals.onnx            # ACTIVE: Numbers 0–9 ONNX runtime (Nov 6, 2026)
│   ├── isl_hindi_vowels.onnx        # ACTIVE: Hindi Vowels ONNX runtime (Nov 6, 2026)
│   ├── yolov8s-worldv2.pt           # ACTIVE: Auxiliary YOLO-World object detection weights
│   ├── realsign_26class_mlp.pth.bak # ARCHIVE: Single-signer backup weights (Oct 30, 2026)
│   ├── isl_sequence_lstm.pth        # ARCHIVE: Unidirectional LSTM baseline (Oct 28, 2026)
│   ├── hf_isl_26class_mlp.pth       # ARCHIVE: HuggingFace baseline weights (Oct 26, 2026)
│   └── isl_mvp_model.npz            # ARCHIVE: MVP Prototype weights (Oct 26, 2026)
├── tasks/                  # Use Case: Pre-compiled MediaPipe task bundles
│   ├── hand_landmarker.task      # MediaPipe 21-hand landmark detector binary
│   └── holistic_landmarker.task  # MediaPipe 543-joint holistic landmarker binary
├── metadata/               # Use Case: Label maps and vocabulary mapping JSONs
│   └── sequence_lstm_label_map.json # Class index to human-readable sign strings
├── reference/              # Use Case: Benchmark baseline models for comparative evaluations
│   ├── BASELINES.md                 # Baseline accuracy logs
│   ├── baseline_89d_realsign_mlp.pth # 89-dim feature ablation model
│   └── realsign_26class_mlp_63d.pt   # 63-dim unquantized reference model
├── registry.py             # USE CASE: Central Model Registry providing dynamic model switching
└── WEIGHTS_REGISTRY.md     # USE CASE: Comprehensive dates, sizes, architectures & accuracy index
```

---

### Adapter Pattern Explained

Instead of having the WebSocket code directly deal with raw PyTorch tensor shapes or ONNX runtime sessions:
1. Every model is wrapped inside an **Adapter class** (in `models/adapters/`).
2. Each adapter exposes a standardized method: `predict(features) -> PredictionResult`.
3. `PredictionResult` always returns:
   - `label`: String name of the predicted sign (e.g. `"C"` or `"thank_you"`).
   - `confidence`: Float probability between $0.0$ and $1.0$.
   - `is_known_sign`: Boolean indicating whether the confidence meets the threshold.
   - `probabilities`: Dictionary mapping all candidate classes to their softmax scores.
