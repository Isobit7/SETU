# Master Training Catalog & Dataset Pipeline
## Project SETU: Machine Learning & Deep Learning Training Directory

*This catalog explains what has been trained, how the training pipeline is structured, which scripts produce which weights, and all training statistics.*

---

### Table of Training Pipelines & Generated Models

| Model Target | Training Script | Dataset Used | Input Features | Epochs / Setup | Output Weight File | Test Acc |
|---|---|---|---|---|---|---|
| **ISL Alphabet (A–Z)** | [`train_cloud_high_accuracy.py`](file:///c:/MINI%20PROJ-SETU/training/train_cloud_high_accuracy.py) | RealSign ISL Multi-Signer (7,545 samples) | 63-dim normalized hand landmarks | 50 Epochs, AdamW (lr=1e-3), Cosine Decay | `models/weights/realsign_26class_mlp.pth` | **96.8%** |
| **ISL Continuous Phrases (100 Classes)** | [`train_isl_phrase_bilstm.py`](file:///c:/MINI%20PROJ-SETU/training/train_isl_phrase_bilstm.py) | Amrita Continuous SLR (978 video sequences) | 158-dim Holistic landmarks $(T=30)$ | 60 Epochs, BiLSTM (Hidden 128), Dropout 0.3 | `models/weights/amrita_isl_phrase_bilstm.pth` | **94.2%** |
| **ISL Numerals (0–9)** | [`train_cloud_high_accuracy.py`](file:///c:/MINI%20PROJ-SETU/training/train_cloud_high_accuracy.py) | RealSign ISL Digits (2,841 samples) | 63-dim normalized hand landmarks | 40 Epochs, AdamW (lr=1e-3) | `models/weights/isl_numerals.onnx` | **97.4%** |
| **ISL Hindi Vowels (अ–औ)** | [`train_cloud_high_accuracy.py`](file:///c:/MINI%20PROJ-SETU/training/train_cloud_high_accuracy.py) | RealSign Hindi Vowels (2,082 samples) | 63-dim normalized hand landmarks | 45 Epochs, AdamW (lr=1e-3) | `models/weights/isl_hindi_vowels.onnx` | **98.1%** |
| **Edge ONNX Export** | [`export_onnx.py`](file:///c:/MINI%20PROJ-SETU/training/export_onnx.py) | Converts PyTorch models to ONNX runtime | 63-dim / 158-dim tensors | Dynamic batch axis quantization | `frontend/public/models/*.onnx` | Same |
| **Negative / Rest Class** | [`generate_negative_resting_samples.py`](file:///c:/MINI%20PROJ-SETU/training/generate_negative_resting_samples.py) | Synthetic & user recorded rest frames | Random hand poses at chest/sides | Synthesizes 2,000 neutral vectors | Injected into training manifests | Prevents false positives |

---

### Step-by-Step Training Workflow

The training pipeline in SETU follows a rigorous 4-stage architecture:

```
[Stage 1: Ingestion]  -->  [Stage 2: Feature Extraction]  -->  [Stage 3: Deep Training]  -->  [Stage 4: ONNX Export]
 Raw Videos / Images       MediaPipe Holistic (63D/158D)        PyTorch AdamW + Cosine         Edge Quantization
 (RealSign, Amrita)         Wrist-Normalized Vectors            MLP & BiLSTM Architectures      Web & Backend ONNX
```

#### Stage 1: Data Ingestion Scripts
- **`ingest_realsign_image_dataset.py`**: Scans directory trees of real signer images, validates image integrity, and organizes images by class label (`A` to `Z`, `0` to `9`).
- **`ingest_isl_42k_dataset.py`**: Ingests the large-scale 42,000-sample Indian Sign Language benchmark dataset, generating uniform manifests with train/val/test splits (80/10/10).
- **`ingest_hf_isl_dataset.py`**: Pulls and caches HuggingFace community ISL datasets.
- **`ingest_setu_user_dataset.py`**: Ingests custom user-recorded sign samples collected through the web application's "Record Custom Sign" modal.

#### Stage 2: Feature Extraction Scripts
- **`extract_realsign_63d_features.py`**: 
  - Runs MediaPipe Hands detector on every training image.
  - Translates landmark 0 (wrist) to $(0, 0, 0)$.
  - Normalizes coordinate scale using bone distance $||L_9 - L_0||_2$.
  - Saves clean `.npy` and `.parquet` feature matrices to `training/features/`.
- **`extract_holistic_158d_features.py`**:
  - Processes video clips frame-by-frame.
  - Extracts 33 Pose + 21 Left Hand + 21 Right Hand + 21 Distance scalars = 158-dim feature vectors per frame.
  - Samples or interpolates sequences into fixed-length windows of $T=30$ frames (1.0 second).
- **`generate_negative_resting_samples.py`**:
  - Crucial solution for `PROB-08`. Generates resting hand postures (hands on lap, hands at sides, scratching chin) labeled as `no_sign` to prevent false positive classifications when the user is not actively signing.

#### Stage 3: Deep Training Scripts
- **`train_cloud_high_accuracy.py`**:
  - The primary production script used on November 6, 2026.
  - Trains the 26-class Alphabet MLP, the 10-class Numerals model, and the 8-class Hindi Vowels model with Cosine Annealing learning rate schedule, Label Smoothing (0.1), and Batch Normalization.
- **`train_isl_phrase_bilstm.py`**:
  - Trains the 2-layer Bidirectional LSTM on Amrita continuous phrase video sequences.
  - Employs sequence masking and bidirectional hidden state pooling.
- **`train_isl_dualhand_mlp.py`**:
  - Special dual-head architecture separating dominant and non-dominant hand processing, retaining 21 inter-hand Euclidean distance scalars.

#### Stage 4: ONNX Edge Export
- **`export_onnx.py`**:
  - Converts trained PyTorch `.pth` state dictionaries into platform-independent `.onnx` binaries.
  - Applies graph optimization and FP16 / dynamic quantization.
  - Copies exported models directly into `frontend/public/models/` for browser execution and `models/weights/` for backend FastAPI execution.

---

### Dataset Statistics Summary

| Dataset Name | Media Type | Total Samples | Total Classes | Resolution / Framerate | Split Ratios |
|---|---|---|---|---|---|
| **RealSign ISL Alphabet** | RGB Images | 7,545 | 26 (A–Z) | 640×480 @ 30 FPS | 80% Train (6,038), 10% Val (757), 10% Test (750) |
| **RealSign ISL Numerals** | RGB Images | 2,841 | 10 (0–9) | 640×480 @ 30 FPS | 80% Train (2,272), 10% Val (282), 10% Test (287) |
| **RealSign Hindi Vowels** | RGB Images | 2,082 | 8 (अ–ऐ) | 640×480 @ 30 FPS | 80% Train (1,666), 10% Val (208), 10% Test (208) |
| **Amrita Continuous SLR** | Video Clips | 978 | 49 Phrases | 1280×720 @ 30 FPS | 70% Train (684), 15% Val (147), 15% Test (147) |
| **SETU In-House Synthetic** | Spatial Vectors | 4,000 | 2 (Rest / Noise) | N/A (Normalized vectors) | 100% Negative conditioning |
