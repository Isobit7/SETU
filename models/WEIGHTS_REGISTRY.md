# Model Weights Registry & Chronological Version Catalog
## Project SETU: Multi-Model Indian Sign Language Recognition Engine

*This registry documents all active production weights and historical baseline weights in the repository. Use this table to understand which weights are currently powering the engine, when they were trained, and what accuracy they achieve.*

---

### Master Weights Summary Table

| Weight File | Format | Trained Date | Status | Architecture | Target Classes | Test Accuracy | Primary Code Consumer |
|---|---|---|---|---|---|---|---|
| **`amrita_isl_phrase_bilstm.pth`** | PyTorch (`.pth`) | **2026-11-06** | **ACTIVE (Current)** | 2-Layer BiLSTM (Hidden: 128) | 100 Continuous Phrases | **94.2%** | `backend/app/services/classifier.py` |
| **`realsign_26class_mlp.pth`** | PyTorch (`.pth`) | **2026-11-06** | **ACTIVE (Current)** | 4-Layer Dense MLP (63-dim input) | 26 ISL Alphabets (A–Z) | **96.8%** | `models/adapters/realsign.py` |
| **`realsign_multisigner.onnx`** | ONNX (`.onnx`) | **2026-11-06** | **ACTIVE (Current)** | Quantized Dense Net (63-dim) | 26 ISL Alphabets (Multi-signer) | **96.2%** | `backend/app/services/realsign_classifier_service.py` |
| **`isl_hindi_vowels.onnx`** | ONNX (`.onnx`) | **2026-11-06** | **ACTIVE (Current)** | 3-Layer Dense Net (63-dim) | 11 Hindi Vowels (अ to औ) | **98.1%** | `backend/app/services/classifier.py` |
| **`isl_numerals.onnx`** | ONNX (`.onnx`) | **2026-11-06** | **ACTIVE (Current)** | 3-Layer Dense Net (63-dim) | 10 ISL Digits (0–9) | **97.4%** | `models/adapters/digits.py` |
| **`isl_hindi_vowels_mlp.pth`** | PyTorch (`.pth`) | **2026-11-06** | **ACTIVE (Training)** | PyTorch source for ONNX export | 11 Hindi Vowels | **98.1%** | `training/train_cloud_high_accuracy.py` |
| **`isl_numerals_mlp.pth`** | PyTorch (`.pth`) | **2026-11-06** | **ACTIVE (Training)** | PyTorch source for ONNX export | 10 ISL Digits | **97.4%** | `training/train_cloud_high_accuracy.py` |
| `realsign_26class_mlp.pth.bak` | PyTorch (`.pth`) | 2026-10-30 | *PREVIOUS (Backup)* | 3-Layer MLP | 26 Alphabets (Single Signer) | 88.4% | Kept for regression validation |
| `isl_sequence_lstm.pth` | PyTorch (`.pth`) | 2026-10-28 | *PREVIOUS (Baseline)* | 1-Layer Unidirectional LSTM | 25 Common Signs | 82.1% | Superseded by Amrita BiLSTM |
| `hf_isl_26class_mlp.pth` | PyTorch (`.pth`) | 2026-10-26 | *PREVIOUS (Legacy)* | 2-Layer MLP (HuggingFace data) | 26 Alphabets | 76.5% | Replaced by RealSign 63D |
| `isl_mvp_model.npz` | NumPy (`.npz`) | 2026-10-26 | *PREVIOUS (Prototype)* | Ridge / Logistic Regression | 5 Basic Signs | 68.0% | Initial concept prototype |
| `yolov8s-worldv2.pt` | Ultralytics (`.pt`)| 2026-10-29 | **ACTIVE (Auxiliary)** | YOLO-World Open-Vocab Vision | Contextual Objects & Props | mAP 52.4 | `backend/app/services/vision/object_detector.py` |

---

### Detailed Model Architecture & File Descriptions

#### 1. Current Active Weights (Released 2026-11-06)

##### `amrita_isl_phrase_bilstm.pth`
- **Location:** `models/weights/amrita_isl_phrase_bilstm.pth`
- **Release Date:** November 6, 2026 (14:47:09)
- **Size:** 3.64 MB
- **Input Dimension:** Shape $(B, 30, 158)$ where $B$ is batch size, 30 is the temporal sequence window (1.0 second at 30 FPS), and 158 is the concatenated Holistic feature vector:
  - 33 Pose landmarks $(x, y, z)$
  - 21 Dominant hand landmarks $(x, y, z)$
  - 21 Non-dominant hand landmarks $(x, y, z)$
  - 21 Inter-hand joint distance scalars
- **Hidden Layers:** 2-layer Bidirectional LSTM, hidden size 128 (effective 256 per time-step), dropout 0.3, followed by a fully-connected layer to 100 classes.
- **Dataset:** Amrita Indian Sign Language Continuous Phrase Dataset (100 distinct vocabulary phrases with multi-signer co-articulation).
- **Performance:** 94.2% test accuracy, 0.938 macro F1-score.

##### `realsign_26class_mlp.pth` & `realsign_multisigner.onnx`
- **Location:** `models/weights/realsign_26class_mlp.pth` & `models/weights/realsign_multisigner.onnx`
- **Release Date:** November 6, 2026 (15:19:58)
- **Size:** 114 KB (`.pth`), 109 KB (`.onnx`)
- **Input Dimension:** 63-dimensional normalized hand vector:
  $$\vec{v} = [x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$$
  where $(x_0, y_0, z_0) = (0, 0, 0)$ (wrist-centered) and scaled by $1 / ||L_9 - L_0||_2$.
- **Architecture:** Linear(63, 256) $\to$ BatchNorm $\to$ ReLU $\to$ Dropout(0.25) $\to$ Linear(256, 128) $\to$ BatchNorm $\to$ ReLU $\to$ Linear(128, 64) $\to$ ReLU $\to$ Linear(64, 26).
- **Target Classes:** English alphabets A through Z in Indian Sign Language.
- **Performance:** 96.8% test accuracy across unseen signer hands.

##### `isl_hindi_vowels.onnx` & `isl_hindi_vowels_mlp.pth`
- **Location:** `models/weights/isl_hindi_vowels.onnx`
- **Release Date:** November 6, 2026 (15:26:15)
- **Size:** 105 KB
- **Input Dimension:** 63-dim normalized hand landmarks.
- **Target Classes:** 11 Hindi vowels (अ, आ, इ, ई, उ, ऊ, ऋ, ए, ऐ, ओ, औ).
- **Performance:** 98.1% test accuracy.

##### `isl_numerals.onnx` & `isl_numerals_mlp.pth`
- **Location:** `models/weights/isl_numerals.onnx`
- **Release Date:** November 6, 2026 (15:25:07)
- **Size:** 105 KB
- **Input Dimension:** 63-dim normalized hand landmarks.
- **Target Classes:** Digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 in ISL.
- **Performance:** 97.4% test accuracy.

---

### 2. Historical & Previous Weights (Archive)

##### `realsign_26class_mlp.pth.bak` (Trained: 2026-10-30)
- **Size:** 116 KB
- **Why archived:** Trained on single-signer dataset. Suffered accuracy degradation when tested on signers with different finger proportions (PROB-10). Replaced by multi-signer augmented version on Nov 6.

##### `isl_sequence_lstm.pth` (Trained: 2026-10-28)
- **Size:** 3.86 MB
- **Why archived:** Unidirectional 1-layer LSTM trained on early 25-sign prototype. Lacked reverse temporal context and inter-hand distance features. Replaced by the 100-phrase BiLSTM on Nov 6.

##### `hf_isl_26class_mlp.pth` (Trained: 2026-10-26)
- **Size:** 114 KB
- **Why archived:** Trained on uncleaned HuggingFace community landmark dataset with significant noise in z-depth and incorrect handedness mirroring. Replaced by verified RealSign dataset.

##### `isl_mvp_model.npz` (Trained: 2026-10-26)
- **Size:** 38 KB
- **Why archived:** Linear Ridge regression model used for Day 1 proof-of-concept. Replaced by deep neural MLP.

