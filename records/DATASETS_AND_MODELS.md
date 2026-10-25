# SETU — Machine Learning Registry: Datasets & Models

> **System:** SETU (Indian Sign Language Translator)  
> **Status:** Active ML Catalog & Training Logs  
> **Target Frameworks:** PyTorch 2.x, ONNX Runtime, NumPy, MediaPipe Tasks  
> **Last Updated:** October 2026  

---

## 1. Datasets Catalog & Provenance

| Dataset ID | Full Name | Source / Provider | Total Samples | Classes | Input Modality | Storage Location |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DS-REALSIGN` | RealSign ISL Alphabet Dataset | RealSign Open Research Corpus | ~31,200 frames | 26 (A–Z) | Dual-Hand RGB Images / 126-D Features | `data/raw/realsign/`, `data/processed/realsign_landmarks_26class.npz` |
| `DS-INCLUDE-MVP` | AI4Bharat INCLUDE (Subset) | AI4Bharat / IIT Madras (7 Deaf Schools) | 15 verified video sequences | 5 conversational signs | Video MP4 / 158-D Holistic Features | `data/raw/include/`, `data/processed/include_*.npz` |
| `DS-INCLUDE-263` | AI4Bharat INCLUDE (Full) | AI4Bharat Open Dataset | 4,287 videos | 263 ISL words | 25 FPS Video Clips | `models/pretrained/include_no_cnn_lstm.pth` |
| `DS-NO-SIGN` | SETU Background Negative Class | In-House Webcam Recordings | ~1,200 frames | 1 (`no_sign`) | Holistic 158-D Coordinates | `data/raw/real_no_sign/`, `data/processed/no_sign_*.npz` |
| `DS-CUSTOM` | Interactive Calibrated Takes | SETU Live Calibration Pipeline | Variable (5 takes/sign) | User-defined | Live Webcam Frames | `data/processed/custom_*.npz` |

### Detailed Dataset Manifests
1. **RealSign Alphabet (`data/manifests/realsign_clean_manifest.json`):**
   - Covers all 26 uppercase English letters expressed in genuine Indian Sign Language (ISL). Note that ISL utilizes two-handed gestures for multiple characters (e.g., 'B', 'D', 'P', 'Q') distinct from American Sign Language (ASL).
   - Strict signer-independent partitioning (70% train, 15% validation, 15% test).
2. **AI4Bharat INCLUDE (`data/manifests/setu_isl_mvp_manifest.json`):**
   - Standard benchmark recorded by native deaf signers across multiple Indian states.
   - Core conversational subset: `hello`, `thank_you`, `good_morning`, `i`, `you`.
3. **Negative Class (`no_sign`):**
   - Natural resting poses, hands on lap, face scratch, idle sitting. Critical for suppressing false-positive sign detections in conversational pauses.

---

## 2. Model Architectures & Version Checkpoints

### Model 1: RealSignMLP (26-Class Alphabet Classifier)
- **Version Checkpoint:** `v1.2-clean`
- **Weights Location:** 
  - Canonical Directory: `models/weights/`
  - PyTorch: `models/weights/realsign_26class_mlp.pth` (75.9 KB)
  - TorchScript: `models/weights/realsign_26class_mlp.pt` (75.5 KB)
  - ONNX Runtime: `models/weights/realsign_26class_mlp.onnx` (73.3 KB)
  - *(Backward-compatibility symlinks/aliases preserved in `models/` root)*
- **Input Dimension:** 126 features
  - Left Hand: 21 landmarks × (x, y, z) = 63 values (wrist-centered, scaled)
  - Right Hand: 21 landmarks × (x, y, z) = 63 values (wrist-centered, scaled)
- **Layer Architecture:**
  ```python
  Sequential(
      Linear(in_features=126, out_features=256),
      BatchNorm1d(256),
      ReLU(),
      Dropout(p=0.2),
      Linear(in_features=256, out_features=128),
      BatchNorm1d(128),
      ReLU(),
      Dropout(p=0.2),
      Linear(in_features=128, out_features=26)
  )
  ```
- **Inference Latency:** 1.8 ms on standard Intel/AMD CPU; 0.6 ms with ONNX Runtime.
- **Evaluation Metrics:**
  - Training Accuracy: 99.82%
  - Validation Accuracy: **99.11%**
  - Test Accuracy: 98.74%
  - Macro F1-Score: 0.988

---

### Model 2: ISLMVPClassifier (5-Sign Conversational Model)
- **Version Checkpoint:** `v1.0-mvp`
- **Weights Location:** `models/weights/isl_mvp_model.npz` (38.8 KB) *(alias in `models/isl_mvp_model.npz`)*
- **Inference Runtime:** Pure NumPy matrix multiplication (zero PyTorch dependency during inference)
- **Input Dimension:** 158 features
  - Upper body pose: 8 landmarks (shoulders, elbows, wrists, nose, torso center) × 4 = 32 values
  - Left hand: 21 landmarks × 3 = 63 values
  - Right hand: 21 landmarks × 3 = 63 values
- **Algorithm:** Linear Ridge Classifier with L2 Regularization ($\alpha = 1.0$)
- **Evaluation Metrics:**
  - Validation Accuracy: 80.0% (strict signer isolation)
  - Negative Class Rejection: 95.2% rejection of idle frames

---

### Model 3: ISLSequenceLSTM (Dynamic Gesture Model)
- **Version Checkpoint:** `v0.8-bilstm`
- **Weights Location:** `models/weights/isl_sequence_lstm.pth` (3.86 MB) *(alias in `models/isl_sequence_lstm.pth`)*
- **Input Dimension:** `(batch_size, sequence_length=15..30, features=158)`
- **Layer Architecture:**
  - Layer 1: Bidirectional LSTM (`input_size=158`, `hidden_size=128`, `batch_first=True`)
  - Layer 2: Bidirectional LSTM (`input_size=256`, `hidden_size=64`, `dropout=0.3`)
  - Linear Projection: `Linear(128, 64) -> ReLU() -> Linear(64, num_classes)`
- **Evaluation Metrics:**
  - Sequence Accuracy: 82.4% on 5-sign dynamic transitions

---

### Model 4: AI4Bharat INCLUDE Pretrained Sequence Model
- **Version Checkpoint:** `v1.0-pretrained-no-cnn`
- **Weights Location:** `models/pretrained/include_no_cnn_lstm.pth` (86.9 MB)
- **Labels:** `models/pretrained/label_map_include.json` (263 classes)
- **Task:** Full-word ISL recognition from uploaded video files.

---

### Vision Backbone: Google MediaPipe Task Bundles
- **Hand Landmarker:** `models/tasks/hand_landmarker.task` (7.8 MB) — High-precision hand skeleton extraction.
- **Holistic Landmarker:** `models/tasks/holistic_landmarker.task` (13.6 MB) — Full upper-body pose + dual-hand + face mesh.

---

## 3. Training Errors Encountered & Engineering Resolutions

### Error 1: Signer-Split Data Leakage in Initial RealSign Training
- **Symptom:** Validation accuracy reached an unrealistic 99.9% while real-world webcam tests performed poorly.
- **Root Cause:** Random shuffle split placed frames from the same video recording session into both train and validation sets, allowing the model to overfit on background lighting and signer clothing rather than hand shapes.
- **Resolution:** Authored `scripts/audit_realsign_leakage.py` and `scripts/clean_realsign_split.py`. Partitioned data strictly by unique signer ID. Retrained `RealSignMLP` to 99.11% on genuinely unseen signers.

### Error 2: ASL vs. ISL Dual-Hand Conflict
- **Symptom:** Alphabet models trained on standard datasets failed on ISL letters 'B', 'D', 'P', 'Q'.
- **Root Cause:** Standard open-source sign language datasets predominantly use American Sign Language (ASL), which is strictly single-handed. Indian Sign Language (ISL) requires both hands for several letters.
- **Resolution:** Re-engineered the feature extractor from single-hand (63-D) to dual-hand (126-D) with zero-padding when only one hand is active, allowing seamless single-hand and double-hand alphabet recognition.

### Error 3: OpenCV VideoCapture Hang on Windows Systems
- **Symptom:** Camera initialization stalled indefinitely or crashed during custom sign recording.
- **Root Cause:** Default Windows MSMF (Media Foundation) backend deadlocks when querying webcam hardware capabilities.
- **Resolution:** Enforced `cv2.CAP_DSHOW` (DirectShow) flag across all camera opening scripts (`cv2.VideoCapture(index, cv2.CAP_DSHOW)`).

### Error 4: False Positive Commits During Hand Return
- **Symptom:** As signers lowered their hands after completing a sign, intermediate hand shapes triggered false letters.
- **Root Cause:** Classification was evaluated on every single frame without temporal confirmation.
- **Resolution:** Implemented `StreakFilter` (requiring 2–3 consecutive identical predictions) followed by `CooldownGate` (350ms lock) in `setu/nlp/accumulator.py`.

---

## 4. Identified Biases & Structural Limitations

1. **Illumination & Background Sensitivity:** MediaPipe landmark detection reliability degrades under backlighting (< 50 lux) or when hands pass across patterned shirts matching skin tones.
2. **Two-Hand Occlusion:** When hands overlap directly in front of the chest, landmark depth ($z$-axis) coordinates can invert momentarily.
3. **Speed of Articulation:** Extremely rapid fingerspelling (> 4 signs per second) exceeds the 15–20 FPS webcam sampling rate, leading to skipped letters.
4. **Signer Body Proportions:** Wrist-centered scaling normalizes hand size well, but arm reach variations can affect upper-body pose normalization in 158-D sequence models.

---

## 5. Hyperparameter Evolution & Iteration History

| Iteration | Model | Epochs | Batch Size | Optimizer | Learning Rate | Train Loss | Val Loss | Val Acc | Key Adjustments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iter-01** | RealSign MLP | 30 | 128 | SGD (momentum=0.9) | 0.01 | 0.452 | 0.412 | 88.4% | Baseline single-hand 63-D architecture |
| **Iter-02** | RealSign MLP | 50 | 64 | Adam | 0.001 | 0.118 | 0.145 | 96.2% | Switched to dual-hand 126-D features + BatchNorm |
| **Iter-03** | RealSign MLP | 60 | 64 | AdamW (decay=1e-4) | 0.001 | 0.032 | 0.041 | **99.11%** | Added Dropout(0.2), leak-free signer split, ONNX export |
| **Iter-04** | ISL MVP Ridge | N/A | Full | Closed-form Ridge | $\alpha=1.0$ | N/A | N/A | **80.0%** | Pure NumPy weights matrix for instant zero-dependency CPU inference |
| **Iter-05** | Sequence BiLSTM | 40 | 32 | Adam | 0.0005 | 0.241 | 0.289 | **82.4%** | 2-layer BiLSTM on 15-frame rolling window |
