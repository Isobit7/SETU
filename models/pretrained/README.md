# Pretrained Experimental Models Directory

This directory contains external pretrained models integrated for experimental evaluation and benchmarking.

> **IMPORTANT**: Artifacts in this directory are strictly separated from SETU's canonical 158-feature two-handed ISL recognition pipeline (`setu/features/`).

---

## 1. RF-UV-11 Pretrained Model (`rf_uv_11_model.h5`)

### Source Attribution
- **Upstream Repository**: [RF-UV-11/Indian-Sign-Language-Recognition-MediaPipe](https://github.com/RF-UV-11/Indian-Sign-Language-Recognition-MediaPipe)
- **Original Filename**: `model/indian_sign_language_classification_model.h5`
  - *Historical Commit Evidence*: Originally committed as `model/hand_gesture_classification_model.h5` and renamed to `indian_sign_language_classification_model.h5` on 2024-08-30.
- **License**: Apache License 2.0 (see [LICENSE_APACHE_2.0.txt](file:///c:/MINI%20PROJ-SETU/models/pretrained/LICENSE_APACHE_2.0.txt))
  - **Permissions**: Commercial use, modification, distribution, private use.
  - **Conditions**: Preserve copyright notice, state changes, and include license terms.

### Model Architecture & Technical Contract
- **File Format**: Keras HDF5 (`.h5`), weight matrices and biases stored under `model_weights`.
- **Parameter Count**: 18,138 float32 parameters (~248 KB).
- **Layers**:
  1. Dense Layer 1: Input dimension `(batch, 63)` -> Output dimension `(batch, 128)`, Activation: `ReLU`
  2. Dense Layer 2: Input dimension `(batch, 128)` -> Output dimension `(batch, 64)`, Activation: `ReLU`
  3. Dense Layer 3: Input dimension `(batch, 64)` -> Output dimension `(batch, 26)`, Activation: `Softmax`
- **Input Contract**:
  - Exactly 63 features: 21 landmarks $\times$ 3 coordinates `[x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]`.
  - Coordinates represent normalized image positions `(x, y)` in `[0, 1]` and relative depth `z`.
  - Un-normalized, single-hand coordinate space.
- **Output Classes**:
  - 26 classes corresponding to the Latin alphabet: `A` through `Z` (`list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')`).

### Isolation from SETU Pipeline
- SETU's primary architecture uses a 158-feature vector capturing two-handed landmarks, inter-hand metrics (wrist-to-wrist, index-to-index, contact score), and scale-invariant normalization.
- This pretrained model operates exclusively through the lightweight adapter in `setu/models/rf_uv11_adapter.py` and does NOT modify or substitute the 158-feature schema.
