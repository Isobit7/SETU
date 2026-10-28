# Pretrained Models Directory

This directory contains external reference and benchmark models integrated for experimental evaluation against SETU's native architecture.

> **CRITICAL ARCHITECTURAL ISOLATION**: Artifacts in this directory are strictly separated from SETU's native 158-feature holistic recognition pipeline (`training/features/`). They operate via dedicated adapters in `models/adapters/`.

---

## 1. RF-UV-11 Single-Hand Baseline (`rf_uv_11_model.h5` & `rf_uv11_weights.npz`)

### Source Attribution
- **Upstream Repository**: [RF-UV-11/Indian-Sign-Language-Recognition-MediaPipe](https://github.com/RF-UV-11/Indian-Sign-Language-Recognition-MediaPipe)
- **Original Filename**: `model/indian_sign_language_classification_model.h5`
- **License**: Apache License 2.0 (see [LICENSE_APACHE_2.0.txt](file:///c:/MINI%20PROJ-SETU/models/pretrained/LICENSE_APACHE_2.0.txt))
  - **Permissions**: Commercial use, modification, distribution, private use.
  - **Conditions**: Preserve copyright notice, state changes, and include license terms.

### Model Specification

| Parameter | Value |
| :--- | :--- |
| **File Format** | Keras HDF5 (`.h5`) & Extracted NumPy matrices (`.npz`) |
| **Parameter Count** | 18,138 float32 parameters (~248 KB) |
| **Input Feature Vector** | 63-D Single Hand Coordinates: $21 \text{ landmarks} \times 3 \text{ coords } (x, y, z)$ |
| **Network Architecture** | Dense(63 $\to$ 128, ReLU) $\to$ Dense(128 $\to$ 64, ReLU) $\to$ Dense(64 $\to$ 26, Softmax) |
| **Output Classes** | 26 Latin Alphabet letters (`A` through `Z`) |
| **Forward Adapter** | `models/adapters/rf_uv11.py` (`RFUV11Adapter`) |
| **Inference Engine** | NumPy Matrix Multiplication (zero-dependency, ~0.9 ms CPU) |

---

## 2. AI4Bharat INCLUDE Pretrained Sequence Model (`include_no_cnn_lstm.pth`)

### Source Attribution
- **Dataset / Provider**: AI4Bharat / IIT Madras (INCLUDE Open Research Corpus)
- **Primary Source**: 4,287 sign videos recorded across 7 Indian deaf schools.
- **License**: Open Research Dataset (Academic / Research Non-Commercial).

### Model Specification

| Parameter | Value |
| :--- | :--- |
| **File Format** | PyTorch Checkpoint (`.pth`) |
| **Target Vocabulary** | 263 ISL Conversational Signs and Words |
| **Input Shape** | 158-D Holistic Coordinate Sequences &times; Variable Frame Length |
| **Network Architecture** | 2-Layer Bidirectional LSTM with Temporal Pooling |
| **Forward Adapter** | `models/adapters/include.py` (`INCLUDEAdapter`) |
| **Primary Purpose** | Asynchronous full-video gesture classification via `/api/v1/video/upload` |

---

## 3. Pipeline Isolation Guarantee

1. **Feature Vector Independence:** SETU's canonical real-time pipeline uses a normalized 126-D dual-hand vector (RealSign) and a 158-D holistic vector (ISL-MVP). The 63-D RF-UV11 model operates through an explicit slice adapter and does not contaminate the canonical feature space.
2. **Zero Global State:** Model weights in this directory are loaded lazily and immutably into `models/registry.py` with thread-safe singleton caching.
