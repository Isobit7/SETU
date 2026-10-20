# SETU — Real-Time Indian Sign Language (ISL) Recognition SaaS

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)

> **SETU** ("Bridge" in Sanskrit) is an end-to-end, real-time Indian Sign Language (ISL) translation platform. It bridges communication gaps by interpreting static fingerspelling (A–Z) and dynamic multi-frame phrase sequences into fluent text using computer vision and deep learning.

---

## 🌟 Key Features

| Feature | Description | Tech Stack |
| :--- | :--- | :--- |
| **Real-Time Streaming** | 30 FPS non-blocking WebSocket stream processing with offscreen frame downsampling ($320 \times 240$) | FastAPI Async, OpenCV |
| **A–Z Fingerspelling** | Instant single-frame static letter recognition with wrist-centered palm normalization | 63-D MLP PyTorch Classifier |
| **Dynamic ISL Sequences** | 328-D spatiotemporal sequence modeling over 30-frame rolling windows | Attention-based BiLSTM |
| **Non-Manual Markers (NMM)** | 6-D scale-invariant facial ratio extraction (Eyebrow Elevation, MAR, EAR) for grammatical context | MediaPipe Face Mesh |
| **Noise Filtering** | Temporal debouncing, confidence hysteresis gating, and refractory cooldown | Custom Post-Processing Pipeline |
| **In-App Dataset Recorder** | Interactive CLI and API endpoints to record custom video samples and retrain signs | OpenCV HUD, NumPy `.npz` |

---

## 📐 System Architecture

### Architectural Flowchart

```
                 ┌────────────────────────────────┐
                 │  Client Camera Stream (HD)     │
                 └──────────────┬─────────────────┘
                                │
                                ▼
                 ┌────────────────────────────────┐
                 │ Offscreen Downsample (320x240) │
                 └──────────────┬─────────────────┘
                                │ WebSocket (JPEG binary)
                                ▼
                 ┌────────────────────────────────┐
                 │ FastAPI WebSocket Receiver     │
                 │ (Asynchronous Thread Offload)  │
                 └──────────────┬─────────────────┘
                                │
                                ▼
                 ┌────────────────────────────────┐
                 │ MediaPipe Landmarker Pipeline  │
                 │ Holistic (Pose, Face) + Hands  │
                 └──────────────┬─────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
┌───────────────────────────────┐     ┌─────────────────────────────────┐
│ Alphabet Mode (A–Z)           │     │ Sequence / Phrase Mode          │
│ Dedicated 63-D Hand Vector    │     │ 164-D Spatial + 164-D Velocity  │
│ - Wrist origin translation    │     │ - 158-D Holistic coordinates    │
│ - Palm-scale normalization    │     │ - 6-D Facial NMM ratios         │
│ - Zero sequence buffer delay  │     │ - 30-frame rolling deque        │
└───────────────┬───────────────┘     └─────────────────┬───────────────┘
                │                                       │
                ▼                                       ▼
┌───────────────────────────────┐     ┌─────────────────────────────────┐
│ RealSign 26-Class PyTorch MLP │     │ Attention-based BiLSTM (PyTorch)│
│ Single-frame inference        │     │ Spatiotemporal 328-D modeling   │
└───────────────┬───────────────┘     └─────────────────┬───────────────┘
                │                                       │
                └───────────────────────┬───────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────┐
                 │ Post-Processing & Temporal Stability │
                 │ - Confidence Hysteresis (P >= 0.35)  │
                 │ - Consecutive Streak Verification    │
                 │ - Refractory Cooldown (350ms)        │
                 │ - Idle Baseline Transition Gate      │
                 └──────────────────────┬───────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────┐
                 │ React 18 Glassmorphic UI             │
                 │ - Real-Time Movie Subtitle HUD       │
                 │ - MediaPipe Joint Skeleton Overlay   │
                 │ - Camera Framing & Distance Guidance │
                 │ - Web Speech API Text-to-Speech      │
                 └──────────────────────────────────────┘
```

---

### 1. Spatiotemporal Feature Vector (328-D)

To capture multi-joint bodily movement and temporal inflection without relying on pixel appearance, SETU extracts a **328-dimensional spatiotemporal representation**:

$$\mathbf{F}_{328} = \begin{bmatrix} \mathbf{x}_t^{(164)} & \Delta \mathbf{x}_t^{(164)} \end{bmatrix} \in \mathbb{R}^{328}$$

Where:
- **$\mathbf{x}_t^{(164)}$ (Spatial Component):** 158 Holistic coordinates and inter-hand relations $+$ 6 Facial Non-Manual Marker ratios.
- **$\Delta \mathbf{x}_t^{(164)}$ (Velocity Component):** First-order finite differences $\mathbf{x}_t - \mathbf{x}_{t-1}$ representing the joint velocity and signing acceleration.

#### Dimensional Partition:
```
Index Range   | Dimension | Feature Category
--------------|-----------|---------------------------------------------------------
[0 : 63]      | 63-D      | Right Hand: 21 points (x, y, z) wrist-centered & normalized
[63 : 68]     | 5-D       | Right Hand: Fingertip-to-wrist Euclidean distances
[68 : 73]     | 5-D       | Right Hand: Inter-finger adjacent spread angles
[73 : 136]    | 63-D      | Left Hand: 21 points (x, y, z) wrist-centered & normalized
[136 : 141]   | 5-D       | Left Hand: Fingertip-to-wrist Euclidean distances
[141 : 146]   | 5-D       | Left Hand: Inter-finger adjacent spread angles
[146 : 148]   | 2-D       | Hand Presence Flags (Right present, Left present)
[148 : 151]   | 3-D       | Inter-wrist 3D spatial vector
[151]         | 1-D       | Inter-wrist Euclidean distance
[152 : 155]   | 3-D       | Inter-index fingertip 3D spatial vector
[155]         | 1-D       | Inter-index fingertip Euclidean distance
[156]         | 1-D       | Inter-thumb fingertip Euclidean distance
[157]         | 1-D       | Geometric Hand Contact Score [0.0 - 1.0]
[158 : 164]   | 6-D       | Facial Non-Manual Marker (NMM) Ratios
[164 : 328]   | 164-D     | First-Order Temporal Velocity Features (Δx_t = x_t - x_{t-1})
```

---

### 2. Static A–Z Hand Feature Normalization (63-D)

RealSign alphabet fingerspelling processes isolated hands per frame with zero sequence delay. To maintain invariance against camera distance, hand orientation, and subject positioning:

1. **Wrist Origin Centering:** The wrist coordinate ($P_0$) is subtracted from all 21 hand keypoints:
   $$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0 \quad \forall i \in \{0, \dots, 20\}$$
2. **Scale Invariance:** Normalized coordinates are divided by the subject's palm size (distance from wrist $P_0$ to middle finger metacarpophalangeal joint $P_9$):
   $$S_{\text{palm}} = \max\left(\|\mathbf{p}_9 - \mathbf{p}_0\|_2, \, 10^{-6}\right)$$
   $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}'_i}{S_{\text{palm}}}$$
3. **Vector Flattening:** Flattens $21 \times 3$ into a single 63-D float vector feeding directly into the PyTorch RealSign MLP.

---

### 3. MediaPipe Keypoint Topology & Facial Ratios

#### Hand Landmark Indices (21 Points)
```
          8 (INDEX_TIP)    12 (MID_TIP)   16 (RING_TIP)   20 (PINKY_TIP)
          │                │              │               │
          7 (INDEX_DIP)    11 (MID_DIP)   15 (RING_DIP)   19 (PINKY_DIP)
          │                │              │               │
          6 (INDEX_PIP)    10 (MID_PIP)   14 (RING_PIP)   18 (PINKY_PIP)
4 (THUMB) │                │              │               │
    \     5 (INDEX_MCP)────9 (MID_MCP)───13 (RING_MCP)───17 (PINKY_MCP)
     3 (IP)         \      │             /
      \              \     │            /
       2 (MCP)────────0 (WRIST)────────┘
        \            /
         1 (CMC)────┘
```

#### Facial Non-Manual Marker (NMM) Ratios (6-D)
Facial expressions provide vital grammatical cues (questions, negations, emphasis) in Indian Sign Language:

1. **Mouth Aspect Ratio (MAR):** Measures mouth opening / shape:
   $$\text{MAR} = \frac{\|P_{13} - P_{14}\|_2}{\max(\|P_{61} - P_{291}\|_2, \, 10^{-6})}$$
2. **Left Eyebrow Elevation Ratio (LEER):**
   $$\text{LEER} = \frac{\|P_{105} - P_{159}\|_2}{\max(\|P_{10} - P_{152}\|_2, \, 10^{-6})}$$
3. **Right Eyebrow Elevation Ratio (REER):**
   $$\text{REER} = \frac{\|P_{334} - P_{386}\|_2}{\max(\|P_{10} - P_{152}\|_2, \, 10^{-6})}$$
4. **Mean Eyebrow Elevation Ratio (MEER):** $\frac{\text{LEER} + \text{REER}}{2}$
5. **Left Eye Aspect Ratio (LEAR):**
   $$\text{LEAR} = \frac{\|P_{160} - P_{144}\|_2 + \|P_{158} - P_{153}\|_2}{2 \cdot \max(\|P_{33} - P_{133}\|_2, \, 10^{-6})}$$
6. **Right Eye Aspect Ratio (REAR):**
   $$\text{REAR} = \frac{\|P_{385} - P_{380}\|_2 + \|P_{387} - P_{373}\|_2}{2 \cdot \max(\|P_{362} - P_{263}\|_2, \, 10^{-6})}$$

---

## 🛡️ Post-Processing & Temporal Stability Pipeline

To eliminate jitter, runaway text duplication, and transient hand transition artifacts, SETU applies a 4-tier filtering pipeline:

1. **Confidence Hysteresis Gate:** Softmax probabilities must satisfy $P \ge 0.35$ (dynamically adjustable via the client UI slider). Sub-threshold predictions are mapped to `"Awaiting Sign Input"`.
2. **Consecutive Frame Streak Filter:**
   - **Alphabet Mode (A–Z):** Requires model prediction consistency across **2 consecutive frames** (~66ms at 30 FPS).
   - **Sequence Mode (Phrases):** Requires consistency across **3 consecutive frames** (~100ms).
3. **Refractory Cooldown & Idle Baseline:**
   - Enforces a **350ms cooldown** before allowing the same character or sign to register again.
   - Requires the user's hand to return through an idle resting baseline (`no_sign` or hand relaxed) before accepting consecutive duplicates.
4. **Positioning & Subtitle Collision Guidance:**
   - Monitors shoulder width and nose position: guides users if they are "Too close", "Too far", or if camera tilt needs adjustment.
   - Alerts signers if hands collide with the lower viewport subtitle area ($y \in [0.72, 0.98]$).

---

## 💻 Tech Stack & Dependencies

```
SETU Platform
├── Backend
│   ├── Python 3.12 (Virtualenv)
│   ├── FastAPI & Uvicorn (Asynchronous REST & WebSocket server)
│   ├── PyTorch 2.2+ (CPU & CUDA accelerated neural network execution)
│   ├── MediaPipe 0.10+ (Holistic and HandLandmarker task pipelines)
│   └── OpenCV (imdecode, colorspace transformations, video capture)
│
└── Frontend
    ├── React 18 & TypeScript 5
    ├── Vite 6 (Lightning-fast HMR and production bundle optimization)
    ├── Tailwind CSS 3.4 (Custom glassmorphic HUD styling)
    └── Lucide React (Clean semantic iconography)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python:** 3.10 to 3.12
- **Node.js:** 18+ and `npm`
- **Hardware:** Standard USB or integrated webcam

---

### Quick Launch (Windows)

Launch both backend and frontend servers simultaneously using the automated launcher:
```powershell
.\Start_SETU.bat
```
To run the standalone webcam alphabet demonstration:
```powershell
.\Start_Alphabet_Demo.bat
```

---

### Manual Setup & Execution

#### 1. Backend Setup
```powershell
# 1. Clone repository
git clone https://github.com/Isobit7/SETU.git
cd SETU

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install packages in editable mode
pip install -e .

# 4. Verify test suite (87 tests)
pytest -v

# 5. Start FastAPI WebSocket server
uvicorn setu.api.server:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```powershell
# In a separate terminal:
cd frontend

# Install npm dependencies
npm install

# Start Vite development server
npm run dev
```

Navigate to **`http://localhost:5173`** in your browser.

---

## 📹 Dataset Recording & Model Training CLI

### Record Custom Sign Samples
Record a 30-frame sequence ($30 \times 164$-D) directly with visual countdowns and bounding HUD:
```powershell
.\.venv\Scripts\python scripts/record_custom_signs.py --label namaste --takes 10
```
Files are compressed and saved to `data/processed/custom_<label>_<timestamp>.npz`.

### Train Sequence LSTM Classifier
Trains the Attention-based Bidirectional LSTM on balanced synthetic, benchmark (AI4Bharat INCLUDE), and custom datasets:
```powershell
.\.venv\Scripts\python training/train_sequence_lstm.py --epochs 60 --lr 0.001
```

### Standalone Webcam Demos
- **A–Z RealSign Alphabet:**
  ```powershell
  .\.venv\Scripts\python scripts/demo_realsign_26class_webcam.py --camera 0
  ```
- **ISL MVP 5-Sign Sequence:**
  ```powershell
  .\.venv\Scripts\python scripts/demo_isl_mvp_webcam.py --camera 0
  ```

---

## 📡 REST & WebSocket API Specification

### WebSocket Recognition Stream
- **URL:** `ws://localhost:8000/api/v1/ws/recognize?model=realsign&threshold=0.35`
- **Supported Models:** `realsign` (Alphabet), `isl_sequence_lstm` (Phrases), `isl_mvp` (5-Sign Baseline)
- **Client Inbound:**
  * Binary JPEG image buffer (scaled to $320 \times 240$).
  * JSON commands: `{"action": "set_model", "model": "realsign"}`, `{"action": "set_threshold", "threshold": 0.5}`, `{"action": "reset"}`.
- **Server Outbound:**
  ```json
  {
    "type": "prediction",
    "active_model": "realsign",
    "label": "A",
    "display_name": "LETTER A",
    "confidence": 0.8924,
    "is_known_sign": true,
    "is_stable": true,
    "streak_count": 2,
    "required_streak": 2,
    "hand_detected": true,
    "hands_in_subtitle_zone": false,
    "guidance": "Position good — Ready to translate",
    "fps": 28.5,
    "latency_ms": 14.2,
    "landmarks": { "left_hand": [], "right_hand": [...], "pose": [...], "face": [...] },
    "realsign_prediction": {
      "letter": "A",
      "confidence": 0.8924,
      "top_3": [["A", 0.8924], ["S", 0.0512], ["E", 0.0211]]
    }
  }
  ```

### Key REST Endpoints
- `GET /api/v1/models`: Catalog of active and available AI model services.
- `GET /api/v1/models/realsign/classes`: Class list of 26 alphabet letters.
- `GET /api/v1/vocabulary`: Complete ISL phrase vocabulary with execution descriptions.
- `POST /api/v1/dataset/record_sample`: Upload and persist a $30 \times 164$ feature array for a custom sign.
- `GET /api/v1/dataset/custom_samples`: List all saved custom sign samples.

---

## 🧪 Test Suite & Invariant Verification

SETU maintains a rigorous test suite verifying algorithmic invariants, mathematical scaling, and API contracts:

```powershell
.\.venv\Scripts\pytest.exe -v
```

```
============================== test session starts ==============================
collected 87 items

tests/test_api.py::test_health_endpoint PASSED                           [  1%]
tests/test_api.py::test_websocket_recognize_mock_frame PASSED            [  8%]
tests/test_feature_schema.py::test_facial_expression_ratio_math PASSED   [ 34%]
tests/test_feature_schema.py::test_expanded_164_feature_vector_validation PASSED [ 35%]
tests/test_feature_schema.py::test_velocity_features_328_computation PASSED [ 36%]
tests/test_isl_sequence_lstm_classifier.py::test_classifier_predict_sequence_shape_30x328 PASSED [ 66%]
tests/test_real_data_readiness.py::test_strict_signer_isolation_and_no_leakage PASSED [ 75%]
tests/test_realsign_classifier.py::test_extract_realsign_hand_features PASSED [ 81%]
tests/test_realsign_classifier.py::test_realsign_classifier_predict_direct PASSED [ 82%]
tests/test_vision.py::test_holistic_landmarker_pipeline_with_local_task PASSED [100%]

======================= 87 passed, 1 warning in 25.84s =======================
```

---

## 📂 Repository Directory Structure

```
SETU/
├── configs/                     # Application configurations and constants
├── data/
│   ├── manifests/               # Signer-isolated train/val/test manifests
│   └── processed/               # Preprocessed feature arrays (.npz)
├── docs/                        # Technical specifications & milestone reports
├── frontend/                    # Vite + React + TypeScript + Tailwind frontend
│   ├── src/
│   │   ├── components/          # Viewport, Translation HUD, Settings modals
│   │   ├── hooks/               # useRecognitionStream WebSocket hook
│   │   ├── types.ts             # Data contracts and API schemas
│   │   └── utils/               # Canvas landmark overlay renderer
│   └── package.json
├── models/                      # Serialized PyTorch weights & label mappings
│   ├── isl_sequence_lstm.pth    # Attention-based BiLSTM weights (3.8 MB)
│   ├── realsign_26class_mlp.pth # RealSign 26-Class Alphabet weights (76 KB)
│   └── label_map.json           # Class label dictionary
├── Project Main Files/          # Product Requirements, PRD, and Architecture docs
├── scripts/                     # Custom recorder CLI & webcam demo runners
├── setu/                        # Core Python application package
│   ├── api/                     # FastAPI REST & WebSocket streaming server
│   ├── features/                # 158-D / 164-D / 328-D / 63-D feature extractors
│   ├── models/                  # PyTorch model definitions & classifiers
│   ├── utils/                   # Geometry, distance, angle calculation helpers
│   └── vision/                  # MediaPipe Holistic & Hand landmarker wrappers
├── tests/                       # 87 automated pytest test suites
├── training/                    # Sequence LSTM and RealSign MLP training scripts
├── pyproject.toml               # Python package configuration
├── requirements.txt             # Pip dependencies
└── README.md                    # Primary technical documentation
```

---

## 📄 License & Attribution

- **License:** MIT License.
- **Dataset:** Built with references and subsets from the **AI4Bharat INCLUDE** dataset, licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Vision Models:** MediaPipe Holistic and HandLandmarker by Google LLC (Apache 2.0).
