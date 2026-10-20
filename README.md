# SETU - Real-Time Indian Sign Language (ISL) Recognition SaaS

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-0078D4.svg)](https://developers.google.com/mediapipe)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5+-3178C6.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6+-646CFF.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-87%20Passed-success.svg)]()

> **SETU** (*"Bridge"* in Sanskrit & Hindi) is a production-grade, real-time Indian Sign Language (ISL) recognition and translation platform designed to enable seamless, bidirectional communication between the Deaf community and non-signers.

---

## 🌟 Overview

SETU translates continuous ISL phrases and single-hand fingerspelling into speech and text with sub-100ms latency. The system combines:
1. **Low-Latency Streaming:** WebSocket communication delivering continuous live landmark predictions at 25–30 FPS.
2. **328-D Spatiotemporal Sequence Modeling:** Dynamic bidirectional LSTM with attention for word-level gestures and continuous phrases.
3. **Dedicated Single-Frame Alphabet Recognition:** 63-D wrist-normalized PyTorch MLP for instantaneous A–Z ISL fingerspelling.
4. **Non-Manual Markers (NMMs):** 6-D scale-invariant facial landmark ratios (mouth opening, eyebrow elevation, and eye aspect ratios) capturing grammatical inflections and emotional intent.
5. **Robust Post-Processing:** Confidence hysteresis gating, temporal streak verification, and refractory cooldown to eliminate flicker and runaway duplicate text.

---

## 🛠️ Tech Stack

### Backend
- **Framework:** FastAPI, Uvicorn (ASGI)
- **Deep Learning:** PyTorch (CUDA / CPU acceleration)
- **Computer Vision:** MediaPipe Holistic (Face Mesh, Pose, Hands) & MediaPipe HandLandmarker, OpenCV
- **Numerical Computing:** NumPy, SciPy

### Frontend
- **Core:** React 18, TypeScript
- **Bundler & Tooling:** Vite
- **Styling:** Vanilla CSS & Tailwind CSS, Glassmorphic UI design system
- **Icons:** Lucide React

---

## 🧠 AI / ML Architecture

```
[Webcam Stream 640x480]
          │
          ▼ (Offscreen Downsampling 320x240)
[WebSocket Transmission]
          │
          ▼
[MediaPipe Extraction (Holistic + Hands)]
          ├─────────────────────────────────────────────────┐
          ▼                                                 ▼
[Alphabet Mode (A–Z)]                            [Sequence / Phrase Mode]
Wrist-Centered 63-D Hand Vector                  164-D Holistic + 6-D Facial NMM
- Wrist at origin (0, 0, 0)                      - 158-D Hand & Body Keypoints
- Normalized by palm size (Wrist to MCP 9)       - 6-D Facial Expression Ratios
          │                                                 │
          ▼                                                 ▼
[RealSign 26-Class PyTorch MLP]                  [Temporal Velocity Derivation]
Instantaneous single-frame evaluation            - 30-Frame Sequence Buffer
          │                                      - 164 Position + 164 Velocity = 328-D
          │                                                 │
          │                                                 ▼
          │                                      [Attention Bidirectional LSTM]
          └───────────────────────┬─────────────────────────┘
                                  ▼
         [Post-Processing & Temporal Stability Filter]
         - Confidence Hysteresis Gate (P >= 0.35)
         - Consecutive Frame Streak Counter (2 frames for A-Z, 3 for Sequence)
         - Refractory Cooldown (350ms duplicate suppression)
                                  │
                                  ▼
      [Movie-Style Subtitle HUD & Text-to-Speech Engine]
```

### 1. Spatiotemporal 328-D Feature Vector
- **158-D Holistic Geometry:** Wrist-centered 3D coordinates, finger extension angles, tip-to-wrist euclidean distances, and inter-hand contact metrics.
- **6-D Scale-Invariant Facial Ratios:**
  * Mouth Aspect Ratio (MAR): $\|P_{13} - P_{14}\| / \max(\|P_{61} - P_{291}\|, 10^{-6})$
  * Eyebrow Elevation Ratio (Left, Right, and Mean EER): normalized by facial height.
  * Eye Aspect Ratio (Left & Right EAR).
- **164-D First-Order Velocity:** Inter-frame finite differences ($\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$) capturing motion trajectory and signing speed.

### 2. Dedicated RealSign 63-D Hand Normalizer
- Evaluates individual frames immediately without waiting for sequence buffers.
- Translates coordinates relative to landmark 0 (wrist) and scales by palm distance (wrist to middle MCP 9).

---

## ✨ Key Features

- ⚡ **Ultra-Low Latency:** Offscreen canvas downsampling ($320 \times 240$ JPEG compression) delivers 30 FPS video transmission over local networks with < 50ms round-trip inference latency.
- 🎯 **No False Positives:** Built-in `no_sign` idle baseline and automatic body positioning guidance ("Too close", "Step back", "Raise hands into frame").
- 🔒 **Duplicate Suppression:** Refractory cooldown (350ms) ensures stable sentence building without repetitive letter/word duplication.
- 📹 **Custom Sign Dataset Recorder:** In-app and CLI recording tool (`scripts/record_custom_signs.py`) captures 30-frame 164-D sequence arrays directly from webcam with visual countdown and HUD guides.
- 🧪 **Comprehensive Test Coverage:** 87 automated unit, invariant, and integration tests verifying schema immutability, signer isolation, and API contracts.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Node.js 18+ and npm
- Webcam

### One-Click Launch (Windows)
Run the automated launcher batch script:
```powershell
.\Start_SETU.bat
```
*(Starts the FastAPI backend on port 8000 and the Vite frontend on port 5173)*.

---

### Manual Setup

#### 1. Backend Setup
```powershell
# Navigate to repository root
cd c:\MINI PROJ-SETU

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies and editable setu package
pip install -e .

# Run the backend test suite
pytest -v

# Start the FastAPI WebSocket server
uvicorn setu.api.server:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Frontend Setup
```powershell
# In a separate terminal, navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Build or run Vite development server
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 📊 Dataset Recording & Training CLI

### Record Custom Sign Samples
To add custom signs to the training set:
```powershell
.\.venv\Scripts\python scripts/record_custom_signs.py --label namaste --takes 10
```
Samples are automatically processed and stored in `data/processed/custom_<label>_<timestamp>.npz`.

### Train Sequence LSTM Classifier
```powershell
.\.venv\Scripts\python training/train_sequence_lstm.py --epochs 60 --lr 0.001
```

### Run Standalone Webcam Demos
- **A–Z Alphabet Fingerspelling:**
  ```powershell
  .\.venv\Scripts\python scripts/demo_realsign_26class_webcam.py --camera 0
  ```
- **ISL MVP Phrases:**
  ```powershell
  .\.venv\Scripts\python scripts/demo_isl_mvp_webcam.py --camera 0
  ```

---

## 📂 Repository Structure

```
SETU/
├── configs/                     # Application configurations and constants
├── data/
│   ├── manifests/               # Dataset split manifests (signer-isolated)
│   └── processed/               # Preprocessed feature arrays (.npz)
├── docs/                        # Milestone specifications and architecture reports
├── frontend/                    # Vite + React + TypeScript + Tailwind frontend
│   ├── src/
│   │   ├── components/          # Camera viewport, translation HUD, settings
│   │   ├── hooks/               # useRecognitionStream WebSocket hook
│   │   └── types/               # TypeScript interfaces & API schemas
│   └── package.json
├── models/                      # Serialized PyTorch state dicts & label maps
│   ├── isl_sequence_lstm.pth    # 328-D Attention Sequence LSTM weights
│   ├── realsign_26class_mlp.pth # 63-D RealSign Alphabet MLP weights
│   └── label_map.json           # Class vocabulary mappings
├── Project Main Files/          # Foundation documentation and PRD specs
├── scripts/                     # Operational utilities and webcam demos
├── setu/                        # Core Python application package
│   ├── api/                     # FastAPI REST & WebSocket streaming server
│   ├── features/                # 158-D / 164-D / 328-D / 63-D feature extractors
│   ├── models/                  # PyTorch model architectures & classifiers
│   ├── utils/                   # Geometry and mathematical utilities
│   └── vision/                  # MediaPipe Holistic & Hand landmarker wrappers
├── tests/                       # 87 automated pytest test suites
├── training/                    # Model training and fine-tuning pipelines
├── pyproject.toml               # Python project configuration & packaging
├── requirements.txt             # Pip dependencies
└── README.md
```

---

## 📄 License & Attribution

- **License:** MIT License.
- **AI4Bharat INCLUDE Dataset:** Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **MediaPipe:** Google LLC (Apache 2.0).
