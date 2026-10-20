<div align="center">
  <img src="logo.png" alt="SETU Logo" width="560" />
  <br />

  [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
  [![MediaPipe](https://img.shields.io/badge/MediaPipe-Holistic-00C7B7?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
  [![React](https://img.shields.io/badge/React-18.0+-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
  [![Vite](https://img.shields.io/badge/Vite-5.0+-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
  [![PyTest](https://img.shields.io/badge/PyTest-87%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)]()
</div>

# SETU

SETU is an AI-powered Indian Sign Language (ISL) translation system that turns live webcam input into understandable text and speech. It combines machine learning, computer vision, and a modern web interface to bridge communication between sign language users and spoken/written communication.

The project is designed for real-time fingerspelling, phrase recognition, facial expression-aware parsing, and accessible translation workflows.

## Why SETU

- Real-time recognition from a webcam
- Built for Indian Sign Language workflows
- FastAPI backend + React frontend
- MediaPipe-based landmark processing
- Temporal filtering for stable predictions
- Easy custom-sign recording and retraining pipeline

## What it does

SETU helps interpret:

- Static alphabet signs (A–Z)
- Phrase-level ISL gestures
- Facial non-manual markers and expression cues
- Live subtitle-style translation output
- Spoken text feedback through browser speech synthesis

## Architecture at a glance

```text
Webcam input
  ↓
Frontend captures and downsamples video frames
  ↓
FastAPI WebSocket server receives frames
  ↓
MediaPipe Holistic + Hand landmark extraction
  ↓
Feature engineering + model inference
  ↓
Temporal smoothing + confidence filtering
  ↓
Live translation output + speech synthesis
```

## Tech stack

- Python 3.10+
- FastAPI
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- React 19
- TypeScript
- Vite
- Tailwind CSS

## Key capabilities

### Real-time classification
- Static sign recognition for alphabet fingerspelling
- Phrase recognition for common ISL gestures
- Low-latency streaming pipeline for live interaction

### Robust vision processing
- Hand normalization and landmark extraction
- Facial feature tracking for NMM-inspired cues
- Confidence thresholds and duplicate suppression
- Framing/guidance feedback for better camera setup

### Data and model experimentation
- Custom sign recording from webcam samples
- `.npz` dataset generation for feature-based learning
- Training utilities for extending model behavior

### Frontend experience
- Movie-style subtitle HUD
- Landmark overlays on camera feed
- Adjustable threshold and model switching
- Speech output for translations

## Repository layout

```text
SETU/
├── .agent/                     # project support files
├── configs/                    # configuration and constants
├── data/
│   ├── manifests/              # signer-isolated train/val/test manifests
│   └── processed/              # processed feature arrays (.npz)
├── docs/                       # documentation and notes
├── frontend/                   # React + TypeScript + Vite app
│   ├── src/                   # frontend source files
│   ├── package.json           # frontend scripts and deps
│   └── README.md              # frontend notes
├── models/                     # trained weights and labels
├── Project Main Files/         # core product/design/architecture documentation
├── scripts/                    # custom recording and demo scripts
├── setu/                       # backend Python package
│   ├── api/                   # FastAPI API + WebSocket server
│   ├── features/              # feature engineering logic
│   ├── models/                # classifiers and ML logic
│   ├── utils/                 # helper and geometry utilities
│   ├── vision/                # MediaPipe wrappers and vision utilities
│   ├── config.py              # config values
│   └── __init__.py            # package marker
├── tests/                      # automated test suite
├── training/                   # training and model scripts
├── .env.example                # environment example
├── .gitignore                  # repository ignore rules
├── logo.png                    # SETU logo
├── pyproject.toml              # Python package metadata
├── README.md                   # project overview
├── Start_SETU.bat              # Windows launcher for full app
├── Start_Alphabet_Demo.bat     # Windows alphabet demo launcher
├── SETU_Sign_Language_Translator_Build_Handbook.docx
└── SETU_Foundation_Documents_2026-10-17_Updated.zip
```

## Quick start

### Windows one-click launch

```powershell
./Start_SETU.bat
```

For the standalone alphabet demo:

```powershell
./Start_Alphabet_Demo.bat
```

### Manual setup

#### 1) Backend

```powershell
cd <path-to-SETU>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -v
uvicorn setu.api.server:app --host 0.0.0.0 --port 8000 --reload
```

#### 2) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

The backend is expected to run at:

```text
http://localhost:8000
```

## API overview

The backend exposes REST and WebSocket endpoints for recognition.

### WebSocket stream

```text
ws://localhost:8000/api/v1/ws/recognize?model=realsign&threshold=0.35
```

This endpoint receives webcam frames and responds with prediction payloads containing details like:

- label and confidence
- recognition stability state
- model selection
- camera guidance feedback
- landmark data
- speech-ready translation output

### Common REST routes

- `GET /api/v1/models`
- `GET /api/v1/models/realsign/classes`
- `GET /api/v1/vocabulary`
- `POST /api/v1/dataset/record_sample`
- `GET /api/v1/dataset/custom_samples`

## Training and custom sign recording

SETU supports user-recorded sign sample collection.

Example:

```powershell
.\.venv\Scripts\python scripts/record_custom_signs.py --label namaste --takes 10
```

Example training run:

```powershell
.\.venv\Scripts\python training/train_sequence_lstm.py --epochs 60 --lr 0.001
```

This makes it easier to extend recognition with domain-specific or user-specific signs.

## Project documentation

The `Project Main Files/` directory contains deeper product and system documentation for the project. It is the best place to understand:

- product requirements and goals
- architecture and module boundaries
- design decisions and workspace direction
- phased roadmap and delivery plan
- operating principles and engineering rules

Recommended starting documents:

- `Project Main Files/README.md`
- `Project Main Files/PRD.md`
- `Project Main Files/architecture.md`
- `Project Main Files/design.md`
- `Project Main Files/phase.md`
- `Project Main Files/rules.md`

## Testing

The repo includes automated tests for API behavior, feature extraction, model logic, and vision processing.

```powershell
.\.venv\Scripts\pytest.exe -v
```

## Troubleshooting

Common issues:

- webcam not detected or blocked by browser
- backend not running on port 8000
- low light or poor hand visibility
- unstable predictions caused by low confidence thresholds
- dependency issues during first-time setup

## License

This project is distributed under the MIT license unless otherwise stated in the repository files.

## Contributing

Key areas to explore:

- `setu/features/` for feature extraction improvements
- `setu/models/` for model architecture and inference logic
- `training/` for retraining and dataset experimentation
- `frontend/src/` for UI and interaction improvements
- `Project Main Files/` for architecture and roadmap updates

For a deeper technical understanding, start with `Project Main Files/` and then move into the backend and frontend entry points.
