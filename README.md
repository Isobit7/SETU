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

SETU is an AI-powered Indian Sign Language (ISL) translation platform that turns live webcam input into readable text and speech. The project combines real-time gesture recognition, deep learning inference, and a browser-based interface to help bridge communication gaps for deaf and hard-of-hearing users.

The name "SETU" means "bridge" in Sanskrit, reflecting the project goal: connecting sign-language communication with spoken and written language.

## What SETU does

SETU is designed to:

- Detect and classify static alphabet signs (A–Z) in real time
- Recognize phrase-level ISL gestures using sequence-based models
- Monitor facial non-manual markers such as eyebrow and mouth movement
- Process webcam frames through a lightweight MediaPipe-based pipeline
- Display translated output in a live subtitle-style interface
- Speak translated text aloud using browser speech synthesis
- Record custom signs and retrain models using user-collected samples

## Core system overview

SETU uses a Python backend with FastAPI and a React + TypeScript frontend.

```text
Webcam feed
    ↓
Frontend captures and downsamples frames
    ↓
FastAPI WebSocket server receives video frames
    ↓
MediaPipe Holistic + Hand landmark extraction
    ↓
Feature engineering + model inference
    ↓
Temporal filtering + confidence checks
    ↓
Live translation output + voice synthesis
```

The system includes both:

- Alphabet mode: immediate static finger-spelling recognition
- Sequence mode: phrase and word recognition using temporal models

## Features

### Real-time recognition
- Webcam-based live inference
- Low-latency model pipeline
- Frame downsampling for smoother streaming
- Temporal stabilization to reduce flickering and duplicate outputs

### ISL-focused modeling
- Hand landmark normalization
- Facial non-manual feature tracking
- Sequence modeling for phrase representation
- Post-processing filters for confidence and stability

### Frontend experience
- Movie-style subtitle overlay
- Landmark visualization on live camera feed
- Camera guidance and framing hints
- Adjustable model and threshold settings
- Real-time text-to-speech feedback

### Custom data pipeline
- Record custom signs from the webcam
- Save processed samples in `.npz` format
- Load recorded samples into the training pipeline
- Retrain models with custom dataset additions

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

## Repository layout

```text
SETU/
├── .agent/                      # Agent/project support files
├── configs/                     # configuration and app constants
├── data/
│   ├── manifests/               # signer-isolated train/val/test manifests
│   └── processed/               # processed feature arrays (.npz)
├── docs/                        # project documentation and notes
├── frontend/                    # Vite + React + TypeScript frontend
│   ├── src/                    # app source
│   ├── package.json            # frontend dependencies/scripts
│   └── README.md               # frontend-specific notes
├── models/                      # trained model weights and label mappings
├── Project Main Files/          # canonical product/design docs and specs
├── scripts/                     # recording and demo scripts
├── setu/                        # core backend package
│   ├── api/                    # FastAPI API and websocket handlers
│   ├── features/               # feature extraction logic
│   ├── models/                 # model definitions and classifiers
│   ├── utils/                  # geometry and helper utilities
│   ├── vision/                 # MediaPipe and camera wrappers
│   ├── config.py               # project config
│   └── __init__.py             # package marker
├── tests/                       # automated pytest suite
├── training/                    # training scripts and utilities
├── .env.example                 # environment example file
├── .gitignore                  # git ignore rules
├── logo.png                     # project logo
├── pyproject.toml               # Python package configuration
├── README.md                    # project overview and setup guide
├── Start_SETU.bat               # Windows launcher for the full app
├── Start_Alphabet_Demo.bat      # Windows launcher for alphabet demo
└── SETU_Sign_Language_Translator_Build_Handbook.docx
```

## Quick start

### Windows one-click launch

From the repository root:

```powershell
./Start_SETU.bat
```

For the standalone alphabet demo:

```powershell
./Start_Alphabet_Demo.bat
```

## Manual setup

### 1) Backend

```powershell
cd <path-to-SETU>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pytest -v
uvicorn setu.api.server:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Then open:

```text
http://localhost:5173
```

Backend API is typically served on:

```text
http://localhost:8000
```

## API overview

The backend exposes a FastAPI service with both REST and WebSocket capabilities.

### WebSocket streaming

```text
ws://localhost:8000/api/v1/ws/recognize?model=realsign&threshold=0.35
```

This endpoint receives webcam frames from the client and returns prediction payloads containing:

- active model
- predicted label
- confidence score
- stabilization metadata
- guidance messages
- landmark information
- speech-ready output payloads

### REST endpoints

Examples include:

- `GET /api/v1/models` — model catalog
- `GET /api/v1/models/realsign/classes` — alphabet class information
- `GET /api/v1/vocabulary` — phrase vocabulary metadata
- `POST /api/v1/dataset/record_sample` — custom sample upload
- `GET /api/v1/dataset/custom_samples` — custom recorded samples

## Training and custom sign recording

SETU supports custom sign recording for extending the dataset.

Example:

```powershell
.\.venv\Scripts\python scripts/record_custom_signs.py --label namaste --takes 10
```

This captures multiple labeled samples from the webcam and stores processed feature vectors for later training.

A training command example:

```powershell
.\.venv\Scripts\python training/train_sequence_lstm.py --epochs 60 --lr 0.001
```

## Project documentation

The repository includes deep project documentation in the `Project Main Files/` directory. This is the best place to understand the product requirements, architecture, design goals, and phased roadmap.

Key documents include:

- `Project Main Files/README.md`
- `Project Main Files/PRD.md`
- `Project Main Files/architecture.md`
- `Project Main Files/design.md`
- `Project Main Files/phase.md`
- `Project Main Files/rules.md`
- `Project Main Files/memory.md`

## Testing

SETU includes automated test coverage for API behavior, feature extraction, sequence modeling, and vision pipeline logic.

```powershell
.\.venv\Scripts\pytest.exe -v
```

The project reports a passing test suite in the current repository state.

## Important notes

- SETU is a research/prototype-style system with a strong technical foundation and live demo workflow.
- Some files in the project documentation describe future UI/UX targets and design phases.
- The runtime behavior depends on webcam input quality, lighting, and model availability.

## Troubleshooting

Common checks:

- Ensure your webcam is available and active
- Confirm MediaPipe and OpenCV dependencies are installed
- Check the backend is running on port 8000
- Use a clear background and visible hand/face region
- Increase confidence thresholds if predictions flicker or appear unstable

## License

This project is distributed under the MIT license unless otherwise noted in project files.

## Contributing

This repository is structured to support backend, frontend, data, and model experimentation. If you want to extend the project, the most natural areas to work in are:

- `setu/features/` for new feature extraction logic
- `setu/models/` for classifiers and model definitions
- `training/` for retraining scripts
- `frontend/src/` for UI and webcam integration
- `Project Main Files/` for architecture and product documentation updates

For deeper technical context, start with the documentation in `Project Main Files/` and then inspect the backend/frontend entry points.
