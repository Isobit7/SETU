# SETU: AI-Powered Indian Sign Language Translator

**SETU** is an AI-powered Indian Sign Language (ISL) translation system designed to bridge communication between the Deaf community and non-signers in India.

## Project Structure

```
c:\MINI PROJ-SETU\
├── setu/                # Canonical recognition-engine package
│   ├── vision/          # Camera capture and MediaPipe Tasks wrappers
│   ├── features/        # 158-feature schema, extraction, and normalization
│   ├── utils/           # Shared geometric algorithms
│   ├── models/          # Model architectures and artifact loaders (Phase 2)
│   ├── inference/       # Real-time inference and routing (Phase 2)
│   └── translation/     # Word/sentence composition and state machines (Phase 2)
├── configs/             # Non-secret configuration and threshold definitions
├── training/            # Dataset loaders, augmentation, training, and evaluation
├── tests/               # Automated unit, integration, and model tests
├── docs/                # Architecture, specifications, and setup documentation
└── pyproject.toml       # Python packaging and dependency specification
```

## Quick Start

### 1. Environment Setup
Create and activate a Python 3.12 virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Editable Package Installation
Install `setu` in editable mode:
```powershell
pip install -e .
```

### 3. Run Automated Tests
```powershell
pytest
```
