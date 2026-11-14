# SETU Scripts & Operational Utilities

This directory contains standalone execution tools, interactive demos, diagnostic runners, and launch automation scripts for **SETU**.

---

## Directory Organization

```
scripts/
├── launchers/    # Windows batch launchers (Start_SETU, Stop_SETU, Verify_SETU, run_backend, run_frontend)
├── demos/        # Standalone OpenCV & MediaPipe webcam demo pilots (RealSign, YOLOv11, Hand tracking)
├── audits/       # Empirical accuracy verification, system diagnostics, and leak audits
└── tools/        # Offline dataset extractors, cloud training helpers, and SSL cert utilities
```

---

## 1. Launchers (`scripts/launchers/`)
- **`Start_SETU.bat`**: Launches both FastAPI backend (port 8000) and Vite frontend (port 5173).
- **`Stop_SETU.bat`**: Safely terminates any active processes on ports 8000 and 5173.
- **`Verify_SETU.bat`**: Pre-flight test runner checking Python venv, CUDA, models, and regression tests.
- **`run_backend.bat`**: Starts only the FastAPI ASGI engine.
- **`run_frontend.bat`**: Starts only the React/Vite development server.

---

## 2. Interactive Webcam Demos (`scripts/demos/`)
- **`demo_realsign_26class_webcam.py`**: Standalone OpenCV window tracking 21-joint hand landmarks and predicting alphabet letters A–Z.
- **`demo_conversational_5sign_webcam.py`**: Standalone test for 5 core conversational phrases.
- **`run_standalone_hand_pilot.py`**: Benchmarks MediaPipe HandLandmarker frame latency.
- **`webcam_physical_verification.py`**: Verifies camera capture and lighting conditions.

---

## 3. Audits & Diagnostics (`scripts/audits/`)
- **`deep_system_diagnostic.py`**: Full 100-point diagnostic audit checking pipelines, CUDA, and NLP.
- **`audit_project.py`**: Structural health check of the codebase.
- **`run_accuracy_test.py`**: Runs held-out accuracy evaluation across all active classifiers.
- **`verify_all_endpoints.py`**: Validates all FastAPI REST and WebSocket routes.

---

## 4. Pipeline Helpers (`scripts/`)
These modules provide active learning and feature augmentation utilities imported by the backend and tests:
- `record_custom_signs.py`: Custom gesture recording utility.
- `retrain_feedback_loop.py`: Continuous learning loop from user correction feedback.
- `generate_multisigner_dataset.py`: 3D spatial rotation and scale augmentation.
- `calibrate_and_train_webcam_alphabet.py`: User-specific webcam calibration.
