# SETU — Key Files Index and Component Reference

**Last Updated:** 2026-10-17  
**Status:** Authoritative Code & Asset Index

This index provides direct references to the canonical files, entry points, APIs, models, configurations, and test suites across the SETU project.

---

## 1. Documentation & Foundation Specifications
All canonical documentation resides in the [`Project Main Files/`](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/) directory:

- [Navigator & Overview](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/README.md) (`Project Main Files/README.md`)
- [Product Requirements Document (PRD)](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PRD.md) (`Project Main Files/PRD.md`)
- [System Architecture Specification](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/architecture.md) (`Project Main Files/architecture.md`)
- [UI & Design Conventions](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/design.md) (`Project Main Files/design.md`)
- [Translator Workspace Design](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/translator_workspace_design.md) (`Project Main Files/translator_workspace_design.md`)
- [Button & Control Inventory](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/button.md) (`Project Main Files/button.md`)
- [Project Memory & Decisions](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/memory.md) (`Project Main Files/memory.md`)
- [Phased Delivery Plan](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/phase.md) (`Project Main Files/phase.md`)
- [Engineering & Agent Rules](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/rules.md) (`Project Main Files/rules.md`)
- [Project Structure Blueprint](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PROJECT_STRUCTURE.md) (`Project Main Files/PROJECT_STRUCTURE.md`)
- [Project Status & Health](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PROJECT_STATUS.md) (`Project Main Files/PROJECT_STATUS.md`)

---

## 2. Core Python Package (`setu/`)

| File / Component | Responsibility | Relative Path |
| :--- | :--- | :--- |
| **Package Root** | Package exports and namespace definition | [`setu/__init__.py`](file:///c:/MINI%20PROJ-SETU/setu/__init__.py) |
| **FastAPI Server** | Streaming WebSocket and REST recognition API | [`setu/api/server.py`](file:///c:/MINI%20PROJ-SETU/setu/api/server.py) |
| **Domain Exceptions** | Centralized `SETUError` hierarchy for camera, vision, model & features | [`setu/exceptions.py`](file:///c:/MINI%20PROJ-SETU/setu/exceptions.py) |
| **Feature Schema** | 158-D immutable vector slice constants, indices & validation rules | [`setu/features/schema.py`](file:///c:/MINI%20PROJ-SETU/setu/features/schema.py) |
| **Feature Extractor** | `FeatureExtractor158`: extracts normalized coordinates & relational features | [`setu/features/extractor.py`](file:///c:/MINI%20PROJ-SETU/setu/features/extractor.py) |
| **ISL MVP Classifier** | `ISLMVPClassifier`: pure NumPy multi-phase inference engine (<0.2 ms) | [`setu/models/isl_mvp_classifier.py`](file:///c:/MINI%20PROJ-SETU/setu/models/isl_mvp_classifier.py) |
| **RF-UV-11 Adapter** | `RFUV11Adapter`: isolated reference adapter for 63-D single-hand MLP | [`setu/models/rf_uv11_adapter.py`](file:///c:/MINI%20PROJ-SETU/setu/models/rf_uv11_adapter.py) |
| **Geometric Mathematics** | Landmark normalization, finger angles, Euclidean distances, contact scoring | [`setu/utils/geometry.py`](file:///c:/MINI%20PROJ-SETU/setu/utils/geometry.py) |
| **Camera Capture** | `CameraReader`: hardware acquisition, mirror preview, error boundaries | [`setu/vision/camera.py`](file:///c:/MINI%20PROJ-SETU/setu/vision/camera.py) |
| **Holistic Landmarker** | `HolisticLandmarkerWrapper`: MediaPipe Tasks wrapper for face/pose/hands | [`setu/vision/landmarks.py`](file:///c:/MINI%20PROJ-SETU/setu/vision/landmarks.py) |

---

## 3. Frontend Web Application (`frontend/`)

| File / Component | Responsibility | Relative Path |
| :--- | :--- | :--- |
| **Main App Root** | Master workspace combining video, subtitles, controls & panels | [`frontend/src/App.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/App.tsx) |
| **Camera Workspace** | Live video, canvas landmark overlays, movie-style subtitles | [`frontend/src/components/CameraWorkspace.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/CameraWorkspace.tsx) |
| **Camera Controls** | Start, pause, resume, stop, reset, landmark toggles, mirror | [`frontend/src/components/CameraControls.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/CameraControls.tsx) |
| **Translation Panel** | Live output, copy to clipboard, speech synthesis, history | [`frontend/src/components/TranslationPanel.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/TranslationPanel.tsx) |
| **Navigation Sidebar** | Dark navy branding sidebar with responsive collapse | [`frontend/src/components/Sidebar.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/Sidebar.tsx) |
| **Top Control Bar** | Translation direction, online/offline pill, settings triggers | [`frontend/src/components/TopBar.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/TopBar.tsx) |
| **Status Strip** | Live hardware status, framing feedback, and latency monitor | [`frontend/src/components/StatusStrip.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/StatusStrip.tsx) |
| **Vocabulary Modal** | Reference modal explaining the 5 authentic ISL signs & tips | [`frontend/src/components/VocabularyModal.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/VocabularyModal.tsx) |
| **Settings Modal** | Accessibility font size, landmark layer switches, sensitivity | [`frontend/src/components/SettingsModal.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/SettingsModal.tsx) |
| **WebSocket Stream Hook** | `useRecognitionStream`: manages webcam feed, binary frame transfers | [`frontend/src/hooks/useRecognitionStream.ts`](file:///c:/MINI%20PROJ-SETU/frontend/src/hooks/useRecognitionStream.ts) |
| **Landmark Canvas Drawer** | Color-coded skeleton bones and joint points (no fake dots) | [`frontend/src/utils/landmarksDrawer.ts`](file:///c:/MINI%20PROJ-SETU/frontend/src/utils/landmarksDrawer.ts) |
| **Design System CSS** | HSL tokens, movie subtitle styling, responsive layout | [`frontend/src/index.css`](file:///c:/MINI%20PROJ-SETU/frontend/src/index.css) |

---

## 3. Configuration & Runtime Settings

| File | Purpose | Relative Path |
| :--- | :--- | :--- |
| **Central Settings** | Non-secret system defaults, paths, camera settings, thresholds | [`configs/settings.py`](file:///c:/MINI%20PROJ-SETU/configs/settings.py) |
| **Package Manifest** | Python 3.12 dependencies, pytest options, project build configuration | [`pyproject.toml`](file:///c:/MINI%20PROJ-SETU/pyproject.toml) |
| **Environment Template** | Safe example environment variables (no committed secrets) | [`.env.example`](file:///c:/MINI%20PROJ-SETU/.env.example) |

---

## 4. Models & Pretrained Assets

| Model Asset | Description | File Path |
| :--- | :--- | :--- |
| **ISL MVP Model** | Trained weights, mean, scale, labels for 6-class regularized classifier | [`models/isl_mvp_model.npz`](file:///c:/MINI%20PROJ-SETU/models/isl_mvp_model.npz) |
| **Holistic Task Bundle** | MediaPipe Tasks model for multi-component landmark tracking | [`models/tasks/holistic_landmarker.task`](file:///c:/MINI%20PROJ-SETU/models/tasks/holistic_landmarker.task) |
| **RF-UV-11 Model** | Experimental 63-D MLP single-hand weights (Apache 2.0 reference) | [`models/pretrained/rf_uv11_model.h5`](file:///c:/MINI%20PROJ-SETU/models/pretrained/rf_uv11_model.h5) |
| **RF-UV-11 Report** | Architectural limitation analysis for single-hand ISL models | [`models/pretrained/evidence_report.md`](file:///c:/MINI%20PROJ-SETU/models/pretrained/evidence_report.md) |
| **Legacy ISL Weights** | Historical/reference weights | [`models/reference/isl_model.h5`](file:///c:/MINI%20PROJ-SETU/models/reference/isl_model.h5) |

---

## 5. Datasets & Feature Manifests

| Asset | Description | File Path |
| :--- | :--- | :--- |
| **MVP Manifest** | 33-sample dataset manifest with strict signer-isolation across 6 classes | [`data/manifests/setu_isl_mvp_manifest.json`](file:///c:/MINI%20PROJ-SETU/data/manifests/setu_isl_mvp_manifest.json) |
| **Processed Features** | 33 verified .npz feature files (shape `(30, 158)`, 0 NaNs/Infs) | `data/processed/*.npz` |
| **Raw INCLUDE Videos** | 15 verified authentic ISL videos downloaded from Zenodo (CC BY 4.0) | `data/raw/include/*.mov` |

---

## 6. Runnable Scripts & Executables

| Command / Script | Purpose | File Path |
| :--- | :--- | :--- |
| **ISL MVP Live Demo** | Real-time OpenCV ISL 5-Sign recognition on webcam or video | [`scripts/demo_isl_mvp_webcam.py`](file:///c:/MINI%20PROJ-SETU/scripts/demo_isl_mvp_webcam.py) |
| **RF-UV-11 Demo** | Isolated reference demo for experimental single-hand model | [`scripts/demo_rf_uv11_webcam.py`](file:///c:/MINI%20PROJ-SETU/scripts/demo_rf_uv11_webcam.py) |
| **Model Training** | Regularized softmax training & evaluation across train/val/test splits | [`training/train_mvp_model.py`](file:///c:/MINI%20PROJ-SETU/training/train_mvp_model.py) |
| **INCLUDE Downloader** | Byte-range HTTP subset extractor for AI4Bharat INCLUDE Zenodo archives | [`scripts/download_include_subset.py`](file:///c:/MINI%20PROJ-SETU/scripts/download_include_subset.py) |
| **Feature Extraction** | Batch feature extraction pipeline from raw videos to .npz sequences | [`scripts/process_include_subset.py`](file:///c:/MINI%20PROJ-SETU/scripts/process_include_subset.py) |
| **Camera Verification** | Hardware verification and snapshot generation tool | [`scripts/webcam_physical_verification.py`](file:///c:/MINI%20PROJ-SETU/scripts/webcam_physical_verification.py) |

---

## 7. Automated Test Suites (`tests/`)

| Test File | Verification Scope | Current Status |
| :--- | :--- | :--- |
| [`test_config_exceptions.py`](file:///c:/MINI%20PROJ-SETU/tests/test_config_exceptions.py) | Validates non-secret config immutability and exception hierarchy | 4 passed |
| [`test_feature_schema.py`](file:///c:/MINI%20PROJ-SETU/tests/test_feature_schema.py) | 158-D schema slice invariants, boundary checks, zero-scale guards | 10 passed |
| [`test_isl_mvp_model.py`](file:///c:/MINI%20PROJ-SETU/tests/test_isl_mvp_model.py) | MVP model inference, weight shapes, pooling, hand gating & thresholds | 5 passed |
| [`test_pipeline.py`](file:///c:/MINI%20PROJ-SETU/tests/test_pipeline.py) | Video loading, uniform temporal resampling to 30 frames, feature vectors | 4 passed |
| [`test_real_data_readiness.py`](file:///c:/MINI%20PROJ-SETU/tests/test_real_data_readiness.py) | Verified genuine INCLUDE samples, SHA-256 hashes, zero signer leakage | 5 passed |
| [`test_rf_uv11_adapter.py`](file:///c:/MINI%20PROJ-SETU/tests/test_rf_uv11_adapter.py) | Isolated RF-UV-11 single-hand adapter inference and smoothing checks | 7 passed |
| [`test_vision.py`](file:///c:/MINI%20PROJ-SETU/tests/test_vision.py) | CameraReader lifecycle and HolisticLandmarkerWrapper inference | 4 passed |
| **Total Test Suite** | **Comprehensive automated test coverage** | **39 passed** |
