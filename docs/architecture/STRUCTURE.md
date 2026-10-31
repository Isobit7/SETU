# SETU — Living Master File Index & Architectural Dictionary

> **Project:** SETU (Indian Sign Language Translator)  
> **Repository Root:** `c:\MINI PROJ-SETU`  
> **Architecture Level:** Production Enterprise Multi-Tier (React 19 + FastAPI + PyTorch/ONNX + MediaPipe)  
> **Compliance Standard:** Enterprise Separation of Concerns & Living Master Index  
> **Last Synchronized:** October 2026  
> **Status:** 100% Operational (133/133 Automated Tests Passing | Zero Build Warnings)  

This master document serves as the authoritative, living single-source-of-truth index for every directory and file across the SETU repository. Each directory is documented with **Why the Folder Exists** (its architectural rationale), and every file is accompanied by its **Exact Purpose** in plain, unambiguous language.

---

## 1. Master Project Directory Tree

```
c:/MINI PROJ-SETU/
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Gemini.zip
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
├── Start_SETU.bat
│
├── backend/                             # FastAPI Server & Real-Time Engine
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   └── app/
│       ├── __init__.py
│       ├── main.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── endpoints/
│       │       ├── __init__.py
│       │       ├── dataset.py
│       │       ├── models.py
│       │       ├── predict.py
│       │       ├── video.py
│       │       └── vocabulary.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── predict.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── classifier.py
│       │   ├── pipeline_manager.py
│       │   ├── realsign_classifier_service.py
│       │   ├── nlp/
│       │   │   ├── __init__.py
│       │   │   ├── accumulator.py
│       │   │   └── gloss_parser.py
│       │   └── vision/
│       │       ├── __init__.py
│       │       ├── camera.py
│       │       └── landmarks.py
│       └── websocket/
│           ├── __init__.py
│           └── recognize.py
│
├── frontend/                            # React 19 + TypeScript + Vite UI Workstation
│   ├── Dockerfile
│   ├── .oxlintrc.json
│   ├── index.html
│   ├── nginx.conf
│   ├── package.json
│   ├── package-lock.json
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   ├── public/
│   │   ├── apple-touch-icon.png
│   │   ├── favicon-16x16.png
│   │   ├── favicon-32x32.png
│   │   ├── favicon.ico
│   │   ├── logo.png
│   │   ├── manifest.json
│   │   ├── robots.txt
│   │   └── sitemap.xml
│   └── src/
│       ├── App.css
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       ├── types.ts
│       ├── assets/
│       │   ├── hero.png
│       │   ├── logo.png
│       │   └── react.svg
│       ├── components/
│       │   ├── AlphabetModal.tsx
│       │   ├── CameraControls.tsx
│       │   ├── CameraWorkspace.tsx
│       │   ├── HelpModal.tsx
│       │   ├── HistoryModal.tsx
│       │   ├── IncludeVideoModal.tsx
│       │   ├── LiveDetectionOutput.tsx
│       │   ├── SavedModal.tsx
│       │   ├── SettingsModal.tsx
│       │   ├── TextToSignWorkspace.tsx
│       │   ├── TopNavbar.tsx
│       │   ├── TranslationPanel.tsx
│       │   ├── VisualFramingGuide.tsx
│       │   ├── VocabularyModal.tsx
│       │   └── WebcamAlphabetRecognition.tsx
│       ├── hooks/
│       │   ├── useRecognitionStream.ts
│       │   ├── useSentenceBuilder.ts
│       │   ├── useTextBuffer.ts
│       │   └── useTTS.ts
│       ├── services/
│       │   └── apiService.ts
│       └── utils/
│           ├── framingEvaluator.ts
│           ├── landmarkProcessor.ts
│           └── landmarksDrawer.ts
│
├── models/                              # AI Model Layer (Code & Weights Split)
│   ├── __init__.py
│   ├── registry.py
│   ├── model_registry.py
│   ├── include.py
│   ├── include_video_lstm_adapter.py
│   ├── isl_conversational_mvp.py
│   ├── isl_mvp_model.npz
│   ├── isl_phrase_sequence_lstm_adapter.py
│   ├── isl_sequence_lstm.pth
│   ├── label_map.json
│   ├── mvp.py
│   ├── realsign.py
│   ├── realsign_26class_mlp.onnx
│   ├── realsign_26class_mlp.pt
│   ├── realsign_26class_mlp.pth
│   ├── realsign_alphabet_mlp_adapter.py
│   ├── rf_uv11.py
│   ├── roboflow_yolov11_adapter.py
│   ├── sequence_lstm.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── include.py
│   │   ├── mvp.py
│   │   ├── realsign.py
│   │   ├── rf_uv11.py
│   │   └── sequence_lstm.py
│   ├── pretrained/
│   │   ├── evidence_report.md
│   │   ├── include_no_cnn_lstm.pth
│   │   ├── label_map_include.json
│   │   ├── LICENSE_APACHE_2.0.txt
│   │   ├── README.md
│   │   └── rf_uv_11_model.h5
│   ├── reference/
│   │   ├── isl_model.h5
│   │   └── README.md
│   ├── tasks/
│   │   ├── hand_landmarker.task
│   │   ├── holistic_landmarker.task
│   │   └── README.md
│   └── weights/
│       ├── isl_mvp_model.npz
│       ├── isl_sequence_lstm.pth
│       ├── label_map.json
│       ├── realsign_26class_mlp.onnx
│       ├── realsign_26class_mlp.pt
│       └── realsign_26class_mlp.pth
│
├── training/                            # Machine Learning Training & Processing Pipelines
│   ├── __init__.py
│   ├── build_conversational_5sign_manifest.py
│   ├── build_mvp_manifest.py
│   ├── dataset_manifest.py
│   ├── evaluate.py
│   ├── evaluate_realsign_mlp.py
│   ├── export_onnx.py
│   ├── export_realsign_mlp_to_onnx.py
│   ├── extract_features.py
│   ├── extract_holistic_158d_features.py
│   ├── extract_realsign_63d_features.py
│   ├── feature_extractor.py
│   ├── generate_negative_resting_samples.py
│   ├── ingest_raw.py
│   ├── ingest_raw_datasets.py
│   ├── ingest_realsign_image_dataset.py
│   ├── isl_dataset_manifest_schema.py
│   ├── no_sign_generator.py
│   ├── train.py
│   ├── train_conversational_5sign_ridge.py
│   ├── train_isl_phrase_bilstm.py
│   ├── train_mvp_model.py
│   ├── train_realsign_26class.py
│   ├── train_realsign_26class_mlp.py
│   ├── train_sequence_lstm.py
│   └── features/
│       ├── __init__.py
│       ├── extractor.py
│       ├── geometry.py
│       └── schema.py
│
├── records/                             # Living Master Records & Ledger
│   ├── ARCHITECTURE.md
│   ├── CHECKPOINTS.md
│   ├── DATASETS_AND_MODELS.md
│   ├── MEMORY.md
│   └── STRUCTURE.md
│
├── config/                              # Centralized Environment Settings & Exceptions
│   ├── __init__.py
│   ├── exceptions.py
│   └── settings.py
│
├── reports/                             # Evaluation & Bias Audit Reports
│   ├── isl_sequence_lstm_evaluation.json
│   ├── realsign_26class_evaluation_report.json
│   ├── realsign_audit_report.json
│   ├── realsign_hand_pilot_report.json
│   ├── realsign_leakage_report.json
│   └── realsign_standalone_hand_pilot_report.json
│
├── scripts/                             # Utility Scripts, Harnesses & Standalone Demos
│   ├── audit_project.py
│   ├── audit_realsign_dataset.py
│   ├── audit_realsign_leakage.py
│   ├── build_overview_notebook.py
│   ├── calibrate_and_train_webcam_alphabet.py
│   ├── check_dataset_sources.py
│   ├── check_framing_evaluator.js
│   ├── clean_realsign_split.py
│   ├── demo_conversational_5sign_webcam.py
│   ├── demo_realsign_26class_webcam.py
│   ├── demo_roboflow_yolov11_webcam.py
│   ├── download_include_subset.py
│   ├── download_realsign_dataset.py
│   ├── extract_realsign_landmarks.py
│   ├── fetch_include_info.py
│   ├── fetch_include_splits.py
│   ├── generate_certs.ps1
│   ├── inspect_remote_dataset_zip_headers.py
│   ├── package_gemini.py
│   ├── process_include_subset.py
│   ├── record_custom_signs.py
│   ├── run_controlled_alphabet_test.py
│   ├── run_hand_detector_pilot.py
│   ├── run_standalone_hand_pilot.py
│   ├── Start_Alphabet_Demo.bat
│   ├── test_include_inference.py
│   ├── verify_include_dataset_zenodo_api.py
│   └── webcam_physical_verification.py
│
├── tests/                               # Automated Pytest Suite
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_backend_app_modular.py
│   ├── test_camera_and_holistic_vision.py
│   ├── test_config_exceptions.py
│   ├── test_custom_sign_pipeline.py
│   ├── test_feature_extraction_and_manifest_pipeline.py
│   ├── test_feature_schema.py
│   ├── test_include_model.py
│   ├── test_invariant_features_and_segmentation.py
│   ├── test_isl_mvp_model.py
│   ├── test_isl_sequence_lstm_classifier.py
│   ├── test_nlp_pipeline.py
│   ├── test_real_data_readiness.py
│   ├── test_realsign_classifier.py
│   └── test_rf_uv11_adapter.py
│
├── certs/                               # Local SSL/TLS Development Certificates
│   ├── .gitkeep
│   ├── fullchain.pem
│   └── privkey.pem
│
├── data/                                # Local Datasets, Manifests & Processed Features
│   ├── manifests/
│   │   └── setu_isl_mvp_manifest.json
│   ├── processed/
│   └── raw/
│
├── docs/                                # Technical Specifications & Archival Records
│   ├── aju22_license_resolution.md
│   ├── milestone1_data_acquisition_report.json
│   ├── phase1a_specifications.md
│   ├── phase1b_coordinate_verification.md
│   ├── phase1b_vocabulary_selection.md
│   ├── SETU_DEEP_ARCHITECTURE_GUIDE.md
│   ├── SETU_Sign_Language_Translator_Build_Handbook.docx
│   ├── SETU_system_overview.svg
│   ├── webcam_verification_results.json
│   ├── webcam_verification_snapshot.jpg
│   ├── archive/
│   │   ├── SETU_Foundation_Documents_2026-10-17_Updated.zip
│   │   └── SETU_Project.zip
│   └── user_understanding/              # Conceptual System Architecture Manuals
│       ├── ARCHITECTURE_AND_PIPELINE.md
│       ├── END_TO_END_FLOW.md
│       ├── MODEL_AND_DATASET_SPEC.md
│       ├── PROJECT_OVERVIEW.md
│       ├── ROADMAP_AND_SELF_AUDIT.md
│       └── VISUAL_DASHBOARD.md
│
└── notebooks/                           # Interactive Walkthroughs
    └── overview_of_setu.ipynb
```

---

## 2. Master Component & Architecture Overview Table

| Architectural Subsystem | Directory Path | Core Technical Function | Key Technologies & Anchor Files |
| :--- | :--- | :--- | :--- |
| **Root Workspace** | `/` | Containerization, environment variables, Windows startup launcher, package config | `docker-compose.yml`, `Start_SETU.bat`, `pyproject.toml`, `README.md` |
| **Backend Server** | `/backend` | Asynchronous FastAPI server, REST routers, duplex WebSocket landmark streaming | FastAPI, Uvicorn, WebSockets (`backend/app/main.py`, `recognize.py`) |
| **Frontend Workstation** | `/frontend` | React 19 UI workstation, MediaPipe WebAssembly landmark tracking, dark HUD | React 19, TypeScript, Vite (`App.tsx`, `CameraWorkspace.tsx`) |
| **AI Model Layer** | `/models` | Multi-tier model adapters, weights registry, MediaPipe `.task` bundles | PyTorch, ONNX, MediaPipe (`models/registry.py`, `models/weights/`) |
| **Training & Features** | `/training` | 89-D invariant geometric math, dataset manifests, PyTorch CUDA retraining | PyTorch, Scikit-learn (`training/features/geometry.py`, `train.py`) |
| **Living System Records** | `/records` | Authoritative single-source-of-truth ledgers for architecture, models, audit | `ARCHITECTURE.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`, `MEMORY.md` |
| **Centralized Config** | `/config` | Frozen dataclass settings, environment variables, typed domain exceptions | Python 3.10 Dataclasses (`config/settings.py`, `config/exceptions.py`) |
| **Automated Test Suite** | `/tests` | 13 comprehensive test suites covering API, vision, invariant math, and models | 133 Pytest tests passing in 67s (`test_api.py`, `test_invariant_features...`) |
| **Harnesses & Scripts** | `/scripts` | Sign calibration, data ingestion, webcam verification, release packager | Python, Node.js (`calibrate_and_train_webcam_alphabet.py`) |
| **Audit Reports** | `/reports` | Evaluation matrices, bias audits, data leakage audits, pilot telemetry | JSON evaluation reports (`realsign_26class_evaluation_report.json`) |
| **SSL/TLS Security** | `/certs` | Local SSL/TLS certificates enabling secure WebRTC/MediaDevices browser access | OpenSSL certificates (`fullchain.pem`, `privkey.pem`) |
| **System Documentation**| `/docs` | Architectural deep-dive guides, handbooks, and conceptual user manuals | `docs/user_understanding/`, `SETU_DEEP_ARCHITECTURE_GUIDE.md` |

---

## 3. Master File-by-File Purpose Dictionary

### Root Workspace (`/`)
* **Why the Folder Exists:** The root directory acts as the orchestration anchor for developer setup, container orchestration, environment variable defaults, dependency definitions, packaging configuration, and top-level execution triggers.

1. **`.dockerignore`**: Excludes Python virtual environments, local node_modules, build directories, heavy datasets, and temporary caches from Docker build contexts.
2. **`.env.example`**: Standardized environment variable template defining camera device indices, model paths, CORS origins, and server ports.
3. **`.gitignore`**: Git repository ignore specification protecting binary model weights, datasets, virtual environments, and caches from accidental version control tracking.
4. **`docker-compose.yml`**: Multi-container Docker Compose specification orchestrating the FastAPI backend and Nginx frontend services with volume mounts and health checks.
5. **`Gemini.zip`**: Codebase archive snapshot packaging all active modules, scripts, models, and tests while strictly omitting heavy dependencies.
6. **`LICENSE`**: Standard MIT open-source license granting permissive access, modification, and distribution.
7. **`pyproject.toml`**: Enterprise Python packaging specification configuring package discovery for `backend`, `models`, `training`, and `config` alongside pytest settings.
8. **`README.md`**: Primary repository landing page delivering a comprehensive project overview, quickstart instructions, and a direct link to this master file index.
9. **`requirements.txt`**: Pinned top-level Python dependency specification for backend, ML, computer vision, and API operations.
10. **`Start_SETU.bat`**: Windows double-click launcher checking Python/Node environments and launching both the backend uvicorn server and Vite frontend in dedicated console windows.

---

### `/backend` (FastAPI Server & Real-Time Engine)
* **Why the Folder Exists:** Encapsulates the entire server-side application logic, providing thread-safe RESTful API endpoints, duplex WebSocket video frame streaming handlers, Pydantic validation schemas, and modular vision/NLP service layers.

11. **`backend/Dockerfile`**: Container definition building the Python 3.10-slim runtime, system OpenGL/MediaPipe libraries, and Uvicorn production server.
12. **`backend/requirements.txt`**: Specialized Python dependency manifest pinned for server-side FastAPI, Uvicorn, MediaPipe, and PyTorch inference.
13. **`backend/main.py`**: Backend package entrypoint providing backward-compatible re-exports for `uvicorn backend.main:app`.
14. **`backend/app/__init__.py`**: Root Python package initializer for the modular backend application.
15. **`backend/app/main.py`**: Central FastAPI application factory configuring OpenAPI metadata, CORS middleware, correlation ID tracing (`X-Request-ID`), and latency telemetry headers.

#### `/backend/app/api/endpoints` (RESTful Endpoint Routers)
* **Why the Folder Exists:** Provides clean separation of concern for HTTP endpoints, segregating model cataloging, offline video uploads, dataset collection, vocabulary queries, and single-frame predictions.

16. **`backend/app/api/endpoints/__init__.py`**: Subpackage initializer grouping all endpoint routers for consolidated inclusion in `backend/app/main.py`.
17. **`backend/app/api/endpoints/dataset.py`**: REST endpoints managing user sign calibration recordings, take persistence, and custom dataset management.
18. **`backend/app/api/endpoints/models.py`**: REST endpoint inspecting the active `ModelRegistry`, querying hardware capabilities (CUDA/CPU), and switching active inference engines.
19. **`backend/app/api/endpoints/predict.py`**: High-speed REST endpoint handling single-frame landmark and feature array inference payloads.
20. **`backend/app/api/endpoints/video.py`**: REST endpoint accepting multi-frame MP4/AVI video file uploads for asynchronous MediaPipe landmark extraction and sequence classification.
21. **`backend/app/api/endpoints/vocabulary.py`**: REST endpoint returning active vocabulary lists, class indices, localized English gloss labels, and model categories.

#### `/backend/app/schemas` (Pydantic Data Contracts)
* **Why the Folder Exists:** Enforces strict type validation, request/response data contracts, and boundary safety across all API interactions using Pydantic v2.

22. **`backend/app/schemas/__init__.py`**: Schema subpackage initializer exporting validation models.
23. **`backend/app/schemas/predict.py`**: Pydantic schemas validating landmark coordinate inputs (`LandmarkRequest`), top-k candidates (`TopCandidate`), and classification results (`PredictionResponse`).

#### `/backend/app/services` (Business Logic & Engine Adapters)
* **Why the Folder Exists:** Houses thread-safe inference services, pipeline management, hardware dispatchers, and domain-specific processing engines.

24. **`backend/app/services/__init__.py`**: Service layer package initializer.
25. **`backend/app/services/classifier.py`**: Backward-compatibility facade exposing `classifier_service` and `RealSignMLP` to legacy consumers.
26. **`backend/app/services/pipeline_manager.py`**: High-level orchestrator managing camera readers, holistic landmark extractors, and active sign classifier instances with graceful fallback logic.
27. **`backend/app/services/realsign_classifier_service.py`**: Production inference service loading PyTorch/ONNX RealSign 26-class models with coordinate normalization and thread-safe forward passes.

#### `/backend/app/services/nlp` (Natural Language Processing Subsystem)
* **Why the Folder Exists:** Translates raw sign predictions and stream glosses into fluent English sentences using temporal streak filtering, refractory windows, and SOV-to-SVO grammar reordering.

28. **`backend/app/services/nlp/__init__.py`**: Package initializer exposing `SentenceAccumulator` and `ISLGlossParser`.
29. **`backend/app/services/nlp/accumulator.py`**: Real-time sign stabilizer aggregating consistent frame predictions over time to eliminate single-frame visual flicker.
30. **`backend/app/services/nlp/gloss_parser.py`**: Rule-based linguistic parser transforming Indian Sign Language SOV (Subject-Object-Verb) gloss structures into natural SVO English.

#### `/backend/app/services/vision` (Computer Vision & Landmark Detection)
* **Why the Folder Exists:** Encapsulates physical webcam frame acquisition, DirectShow backend fallbacks, and Google MediaPipe Holistic/Hand landmark extraction.

31. **`backend/app/services/vision/__init__.py`**: Package initializer exposing `CameraReader`, `HolisticLandmarkerWrapper`, and `HandLandmarkerWrapper`.
32. **`backend/app/services/vision/camera.py`**: Robust OpenCV video capture wrapper supporting device index selection, resolution negotiation, and graceful hardware fallbacks.
33. **`backend/app/services/vision/landmarks.py`**: High-performance MediaPipe wrapper extracting 33 pose landmarks, 468 face landmarks, and 21 left/right hand coordinates.

#### `/backend/app/websocket` (Real-Time Duplex Streaming)
* **Why the Folder Exists:** Implements sub-30ms bidirectional streaming over WebSockets, ingesting base64 video frames from web clients and streaming back stabilized sign predictions, confidence scores, and skeletal mesh coordinates.

34. **`backend/app/websocket/__init__.py`**: WebSocket routing package initializer.
35. **`backend/app/websocket/recognize.py`**: Core real-time streaming handler receiving JPEG/base64 frames, invoking the active classifier, maintaining temporal state, and broadcasting JSON predictions.

---

### `/frontend` (React 19 + TypeScript + Vite UI Workstation)
* **Why the Folder Exists:** Hosts the modern, responsive web application enabling signers and observers to view live camera feeds, skeletal landmark overlays, real-time translated subtitles, audio speech synthesis, and custom gesture recording.

36. **`frontend/Dockerfile`**: Multi-stage production container compiling React with Vite and serving production static assets through Nginx.
37. **`frontend/.oxlintrc.json`**: Ultra-fast linting configuration enforcing modern ECMAScript standards and React Hooks best practices.
38. **`frontend/index.html`**: HTML5 root entry document declaring responsive viewports, Google Fonts (Inter, Roboto), and React root mount node.
39. **`frontend/nginx.conf`**: High-performance Nginx web server configuration handling client-side routing, WebSocket reverse proxying, and SSL termination.
40. **`frontend/package.json`**: Node.js project manifest defining React 19, TypeScript, Lucide Icons, Vite, and development dependencies.
41. **`frontend/package-lock.json`**: Cryptographic lockfile pinning deterministic Node dependency tree versions.
42. **`frontend/tsconfig.app.json`**: TypeScript configuration tailored for React application source code compilation.
43. **`frontend/tsconfig.json`**: Top-level TypeScript project configuration referencing application and Node configurations.
44. **`frontend/tsconfig.node.json`**: TypeScript compiler options governing Vite tooling and configuration scripts.
45. **`frontend/vite.config.ts`**: Vite build bundler configuration defining React plugins, dev server ports, and proxy routes.

#### `/frontend/public` (Static Brand & PWA Assets)
* **Why the Folder Exists:** Delivers browser favicons, web app manifests, sitemaps, and static branding assets directly to the browser root.

46. **`frontend/public/apple-touch-icon.png`**: Apple iOS home screen touch icon.
47. **`frontend/public/favicon-16x16.png`**: 16x16 pixel favicon for desktop browser tabs.
48. **`frontend/public/favicon-32x32.png`**: 32x32 pixel standard desktop favicon.
49. **`frontend/public/favicon.ico`**: Multi-resolution legacy ICO favicon.
50. **`frontend/public/logo.png`**: Web application logo asset.
51. **`frontend/public/manifest.json`**: Progressive Web App (PWA) manifest declaring theme colors, display modes, and launcher icons.
52. **`frontend/public/robots.txt`**: Search engine crawler directive file.
53. **`frontend/public/sitemap.xml`**: XML search engine sitemap.

#### `/frontend/src` (Client Application Source)
* **Why the Folder Exists:** Contains all React components, custom hooks, centralized API client services, canvas renderers, and style tokens.

54. **`frontend/src/App.css`**: Application-specific layout overrides and animations.
55. **`frontend/src/App.tsx`**: Central application state controller coordinating camera streams, modal visibility, and workspace switching.
56. **`frontend/src/index.css`**: Core design system defining CSS variables, dark-mode color palettes, typography, and responsive utilities.
57. **`frontend/src/main.tsx`**: React DOM root bootstrap mounting the App component into the document.
58. **`frontend/src/types.ts`**: Shared TypeScript type definitions specifying prediction structures, landmark coordinate arrays, and UI states.

#### `/frontend/src/assets` (Bundled UI Graphics)
* **Why the Folder Exists:** Contains vector graphics and illustrations bundled directly by the Vite build pipeline.

59. **`frontend/src/assets/hero.png`**: Landing page hero illustration.
60. **`frontend/src/assets/logo.png`**: High-resolution brand logo bundled into React components.
61. **`frontend/src/assets/react.svg`**: React vector graphic logo.

#### `/frontend/src/components` (Modular UI Panels & Views)
* **Why the Folder Exists:** Implements self-contained, accessible, and reactive visual interfaces for video feeds, recognition outputs, modal dialogues, and navigation.

62. **`frontend/src/components/AlphabetModal.tsx`**: Interactive reference modal illustrating A-Z sign poses with visual landmark hints.
63. **`frontend/src/components/CameraControls.tsx`**: Control strip managing webcam power, camera flip, audio synthesis toggles, and mesh rendering flags.
64. **`frontend/src/components/CameraWorkspace.tsx`**: Primary viewport rendering live webcam video feeds with synchronized HTML5 Canvas skeletal landmark overlays.
65. **`frontend/src/components/HelpModal.tsx`**: Comprehensive user assistance guide detailing gestures, keyboard shortcuts, and troubleshooting steps.
66. **`frontend/src/components/HistoryModal.tsx`**: Audit modal displaying a chronological log of all translated phrases during the session with export options.
67. **`frontend/src/components/IncludeVideoModal.tsx`**: Video recognition modal allowing users to test inference by uploading MP4 clips.
68. **`frontend/src/components/LiveDetectionOutput.tsx`**: Real-time HUD displaying the active sign classification, confidence meter, and top-3 candidate distribution.
69. **`frontend/src/components/SavedModal.tsx`**: Management modal for reviewing, replaying, and deleting bookmarked translation snippets.
70. **`frontend/src/components/SettingsModal.tsx`**: Configuration modal allowing users to adjust WebSocket endpoints, confidence thresholds, and TTS speech rate.
71. **`frontend/src/components/TextToSignWorkspace.tsx`**: Reverse translator panel converting English text inputs into animated avatar sign gesture sequences.
72. **`frontend/src/components/TopNavbar.tsx`**: Header navigation bar displaying the SETU brand, active model selector, backend health badge, and modal triggers.
73. **`frontend/src/components/TranslationPanel.tsx`**: Dynamic subtitle dock rendering stabilized sentences, gloss history, and one-click speech playback buttons.
74. **`frontend/src/components/VisualFramingGuide.tsx`**: Proactive visual framing overlay rendering semi-transparent upper body silhouette, signing-space reticles, and real-time HUD status badge.
75. **`frontend/src/components/VocabularyModal.tsx`**: Catalog modal displaying the complete list of supported signs and phrases for the currently active model.
76. **`frontend/src/components/WebcamAlphabetRecognition.tsx`**: Specialized client-side recognition panel running live alphabet classification directly in the browser.

#### `/frontend/src/hooks` (Custom Reactive Hooks)
* **Why the Folder Exists:** Decouples complex asynchronous operations (WebSocket duplex streaming, speech synthesis, text accumulation) from UI component rendering.

77. **`frontend/src/hooks/useRecognitionStream.ts`**: Custom hook managing WebSocket connection lifecycle, frame dispatching, and streaming prediction ingestion with framing gating.
78. **`frontend/src/hooks/useSentenceBuilder.ts`**: Reactive hook translating raw sign glosses into finalized English sentences with cooldown timers.
79. **`frontend/src/hooks/useTextBuffer.ts`**: State buffer hook managing letter-by-letter alphabet typing, space insertion, and backspace editing.
80. **`frontend/src/hooks/useTTS.ts`**: Browser SpeechSynthesis API wrapper delivering natural text-to-speech audio with language and pitch configuration.

#### `/frontend/src/services` (Centralized API Client)
* **Why the Folder Exists:** Centralizes all outgoing HTTP requests into a strongly-typed, predictable service layer with automatic error handling.

81. **`frontend/src/services/apiService.ts`**: Centralized API client invoking backend REST endpoints (`/health`, `/models`, `/predict`, `/vocabulary`, `/video`).

#### `/frontend/src/utils` (Canvas Geometry & Drawing)
* **Why the Folder Exists:** Provides ultra-fast, pure TypeScript algorithms for drawing MediaPipe skeletal meshes, calculating joint angles, and parsing raw landmarks.

82. **`frontend/src/utils/framingEvaluator.ts`**: High-performance, zero-allocation client-side CV evaluator computing shoulder distance, centering, headroom, and hand positioning heuristics.
83. **`frontend/src/utils/landmarkProcessor.ts`**: Utility functions formatting landmark coordinate tuples and calculating geometric hand properties.
84. **`frontend/src/utils/landmarksDrawer.ts`**: High-performance HTML5 Canvas renderer drawing hand bones, palm joints, and facial landmarks with custom neon color palettes.

---

### `/models` (AI Model Layer — Code & Weights Split)
* **Why the Folder Exists:** Houses the complete machine learning model layer, strictly separating Python adapter inference code from binary weights, pretrained models, and MediaPipe task bundles.

83. **`models/__init__.py`**: Model layer root package initializer re-exporting `ModelRegistry` and canonical model adapters.
84. **`models/registry.py`**: Central ModelRegistry registering available models, metadata, input shapes, and lazy-loading mechanisms.
85. **`models/model_registry.py`**: Backward-compatibility shim pointing legacy consumers to `models.registry`.
86. **`models/realsign.py`**: Compatibility shim re-exporting `RealSignClassifier`, `RealSignMLP`, and classes from `models.adapters.realsign`.
87. **`models/mvp.py`**: Compatibility shim re-exporting `ISLMvpClassifier` from `models.adapters.mvp`.
88. **`models/include.py`**: Compatibility shim re-exporting `INCLUDEInferenceService` from `models.adapters.include`.
89. **`models/sequence_lstm.py`**: Compatibility shim re-exporting `ISLSequenceLSTMClassifier` from `models.adapters.sequence_lstm`.
90. **`models/rf_uv11.py`**: Compatibility shim re-exporting `RFUV11Adapter` from `models.adapters.rf_uv11`.
91. **`models/isl_conversational_mvp.py`**: Legacy name alias mapping to `models.adapters.mvp`.
92. **`models/include_video_lstm_adapter.py`**: Legacy name alias mapping to `models.adapters.include`.
93. **`models/isl_phrase_sequence_lstm_adapter.py`**: Legacy name alias mapping to `models.adapters.sequence_lstm`.
94. **`models/realsign_alphabet_mlp_adapter.py`**: Legacy name alias mapping to `models.adapters.realsign`.
95. **`models/roboflow_yolov11_adapter.py`**: Legacy name alias mapping to `models.adapters.rf_uv11`.

#### `/models/adapters` (Python Model Adapters)
* **Why the Folder Exists:** Encapsulates model-specific preprocessing, neural network forward graph definitions, and output probability mapping behind a uniform adapter interface.

96. **`models/adapters/__init__.py`**: Subpackage initializer exporting all model adapters.
97. **`models/adapters/include.py`**: Inference service and PyTorch BiLSTM neural network executing predictions across 263 INCLUDE sign classes.
98. **`models/adapters/mvp.py`**: Ultra-fast pure NumPy Ridge classifier executing 5-class conversational sign inference with sub-millisecond latency.
99. **`models/adapters/realsign.py`**: Production 26-class alphabet PyTorch MLP classifier operating on wrist-normalized 63-D hand landmark coordinates.
100. **`models/adapters/rf_uv11.py`**: Lightweight reference adapter executing YOLOv11 sign bounding-box and landmark detection via Roboflow weights.
101. **`models/adapters/sequence_lstm.py`**: Dynamic phrase classifier utilizing bidirectional LSTM with temporal attention across 30-frame landmark sequences.

#### `/models/weights` (Production Binary Weights)
* **Why the Folder Exists:** Houses production-ready machine learning weights, checkpoints, label maps, and ONNX binaries.

102. **`models/weights/isl_mvp_model.npz`**: Compressed NumPy archive storing Ridge regression weight matrices and intercepts for 5 conversational phrases.
103. **`models/weights/isl_sequence_lstm.pth`**: PyTorch checkpoint containing trained weights for the bidirectional sequence LSTM phrase classifier.
104. **`models/weights/label_map.json`**: JSON dictionary mapping numerical class indices to human-readable sign glosses.
105. **`models/weights/realsign_26class_mlp.onnx`**: Optimized ONNX graph of the 26-class alphabet MLP for high-speed cross-platform runtime execution.
106. **`models/weights/realsign_26class_mlp.pt`**: TorchScript compiled binary of the 26-class alphabet MLP.
107. **`models/weights/realsign_26class_mlp.pth`**: PyTorch state dictionary containing trained weights for the 99.11% accuracy alphabet classifier.

#### `/models/tasks` (MediaPipe Binary Task Bundles)
* **Why the Folder Exists:** Stores Google MediaPipe pre-packaged neural models for hand detection and holistic body tracking.

108. **`models/tasks/README.md`**: Documentation explaining MediaPipe task bundle origins, download links, and version hashes.
109. **`models/tasks/hand_landmarker.task`**: Official MediaPipe binary bundle detecting 21 3D hand joints.
110. **`models/tasks/holistic_landmarker.task`**: Official MediaPipe binary bundle detecting body pose, face mesh, and hands simultaneously.

#### `/models/pretrained` & `/models/reference` (External Reference Models)
* **Why the Folder Exists:** Retains external research checkpoints and reference artifacts for baseline comparison and auditing.

111. **`models/pretrained/evidence_report.md`**: Quantitative audit report validating inference latency and accuracy of the external RF-UV11 model.
112. **`models/pretrained/include_no_cnn_lstm.pth`**: Baseline research checkpoint for the INCLUDE sign classifier without CNN feature extractors.
113. **`models/pretrained/label_map_include.json`**: Class label mapping for the 263-class INCLUDE dataset.
114. **`models/pretrained/README.md`**: Technical overview of the pretrained model artifacts.
115. **`models/pretrained/LICENSE_APACHE_2.0.txt`**: Apache 2.0 open source license terms for the RF-UV-11 reference model.
116. **`models/pretrained/rf_uv_11_model.h5`**: HDF5 neural network weights for the RF-UV-11 single-hand 63-D gesture classification baseline.
117. **`models/reference/isl_model.h5`**: Legacy Keras/TensorFlow model retained exclusively as a baseline reference.
118. **`models/reference/README.md`**: Reference model documentation detailing input shapes and historical benchmark context.

---

### `/training` (Machine Learning Training & Processing Pipelines)
* **Why the Folder Exists:** Contains complete, reproducible data pipelines for ingesting raw sign images and videos, extracting invariant geometric features, generating negative samples, training classifiers, evaluating confusion matrices, and exporting to ONNX.

119. **`training/__init__.py`**: Training subsystem package initializer.
120. **`training/dataset_manifest.py`**: Signer-split manifest generation utility managing train/val/test video partitions and metadata schemas.
121. **`training/feature_extractor.py`**: Canonical coordinate normalizer providing wrist-centered origin translation, scale invariance, and 63-D/158-D extraction.
122. **`training/extract_features.py`**: Batch video landmark extraction pipeline converting video collections into structured feature archives.
123. **`training/ingest_raw_datasets.py`**: Raw image ingestion pipeline parsing directory structures, extracting features, and generating stratified splits.
124. **`training/train_realsign_26class.py`**: PyTorch training script for the 26-class alphabet MLP with Cosine Annealing learning rate schedules and early stopping.
125. **`training/evaluate.py`**: Quantitative evaluation tool generating multi-class confusion matrices, per-class F1-scores, and JSON performance reports.
126. **`training/export_onnx.py`**: Automated exporter converting PyTorch checkpoints to ONNX graphs and running CPU inference benchmarks.

#### `/training/features` (Coordinate Mathematics & Schema)
* **Why the Folder Exists:** Defines the mathematical foundations of invariant landmark processing, joint angle calculations, and feature dimension contracts.

127. **`training/features/__init__.py`**: Feature package initializer exposing schema constants and normalizers.
128. **`training/features/extractor.py`**: Low-level feature extractor producing normalized spatial coordinates and inter-joint velocity vectors.
129. **`training/features/geometry.py`**: Vector geometry module computing 3D Euclidean distances, palm normal vectors, and joint angles.
130. **`training/features/schema.py`**: Specification declaring canonical feature dimensions (63-D, 126-D, 158-D, 164-D, 316-D, 328-D).

#### Supplementary Training & Ingestion Facades
* **Why the Folder Exists:** Preserves backward-compatible CLI invocations for specialized model architectures and legacy scripts.

131. **`training/build_conversational_5sign_manifest.py`**: Specialized manifest builder for the 5-sign conversational dataset.
132. **`training/build_mvp_manifest.py`**: Backward-compatibility facade for the conversational manifest builder.
133. **`training/evaluate_realsign_mlp.py`**: Backward-compatibility facade for alphabet model evaluation.
134. **`training/export_realsign_mlp_to_onnx.py`**: Backward-compatibility facade for alphabet ONNX export.
135. **`training/extract_holistic_158d_features.py`**: Pipeline script extracting 158-D holistic body and hand features from video files.
136. **`training/extract_realsign_63d_features.py`**: Pipeline script extracting 63-D single-hand coordinates from RealSign image directories.
137. **`training/generate_negative_resting_samples.py`**: Algorithmic negative sample generator producing resting and out-of-frame samples for `no_sign` robustness.
138. **`training/ingest_raw.py`**: Backward-compatibility facade for raw dataset ingestion.
139. **`training/ingest_realsign_image_dataset.py`**: Backward-compatibility facade for RealSign image dataset processing.
140. **`training/isl_dataset_manifest_schema.py`**: Data schema definition for ISL dataset manifests.
141. **`training/no_sign_generator.py`**: Backward-compatibility facade for negative sample generation.
142. **`training/train.py`**: Backward-compatibility facade for running default alphabet training.
143. **`training/train_conversational_5sign_ridge.py`**: Scikit-learn Ridge regression training script for conversational signs.
144. **`training/train_isl_phrase_bilstm.py`**: PyTorch training script for the bidirectional sequence LSTM model with temporal attention.
145. **`training/train_mvp_model.py`**: Backward-compatibility facade for Ridge regression training.
146. **`training/train_realsign_26class_mlp.py`**: Backward-compatibility facade for RealSign alphabet training.
147. **`training/train_sequence_lstm.py`**: Backward-compatibility facade for sequence LSTM training.

---

### `/records` (Living Master Records & Ledger)
* **Why the Folder Exists:** Serves as the central repository audit tracking center, maintaining architecture specifications, rollback changelogs, model catalogs, and this master file index.

148. **`records/ARCHITECTURE.md`**: Canonical system architecture specification documenting module responsibilities, data flows, and reliability guarantees.
149. **`records/CHECKPOINTS.md`**: Chronological milestone changelog detailing structural modifications, refactoring passes, and rollback procedures.
150. **`records/DATASETS_AND_MODELS.md`**: Machine learning catalog detailing dataset sources, sample counts, neural architectures, and training hyperparameters.
151. **`records/MEMORY.md`**: Active project ledger documenting completed milestones, current codebase status, and strategic roadmap goals.
152. **`records/STRUCTURE.md`**: This document — the living master file-by-file dictionary and directory index.

---

### `/config` (Centralized Settings & Exceptions)
* **Why the Folder Exists:** Centralizes all system-wide environment configuration, hardware camera indices, confidence thresholds, and domain-specific exception hierarchies.

153. **`config/__init__.py`**: Configuration package initializer.
154. **`config/exceptions.py`**: Custom domain exception hierarchy (`SETUError`, `FeatureValidationError`, `LandmarkExtractionError`, `CameraError`, `ModelArtifactError`).
155. **`config/settings.py`**: Immutable dataclass parsing environment variables with type validation and sensible defaults.

---

### `/reports` (Model Evaluation & Verification Reports)
* **Why the Folder Exists:** Stores machine-readable JSON performance reports, confusion matrices, and dataset bias audits.

156. **`reports/isl_sequence_lstm_evaluation.json`**: Evaluation metrics and confusion matrix log for the sequence LSTM phrase model.
157. **`reports/realsign_26class_evaluation_report.json`**: Official performance report card logging 99.11% accuracy for the RealSign alphabet MLP.
158. **`reports/realsign_audit_report.json`**: Dataset balance report confirming frame distributions across all 26 alphabet classes.
159. **`reports/realsign_hand_pilot_report.json`**: Latency and detection rate benchmarks for single-hand landmark trials.
160. **`reports/realsign_leakage_report.json`**: Audit report proving zero image or signer leakage between train, validation, and test splits.
161. **`reports/realsign_standalone_hand_pilot_report.json`**: Benchmarking log assessing standalone MediaPipe HandLandmarker throughput.

---

### `/scripts` (Project Utilities, Benchmarks & Standalone Demos)
* **Why the Folder Exists:** Houses CLI diagnostic tools, hardware verification scripts, remote dataset downloaders, and standalone visual demos.

162. **`scripts/audit_project.py`**: CLI repository health auditor verifying file integrity, weight existence, and import paths.
163. **`scripts/audit_realsign_dataset.py`**: Data quality utility inspecting image headers and reporting unreadable frames.
164. **`scripts/audit_realsign_leakage.py`**: Cryptographic hash auditor ensuring no sample overlap across training partitions.
165. **`scripts/build_overview_notebook.py`**: Generator script programmatically assembling `notebooks/overview_of_setu.ipynb`.
166. **`scripts/calibrate_and_train_webcam_alphabet.py`**: Interactive webcam calibration tool fine-tuning alphabet models on user-specific hand shapes.
167. **`scripts/check_dataset_sources.py`**: Network checker probing availability of external dataset mirror URLs.
168. **`scripts/check_framing_evaluator.js`**: Node test runner validating 8 client-side framing constraints, distance heuristics, and hand-presence gating checks.
169. **`scripts/clean_realsign_split.py`**: Data sanitization script generating leak-free train/val/test split manifests.
169. **`scripts/demo_conversational_5sign_webcam.py`**: Standalone OpenCV demo executing live 5-class conversational sign classification.
170. **`scripts/demo_realsign_26class_webcam.py`**: Standalone OpenCV demo running live 26-class alphabet recognition with HUD overlays.
171. **`scripts/demo_roboflow_yolov11_webcam.py`**: Standalone demo executing sign inference via the reference YOLOv11 model.
172. **`scripts/download_include_subset.py`**: Downloader script pulling the 15-video INCLUDE benchmark subset.
173. **`scripts/download_realsign_dataset.py`**: Downloader script fetching the raw RealSign alphabet image archives.
174. **`scripts/extract_realsign_landmarks.py`**: Landmark extractor generating pre-computed feature arrays from raw RealSign images.
175. **`scripts/fetch_include_info.py`**: Metadata extractor parsing INCLUDE dataset video catalogs and signer annotations.
176. **`scripts/fetch_include_splits.py`**: Downloader script fetching official train/val/test video partition splits for INCLUDE.
177. **`scripts/generate_certs.ps1`**: PowerShell script generating local self-signed SSL/TLS certificates for HTTPS testing.
178. **`scripts/inspect_remote_dataset_zip_headers.py`**: Network diagnostic tool testing HTTP byte-range support on remote dataset servers.
179. **`scripts/package_gemini.py`**: Automated repository packager generating clean `Gemini.zip` code snapshots.
180. **`scripts/process_include_subset.py`**: Video processing script extracting 158-D landmark sequences from INCLUDE MP4 files.
181. **`scripts/record_custom_signs.py`**: Interactive 5-take calibration tool allowing users to record custom gesture samples.
182. **`scripts/run_controlled_alphabet_test.py`**: Automated evaluation runner executing inference benchmarks across test datasets.
183. **`scripts/run_hand_detector_pilot.py`**: Benchmarking tool measuring MediaPipe hand detection frame processing latency.
184. **`scripts/run_standalone_hand_pilot.py`**: Standalone benchmarking script profiling isolated hand tracking performance.
185. **`scripts/Start_Alphabet_Demo.bat`**: Windows batch launcher for the standalone OpenCV alphabet visual demo.
186. **`scripts/test_include_inference.py`**: Verification script validating end-to-end model inference on INCLUDE sample clips.
187. **`scripts/verify_include_dataset_zenodo_api.py`**: API checker validating Zenodo record metadata, deposit files, and checksums for INCLUDE.
188. **`scripts/webcam_physical_verification.py`**: Hardware verification script confirming webcam device availability, resolution, and DirectShow capture.

---

### `/tests` (Automated Pytest Suite)
* **Why the Folder Exists:** Contains modular, automated unit and integration tests enforcing mathematical correctness, API contracts, WebSocket protocols, and model inference fidelity (133/133 passing).

189. **`tests/__init__.py`**: Pytest test suite root package initializer.
190. **`tests/test_api.py`**: Integration tests asserting HTTP endpoints, health checks, and WebSocket handshake behavior.
191. **`tests/test_api_v1_alignment.py`**: Route alignment tests asserting strict `/api/v1/` prefixing and deprecation guards.
192. **`tests/test_backend_app_modular.py`**: Unit tests verifying modular FastAPI endpoints, Pydantic schemas, and inference services.
193. **`tests/test_camera_and_holistic_vision.py`**: Computer vision tests validating OpenCV camera capture, fallback modes, and MediaPipe landmark detection.
194. **`tests/test_config_exceptions.py`**: Unit tests confirming configuration validation, boundary checks, and custom domain exceptions.
195. **`tests/test_custom_sign_pipeline.py`**: Pipeline tests verifying interactive sign calibration, landmark recording, and feature serialization.
196. **`tests/test_feature_extraction_and_manifest_pipeline.py`**: End-to-end integration tests asserting complete flow from video frames to translated English sentences.
197. **`tests/test_feature_schema.py`**: Schema validation tests asserting exact array dimensions and bounds across 158-D and 126-D features.
198. **`tests/test_include_model.py`**: Model tests validating inference accuracy and weight integrity for the 263-class INCLUDE classifier.
199. **`tests/test_invariant_features_and_segmentation.py`**: Mathematical unit tests proving scale and rotation invariance across arbitrary hand poses.
200. **`tests/test_isl_mvp_model.py`**: Unit tests asserting numerical correctness and classification speed of the 5-class NumPy Ridge model.
201. **`tests/test_isl_sequence_lstm_classifier.py`**: Unit tests asserting output shape and temporal sliding-window behavior of the BiLSTM model.
202. **`tests/test_nlp_pipeline.py`**: NLP tests verifying streak accumulation, refractory delays, and SOV-to-SVO grammar parsing.
203. **`tests/test_real_data_readiness.py`**: Data readiness tests checking file existence, feature dimensions, and manifest schemas.
204. **`tests/test_realsign_classifier.py`**: Classifier tests asserting 99%+ accuracy and ONNX runtime parity for the RealSign alphabet model.
205. **`tests/test_rf_uv11_adapter.py`**: Compatibility tests validating prediction outputs from the external RF-UV-11 reference adapter.
206. **`tests/test_ui_audit_and_contract.py`**: Contract tests validating UI payload compatibility, subtitle zone flags, and WebSocket schemas.

---

### `/certs` (Local Development SSL/TLS Certificates)
* **Why the Folder Exists:** Provides local cryptographic keys and certificates allowing modern browsers to grant webcam and microphone permissions over HTTPS during local development.

205. **`certs/.gitkeep`**: Git placeholder ensuring directory structure is preserved across clones.
206. **`certs/fullchain.pem`**: Local development SSL public certificate for HTTPS server binding.
207. **`certs/privkey.pem`**: Local development SSL private key corresponding to `fullchain.pem`.

---

### `/data` (Local Datasets & Manifests — Git-Ignored)
* **Why the Folder Exists:** Houses local raw videos, image directories, processed NumPy landmark arrays, and signer-split dataset manifests.

208. **`data/manifests/setu_isl_mvp_manifest.json`**: Official JSON manifest indexing 100 conversational sign video takes with signer splits.
209. **`data/processed/`**: Directory storing normalized `.npz` feature vectors extracted from video and image datasets.
210. **`data/raw/`**: Directory storing uncompressed video clips and raw RealSign image collections.

---

### `/docs` (Architecture Guides & Archival Records)
* **Why the Folder Exists:** Houses in-depth technical architecture handbooks, licensing resolutions, SVG system diagrams, and archival backups.

211. **`docs/aju22_license_resolution.md`**: Legal and licensing compliance documentation resolving open-source dataset usage terms.
212. **`docs/milestone1_data_acquisition_report.json`**: Quantitative audit log detailing dataset acquisition counts and integrity checks.
213. **`docs/phase1a_specifications.md`**: Phase 1A system engineering specification defining vision pipeline constraints and frame rates.
214. **`docs/phase1b_coordinate_verification.md`**: Mathematical validation report confirming wrist-centered rotation invariance formulas.
215. **`docs/phase1b_vocabulary_selection.md`**: Linguistic rationale document outlining selection criteria for initial conversational ISL vocabulary.
216. **`docs/SETU_DEEP_ARCHITECTURE_GUIDE.md`**: Deep technical architecture manual explaining coordinate normalization, threading, and latency targets.
217. **`docs/SETU_Sign_Language_Translator_Build_Handbook.docx`**: Complete comprehensive build and operational handbook for the SETU project.
218. **`docs/SETU_system_overview.svg`**: High-level vector architectural diagram showing end-to-end data flow from camera to speech synthesis.
219. **`docs/webcam_verification_results.json`**: Verification report logging hardware camera latency and frame acquisition benchmarks.
220. **`docs/webcam_verification_snapshot.jpg`**: Visual test snapshot confirming correct DirectShow webcam initialization.
221. **`docs/archive/SETU_Foundation_Documents_2026-10-17_Updated.zip`**: Archived collection of foundational design and project initiation documents.
222. **`docs/archive/SETU_Project.zip`**: Archived full-project snapshot taken during early development iterations.

#### `/docs/user_understanding` (Conceptual Architecture Guides)
* **Why the Folder Exists:** Provides accessible, high-level conceptual reference manuals explaining end-to-end pipeline mechanics, user flows, and system roadmaps.

223. **`docs/user_understanding/ARCHITECTURE_AND_PIPELINE.md`**: Comprehensive reference manual explaining vision and model pipeline mechanics.
224. **`docs/user_understanding/END_TO_END_FLOW.md`**: Detailed step-by-step trace of frame lifecycle from webcam capture to audio synthesis.
225. **`docs/user_understanding/MODEL_AND_DATASET_SPEC.md`**: Technical specification document detailing dataset origins and neural architectures.
226. **`docs/user_understanding/PROJECT_OVERVIEW.md`**: High-level overview of project goals, technical stack, and social impact.
227. **`docs/user_understanding/ROADMAP_AND_SELF_AUDIT.md`**: Self-audit checklist assessing code health, security, and feature completeness.
228. **`docs/user_understanding/VISUAL_DASHBOARD.md`**: Visual design specification defining color contrast, fonts, and responsive layout grids.

---

### `/notebooks` (Interactive Walkthroughs)
* **Why the Folder Exists:** Provides interactive, executable Jupyter Notebooks for exploring the vision pipeline, feature geometry, and model inference step-by-step.

229. **`notebooks/overview_of_setu.ipynb`**: Complete interactive notebook demonstrating camera capture, landmark extraction, and alphabet inference with visual plots.
