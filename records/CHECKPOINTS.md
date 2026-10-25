# SETU — Milestone Checkpoints & Rollback Registry

> **System:** SETU (Indian Sign Language Translator)  
> **Status:** Active Audit & Rollback Registry  
> **Repository:** `c:\MINI PROJ-SETU`  
> **Last Updated:** October 2026  

This document serves as the canonical audit trail and rollback safety net for the SETU project. If any component, model weight, or UI feature suffers corruption, degradation, or regression, developers and agents can inspect the exact commit hash, review what changed, and execute precise rollback procedures.

---

## 1. Rollback Execution Guidelines

To restore the project to any verified milestone below, execute the corresponding git checkout commands from PowerShell or Git Bash within the root repository:

```powershell
# 1. Stash or inspect current uncommitted work
git status
git stash save "work_in_progress_before_rollback"

# 2. Checkout the desired checkpoint tag or commit hash
git checkout <COMMIT_HASH>

# 3. If verifying or testing in detached HEAD state, reinstall dependencies if needed
pip install -r requirements.txt
npm --prefix frontend install

# 4. To revert cleanly to the main branch after inspection
git checkout main
git stash pop
```

---

### Checkpoint 22: Root Micro-Cleanup Pass & Documentation Consolidation
- **Date:** October 2026
- **Scope & Changes:**
  - **Root File Sanitization:**
    - Removed redundant root `logo.png` (branding assets standardized in `frontend/public/logo.png` and `frontend/src/assets/logo.png`).
    - Removed legacy root `main.py` (FastAPI server execution standardized inside `backend/app/main.py` and `backend/main.py`).
    - Updated `backend/Dockerfile` to eliminate legacy `COPY main.py .` in favor of `COPY config/ ./config/`.
    - Deleted `setu.egg-info/` build directory; ensured `setu.egg-info/` is explicitly ignored in `.gitignore` and excluded from `scripts/package_gemini.py`.
  - **Documentation Consolidation:**
    - Relocated `user_understanding/` into `docs/user_understanding/` (`ARCHITECTURE_AND_PIPELINE.md`, `END_TO_END_FLOW.md`, `MODEL_AND_DATASET_SPEC.md`, `PROJECT_OVERVIEW.md`, `ROADMAP_AND_SELF_AUDIT.md`, `VISUAL_DASHBOARD.md`), preserving the repository root exclusively for core environment files, Docker Compose, package definitions, and launchers.
  - **Living Master File Index Update (`records/STRUCTURE.md`):**
    - Updated ASCII directory tree to eliminate root `logo.png` and `main.py`, and nested `user_understanding/` under `docs/`.
    - Re-indexed all 228 active repository files with contiguous sequential numbering (1 to 228) and 100% disk path verification.
  - **Verification & Build:**
    - Full automated test suite: 125/125 tests passing.
    - Frontend production build: 0 errors, clean Vite compilation.
- **Affected Files:**
  - `README.md` (updated logo link to `frontend/public/logo.png`)
  - `backend/Dockerfile` (updated copy directives)
  - `.gitignore` (added `setu.egg-info/`)
  - `docs/user_understanding/` (consolidated 6 architecture manuals)
  - `records/STRUCTURE.md`, `records/ARCHITECTURE.md`, `records/CHECKPOINTS.md`, `records/MEMORY.md`

---

### Checkpoint 21: Enterprise Separation of Concerns, Legacy Elimination & Master Index Standard
- **Date:** October 2026
- **Scope & Changes:**
  - **Zero Root Clutter & Enterprise Target Layout:** Restructured repository to strict enterprise layout:
    - `backend/`: FastAPI server, modular endpoint routers (`dataset`, `models`, `predict`, `video`, `vocabulary`), schemas, vision (`camera.py`, `landmarks.py`), and NLP services (`accumulator.py`, `gloss_parser.py`).
    - `models/`: Strictly split into `models/adapters/` (code), `models/weights/` (binaries), `models/tasks/` (MediaPipe task bundles), with central `models/registry.py`.
    - `training/`: Canonical pipelines (`dataset_manifest.py`, `feature_extractor.py`, `extract_features.py`, `ingest_raw_datasets.py`, `train_realsign_26class.py`, `evaluate.py`, `export_onnx.py`) with invariant geometry math in `training/features/`.
    - `records/`: Auto-updating living ledger (`ARCHITECTURE.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`, `MEMORY.md`, `STRUCTURE.md`).
    - `config/`: Centralized settings (`settings.py`) and exception hierarchy (`exceptions.py`).
  - **Elimination of Legacy Redundancies:** Safely pruned obsolete, duplicate directories:
    - Removed `ml/` (superseded by `training/`).
    - Removed `configs/` (superseded by `config/`).
    - Removed `setu/` (fully partitioned into `backend/app/services/`, `training/features/`, `config/exceptions.py`).
    - Removed `Project Main Files/` (superseded by canonical living documentation in `records/`).
  - **Master File Index Rule Enforcement (`records/STRUCTURE.md`):** Complete recursive catalog of every active file and directory with architectural rationale ("Why the Folder Exists") and exact plain-language purpose description.
  - **Verification & Parity:**
    - Automated Test Suite: **125/125 passing tests** in 31.09s with zero failures.
    - Frontend Production Build: **1941 modules transformed**, 0 errors, gzip size 102kB in 989ms.
    - Editable package install updated via `pip install -e .` with clean `pyproject.toml` package discovery.
- **Affected Files:**
  - `backend/` (`app/services/vision/`, `app/services/nlp/`, `app/websocket/recognize.py`)
  - `models/` (`adapters/`, `weights/`, `registry.py`)
  - `training/` (`feature_extractor.py`, `features/`)
  - `config/` (`exceptions.py`, `settings.py`)
  - `records/` (`STRUCTURE.md`, `ARCHITECTURE.md`, `CHECKPOINTS.md`, `MEMORY.md`)
  - `README.md`
  - `Start_SETU.bat`

---

### Checkpoint 20: Principal Architectural Refactoring, Enterprise Relocation & Semantic Hardening
- **Date:** October 2026
- **Scope & Changes:**
  - **Zero Root Clutter:** Established the canonical enterprise living documentation directory `records/` and relocated all 5 living project tracking files (`ARCHITECTURE.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`, `MEMORY.md`, `STRUCTURE.md`) into it.
  - **Model Architecture Semantic Disambiguation:** Upgraded all model adapters in `models/` with explicit semantic naming (`models/include_video_lstm_adapter.py`, `models/isl_phrase_sequence_lstm_adapter.py`, `models/model_registry.py`, `models/realsign_alphabet_mlp_adapter.py`, `models/isl_conversational_mvp.py`, `models/roboflow_yolov11_adapter.py`), backed by 100% transparent zero-overhead backward-compatibility facades.
  - **Training Pipeline Decoupling & Semantic Standardization:** Created canonical explicit training modules in `training/` (`isl_dataset_manifest_schema.py`, `extract_holistic_158d_features.py`, `extract_realsign_63d_features.py`, `evaluate_realsign_mlp.py`, `export_realsign_mlp_to_onnx.py`, `generate_negative_resting_samples.py`, `ingest_realsign_image_dataset.py`, `train_realsign_26class_mlp.py`, `train_conversational_5sign_ridge.py`, `train_isl_phrase_bilstm.py`, `build_conversational_5sign_manifest.py`), providing comprehensive backward-compatibility facades for older training script calls.
  - **Domain-Structured Test Disambiguation:** Superseded ambiguous generic test files (`test_pipeline.py` -> `test_feature_extraction_and_manifest_pipeline.py`, `test_vision.py` -> `test_camera_and_holistic_vision.py`).
  - **Verification & Parity:** 100% test pass rate (**125 passed of 125 tests** in 30.23s) and successful production frontend compilation (**0 errors**, 959ms).
- **Affected Files:**
  - `records/` (`ARCHITECTURE.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`, `MEMORY.md`, `STRUCTURE.md`)
  - `models/` (`include_video_lstm_adapter.py`, `isl_phrase_sequence_lstm_adapter.py`, `model_registry.py`, `realsign_alphabet_mlp_adapter.py`, `isl_conversational_mvp.py`, `roboflow_yolov11_adapter.py`)
  - `training/` (`isl_dataset_manifest_schema.py`, `extract_holistic_158d_features.py`, `extract_realsign_63d_features.py`, `evaluate_realsign_mlp.py`, `export_realsign_mlp_to_onnx.py`, `generate_negative_resting_samples.py`, `ingest_realsign_image_dataset.py`, `train_realsign_26class_mlp.py`, `train_conversational_5sign_ridge.py`, `train_isl_phrase_bilstm.py`, `build_conversational_5sign_manifest.py`, plus facades)
  - `tests/` (`test_feature_extraction_and_manifest_pipeline.py`, `test_camera_and_holistic_vision.py`)
  - `Gemini.zip`

---

### Checkpoint 19: Systematic Codebase Renaming, Disambiguation & Import Harmonization
- **Date:** October 2026
- **Scope & Changes:**
  - Audited full repository to eliminate vague, generic, or ambiguous filenames (`utils.py`, `helper.js`, `classifier.py`, `mvp.py`, `rf_uv11.py`, `button.md`, `phase.md`).
  - Renamed backend classification service from `classifier.py` to `realsign_classifier_service.py` to disambiguate within a multi-model system.
  - Renamed `models/mvp.py` to `models/isl_conversational_mvp.py` and `models/rf_uv11.py` to `models/roboflow_yolov11_adapter.py`.
  - Renamed 5 cryptic/vague scripts to clear, descriptive names: `verify_include_dataset_zenodo_api.py`, `inspect_remote_dataset_zip_headers.py`, `calibrate_and_train_webcam_alphabet.py`, `demo_conversational_5sign_webcam.py`, and `demo_roboflow_yolov11_webcam.py`.
  - Renamed generic markdown files in `Project Main Files/` to self-explanatory titles: `workspace_button_interaction_spec.md`, `phased_delivery_roadmap.md`, `engineering_working_rules.md`, `design_system_and_ux_guidelines.md`, `historical_sprint_memory_20261017.md`, `baseline_architecture_spec_20261017.md`, and `project_specifications_readme.md`.
  - Maintained zero-downtime backward-compatibility facades and shims on all original paths so no external or legacy workflows break.
  - Updated all references across `KEY_FILES_INDEX.md`, `PROJECT_STATUS.md`, `PROJECT_STRUCTURE.md`, root `README.md`, `STRUCTURE.md`, `ARCHITECTURE.md`, `MEMORY.md`, and refreshed `Gemini.zip`.
- **Affected Files:**
  - `backend/app/services/realsign_classifier_service.py` (created) & `classifier.py` (facade)
  - `models/isl_conversational_mvp.py` (created) & `models/mvp.py` (facade)
  - `models/roboflow_yolov11_adapter.py` (created) & `models/rf_uv11.py` (facade)
  - `scripts/verify_include_dataset_zenodo_api.py` & `scripts/check_zenodo.py`
  - `scripts/inspect_remote_dataset_zip_headers.py` & `scripts/inspect_zip_range.py`
  - `scripts/calibrate_and_train_webcam_alphabet.py` & `scripts/train_az_realsign.py`
  - `scripts/demo_conversational_5sign_webcam.py` & `scripts/demo_isl_mvp_webcam.py`
  - `scripts/demo_roboflow_yolov11_webcam.py` & `scripts/demo_rf_uv11_webcam.py`
  - `Project Main Files/*.md` (7 files renamed with pointers)
  - `STRUCTURE.md`, `ARCHITECTURE.md`, `MEMORY.md`, `CHECKPOINTS.md`, `Gemini.zip`

---

### Checkpoint 18: Exhaustive Project File Mapping, Loose File Cleanup & Packaging
- **Date:** October 2026
- **Scope & Changes:**
  - Created `STRUCTURE.md` providing an exhaustive recursive project tree and individual 1-sentence purpose descriptions for all 241 project files.
  - Relocated loose root artifacts: moved `SETU_Sign_Language_Translator_Build_Handbook.docx` to `docs/`, legacy snapshot archives to `docs/archive/`, and removed redundant root `overview_of_setu.ipynb` (consolidated in `notebooks/`).
  - Consolidated training, feature extraction, evaluation, and ONNX graph export pipelines directly inside `training/` with clean backward-compatibility shims in `ml/`.
  - Added export validations in `training/__init__.py` and integration tests in `tests/test_backend_app_modular.py`.
  - Packaged complete repository into `Gemini.zip` excluding non-essential build/cache folders (`.git`, `.venv`, `node_modules`, `dist`, `__pycache__`).
- **Affected Files:**
  - `STRUCTURE.md` (created)
  - `docs/SETU_Sign_Language_Translator_Build_Handbook.docx` (relocated)
  - `docs/archive/SETU_Project.zip` (relocated)
  - `docs/archive/SETU_Foundation_Documents_2026-10-17_Updated.zip` (relocated)
  - `training/feature_extractor.py` (added)
  - `training/evaluate.py` (consolidated)
  - `training/export_onnx.py` (consolidated)
  - `training/train.py` (added)
  - `training/ingest_raw.py` (added)
  - `ml/__init__.py` & shims (updated)
  - `Gemini.zip` (packaged)

---

### Checkpoint 17: UI Decoupling, Dead Code Purge & Structured Logging (Phase 3 & 4)
- **Date:** October 2026
- **Scope & Changes:**
  - Extracted temporal debouncing, refractory hysteresis, and sentence accumulation logic out of `App.tsx` into a decoupled custom hook `useSentenceBuilder.ts`.
  - Created centralized `apiService.ts` in `frontend/src/services/` for typed REST requests, timeout protection, and standardized error normalization.
  - Purged orphaned, unused legacy components (`Header.tsx`, `Sidebar.tsx`, `StatusStrip.tsx`, `TopBar.tsx`).
  - Added request correlation ID middleware (`X-Request-ID` UUID) and latency tracking (`X-Response-Time-MS`) in `backend/app/main.py`.
  - Full test suite verification: 125 of 125 tests passed in 23.38s. Frontend build clean with 0 TypeScript errors.
- **Affected Files:**
  - `frontend/src/hooks/useSentenceBuilder.ts`
  - `frontend/src/services/apiService.ts`
  - `frontend/src/App.tsx`
  - `frontend/src/components/Header.tsx` (deleted)
  - `frontend/src/components/Sidebar.tsx` (deleted)
  - `frontend/src/components/StatusStrip.tsx` (deleted)
  - `frontend/src/components/TopBar.tsx` (deleted)
  - `backend/app/main.py`

---

### Checkpoint 16: Enterprise Separation of Concerns & Model Registry (Phase 2)
- **Date:** October 2026
- **Scope & Changes:**
  - Reorganized repository into canonical enterprise structure:
    - `/backend`: Server logic, modular routers (`predict`, `models`, `video`, `dataset`, `vocabulary`), WebSocket connection handler, and `pipeline_manager.py`.
    - `/models`: Decoupled ML model inference scripts (`realsign.py`, `mvp.py`, `sequence_lstm.py`, `include.py`, `rf_uv11.py`), weights, and swappable `ModelRegistry`.
    - `/training`: Dataset manifest generation, feature extraction, NO_SIGN generation, model training, evaluation reports, and ONNX graph export.
    - `/config`: Centralized frozen dataclass configuration with environment variable validation (`config/settings.py`).
  - Maintained backward compatibility shim in `setu.api.server` and `setu.models` for `Start_SETU.bat` and legacy test suites.
- **Affected Files:**
  - `backend/app/api/endpoints/`
  - `backend/app/services/pipeline_manager.py`
  - `backend/app/websocket/recognize.py`
  - `backend/app/main.py`
  - `models/registry.py`
  - `models/__init__.py`
  - `models/realsign.py`, `models/mvp.py`, `models/sequence_lstm.py`, `models/include.py`, `models/rf_uv11.py`
  - `training/__init__.py`, `training/evaluate.py`, `training/export_onnx.py`
  - `config/settings.py`, `configs/settings.py`

---

### Checkpoint 15: Architectural Mapping & Living Documentation (Phase 1)
- **Date:** October 2026
- **Scope & Changes:**
  - Created 4 living core tracking documents at root level:
    - `ARCHITECTURE.md`: 6-layer architecture map, Mermaid data flow, module dictionary, environment configurations, and security model.
    - `MEMORY.md`: Continuous project memory tracking delivery status, milestones, technical debt, and maintenance rules.
    - `CHECKPOINTS.md`: Comprehensive audit trail and rollback procedure across all 17 milestones.
    - `DATASETS_AND_MODELS.md`: Complete ML registry with datasets, architectures, biases, errors/resolutions, and training hyperparameters.
- **Affected Files:**
  - `ARCHITECTURE.md`
  - `MEMORY.md`
  - `CHECKPOINTS.md`
  - `DATASETS_AND_MODELS.md`

---

### Checkpoint 14: Enterprise Micro-Animations & Tactile Physics
- **Date:** October 2026
- **Commit Reference:** Current Working Tree (Pre-Restructuring)
- **Scope & Changes:**
  - Implemented `@keyframes pulseGlow` breathing animation on active WebSocket status beacon and live camera beacon dot.
  - Implemented 300ms ease-out opacity transitions on live video feed activation and standby card.
  - Implemented subtle pulsing border (`@keyframes cameraInitPulse` with `--setu-accent-primary`) during camera initialization.
  - Added uniform `active:scale-95` micro-interaction with 100ms duration across all buttons, tabs, and toggles.
  - Added smooth 300ms 180° rotation on the 2-way language toggle arrow icon.
  - Added scanning shimmer track with animated beam and skeleton placeholder lines in sentence buffer.
  - Added GPU-accelerated 200ms backdrop blur fades and scale-in transforms (`scale(0.98) translateY(4px) to scale(1) translateY(0)`) on all modals and dropdowns.
  - Added strict `prefers-reduced-motion` override for complete accessibility compliance.
- **Affected Files:**
  - `frontend/src/index.css`
  - `frontend/src/components/CameraWorkspace.tsx`
  - `frontend/src/components/TranslationPanel.tsx`
  - `frontend/src/components/TopNavbar.tsx`
- **Verification Status:** Verified via Vite production build (`npm run build`, 1.06s) and Playwright automated browser test.

---

### Checkpoint 13: Official Brand Favicon Suite & Minimalist Standby View
- **Date:** October 2026
- **Commit Reference:** Working Tree
- **Scope & Changes:**
  - Extracted core SETU abstract bridge symbol into `favicon.svg`, multi-resolution `favicon.ico`, `favicon-32x32.png`, `favicon-16x16.png`, and `apple-touch-icon.png`.
  - Replaced camera standby cluttered badges and heavy constellations with minimalist 56px aperture icon and direct "Start Camera" CTA.
  - Pinned movie-style subtitles permanently to bottom of camera viewport during live detection to achieve zero Cumulative Layout Shift (CLS = 0).
  - Streamlined right navbar pills: removed redundant wifi icon, standardized heights (28px) and border radii (6px).
- **Affected Files:**
  - `frontend/index.html`
  - `frontend/public/` (all favicon assets)
  - `frontend/src/components/CameraWorkspace.tsx`
  - `frontend/src/components/TopNavbar.tsx`
  - `frontend/src/index.css`

---

### Checkpoint 12: 5-Take ISL Alphabet Calibration & Windows Video Capture
- **Commit Reference:** `672f931` (and `db2898e`, `42426a8`, `f288536`, `4bc1bec`)
- **Scope & Changes:**
  - Configured 5 takes default with spacebar-confirmed progression for custom sign data recording.
  - Added DirectShow `cv2.CAP_DSHOW` video capture backend on Windows to eliminate camera initialization freezes.
  - Added hot-reload capability and non-fatal KeyboardInterrupt handling to prevent data loss during long calibration runs.
- **Affected Files:**
  - `scripts/record_custom_signs.py`
  - `setu/vision/camera.py`
  - `training/ingest_raw_datasets.py`

---

### Checkpoint 11: Real-Time 26-Class Alphabet Recognition Full-Stack
- **Commit Reference:** `2848486` (and `6671467`, `09142a3`)
- **Scope & Changes:**
  - Ingested RealSign dataset and trained PyTorch `RealSignMLP` to 99.11% validation accuracy across 26 alphabet classes (A–Z).
  - Exported model to ONNX runtime format (`models/realsign_26class_mlp.onnx`).
  - Implemented 126-D hand-centric invariant normalization (63 left + 63 right wrist-normalized coordinates).
  - Added real-time alphabet classification route and modal in frontend.
- **Affected Files:**
  - `models/realsign_26class_mlp.pth`
  - `models/realsign_26class_mlp.onnx`
  - `setu/models/realsign_classifier.py`
  - `frontend/src/components/AlphabetModal.tsx`
  - `training/train_realsign_26class.py`

---

### Checkpoint 10: Vision Pipeline Stabilization & MediaPipe Tasks
- **Commit Reference:** `9198947`
- **Scope & Changes:**
  - Updated MediaPipe holistic and hand landmarker tasks to stable task bundles.
  - Containerized backend services with Docker and docker-compose.
- **Affected Files:**
  - `models/tasks/holistic_landmarker.task`
  - `models/tasks/hand_landmarker.task`
  - `setu/vision/landmarks.py`
  - `docker-compose.yml`

---

### Checkpoint 9: NLP Gloss Parser, Invariant Geometry & TTS Integration
- **Commit Reference:** `76b9a10`
- **Scope & Changes:**
  - Added `ISLGlossParser` to convert ISL SOV syntax into standard English SVO sentences.
  - Added `SentenceAccumulator` to manage streak-based sign commits and cooldown buffering.
  - Integrated browser-native SpeechSynthesis TTS engine.
- **Affected Files:**
  - `setu/nlp/gloss_parser.py`
  - `setu/nlp/accumulator.py`
  - `frontend/src/components/TranslationPanel.tsx`

---

### Checkpoint 8: Architecture Guides & Visual Documentation Overhaul
- **Commit Reference:** `64aadbc`, `5166591`, `a36f32f`, `1bc95eb`
- **Scope & Changes:**
  - Comprehensive documentation overhaul with visual SVG diagrams, mathematical feature formulation, and diagnostic benchmarks.
- **Affected Files:**
  - `README.md`
  - `docs/SETU_DEEP_ARCHITECTURE_GUIDE.md`
  - `docs/SETU_system_overview.svg`

---

### Checkpoint 7: Initial SETU ISL Recognition SaaS Implementation
- **Commit Reference:** `79826de`
- **Scope & Changes:**
  - Initial end-to-end integration of React frontend with FastAPI WebSocket server.
  - Streaming recognition of signs over local WebSocket connection.
- **Affected Files:**
  - `frontend/src/App.tsx`
  - `setu/api/server.py`
  - `Start_SETU.bat`

---

### Checkpoint 6: Real ISL Subset & Signer Isolation (Phase 1B)
- **Commit Reference:** `e4639a6`
- **Scope & Changes:**
  - Acquired real ISL dataset samples from AI4Bharat INCLUDE.
  - Enforced strict signer-independent splits to eliminate data leakage.
  - Validated physical webcam invariant features.
- **Affected Files:**
  - `data/manifests/setu_isl_mvp_manifest.json`
  - `tests/test_real_data_readiness.py`

---

### Checkpoint 5: Schema v1.0.0 Freeze & NO_SIGN Negative Class Generator
- **Commit Reference:** `09045de`
- **Scope & Changes:**
  - Froze 158-D feature schema contract (`FeatureExtractor158`).
  - Added synthetic and webcam-recorded `no_sign` background negative class generator.
- **Affected Files:**
  - `setu/features/schema.py`
  - `setu/features/extractor.py`
  - `training/no_sign_generator.py`

---

### Checkpoint 4: Modularization of Core Engine (Phase 1)
- **Commit Reference:** `6777562`
- **Scope & Changes:**
  - Separated vision detection, feature extraction, model inference, configuration, and exceptions into modular subpackages.
- **Affected Files:**
  - `setu/vision/`
  - `setu/features/`
  - `setu/models/`
  - `setu/config.py`
  - `setu/exceptions.py`

---

### Checkpoint 1: Initial Repository Baseline (Phase 0)
- **Commit Reference:** `277db72`
- **Scope & Changes:**
  - Initial repository setup and asset preservation.
- **Affected Files:**
  - Base repository configuration and dependencies.

---

## 3. Quick Rollback Command Reference

| Milestone Goal | Target Commit Hash | Direct Command |
| :--- | :--- | :--- |
| **Restore Pre-Restructuring State** | Current HEAD | `git checkout HEAD` |
| **Restore 5-Take Calibration State** | `672f931` | `git checkout 672f931` |
| **Restore Alphabet 26-Class State** | `2848486` | `git checkout 2848486` |
| **Restore NLP & Gloss Parser State** | `76b9a10` | `git checkout 76b9a10` |
| **Restore Initial SaaS MVP State** | `79826de` | `git checkout 79826de` |
| **Restore Schema Freeze State** | `09045de` | `git checkout 09045de` |
| **Restore Initial Baseline** | `277db72` | `git checkout 277db72` |
