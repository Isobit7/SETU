# SETU — Active Project Memory Log

> **System:** SETU (Indian Sign Language Translator)  
> **Repository:** `c:\MINI PROJ-SETU`  
> **Last Synced:** October 2026  
> **Status:** Phase 1–4 Complete (Architectural Mapping, Enterprise Restructuring, UI Decoupling, Error Hardening Active)  
> **Maintainer Protocol:** This file must be updated at the conclusion of every session or major milestone.

---

## 1. Project Overview & Mission

**SETU** is an authentic, local-first Indian Sign Language (ISL) to English real-time translation web application. It combines:
1. **Frontend:** React 19 + TypeScript + Vite responsive workstation with enterprise dark theme (modeled after Google Gemini and ChatGPT).
2. **Backend:** FastAPI high-throughput WebSocket engine streaming at 15–20 FPS with < 30ms latency.
3. **ML Inference:** Multi-tier model pipeline including 26-class PyTorch RealSignMLP (99.11% validation accuracy), 5-class ISL MVP NumPy model (80% test accuracy), and 263-class AI4Bharat INCLUDE sequence classifier.
4. **NLP Translation:** ISLGlossParser converting sign syntax (SOV) into natural English grammar (SVO) with text-to-speech (TTS) audio synthesis.

---

## 2. Project Delivery Phase Tracker

| Phase | Title | Status | Primary Outcome / Artifact |
| :--- | :--- | :--- | :--- |
| **P0** | **Preserve & Audit** | ✅ Complete | Repo baseline frozen, uncommitted state cataloged, dependencies verified. |
| **P1** | **Camera & Vision Foundation** | ✅ Complete | MediaPipe Holistic & Hand tasks integrated with DirectShow Windows fix. |
| **P2** | **Candidate Reuse Feasibility** | ✅ Complete | Isolated RF-UV-11 reference adapter; rejected synthetic models as primary. |
| **P3** | **Real-Data Five-Sign MVP** | ✅ Complete | AI4Bharat INCLUDE subset (`hello`, `thank_you`, `good_morning`, `i`, `you`) trained. |
| **P4** | **Translator Frontend & API** | ✅ Complete | Unified workstation, real landmark canvas, movie-style subtitles, auto-TTS. |
| **P5** | **RealSign 26-Class Alphabet** | ✅ Complete | Trained PyTorch RealSignMLP to 99.11% accuracy; exported to ONNX. |
| **P6** | **UI/UX Enterprise Polish** | ✅ Complete | Google Gemini dark palette, minimalist standby view, full favicon suite, micro-animations. |
| **P7** | **Codebase Restructuring & Hygiene** | ✅ Complete | Decoupled `/backend`, `/models`, `/training`, `/config`; purged dead code; 125/125 tests passing. |
| **P8** | **Exhaustive Mapping & Packaging** | ✅ Complete | Generated `STRUCTURE.md` (241 files cataloged), relocated loose files, generated `Gemini.zip`. |
| **P9** | **Architectural Refactoring & Relocation** | ✅ Complete | Zero root clutter, `records/` living documentation hub, full semantic renaming pass, 125/125 tests passing. |
| **P10**| **Enterprise Layout & Master File Index** | ✅ Complete | Purged legacy folders (`ml/`, `configs/`, `setu/`, `Project Main Files/`); enforced `models/weights/` & `models/adapters/` split; living master dictionary in `records/STRUCTURE.md`; 125/125 tests passing. |
| **P11**| **Root Micro-Cleanup & Doc Consolidation** | ✅ Complete | Sanitized root (`logo.png`, `main.py`, `setu.egg-info/` pruned); consolidated `docs/user_understanding/`; 228 files verified in index; 125/125 tests passing. |

---

## 3. Recent Major Accomplishments (Session Log)

### A. Root Micro-Cleanup Pass & Documentation Consolidation (October 2026)
- **Root File Sanitization:** Pruned redundant root-level `logo.png` (branding managed inside `frontend/public/logo.png`) and legacy `main.py` (server startup standardized inside `backend/app/main.py` and `backend/main.py`).
- **Build Directory Cleanup:** Deleted `setu.egg-info/` artifact directory and added explicit ignore rules to `.gitignore` and `scripts/package_gemini.py`.
- **Documentation Consolidation:** Relocated `user_understanding/` into `docs/user_understanding/` (`ARCHITECTURE_AND_PIPELINE.md`, `END_TO_END_FLOW.md`, `MODEL_AND_DATASET_SPEC.md`, `PROJECT_OVERVIEW.md`, `ROADMAP_AND_SELF_AUDIT.md`, `VISUAL_DASHBOARD.md`). Root directory is now strictly reserved for environment templates, orchestration files, package configs, and launchers.
- **Living Master File Index Update (`records/STRUCTURE.md`):** Updated the ASCII tree and purpose dictionary to reflect all 228 verified files on disk with strictly contiguous 1..228 numbering.
- **Full Verification:** Test suite confirmed 100% operational with 125/125 passing tests; frontend build clean with zero errors.

### B. Enterprise Separation of Concerns & Living Master Index Standard (October 2026)
- **Zero Root Clutter & Strict Separation of Concerns:** Enforced the target enterprise directory structure across `backend/`, `frontend/`, `models/`, `training/`, `records/`, `config/`, `scripts/`, `reports/`, `tests/`, `certs/`, and `data/`.
- **Model Layer Code vs. Weights Split:** Segmented the AI model layer cleanly into `models/adapters/` (pure Python forward adapters), `models/weights/` (production binaries: `.pth`, `.onnx`, `.pt`, `.npz`, `label_map.json`), and `models/tasks/` (MediaPipe `.task` bundles).
- **Backend Services Modularization:** Created structured service domains: `backend/app/services/vision/` (`camera.py`, `landmarks.py`) and `backend/app/services/nlp/` (`accumulator.py`, `gloss_parser.py`).
- **Elimination of Legacy Duplications:** Pruned obsolete directories: `ml/` (merged into `training/`), `configs/` (merged into `config/`), `setu/` (partitioned cleanly into backend services, training features, and config exceptions), and `Project Main Files/` (superseded by `records/`).
- **Living Master File Index (`records/STRUCTURE.md`):** Complete recursive catalog of every active file in the repository, explicitly documenting both **Why the Folder Exists** and the **Exact Purpose** of each file.
- **Verification & Parity:** Verified 100% test suite pass rate (**125 passed of 125 tests** in 31.09s) and verified production frontend compilation (**0 errors**, 989ms).

### B. Principal Architectural Refactoring, Enterprise Relocation & Semantic Hardening (October 2026)
- **Zero Root Clutter & Living Documentation Hub (`records/`):** Relocated all 5 core tracking records (`ARCHITECTURE.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`, `MEMORY.md`, `STRUCTURE.md`) into the dedicated `records/` directory. Root is completely sanitized to essential project orchestration files.
- **Model Architecture Semantic Disambiguation:** Disambiguated all adapters in `models/` into explicit domain modules (`include_video_lstm_adapter.py`, `isl_phrase_sequence_lstm_adapter.py`, `model_registry.py`, `realsign_alphabet_mlp_adapter.py`, `isl_conversational_mvp.py`, `roboflow_yolov11_adapter.py`) with 100% backward-compatibility facades.
- **Training Pipeline Decoupling & Semantic Standardization:** Replaced abbreviated scripts with canonical explicit implementations (`isl_dataset_manifest_schema.py`, `extract_holistic_158d_features.py`, `extract_realsign_63d_features.py`, `evaluate_realsign_mlp.py`, `export_realsign_mlp_to_onnx.py`, `generate_negative_resting_samples.py`, `ingest_realsign_image_dataset.py`, `train_realsign_26class_mlp.py`, `train_conversational_5sign_ridge.py`, `train_isl_phrase_bilstm.py`, `build_conversational_5sign_manifest.py`).
- **Domain-Structured Test Refactoring:** Upgraded generic test files to `test_feature_extraction_and_manifest_pipeline.py` and `test_camera_and_holistic_vision.py`.
- **Complete Test & Build Verification:** 100% test pass rate (**125 passed of 125 tests**) and successful production frontend build (**0 errors**, 959ms).

### B. Exhaustive File Mapping & Clean Packaging (October 2026)
- **Comprehensive Structure Dictionary:** Created `STRUCTURE.md` containing recursive tree of all 241 project files with explicit 1-sentence purpose descriptions.
- **Root Sanitization & Archive Relocation:** Relocated loose Word documentation (`SETU_Sign_Language_Translator_Build_Handbook.docx`) to `docs/`, legacy snapshot archives to `docs/archive/`, and removed duplicate root notebook.
- **Training Consolidation:** Migrated all machine learning training pipelines, feature extractors, and ONNX graph exporters directly into `training/`, providing clean backward-compatibility shims in `ml/`.
- **Gemini.zip Generation:** Packaged all source code, models, configs, training code, and documentation into a single clean root archive `Gemini.zip` excluding build/cache artifacts.

### B. Enterprise Restructuring, Decoupling & Error Hardening (October 2026)
- **Five Core Living Documents:** Maintained `STRUCTURE.md`, `ARCHITECTURE.md`, `MEMORY.md`, `CHECKPOINTS.md`, and `DATASETS_AND_MODELS.md` at root level.
- **Strict Separation of Concerns:**
  - `/backend`: Modular routers for predict, models, video recognition, custom datasets, vocabulary, and WebSocket streaming. Pipeline manager isolated from routing.
  - `/models`: Swappable `ModelRegistry` pattern with decoupled inference scripts (`realsign.py`, `mvp.py`, `sequence_lstm.py`, `include.py`, `rf_uv11.py`).
  - `/training`: Consolidated dataset manifest generation, feature extraction, training, evaluation reporting, and ONNX graph export.
  - `/config`: Centralized frozen dataclass settings with environment variable parsing (`config/settings.py`).
- **UI Decoupling & Code Hygiene:**
  - Extracted temporal streak debouncing and sentence assembly out of `App.tsx` into `useSentenceBuilder.ts`.
  - Created centralized `apiService.ts` for typed REST operations and timeout resilience.
  - Purged 4 dead/orphaned legacy components (`Header.tsx`, `Sidebar.tsx`, `StatusStrip.tsx`, `TopBar.tsx`).
- **Standardized Error Handling & Observability:**
  - Request logging middleware in FastAPI with `X-Request-ID` UUID correlation and millisecond execution tracking (`X-Response-Time-MS`).
  - Validated 100% of the unit and integration test suite: **125 of 125 tests passed in 23.38s**.
  - Frontend production build succeeded with **0 TypeScript errors** in 855ms.

### C. Systematic Codebase Renaming & Disambiguation (October 2026)
- **Eliminated Ambiguous and Poorly Named Files:**
  - `backend/app/services/classifier.py` $\to$ `backend/app/services/realsign_classifier_service.py` (explicit domain responsibilities in multi-classifier system).
  - `models/mvp.py` $\to$ `models/isl_conversational_mvp.py` (clarified 5-sign conversational ridge model).
  - `models/rf_uv11.py` $\to$ `models/roboflow_yolov11_adapter.py` (self-explanatory Roboflow YOLOv11 adapter).
  - `scripts/check_zenodo.py` $\to$ `scripts/verify_include_dataset_zenodo_api.py`.
  - `scripts/inspect_zip_range.py` $\to$ `scripts/inspect_remote_dataset_zip_headers.py`.
  - `scripts/train_az_realsign.py` $\to$ `scripts/calibrate_and_train_webcam_alphabet.py`.
  - `scripts/demo_isl_mvp_webcam.py` $\to$ `scripts/demo_conversational_5sign_webcam.py`.
  - `scripts/demo_rf_uv11_webcam.py` $\to$ `scripts/demo_roboflow_yolov11_webcam.py`.
  - Relocated and clarified `Project Main Files/` docs: `button.md` $\to$ `workspace_button_interaction_spec.md`, `phase.md` $\to$ `phased_delivery_roadmap.md`, `rules.md` $\to$ `engineering_working_rules.md`, `design.md` $\to$ `design_system_and_ux_guidelines.md`, `memory.md` $\to$ `historical_sprint_memory_20261017.md`, `architecture.md` $\to$ `baseline_architecture_spec_20261017.md`, `README.md` $\to$ `project_specifications_readme.md`.
- **Zero-Downtime Backward Compatibility:** Maintained non-breaking re-export facades and runpy redirection shims on all original file paths so no existing imports or external scripts break.
- **Synchronized System Documentation:** Updated `KEY_FILES_INDEX.md`, `PROJECT_STATUS.md`, `PROJECT_STRUCTURE.md`, root `README.md`, `STRUCTURE.md`, `ARCHITECTURE.md`, `MEMORY.md`, and `CHECKPOINTS.md`.

### B. Complete Enterprise UI/UX Refactor (October 2026)
- **Palette Harmonization:** Replaced fragmented colors with standard `:root` design tokens:
  - `--setu-bg-base: #131314`
  - `--setu-bg-surface: #1E1F20`
  - `--setu-border-subtle: #323538`
  - `--setu-accent-primary: #8AB4F8`
  - `--setu-accent-hover: #A8C7FA`
  - `--setu-text-main: #E3E3E3`
  - `--setu-text-muted: #9AA0A6`
- **Camera Standby Decluttering:** Removed all redundant kicker badges, capabilities chips, and heavy constellations. Replaced with an ultra-clean 56px aperture icon, authentic title, and clear "Start Camera" CTA button.
- **Subtitle Dock Pinning:** Completely eliminated layout shifts (CLS = 0) by docking movie-style subtitles permanently at the bottom during live detection.
- **Right-Side Navigation Pill Polish:** Removed redundant nested wifi icon from Online pill to leave a single luminous status beacon. Standardized pill heights (28px) and border radius (6px).
- **Official Brand Favicon Suite:** Extracted the core SETU logo mark into crisp, multi-scale assets:
  - `favicon.svg` (crisp scalable vector)
  - `favicon.ico` (multi-resolution 16x16, 32x32, 48x48)
  - `favicon-32x32.png` & `favicon-16x16.png`
  - `apple-touch-icon.png` (180x180 for iOS/macOS)
  - `site.webmanifest`
- **Enterprise Micro-Animations & Physics:**
  - `pulseGlow` breathing animation on active WebSocket status beacon.
  - Smooth 300ms ease-out opacity transitions on camera feed activation and standby.
  - Active-state micro-interactions (`transform: scale(0.95)` with 100ms duration) across all buttons, tabs, and toggles.
  - Smooth 300ms 180° rotation on the 2-way language toggle arrow icon.
  - Scanning shimmer track with moving beam and skeleton placeholder lines inside sentence buffer.
  - GPU-accelerated 200ms backdrop blur fades and scale-in transforms (`scale(0.98) translateY(4px) to scale(1) translateY(0)`) on all modals and dropdowns.
  - Strict `prefers-reduced-motion` override for full accessibility compliance.

---

## 4. Architectural Rules & Active Constraints

1. **Radical Authenticity:** Never inject mock metrics, simulated hardware states, or fake confidence percentages. If hands are absent or confidence is low, truthfully display "Awaiting sign input".
2. **Local Frame Processing:** Video frames are never stored or streamed to third-party endpoints. All MediaPipe detection and model inference occurs on the user's host machine.
3. **No Breaking Regressions:** Any restructuring must keep `Start_SETU.bat` fully operational, ensure `uvicorn` backend launches cleanly, and ensure `npm run build` succeeds with zero errors.
4. **Signer Independence:** Machine learning evaluations must enforce strict signer splits to avoid data leakage.

---

## 5. Technical Debt & Clutter Inventory

1. **Root Directory Clutter:** Multiple standalone scripts (`app.py`, `main.py`, batch files, legacy zips) scattered at root level.
2. **Parallel Backend Structures:** Both `backend/app/` and `setu/api/` exist. The active server used by `Start_SETU.bat` is `setu.api.server:app`. The project needs a single unified `/backend` structure.
3. **Model & Weight Organization:** Model scripts exist in `setu/models/` while weights exist in `models/`. These need a clean separation where `/models` provides decoupled inference adapters and weights.
4. **Training Scripts:** Training code is split between `training/`, `ml/`, and `scripts/`.
5. **Config Centralization:** Runtime configuration exists in `setu/config.py` and `configs/settings.py`.

---

## 6. Immediate Action Items (Protocol Roadmap)

- [x] **Phase 1: Architectural Mapping & Living Documentation**
  - [x] Create `ARCHITECTURE.md` (System topology, request flows, directory map)
  - [x] Create `MEMORY.md` (Active state, recent accomplishments, constraints)
  - [x] Create `CHECKPOINTS.md` (Changelog, git commit references, rollbacks)
  - [x] Create `DATASETS_AND_MODELS.md` (Dataset origins, model registry, metrics, biases)
- [x] **Phase 2: Strict Separation of Concerns & Folder Restructuring**
  - [x] Consolidate backend routing into `/backend`
  - [x] Decouple model weights and inference engines into `/models`
  - [x] Consolidate dataset utilities and training pipelines into `/training`
  - [x] Centralize environment validation into `/config`
- [x] **Phase 3: Decoupling & Code Hygiene**
  - [x] Extract socket event listeners and API calls out of React UI components into hooks/services
  - [x] Purge orphaned and duplicate files
  - [x] Standardize camelCase/PascalCase naming conventions
- [x] **Phase 4: Error Handling & Debugging Standardization**
  - [x] Unified structured logging across backend with X-Request-ID and latency headers
  - [x] Graceful UI error boundaries, stream reconnection, and camera permission handling

---

## 7. Session Maintenance Protocol

Whenever a development session finishes or a major milestone is reached:
1. Update Section 2 (**Project Delivery Phase Tracker**) with new phase progress.
2. Record specific technical diffs in Section 3 (**Recent Major Accomplishments**).
3. Review and update Section 5 (**Technical Debt**) and Section 6 (**Immediate Action Items**).
4. Record the corresponding Git commit reference in `CHECKPOINTS.md`.
