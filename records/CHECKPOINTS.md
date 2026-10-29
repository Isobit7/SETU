# SETU — Milestone Checkpoints & Rollback Registry

> **System:** SETU (Indian Sign Language Translator)  
> **Status:** Active Audit & Rollback Registry  
> **Repository:** `c:\MINI PROJ-SETU`  
> **Last Updated:** October 2026  
> **Test Status:** 100% Passing (160/160 Backend Unit & Integration Tests | Clean TypeScript Vite Build)

This document serves as the canonical audit trail and rollback safety net for the SETU project. If any component, model weight, or UI feature suffers corruption, degradation, or regression, developers, teammates, and evaluators can inspect the exact commit hash, review what changed, and execute precise rollback procedures.

---

## 1. Executive Milestones Master Index Table

| Ckpt # | Milestone Focus | Core Technical Deliverable | Rollback Target |
| :---: | :--- | :--- | :---: |
| **27** | **YOLO-World, Qwen2.5 SLM & CTC Decoding** | Zero-shot open-vocabulary YOLO-World v2 detector, on-device Qwen2.5-0.5B causal SLM on CUDA, continuous CTC dynamic greedy decoding (zero frozen lockout, stutter collapsed, handles double letters), dual-hand horizontal symmetry ($x' = -x$), streamlined HUD, 160 passing tests. | `git checkout <commit_27>` |
| **26** | **Multimodal AI, Disambiguation & Low-Light** | 80-Class COCO detector, Hand-lift Disambiguation HUD, 176D body-anchored geometry, CLAHE low-light enhancer, fine-tune capture pipeline. | `git checkout <commit_26>` |
| **25** | **89-D Invariant Geometric Features & MLP** | Added 15 joint angles, 17 inter-finger distances; achieved **99.34% test accuracy**. | `git checkout <commit_25>` |
| **24** | **Custom User Dataset Ingestion & Training** | Ingested 89 images + 92 videos from user/peer; **98.44% test accuracy**. | `git checkout <commit_24>` |
| **23** | **Visual Framing Guide & Recognition Gate** | SVG viewfinder reticle, posture heuristic, and dynamic HUD status pill. | `git checkout <commit_23>` |
| **22** | **Root Micro-Cleanup & Documentation Hub** | Removed root `logo.png` & `main.py`; consolidated `docs/user_understanding/`. | `git checkout <commit_22>` |
| **21** | **Enterprise Layout & Master File Index** | Decoupled `/backend`, `/models/weights` vs `/models/adapters`, `/training`. | `git checkout <commit_21>` |
| **20** | **Architectural Refactoring & Hardening** | Disambiguated all model adapters and training scripts with canonical naming. | `git checkout <commit_20>` |
| **19** | **Systematic Codebase Disambiguation** | Audited and renamed cryptic files (`utils.py`, `mvp.py`, `rf_uv11.py`). | `git checkout <commit_19>` |
| **18** | **Exhaustive Project Mapping & Packaging** | Created recursive tree in `STRUCTURE.md` (241 files); packaged `Gemini.zip`. | `git checkout <commit_18>` |
| **17** | **UI Decoupling, Purge & Structured Logs** | Extracted `useSentenceBuilder.ts`; built `apiService.ts`; purged dead UI components. | `git checkout <commit_17>` |
| **16** | **Enterprise Modular Architecture (Phase 2)**| Decoupled backend routers, model registry, training pipelines, frozen config. | `git checkout <commit_16>` |
| **15** | **Architectural Mapping & Living Docs** | Created 4 core living records: `ARCHITECTURE.md`, `MEMORY.md`, `CHECKPOINTS.md`, etc. | `git checkout <commit_15>` |
| **14** | **Micro-Animations & Tactile Physics** | Glassmorphism blur, `@keyframes pulseGlow` beacon, active scale buttons. | `git checkout <commit_14>` |
| **13** | **Brand Favicon Suite & Minimalist Standby**| Multi-res favicon suite; minimalist 56px aperture standby view (CLS = 0). | `git checkout <commit_13>` |
| **12** | **5-Take Calibration & Windows DirectShow** | 5-take custom sign recorder; `cv2.CAP_DSHOW` backend fixing Windows camera hangs. | `git checkout 672f931` |
| **11** | **Real-Time 26-Class Alphabet Full-Stack** | Trained PyTorch RealSignMLP to 99.11% accuracy; exported ONNX graph (20.8 µs). | `git checkout 2848486` |
| **10** | **Vision Stabilization & MediaPipe Tasks** | Stabilized MediaPipe task bundles; containerized services via Docker Compose. | `git checkout 9198947` |
| **9** | **NLP Gloss Parser & Web Speech TTS** | Added `ISLGlossParser` (SOV to SVO translation) and browser SpeechSynthesis TTS. | `git checkout 76b9a10` |
| **8** | **Architecture Guides & Visual Overhaul** | Authored `SETU_DEEP_ARCHITECTURE_GUIDE.md` and `SETU_system_overview.svg`. | `git checkout 64aadbc` |
| **7** | **Initial SETU ISL Recognition SaaS** | Full-stack connection of React frontend to FastAPI server via duplex WebSockets. | `git checkout 79826de` |
| **6** | **Real ISL Subset & Signer Isolation** | Ingested AI4Bharat INCLUDE subset with strict signer-independent splits. | `git checkout e4639a6` |
| **5** | **Schema Freeze & NO_SIGN Generator** | Froze 158-D feature schema; built negative resting-hand class generator. | `git checkout 09045de` |
| **4** | **Modularization of Core Engine** | Partitioned engine into `setu/vision/`, `setu/features/`, `setu/models/`. | `git checkout 6777562` |
| **1** | **Initial Repository Baseline** | Initial repository setup, baseline MediaPipe pipelines, and dependencies. | `git checkout 277db72` |

---

## 2. Rollback Execution Guidelines

To restore the project to any verified milestone, execute the corresponding git checkout commands from PowerShell or Git Bash within the root repository:

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

## 3. Detailed Milestone Audit Cards

| Checkpoint | Checkpoint 26: Multimodal AI, Disambiguation & Scientific Deep Audit |
| :--- | :--- |
| **Date** | October 2026 |
| **Core Innovation** | 80-Class MobileNetV3 SSDLite COCO detector for Show-and-Sign interaction; Left/Right Hand-Lift Disambiguation HUD; 176-D body-anchored geometry; Adaptive LAB CLAHE low-light boost; Retraining capture pipeline. |
| **Scientific Audit** | Automated diagnostic suite (`scripts/deep_system_diagnostic.py`) executed across 7 subsystems: 0 NaN/Inf across 1M weights, zero blank-frame hallucination, 98.30% holdout accuracy, 98.43% precision, 98.36% F1, 152.7 µs latency. |
| **Verification** | 134/134 backend pytest unit and integration tests passing (100%); clean Vite TypeScript bundle in 1.46s; zero layout shifts. |
| **Archived Reports**| `reports/diagnostics/2026-10-28_setu_master_deep_system_diagnostic_audit.md`, `reports/diagnostics/2026-10-28_setu_deep_system_diagnostic_telemetry.json`. |
| **Key Files** | `backend/app/services/vision/object_detector.py`, `backend/app/services/nlp/multimodal_fuser.py`, `backend/app/services/vision/enhancer.py`, `frontend/src/components/DisambiguationOverlay.tsx`. |

---

| Checkpoint | Checkpoint 25: 89-D Invariant Geometric Features & MLP Retraining |
| :--- | :--- |
| **Date** | October 2026 |
| **Core Innovation** | 89-D Geometric Feature Engine (`training/features/geometry.py`) eliminating zero-variance dimensions, adding 15 joint angles, 5 radial distances, 4 thumb-tip distances, and 4 adjacent fingertip spreads. |
| **Model Retraining** | PyTorch RealSignMLP upgraded with `BatchNorm1d` before ReLU. Trained 40 epochs on CUDA with AdamW and Cosine LR. |
| **Accuracy & F1** | **99.34% held-out test accuracy** (8,302/8,357 correct) and **99.34% macro F1**. Decoupled M vs W errors to 3, M vs H to 0. |
| **Frontend & API** | Dynamic 89D/63D adapter dimension auto-detection; secondary hand landmarks schema supported for bimanual ISL signs. |
| **Verification** | 133/133 backend pytest unit tests passing (100%); clean Vite TypeScript build with zero errors. |
| **Key Files** | `training/features/geometry.py`, `training/train_realsign_26class.py`, `models/adapters/realsign.py`, `backend/app/websocket/recognize.py`. |

---

| Checkpoint | Checkpoint 24: Custom User Dataset Ingestion & Model Personalization |
| :--- | :--- |
| **Date** | October 2026 |
| **Core Innovation** | Ingested custom dataset recorded by the user and peer (24 classes A–X, 89 images, 92 video clips in `data/raw/setu_custom/SETU`). |
| **Feature Extraction** | Extracted 1,109 invariant 63-D landmark vectors; merged with base 26-class dataset to preserve Y and Z, yielding 77,508 samples. |
| **Model Retraining** | Retrained RealSignMLP (63 -> 128 -> 64 -> 26) on GPU for 35 epochs. |
| **Accuracy & F1** | **98.76% validation accuracy**, **98.44% test accuracy**, and **89.26% direct accuracy** on custom user recordings. |
| **Latency** | Sub-millisecond CPU inference latency (15.5 microseconds per prediction; > 64,000 FPS throughput). |
| **Key Files** | `training/ingest_setu_user_dataset.py`, `training/train_realsign_26class.py`, `training/export_onnx.py`. |

---

| Checkpoint | Checkpoint 23: Proactive Visual Framing Guide & Recognition Gate |
| :--- | :--- |
| **Date** | October 2026 |
| **Visual Guide** | SVG target bounding box with rounded viewfinder corners, semi-transparent upper-body silhouette, and chest signing zone. |
| **Heuristic Evaluator** | Zero-allocation client-side algorithm checking shoulder span (`0.17 <= span <= 0.56`), centering (`0.33 <= center <= 0.67`), headroom, and hand height. |
| **Recognition Gate** | Real-time HUD pill with status guidance ("Step Closer", "Move Back", "Center Yourself"). Optional toggle to pause predictions until clean framing is achieved. |
| **Verification** | 8/8 automated evaluator checks passing; Vite production bundle compiled in < 500ms; 125/125 pytest tests passing. |
| **Key Files** | `frontend/src/components/VisualFramingGuide.tsx`, `frontend/src/utils/framingEvaluator.ts`, `frontend/src/components/CameraWorkspace.tsx`. |

---

| Checkpoint | Checkpoint 22: Root Micro-Cleanup & Documentation Hub Consolidation |
| :--- | :--- |
| **Date** | October 2026 |
| **Sanitization** | Removed root `logo.png` (standardized in `frontend/public/logo.png`) and legacy root `main.py` (standardized in `backend/app/main.py`). |
| **Build Hygiene** | Removed `setu.egg-info/` build directory; updated `.gitignore` and `scripts/package_gemini.py` ignore lists. |
| **Doc Consolidation**| Relocated manuals into `docs/user_understanding/` (`ARCHITECTURE_AND_PIPELINE.md`, `END_TO_END_FLOW.md`, `PROJECT_OVERVIEW.md`, etc.). |
| **Verification** | Re-indexed all 228 active files with contiguous sequential numbering (1 to 228); 125/125 tests passing. |
| **Key Files** | `backend/Dockerfile`, `.gitignore`, `docs/user_understanding/`, `records/STRUCTURE.md`. |

---

| Checkpoint | Checkpoint 21: Enterprise Separation of Concerns & Master Index Standard |
| :--- | :--- |
| **Date** | October 2026 |
| **Architecture** | Target enterprise layout: `/backend`, `/models/adapters`, `/models/weights`, `/models/tasks`, `/training`, `/records`, `/config`. |
| **Legacy Pruning** | Eliminated redundant directories: `ml/` (merged into `training/`), `configs/` (merged into `config/`), `setu/`, `Project Main Files/`. |
| **Living Index** | Established `records/STRUCTURE.md` cataloging every active file with "Why the Folder Exists" and exact 1-sentence purpose. |
| **Verification** | 125/125 pytest tests passing in 31.09s; 1,941 frontend modules compiled cleanly in 989ms. |
| **Key Files** | `backend/app/services/vision/`, `backend/app/services/nlp/`, `models/registry.py`, `records/STRUCTURE.md`. |

---

| Checkpoint | Checkpoint 20: Semantic Disambiguation & Adapter Hardening |
| :--- | :--- |
| **Date** | October 2026 |
| **Documentation** | Established canonical `records/` directory; relocated all 5 living records into it. |
| **Model Adapters** | Disambiguated all adapters with explicit domain names (`include_video_lstm_adapter.py`, `isl_phrase_sequence_lstm_adapter.py`, etc.). |
| **Training Pipelines**| Explicit training modules (`extract_holistic_158d_features.py`, `train_realsign_26class_mlp.py`, etc.) with backward-compatible facades. |
| **Verification** | 125/125 pytest tests passing; 0 TypeScript build errors. |
| **Key Files** | `records/`, `models/adapters/`, `training/`, `tests/`. |

---

| Checkpoint | Checkpoint 19: Systematic Codebase Renaming & Import Harmonization |
| :--- | :--- |
| **Date** | October 2026 |
| **Renaming Pass** | Eliminated ambiguous names: `classifier.py` -> `realsign_classifier_service.py`, `mvp.py` -> `isl_conversational_mvp.py`, `rf_uv11.py` -> `roboflow_yolov11_adapter.py`. |
| **Scripts & Docs** | Renamed 5 cryptic scripts and 7 markdown files in `Project Main Files/` to self-explanatory names with redirection shims. |
| **Parity** | Zero-downtime backward compatibility preserved across all original paths. |
| **Key Files** | `backend/app/services/`, `models/`, `scripts/`. |

---

| Checkpoint | Checkpoint 18: Exhaustive Project Mapping & Clean Packaging |
| :--- | :--- |
| **Date** | October 2026 |
| **File Mapping** | Created `STRUCTURE.md` providing an exhaustive recursive tree and 1-sentence descriptions for all 241 files. |
| **Sanitization** | Relocated loose handbook to `docs/`, legacy snapshots to `docs/archive/`, and consolidated notebooks. |
| **Packaging** | Packaged complete clean codebase into `Gemini.zip` excluding `.git`, `.venv`, and `node_modules`. |
| **Key Files** | `STRUCTURE.md`, `docs/archive/`, `training/`, `Gemini.zip`. |

---

| Checkpoint | Checkpoint 17: UI Decoupling, Dead Code Purge & Structured Logging |
| :--- | :--- |
| **Date** | October 2026 |
| **Custom Hooks** | Extracted streak debouncing and sentence assembly out of `App.tsx` into `useSentenceBuilder.ts`. |
| **API Client** | Built centralized `apiService.ts` with typed requests, timeout handling, and normalized errors. |
| **Dead Code Purge** | Deleted orphaned components: `Header.tsx`, `Sidebar.tsx`, `StatusStrip.tsx`, `TopBar.tsx`. |
| **Observability** | FastAPI correlation ID middleware (`X-Request-ID` UUID) and latency tracking (`X-Response-Time-MS`). |
| **Verification** | 125/125 tests passing in 23.38s; clean Vite build. |
| **Key Files** | `frontend/src/hooks/useSentenceBuilder.ts`, `frontend/src/services/apiService.ts`, `backend/app/main.py`. |

---

| Checkpoint | Checkpoint 16: Enterprise Modular Architecture & Model Registry |
| :--- | :--- |
| **Date** | October 2026 |
| **Backend Routers** | Modular endpoints for predict, models, video, dataset, vocabulary, and WebSocket streaming. |
| **Model Registry** | Swappable `ModelRegistry` pattern with decoupled inference scripts and dynamic loading. |
| **Configuration** | Centralized frozen dataclass settings in `config/settings.py` with environment variable overrides. |
| **Key Files** | `backend/app/api/endpoints/`, `backend/app/websocket/recognize.py`, `models/registry.py`, `config/settings.py`. |

---

| Checkpoint | Checkpoint 15: Living Documentation Suite Foundation |
| :--- | :--- |
| **Date** | October 2026 |
| **Deliverables** | Created 4 core living records (`ARCHITECTURE.md`, `MEMORY.md`, `CHECKPOINTS.md`, `DATASETS_AND_MODELS.md`) as authoritative system single source of truth. |
| **Key Files** | `records/ARCHITECTURE.md`, `records/MEMORY.md`, `records/CHECKPOINTS.md`, `records/DATASETS_AND_MODELS.md`. |

---

| Checkpoint | Checkpoint 14: Enterprise Micro-Animations & Tactile Physics |
| :--- | :--- |
| **Date** | October 2026 |
| **Motion Design** | `@keyframes pulseGlow` live camera beacon, 300ms feed activation transitions, `active:scale-95` tactile press feedback, 180° language swap rotation, skeleton shimmer loading. |
| **Accessibility** | Strict `prefers-reduced-motion` overrides across all animations for complete WCAG compliance. |
| **Key Files** | `frontend/src/index.css`, `frontend/src/components/CameraWorkspace.tsx`, `frontend/src/components/TranslationPanel.tsx`. |

---

| Checkpoint | Checkpoint 13: Official Brand Favicon Suite & Minimalist Standby View |
| :--- | :--- |
| **Date** | October 2026 |
| **Brand Assets** | High-res vector favicon suite (`svg`, `ico`, `png`) for tab, bookmark, and mobile home screen icons. |
| **Standby View** | Clean 56px aperture icon with direct "Start Camera" CTA replacing cluttered constellation overlays. |
| **Subtitles** | Pinned movie-style subtitles permanently to bottom of camera viewport (CLS = 0). |
| **Key Files** | `frontend/public/`, `frontend/src/components/CameraWorkspace.tsx`, `frontend/src/components/TopNavbar.tsx`. |

---

| Checkpoint | Checkpoint 12: 5-Take ISL Alphabet Calibration & Windows Video Capture |
| :--- | :--- |
| **Date** | October 2026 |
| **Calibration** | 5-take progression with spacebar confirmation for custom user sign recording. |
| **Hardware Fix** | `cv2.CAP_DSHOW` DirectShow backend on Windows eliminating webcam initialization hangs. |
| **Key Files** | `scripts/record_custom_signs.py`, `backend/app/services/vision/camera.py`. |

---

| Checkpoint | Checkpoint 11: Real-Time 26-Class Alphabet Recognition Full-Stack |
| :--- | :--- |
| **Date** | October 2026 |
| **Model Retraining** | Trained PyTorch `RealSignMLP` to 99.11% validation accuracy across 26 classes (A–Z). |
| **ONNX Runtime** | Exported ONNX graph delivering 20.8 µs CPU latency (> 48,000 FPS throughput). |
| **UI Integration** | 126-D hand invariant normalization and real-time alphabet recognition modal in frontend. |
| **Key Files** | `models/weights/realsign_26class_mlp.onnx`, `frontend/src/components/AlphabetModal.tsx`. |

---

| Checkpoint | Checkpoint 10: Vision Pipeline Stabilization & MediaPipe Tasks |
| :--- | :--- |
| **Date** | October 2026 |
| **Vision Stability**| Standardized MediaPipe holistic and hand landmarker `.task` bundles. |
| **Containerization**| Dockerized backend and frontend services with Docker Compose configuration. |
| **Key Files** | `models/tasks/`, `docker-compose.yml`, `backend/Dockerfile`. |

---

| Checkpoint | Checkpoint 9: NLP Gloss Parser, Invariant Geometry & TTS Integration |
| :--- | :--- |
| **Date** | October 2026 |
| **Grammar Parser** | Built `ISLGlossParser` converting Indian Sign Language SOV syntax into natural English SVO syntax. |
| **Speech Audio** | Integrated browser-native SpeechSynthesis for instant spoken sentence playback. |
| **Key Files** | `backend/app/services/nlp/gloss_parser.py`, `frontend/src/components/TranslationPanel.tsx`. |

---

| Checkpoint | Checkpoint 8: Architecture Guides & Visual Documentation Overhaul |
| :--- | :--- |
| **Date** | October 2026 |
| **Visual Assets** | Authored `SETU_DEEP_ARCHITECTURE_GUIDE.md` and created `SETU_system_overview.svg`. |
| **Key Files** | `docs/SETU_DEEP_ARCHITECTURE_GUIDE.md`, `docs/SETU_system_overview.svg`. |

---

| Checkpoint | Checkpoint 7: Initial SETU ISL Recognition SaaS Implementation |
| :--- | :--- |
| **Date** | October 2026 |
| **Milestone** | Initial full-stack streaming integration connecting React frontend to FastAPI WebSocket server. |
| **Key Files** | `frontend/src/App.tsx`, `backend/app/websocket/recognize.py`, `Start_SETU.bat`. |

---

| Checkpoint | Checkpoint 6: Real ISL Subset & Signer Isolation |
| :--- | :--- |
| **Date** | October 2026 |
| **Dataset Ingress**| Acquired AI4Bharat INCLUDE dataset subset; enforced signer-independent splits to prevent leakage. |
| **Key Files** | `data/manifests/setu_isl_mvp_manifest.json`, `tests/test_real_data_readiness.py`. |

---

| Checkpoint | Checkpoint 5: Schema v1.0.0 Freeze & NO_SIGN Generator |
| :--- | :--- |
| **Date** | October 2026 |
| **Schema Freeze** | Froze 158-D holistic feature schema; implemented negative resting-hand class generator. |
| **Key Files** | `training/features/schema.py`, `training/no_sign_generator.py`. |

---

| Checkpoint | Checkpoint 4: Modularization of Core Engine |
| :--- | :--- |
| **Date** | October 2026 |
| **Separation** | Partitioned engine into distinct vision, feature, model, and config subsystems. |
| **Key Files** | `backend/app/services/vision/`, `training/features/`, `config/`. |

---

| Checkpoint | Checkpoint 1: Initial Repository Baseline |
| :--- | :--- |
| **Date** | October 2026 |
| **Baseline** | Initial repository setup, dependency freezing, and project architecture baseline. |
| **Key Files** | `requirements.txt`, `pyproject.toml`, `README.md`. |

---

## 4. Quick Rollback Command Reference

| Milestone Goal | Target Commit / Branch | Direct Execution Command | Expected Restored Capability |
| :--- | :--- | :--- | :--- |
| **Latest Stable State (89-D MLP & Framing)** | `main` | `git checkout main` | 99.34% alphabet accuracy, visual framing guide, 133 passing tests. |
| **Custom Dataset Retrained Baseline** | `Checkpoint 24` | `git checkout <commit_24>` | 98.44% accuracy on custom user dataset recordings. |
| **Visual Framing Guide Baseline** | `Checkpoint 23` | `git checkout <commit_23>` | SVG viewfinder, real-time posture heuristic, HUD status pill. |
| **Enterprise Layout Restructured State** | `Checkpoint 21` | `git checkout <commit_21>` | Strict `/backend`, `/models`, `/training`, `/records` separation. |
| **26-Class Alphabet Initial Release** | `2848486` | `git checkout 2848486` | 99.11% accuracy baseline alphabet recognition with ONNX export. |
| **NLP Gloss Translation Baseline** | `76b9a10` | `git checkout 76b9a10` | SOV to SVO translation with Web Speech synthesis. |
| **Initial SaaS MVP Baseline** | `79826de` | `git checkout 79826de` | WebSocket live streaming camera prototype. |
| **Initial Project Baseline** | `277db72` | `git checkout 277db72` | Original project initialization state. |
