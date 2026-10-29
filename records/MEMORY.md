# SETU — Active Project Memory Log

> **System:** SETU (Indian Sign Language Translator)  
> **Repository:** `c:\MINI PROJ-SETU`  
> **Last Synchronized:** October 2026  
> **Operational Status:** Fully Operational (160/160 Pytest Tests Passing | Clean Vite Build | Live Daemons Active)  
> **Protocol:** This living memory log tracks architectural commitments, completed milestones, resolved debt, and operational constraints.

---

## 1. Executive System Vitals Table

| Metric / Domain | Current Specification | Operational Standard |
| :--- | :--- | :--- |
| **Primary Mission** | Real-time Indian Sign Language (ISL) to English translation | Local-first, privacy-preserving, zero cloud dependency. |
| **Frontend Workstation** | React 19 + TypeScript + Vite | Dark theme (`#131314`), streamlined HUD, camera selector, continuous CTC display, movie subtitles, CLS = 0. |
| **Backend API & Streaming** | FastAPI (ASGI) + Uvicorn + WebSockets | Duplex streaming at 20–30 FPS with < 30ms processing latency. |
| **Alphabet ML Classifier** | PyTorch / ONNX RealSignMLP (89-D / 63-D) | **99.34% test accuracy**, 99.34% macro F1 across 26 alphabet classes (A–Z) with Left/Right symmetry. |
| **Phrase / Word ML** | AI4Bharat INCLUDE + ISL Sequence BiLSTM (176-D) | 12-D Upper-Body Anchored, hysteresis margin gating against misclassification. |
| **Object Detection & Fusion** | YOLO-World v2 (13M Params) + Multimodal Fuser | Open-vocabulary prompt-grounded detection (watches, phones, bottles, cups, books, pens) fused with signed modifiers. |
| **SLM Language Engine** | On-Device Qwen2.5-0.5B Causal SLM (CUDA) | Context-aware predictive autocomplete + neural ISL-to-English translation with LRU caching. |
| **Fingerspelling Decoder** | Continuous CTC Greedy Token Decoder | Stutter-collapsing dynamic decoder: zero 0.45s frozen lockout, fluid double letters (*"BOOK"*). |
| **Low-Light Enhancement** | Adaptive LAB CLAHE + Bilateral Denoising | Automatically boosts dim webcam feeds (< 75 lum) for landmark stability. |
| **Audio Speech** | Native Web SpeechSynthesis API | Instant auditory spoken playback with cancel and re-trigger controls. |
| **Test Suite Health** | 160 / 160 tests passing (100%) | Automated pytest execution across all vision, model, NLP, and regression suites. |

---

## 2. Project Delivery Phase Tracker

| Phase | Title | Status | Primary Technical Deliverable / Outcome |
| :---: | :--- | :---: | :--- |
| **P0** | **Baseline & Environment Audit** | ✅ Complete | Repository baseline frozen, dependencies cataloged, git tree secured. |
| **P1** | **Camera & Vision Ingress** | ✅ Complete | MediaPipe Holistic & Hand landmarker tasks with DirectShow Windows fix. |
| **P2** | **Candidate Model Feasibility** | ✅ Complete | Isolated RF-UV-11 reference adapter; rejected synthetic models as primary. |
| **P3** | **Real-Data Five-Sign MVP** | ✅ Complete | AI4Bharat INCLUDE subset (`hello`, `thank_you`, `good_morning`, `i`, `you`) trained. |
| **P4** | **Translator UI Workstation** | ✅ Complete | Unified camera feed, 3D skeletal canvas overlay, movie subtitles, auto-TTS. |
| **P5** | **RealSign 26-Class Alphabet** | ✅ Complete | Trained PyTorch RealSignMLP to 99.11% accuracy; exported to ONNX runtime. |
| **P6** | **UI/UX Enterprise Refactor** | ✅ Complete | Dark palette (`#131314`), minimalist standby view, favicon suite, micro-animations. |
| **P7** | **Modular Codebase Restructuring** | ✅ Complete | Decoupled `/backend`, `/models`, `/training`, `/config`; purged dead code; 125 tests passing. |
| **P8** | **Exhaustive Mapping & Packaging** | ✅ Complete | Created `STRUCTURE.md` (241 files), relocated handbook, generated `Gemini.zip`. |
| **P9** | **Architectural Refactoring** | ✅ Complete | Zero root clutter, `records/` living ledger, semantic renaming, 125 tests passing. |
| **P10**| **Enterprise Layout Standard** | ✅ Complete | Pruned legacy folders (`ml/`, `configs/`, `setu/`); enforced code vs weights split. |
| **P11**| **Root Micro-Cleanup & Docs** | ✅ Complete | Pruned root `logo.png` & `main.py`; consolidated `docs/user_understanding/`. |
| **P12**| **Visual Framing Guide & Gate** | ✅ Complete | SVG reticle, posture heuristic (`0.17 <= span <= 0.56`), HUD badge, gating telemetry. |
| **P13**| **Custom Dataset & 89-D MLP** | ✅ Complete | Ingested user signs (SETU.zip); engineered 89-D features; achieved **99.34% test accuracy**. |
| **P14**| **Living Documentation Polish** | ✅ Complete | Upgraded documentation to executive table-driven clarity and single-source truth. |
| **P15**| **Multimodal AI & Disambiguation** | ✅ Complete | 80-Class COCO detector, Hand-lift HUD, 176-D geometry, low-light CLAHE, capture CLI. |
| **P16**| **YOLO-World & SLM Modernization** | ✅ Complete | YOLO-World v2 open-vocabulary grounding, Qwen2.5-0.5B causal SLM, Continuous CTC decoding, Hand symmetry, 160 tests passing. |

---

## 3. Key Architectural Innovations Matrix

| Innovation Area | Problem Solved | Technical Implementation | Verified Outcome |
| :--- | :--- | :--- | :--- |
| **89-D Invariant Geometry** | Raw coordinates vary wildly with camera distance, arm tilt, and sitting position; M/W/N confusion. | `training/features/geometry.py` extracts 57 active coordinates, 15 joint angles, 5 radial distances, 4 thumb proximities, 4 fingertip spreads. | **99.34% test accuracy**; M vs W confusion dropped from 37 errors to 3. |
| **Custom Dataset Ingestion** | Models trained solely on synthetic/public data struggle on real user webcams. | Extracted 1,109 invariant landmark vectors from user-provided images/videos (`SETU.zip`) and merged with base dataset. | **89.26% direct accuracy** on previously unseen user recordings. |
| **Proactive Framing Guide** | Users sign too far, too close, or off-center, leading to degraded tracking. | Pure client-side heuristic in `framingEvaluator.ts` evaluating shoulder span, horizontal center, and hand elevation. | Instant HUD feedback ("Step Closer", "Move Back", "Perfect Framing"); 8/8 tests passing. |
| **Dynamic Model Adapter** | Deploying new feature dimensions usually breaks backwards compatibility. | `models/adapters/realsign.py` auto-detects 89-D vs 63-D weights at load time, branching cleanly without API breaks. | Seamless fallback; both 89-D and 63-D ONNX/PyTorch models load cleanly. |
| **Sub-Millisecond Inference** | Cloud GPU calls are slow (200ms+), expensive, and leak video frames over the internet. | ONNX Runtime CPU graph execution with dead-code elimination and constant folding. | **15.5 µs to 20.8 µs per prediction** (> 48,000 FPS throughput on CPU). |
| **Temporal Hold-Gate UX** | Involuntary hand transitions between signs cause accidental letter spam. | `useTextBuffer.ts` enforces 600ms continuous dwell gate and 400ms lockout cooldown. | Smooth, predictable letter accumulation with 0% accidental repeated characters. |

---

## 4. Architectural Rules & Active Constraints

| Rule Name | Constraint Description | Concrete Enforcement Mechanism |
| :--- | :--- | :--- |
| **Radical Authenticity** | Never fabricate mock predictions, fake confidence scores, or simulated hardware states. | If confidence < threshold or hands are absent, the engine explicitly returns empty candidates and status "Awaiting sign input". |
| **Local Frame Ingress** | Never transmit raw video frames to external third-party cloud services. | Video processing happens client-side via MediaPipe WebAssembly; only lightweight coordinate arrays are transmitted. |
| **Zero Regression Policy** | Restructuring or optimization must never break existing launchers or test suites. | All changes are validated against the 133-test automated pytest suite and Vite production compilation. |
| **Code / Weights Separation**| Pure Python code must never be mixed in the same directory as multi-megabyte binary weights. | Code resides in `models/adapters/`; binary weights reside exclusively in `models/weights/`. |
| **User Confirmation Gate** | Frontend component modifications must be preceded by an approved structured plan. | Implementation plans are drafted, reviewed in table format, and approved before touching React component files. |

---

## 5. Technical Debt & Resolution Status Matrix

| Debt Item | Initial Problem State | Concrete Resolution Applied | Current Verification Status |
| :--- | :--- | :--- | :--- |
| **Root Directory Clutter** | Loose scripts, legacy ZIPs, and duplicate `main.py` files at repository root. | Relocated all archives to `docs/archive/`, consolidated manuals into `docs/user_understanding/`, purged loose root files. | Clean root with only orchestration configs; 228 files indexed. |
| **Split Backend Architecture** | Monolithic `setu/api/` coexisting with modular `backend/app/`. | Fully consolidated into `backend/app/` with modular endpoint routers and service layers; legacy imports redirected cleanly. | Uvicorn launches cleanly on port 8000; 133/133 tests passing. |
| **Ambiguous Script Names** | Generic filenames like `mvp.py`, `rf_uv11.py`, `utils.py`. | Disambiguated to `isl_conversational_mvp.py`, `roboflow_yolov11_adapter.py`, `realsign_classifier_service.py`. | 100% backward-compatibility shims verified. |
| **Unanchored Subtitle Layout**| Subtitles caused layout shifts as confidence badges popped in and out. | Pinned movie-style subtitle bar permanently to bottom of camera viewport during live streaming. | **Cumulative Layout Shift (CLS) = 0.00**. |
| **Dense Documentation Text**| Long narrative text blocks made documents difficult to read and understand. | Systematic overhaul converting all records into structured comparison tables, ASCII wireframes, and Mermaid diagrams. | Overhaul active across `records/` and `docs/plans/`. |

---

## 6. Immediate Action Roadmap

| Item # | Objective & Files | Status | Success Metric |
| :---: | :--- | :---: | :--- |
| **1** | **Table-Driven Records Overhaul** (`records/*.md`) | ✅ Complete | 100% of records transformed into scannable tables and cards. |
| **2** | **User Understanding Docs Overhaul** (`docs/user_understanding/*.md`)| ✅ Complete | High-clarity visual tables and plain-language summaries in all 6 manuals. |
| **3** | **Phase-by-Phase Plan Table Polish** (`docs/plans/*.md`) | ⏳ Queued | Complete transformation plan converted to executive tables. |
| **4** | **Settings Modal UX Streamlining** (`SettingsModal.tsx`) | ⏸️ Pending Approval | Tabbed modal layout, confidence slider relocated, developer jargon removed. |
| **5** | **Camera Toolbar & Model Popover** (`CameraWorkspace.tsx`) | ⏸️ Pending Approval | Clean popover menu replacing cluttered raw buttons; plain model labels. |
