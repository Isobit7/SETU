# Project Milestones & Model Checkpoints
## Project SETU: Real-Time Indian Sign Language (ISL) Recognition Engine

*This document records all system checkpoints, model training releases, architecture milestones, and git tags in reverse chronological order.*

---

### Checkpoints Timeline

```
[Oct 24] MVP Scaffolding ---> [Oct 26] Single-Signer Prototype ---> [Oct 28] Temporal BiLSTM
        |                                                                 |
        v                                                                 v
[Oct 30] Multi-Signer RealSign ---> [Nov 01] Indic NLP Engine ---> [Nov 06] High-Accuracy Cloud Models
        |                                                                 |
        v                                                                 v
[Nov 07] Multimodal Gesture/Avatar ---> [Nov 08] Enterprise UI & 3D Practice ---> [Nov 09] Codebase Re-Org
```

---

### Detailed Checkpoint Catalog

#### [Checkpoint 10] 2026-11-09 — Complete Codebase Line-by-Line Annotation & Architecture Registry
- **Git Commit:** `HEAD (nov6-working)`
- **Key Deliverables:**
  - Full codebase line-by-line documentation added across all critical backend and frontend pipelines.
  - Master Weights Registry created (`models/WEIGHTS_REGISTRY.md`) with training dates and performance metrics.
  - Master Reports Index created (`reports/REPORTS_INDEX.md`) linking solved problem reports PROB-01 through PROB-25.
  - Dedicated User Understanding Portal created in `docs/user_understanding/` for effortless viva defense and exam review.
  - Project governance documents initialized: `DECISIONS.md`, `CHECKPOINTS.md`, `MEMORY.md`.

#### [Checkpoint 09] 2026-11-08 — Enterprise Settings Redesign & 3D Skeletal Practice Studio
- **Git Commit:** `799383d`
- **Key Deliverables:**
  - Redesigned `SettingsModal.tsx` following modern Apple/Linear UI guidelines (4 organized tabs: Video & Camera, Recognition, Audio & Speech, Captions & Display).
  - Built interactive `Practice3DHandViewer.tsx` featuring canvas-based 3D projection of hand anatomy with yaw/pitch rotation, depth perspective, and multi-angle presets (`3/4 View`, `Side Profile`, `Front Palm`, `Signer POV`).
  - Added full A–Z alphabet guidance table with anatomical descriptions and pro-tips for two-handed vs single-handed signs.
  - Verified zero-error frontend build (`tsc -b && vite build` in 882ms).

#### [Checkpoint 08] 2026-11-07 — Multimodal Gesture Fusion & Indic Translation Expansion
- **Key Deliverables:**
  - Integrated facial non-manual marker detection (`backend/app/services/vision/non_manual_markers.py`) for affirmative head nods and question eyebrows.
  - Implemented real-time Indic grammar transformer (`backend/app/services/nlp/indic_transformer.py`) supporting Hindi, Tamil, Telugu, and Kannada with Subject-Object-Verb (SOV) restructuring.
  - Added client-side edge ONNX inference pipeline (`frontend/public/models/`) for zero-latency offline alphabet recognition.

#### [Checkpoint 07] 2026-11-06 — High-Accuracy Cloud Training & Final Model Release
- **Model Checkpoints Released:**
  - `models/weights/amrita_isl_phrase_bilstm.pth` (3.64 MB, 94.2% test accuracy on 100 continuous phrases)
  - `models/weights/realsign_26class_mlp.pth` (114 KB, 96.8% test accuracy on 26 static ISL alphabets)
  - `models/weights/realsign_multisigner.onnx` (109 KB, 96.2% test accuracy across 12 unique signers)
  - `models/weights/isl_hindi_vowels.onnx` (105 KB, 98.1% test accuracy on Hindi vowels अ, आ, इ, ई, etc.)
  - `models/weights/isl_numerals.onnx` (105 KB, 97.4% test accuracy on numbers 0–9)
- **Evaluation Reports Generated:**
  - `reports/amrita_slr_evaluation_report.json`
  - `reports/static_isl_alphabet_evaluation_report.json`
  - `reports/static_isl_hindi_vowels_evaluation_report.json`
  - `reports/static_isl_numerals_evaluation_report.json`

#### [Checkpoint 06] 2026-11-04 — 1-Euro Adaptive Filter & Dual-Hand Distance Vectors
- **Key Deliverables:**
  - Solved PROB-02 (Dual-hand relative distance erasure) by concatenating 21 inter-hand joint Euclidean distance features.
  - Solved PROB-16 (Holistic jitter) via `OneEuroFilter` suppressing landmark noise during stationary hand holds.
  - Added audio context memory leak fix (`PROB-13`) and Windows tempfile handle leak fix (`PROB-14`).

#### [Checkpoint 05] 2026-11-01 — RealSign 63D Feature Pipeline & Canonical Landmark Normalization
- **Key Deliverables:**
  - Implemented `extract_realsign_63d_features.py` converting raw pixel coordinates into wrist-centered scale-invariant coordinates.
  - Exported ONNX edge models for in-browser inference (`realsign_26class_mlp_63d.onnx`).
  - Added `backend/app/services/pipeline_manager.py` coordinating vision stream with NLP gloss accumulator.

#### [Checkpoint 04] 2026-10-30 — Multi-Signer Invariant Feature Augmentation
- **Key Deliverables:**
  - Trained `realsign_multisigner_mlp.pth` incorporating multi-signer hand morphology jitter.
  - Resolved single-signer overfitting; tested across diverse hand shapes and skin tones.
  - Solved `PROB-05` (MediaPipe handedness flapping in mirror mode).

#### [Checkpoint 03] 2026-10-28 — Temporal BiLSTM Sequence Model
- **Key Deliverables:**
  - Trained `isl_sequence_lstm.pth` (3.86 MB) using a 30-frame sliding window for continuous conversational phrases.
  - Implemented `backend/app/services/nlp/accumulator.py` for debounce and sign aggregation.

#### [Checkpoint 02] 2026-10-26 — MVP Prototype Classifier
- **Key Deliverables:**
  - Trained initial baseline `isl_mvp_model.npz` (38 KB) and `hf_isl_26class_mlp.pth` (114 KB).
  - Proved feasibility of MediaPipe landmark extraction over WebSockets in FastAPI.

#### [Checkpoint 01] 2026-10-24 — Initial Architecture Scaffolding
- **Key Deliverables:**
  - FastAPI asynchronous backend initialized with WebSocket duplex streaming at `/ws/recognize`.
  - Vite + React + TypeScript frontend scaffolded with high-FPS webcam canvas capture.
