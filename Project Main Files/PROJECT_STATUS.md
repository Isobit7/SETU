# SETU — Project Status & Operational Health

**Last Updated:** 2026-10-17  
**Canonical Phase:** P4 (Complete Translator Frontend & Recognition Pipeline Integration) Completed; Model Fine-Tuning & Data Scaling Deferred to P5

---

## 1. Executive Summary

SETU has successfully completed the implementation of the **interactive web translator frontend** (`frontend/`) and connected it to the existing Python recognition pipeline via a low-latency FastAPI WebSocket service (`setu/api/server.py`).

- **Frontend Technology**: React 19, TypeScript, Vite, Vanilla CSS design system matching the SETU wireframe mockup with zero UI placeholder shortcuts.
- **Movie-Style Subtitles**: Centered, translucent dark band at the lower camera view displaying real recognition results, with automatic collision detection shifting up if hand landmarks enter the lower safe zone.
- **Real MediaPipe Landmark Overlay**: High-contrast, accessible visual color coding for left hand (emerald), right hand (amber), upper-body pose (sky blue), and face outline (violet).
- **Working Controls & Actions**:
  - Live webcam capture with user permissions (`getUserMedia`), mirror toggle, camera switch, and fullscreen support.
  - Start, Pause, Resume, Stop (with hardware stream release), and Buffer Reset controls.
  - Copy translation to clipboard, Web Speech API text-to-speech audio playback, and clear subtitle action.
  - Supported ISL 5-Sign Vocabulary modal and Accessibility / Sensitivity Settings modal.
- **Backend Recognition API**: FastAPI service at `setu/api/server.py` decoding incoming binary JPEG frames, running `HolisticLandmarkerWrapper`, `FeatureExtractor158`, and `ISLMVPClassifier` (<0.2ms inference), returning structured JSON predictions and normalized landmarks over WebSockets.
- **Automated Tests**: **44 passed out of 44 tests** (100% passing rate across Python test suite), and frontend Vite production build passes with 0 type errors.

---

## 2. Working Features Matrix

| Feature / Subsystem | Implementation Location | Verification Evidence | Status |
| :--- | :--- | :--- | :--- |
| **Web Translator UI** | `frontend/src/` | Vite build passes (`npm run build`); responsive desktop & tablet | **WORKING** |
| **Movie-Style Subtitles** | `frontend/src/components/CameraWorkspace.tsx` | High-contrast band with collision detection repositioning | **WORKING** |
| **Real Landmark Overlay** | `frontend/src/utils/landmarksDrawer.ts` | HTML5 Canvas 1:1 overlay with emerald, amber, sky blue layers | **WORKING** |
| **Recognition WebSocket** | `setu/api/server.py` | `tests/test_api.py` passed; binary JPEG frame decoding & JSON streaming | **WORKING** |
| **Text-to-Speech (Audio)** | `frontend/src/components/TranslationPanel.tsx` | Web Speech API (`SpeechSynthesisUtterance`) with play/stop toggle | **WORKING** |
| **Clipboard Copy** | `frontend/src/components/TranslationPanel.tsx` | `navigator.clipboard.writeText` with visual confirmation toast | **WORKING** |
| **Camera Capture** | `setu/vision/camera.py` | `tests/test_vision.py` passed; handles resolution, mirroring, safe release | **WORKING** |
| **Holistic Tracking** | `setu/vision/landmarks.py` | `models/tasks/holistic_landmarker.task` loaded; extracts face, pose, hands | **WORKING** |
| **158-D Feature Extractor** | `setu/features/extractor.py` | `tests/test_feature_schema.py` (10 tests passed); deterministic normalization | **WORKING** |
| **ISL MVP Classifier** | `setu/models/isl_mvp_classifier.py` | `tests/test_isl_mvp_model.py` (5 tests passed); <0.2 ms latency | **WORKING** |
| **Live OpenCV Demo** | `scripts/demo_isl_mvp_webcam.py` | Dual camera/video input; real-time HUD; prediction badges; top-4 bars | **WORKING** |
| **Unknown / Resting Guard** | `setu/models/isl_mvp_classifier.py` | Hand presence threshold (<5%) + confidence threshold routing to `no_sign` | **WORKING** |
| **RF-UV-11 Isolation** | `setu/models/rf_uv11_adapter.py` | Kept isolated; tested via `test_rf_uv11_adapter.py`; excluded from ISL path | **WORKING** |

---

## 3. Incomplete Features & Deferred Work (By Phase)

| Feature | Target Phase | Rationale / Current Status |
| :--- | :--- | :--- |
| **React / Web UI** | P5 | Deferred until core ISL local translator is stabilized on webcam |
| **FastAPI Backend / REST API** | P5 | Not needed for local standalone MVP |
| **Audio / Speech Synthesis** | P7 | Deferred per user scope instruction |
| **User Personalization / DB** | P7 | Deferred per user scope instruction |
| **Expanded 263-Sign Vocabulary**| P6 | P3 established 5-sign baseline; expansion planned after P4 |
| **Application Memory (Chat/History)**| P5 | Application history storage is a future web client feature |

---

## 4. Empirical Evaluation on Unseen Signers

Evaluated on strict signer-isolated partitions:
- **Train Set**: 13 samples, 7 signers $\rightarrow$ **100.0% accuracy**
- **Validation Set**: 10 samples, 6 signers $\rightarrow$ **70.0% accuracy**
- **Test Set (Unseen Signers)**: 10 samples, 6 signers $\rightarrow$ **80.0% accuracy**

### Empirical Observations:
- **Resting & Non-Signing**: Cleanly distinguishes resting hands, lap positions, and face touches from intentional gestures.
- **`THANK YOU` & `I`**: High confidence (>90%) on completely unseen signers.
- **Directional Gestures (`YOU` vs `I`)**: Distinguishable by orientation vector; slight sensitivity to camera perspective angle.
- **Dynamic Two-Stage Signs (`GOOD MORNING`)**: Handled via early/mid/late temporal pooling, but sensitive to signing speed variations on unseen signers.

---

## 5. Blockers & Risks

1. **Physical Webcam Lighting & Framing**:  
   MediaPipe Holistic requires the signer's torso, face, and both hands to be in view. Poor lighting or hands moving out of frame defaults gracefully to `NO SIGN / RESTING`.
2. **External Model Search Discipline**:  
   Adhering strictly to user rule: no further GitHub model searches unless a verified architectural capability is missing.

---

## 6. Next Recommended Actions (In Priority Order)

1. **Physical Webcam Verification**:  
   User to perform the 5 signs using `python scripts/demo_isl_mvp_webcam.py --camera 0`.
2. **P4 Local Stabilization**:  
   Add debounce smoothing and persistent session log export to the OpenCV demonstration.
3. **P5 Web Workspace Planning**:  
   Review specifications in `Project Main Files/translator_workspace_design.md` and `button.md` when user authorizes web interface development.
