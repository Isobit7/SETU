# SETU — Pretrained Model Integration & Live Test Evidence Report

See the complete artifact documentation at [`rf_uv11_evidence_report.md`](file:///C:/Users/sathw/.gemini/antigravity-ide/brain/05942c16-8a3b-4300-a8dd-9f9548ee05fb/rf_uv11_evidence_report.md).

### Summary Highlights:
- **Model**: `models/pretrained/rf_uv_11_model.h5` (Apache-2.0, 18,138 parameters).
- **Adapter**: `setu/models/rf_uv11_adapter.py` (63 features, pure NumPy, 0.14 ms latency).
- **Vision Reuse**: Fully reuses `setu.vision.camera.CameraReader` and `setu.vision.landmarks.HolisticLandmarkerWrapper`.
- **Demo Script**: `scripts/demo_rf_uv11_webcam.py` (live feed, bone skeleton, letter badge, smoothed vote, hands-absent pause).
- **Authenticity Audit**: Upstream Git history commit `3919e1f` revealed the model was renamed from `hand_gesture_classification_model.h5`. Its single-hand input cannot encode the inter-hand spatial dynamics required for two-handed ISL signs.
- **Decision Gate Verdict**: **REJECT as primary SETU ISL baseline**. Retained strictly as an experimental single-hand reference under `models/pretrained/`.
