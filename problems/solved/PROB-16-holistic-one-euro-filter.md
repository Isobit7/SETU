---
id: PROB-16
title: "HolisticLandmarker Still Using Constant-Alpha Smoothing Instead of One-Euro Filter"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Vision Landmarker & Signal Smoothing
---

# PROB-16: HolisticLandmarker Still Using Constant-Alpha Smoothing Instead of One-Euro Filter

## 1. Problem Description
While `HandLandmarkerWrapper` uses the adaptive `OneEuroFilter` to eliminate high-frequency hand jitter, `HolisticLandmarkerWrapper` (used for sequence models and word classification) was left using a fixed exponential moving average ($\alpha = 0.70$). This static filter either lets through camera sensor jitter when stationary or produces motion lag during rapid dynamic gestures.

## 2. Root Cause Analysis
In `backend/app/services/vision/landmarks.py` (lines 120–138):
```python
def _smooth_landmark_list(
    self,
    key: str,
    landmarks: Optional[list[Any]],
    alpha: float = 0.70,
) -> None:
    ...
    lm.x = float(alpha * lm.x + (1.0 - alpha) * prev[i][0])
```
The fixed alpha filter does not scale with velocity ($dx/dt$), unlike the `OneEuroFilter` which dynamically shifts its cutoff frequency based on movement speed.

## 3. Implementation Plan & Proposed Fix
1. In `backend/app/services/vision/landmarks.py`:
   - Initialize `OneEuroFilter` instances for pose, face, left hand, and right hand inside `HolisticLandmarkerWrapper`.
   - Filter incoming landmark arrays with adaptive speed-based smoothing.
   - Reset filter states cleanly in `reset_smoothing()`.

## 4. Acceptance Criteria & Verification
- [x] Stationary holistic landmarks show reduced jitter variance ($> 30\%$ reduction).
- [x] Rapid motion does not lag behind physical gestures.
- [x] PyTest test suite remains 100% passing (`tests/test_camera_and_holistic_vision.py` passes 4/4).
