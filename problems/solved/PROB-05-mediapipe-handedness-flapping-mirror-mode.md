---
id: PROB-05
title: "MediaPipe Handedness Flapping in Mirror Mode"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Frontend & Websocket Ingestion
---

# PROB-05: MediaPipe Handedness Flapping in Mirror Mode

## 1. Problem Description
During live webcam signing (especially when the user signs near the center of the frame or crosses hands), MediaPipe intermittently flips its label between "Left" and "Right" across consecutive video frames (e.g., Frame 1: Right Hand, Frame 2: Left Hand, Frame 3: Right Hand). This triggers violent prediction flickering between unrelated letters because single-handed left-hand signing undergoes sagittal reflection ($x \to -x$).

## 2. Root Cause Analysis
1. MediaPipe's single-image classification head calculates handedness probability per frame without temporal memory.
2. In frontend mirror mode (where the video feed is horizontally flipped with CSS `scaleX(-1)`), MediaPipe receives coordinates that are mirror-inverted unless explicitly un-mirrored before sending to the backend.
3. When MediaPipe flips the label:
   - Hand coordinates intended for slots `63:126` (Right Hand) are placed into `0:63` (Left Hand).
   - In `training/features/geometry.py`, Left and Right hands undergo sagittal reflection ($x \to -x$).
   - A single-frame label flip completely reverses the sign's geometry and flips the feature vector between hand slots, destroying temporal smoothing filters (such as the One-Euro filter).

## 3. Implementation Plan & Proposed Fix
1. In `backend/app/services/vision/landmarks.py`:
   - Built `HandednessTracker` with spatial wrist proximity tracking and 4-frame hysteresis.
   - For single-hand signs, single-frame or 1-3 frame label flaps are suppressed if the hand remains in the same spatial vicinity ($\Delta < 0.25$).
   - 4 consecutive frames of opposite classification are required before confirming a deliberate hand switch.
2. In `backend/app/websocket/recognize.py`:
   - Wired `HandednessTracker` into both Holistic and fallback HandLandmarker branches.
   - Reset tracker memory on WebSocket buffer reset and long idle ($> 0.40$s).

## 4. Acceptance Criteria & Verification
- [x] No single-frame handedness inversions occur when hand is held steady near the center line.
- [x] Spatial continuity ensures deliberate hand changes are cleanly acknowledged after 4 consecutive frames.
- [x] PyTest test `test_handedness_tracker_hysteresis_suppresses_mirror_flaps` in `tests/test_live_bugfixes.py` passes 100%.
