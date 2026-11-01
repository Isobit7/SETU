---
id: PROB-05
title: "MediaPipe Handedness Flapping in Mirror Mode"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Frontend & Websocket Ingestion
---

# PROB-05: MediaPipe Handedness Flapping in Mirror Mode

## 1. Problem Description
During live webcam signing (especially when the user signs near the center of the frame or crosses hands), MediaPipe intermittently flips its label between "Left" and "Right" across consecutive video frames (e.g., Frame 1: Right Hand, Frame 2: Left Hand, Frame 3: Right Hand). This triggers violent prediction flickering between unrelated letters.

## 2. Root Cause Analysis
1. MediaPipe's single-image classification head calculates handedness probability per frame without temporal memory.
2. In frontend mirror mode (where the video feed is horizontally flipped with CSS `scaleX(-1)`), MediaPipe receives coordinates that are mirror-inverted unless explicitly un-mirrored before sending to the backend.
3. When MediaPipe flips the label:
   - Hand coordinates intended for slots `63:126` (Right Hand) are placed into `0:63` (Left Hand).
   - In `backend/app/kinetics/geometry.py`, Left and Right hands undergo sagittal reflection ($x \to -x$).
   - A single-frame label flip completely reverses the sign's geometry and flips the feature vector between hand slots, destroying temporal smoothing filters (such as the One-Euro filter).

## 3. Implementation Plan & Proposed Fix
1. In `backend/app/websocket/recognize.py`:
   - Implement a **Spatial-Temporal Handedness Tracker**:
     - Maintain an exponential moving average (EMA) or Kalman/Hungarian tracking filter for wrist positions $(x_t, y_t)$.
     - Associate detections with existing hand tracks based on spatial proximity rather than trusting MediaPipe's raw string label alone.
     - Require at least 4 consecutive frames of opposite classification before switching hand identity (hysteresis).
2. Explicitly document mirror-mode transformation in frontend `CameraFeed.tsx` and standardize coordinate orientation.

## 4. Acceptance Criteria & Verification
- [ ] No single-frame handedness inversions occur when hand is held steady near the center line.
- [ ] Crossing hands or moving hand across center does not trigger spurious letter flickering.

