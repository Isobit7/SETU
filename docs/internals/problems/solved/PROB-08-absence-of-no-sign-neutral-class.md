---
id: PROB-08
title: "Absence of Negative / NO_SIGN Class in Alphabet Model"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Model Architecture & Classification Head
---

# PROB-08: Absence of Negative / NO_SIGN Class in Alphabet Model

## 1. Problem Description
When a user was simply resting their hands on the desk/lap or gesturing casually, the system previously produced random letter predictions from 'A' to 'Z' rather than staying silent because the softmax distribution across 26 classes always normalized to $1.0$.

## 2. Root Cause Analysis
The RealSign MLP classifier outputs a 26-element logit vector corresponding to 'A'–'Z'. When a resting hand was detected without explicit resting pose detection, the highest logit produced spurious emissions whenever random movements slightly crossed confidence thresholds.

## 3. Implementation Plan & Proposed Fix
1. In `training/features/geometry.py`:
   - Built `is_resting_or_neutral_pose(landmarks, y_threshold=0.82)` to detect hands resting low in the lap or desk area ($y \ge 0.82$).
2. In `backend/app/websocket/recognize.py`:
   - Intercepted resting hand poses before alphabet classification:
     - Sets prediction to empty (`_empty_prediction()`).
     - Emits `no_sign` token to `SentenceAccumulator` to maintain baseline silence.
     - Resets classifier smoothing buffer to prevent stale logits from lingering.

## 4. Acceptance Criteria & Verification
- [x] Relaxed resting hands in camera view trigger `no_sign` baseline silence.
- [x] Valid raised signs ($y < 0.82$) execute standard recognition with high accuracy.
- [x] PyTest test `test_prob08_resting_hand_suppression` in `tests/test_live_bugfixes.py` passes 100%.
