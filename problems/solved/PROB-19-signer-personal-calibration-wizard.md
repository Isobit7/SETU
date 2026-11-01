---
id: PROB-19
title: "5-Second Signer Personal Calibration and Rest Pose Wizard"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:22
severity: Medium
category: Vision & UX
---

# PROB-19: 5-Second Signer Personal Calibration and Rest Pose Wizard

## 1. Problem Description
The resting pose threshold in SETU is hardcoded to $y \ge 0.82$ (`training/features/geometry.py`), and camera distance guidance is hardcoded to fixed shoulder widths ($0.16 \le w \le 0.65$). Different users have varying camera heights, torso sizes, desk arrangements, and natural arm rests. A tall user or someone signing from a low laptop camera might have their resting hands at $y = 0.74$, causing resting hands to be falsely recognized as active signs.

## 2. Root Cause Analysis
- Monocular cameras lack absolute metric depth. Normalized coordinates vary depending on camera FOV and user positioning.
- No personalized baseline calibration exists: the pipeline assumes all humans match a single fixed geometric profile.

## 3. Implementation Plan & Proposed Architecture
1. **Interactive Calibration HUD in Frontend**:
   - A 5-second step-by-step onboarding flow:
     - **Step 1 (2 sec):** "Rest hands naturally on desk/lap" $\to$ records resting wrist $y$-coordinate ($y_{\text{rest}}$).
     - **Step 2 (3 sec):** "Hold up your dominant signing hand" $\to$ records active signing box, focal scale ($s_{\text{ref}}$), and dominant hand choice ('left' vs 'right').
2. **Dynamic Calibration Payload over WebSocket**:
   - Client sends control action:
     ```json
     {"action": "set_calibration", "rest_y": 0.76, "dominant_hand": "right", "focal_scale": 0.22}
     ```
   - `backend/app/websocket/recognize.py` dynamically adjusts `is_resting_or_neutral_pose(landmarks, y_threshold=calibrated_rest_y)`.
3. **Persist in Browser `localStorage`**:
   - Saved across sessions so users only calibrate once per device.

## 4. Acceptance Criteria & Verification
- [ ] Custom `rest_y` parameter passed over WebSocket successfully overrides default $0.82$ threshold.
- [ ] Users with low desk cameras ($y_{\text{rest}} = 0.72$) experience 0 false positive triggers when resting.
- [ ] Unit tests verify dynamic threshold clamping and recovery.
