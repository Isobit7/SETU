---
id: PROB-09
title: "Inference Running on Transitional Movement Frames"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01 12:01
severity: High
category: Websocket Pipeline & Kinetic Gating
---

# PROB-09: Inference Running on Transitional Movement Frames

## 1. Problem Description
When signing words or consecutive letters (e.g. spelling "HELLO"), the recognizer produced a stream of unintended intermediate letters (e.g. "H -> B -> P -> E -> L -> T -> L -> O") because the classifier executed on moving frames while the signer was transitioning between target hand shapes.

## 2. Root Cause Analysis
In `backend/app/websocket/recognize.py`:
- RealSign alphabet classification was allowed to accumulate streaks on every frame where hands were detected, regardless of kinetic velocity.
- Rapid hand transit between signs produces instantaneous velocities $E_k > 0.08$. During this transit, intermediate finger shapes resemble false letters.
- Because streak requirement was only 3–4 frames ($\sim 100\text{ ms}$), moving through intermediate configurations triggered spurious letter commitments to the continuous sentence accumulator.

## 3. Resolution & Code Changes
1. **`backend/app/websocket/recognize.py`**:
   - Added **Kinetic Velocity Gating** for static alphabet recognition:
     ```python
     alphabet_in_motion = bool(active_model == "realsign" and kinetic_energy > 0.055)
     ```
   - When `alphabet_in_motion` is True:
     - `alphabet_streak_count` is reset to 0.
     - `alphabet_streak_candidate` is reset to `"Awaiting Sign Input"`.
     - `res_display` shows `"Moving..."` and `res_label` is `"Transitioning"`.
     - In-flight frames are completely suppressed from committing tokens to the SentenceAccumulator.
   - When hand settles into the intended letter ($E_k \le 0.055$), streaks accumulate cleanly and commit within 3 settled frames without false intermediate letter noise.
   - Updated websocket payload `motion_state` to accurately report `"moving"` vs `"settled"`.
2. **`tests/test_live_bugfixes.py`**:
   - Added permanent regression test `test_kinetic_gating_suppresses_intermediate_transition_frames` verifying that in-flight motion does not emit intermediate letters.

## 4. Acceptance Criteria & Verification
- [x] Rapid movement between signs resets streak and suppresses false intermediate letters.
- [x] Held static signs emit cleanly once stabilized with zero additional latency.
- [x] Regression test added in `tests/test_live_bugfixes.py` and 8/8 tests passing.
