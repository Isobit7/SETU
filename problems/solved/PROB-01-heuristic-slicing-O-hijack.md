---
id: PROB-01
title: "Heuristic Slicing Bug - False 'O' Logit Hijack"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01 11:58
severity: Critical
category: Runtime Heuristics & Classifier Adapter
---

# PROB-01: Heuristic Slicing Bug - False 'O' Logit Hijack

## 1. Problem Description
During single-hand signing (where the user signs using only their primary Right Hand), the model prediction was almost universally hijacked by the letter `'O'` with $94\%\text{--}99\%$ artificial confidence, overriding correct letters like `'A'`, `'C'`, `'E'`, or `'L'`.

## 2. Root Cause Analysis
In `models/adapters/realsign.py`:
```python
if apply_heuristic_boost:
    # Heuristic boost uses 57-dim vector of the hand landmarks
    hand_landmarks = feat_vec[:57]  # <--- CRITICAL BUG
    boost_letter = evaluate_alphabet_heuristics(hand_landmarks)
```
In the 126-D feature vector, indices `0:63` represent the **Left Hand** (21 landmarks $\times$ 3 coords), while indices `63:126` represent the **Right Hand**.
1. When a user signs with their right hand only, the left hand landmarks in `0:63` are all zeros (`0.0`).
2. Slicing `feat_vec[:57]` extracted all zeros.
3. In `evaluate_alphabet_heuristics`, `middle_mcp` was reconstructed with $y = -1.0$, creating `palm_len = 1.0`, but all other landmarks (including thumb tip and index tip) were $(0,0,0)$.
4. Aperture distance evaluated to $0.0$, matching the curled-finger heuristic for `'O'` ($d \le 0.16$).
5. It returned `'O'`, artificially boosting the `'O'` logit by $+2.0$, hijacking predictions.

## 3. Resolution & Code Changes
1. **`training/features/geometry.py`**:
   - Added a validation guard in `evaluate_alphabet_heuristics`:
     ```python
     if np.count_nonzero(np.abs(arr) > 1e-4) < 15:
         return None
     ```
     This rejects any degenerate, zero-padded, or single-landmark mock arrays before heuristic computation can run.
2. **`models/adapters/realsign.py`**:
   - In 126-D mode, inspects both Left and Right hand energy (`lh_energy`, `rh_energy`).
   - Slices the active hand landmarks: picks Right Hand (`feat_vec[63:126]`) when active, or Left Hand (`feat_vec[0:63]`) if LH is active.
   - If neither hand meets energy threshold $> 1e-3$, skips heuristic boosting completely.
3. **`tests/test_realsign_classifier.py`**:
   - Added permanent regression test `test_realsign_no_o_hijack_on_single_hand` verifying that single-handed input with zeroed LH does NOT evaluate to `'O'`.

## 4. Acceptance Criteria & Verification
- [x] Feeding all-zero Left Hand with arbitrary Right Hand feature vectors does NOT return `'O'` from `evaluate_alphabet_heuristics`.
- [x] Single-hand letters ('A', 'L', 'C', 'V') retain their true model logits without $+2.0$ bias towards 'O'.
- [x] Regression test added in `tests/test_realsign_classifier.py` and 15/15 tests passing.
