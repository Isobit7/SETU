---
id: PROB-18
title: "Numbers 0-9 and Currency Signs Recognition"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:26
severity: High
category: Dataset & Vocabulary
---

# PROB-18: Numbers 0-9 and Currency Signs Recognition

## 1. Problem Description
SETU currently only recognizes 26 English alphabet letters (`A`–`Z`) and 263 isolated word signs from AI4Bharat INCLUDE. It completely lacks the ability to recognize numbers (`0` through `9`), prices, phone numbers, ages, calendar dates, or Indian currency denominations (`₹10`, `₹100`, `₹500`), severely limiting its usefulness in everyday transactions (shops, clinics, transit).

## 2. Root Cause Analysis
1. `models/weights/realsign_26class_mlp.pth` and `realsign_multisigner_mlp.pth` are hardcoded to 26 classes.
2. No training data exists in `data/processed/` for numerical finger configurations in Indian Sign Language (e.g. 1 = index extended, 2 = index + middle, 3 = thumb + index + middle in ISL counting, etc.).
3. `SentenceAccumulator` does not distinguish numerical tokens from letters when forming phone numbers or quantities.

## 3. Implementation Plan & Proposed Architecture
1. **Define ISL Numerical Digit Morphology**:
   - Digits 0–9 follow standardized ISL digit forms (finger counting).
   - Generate multi-signer synthetic landmark dataset for classes `0` through `9` using `augment_hand_landmarks()` and verified anatomical keypoints.
2. **Train Unified 36-Class Model (`A`–`Z` + `0`–`9`)**:
   - Extend `RealSignMLP` to 36 classes (input 126-D, hidden 128/64, output 36).
   - Alternatively, add dedicated `NumberClassifier` mode with automatic digit detection heuristic.
3. **Currency & Quantity Syntax Parser**:
   - In `SentenceAccumulator`, when digit tokens occur sequentially (`9`, `8`, `4`, `5`), group them into numerical values (`9845`).
   - Support currency prefixes (`RUPEES` + `50` -> `₹50`).

## 4. Acceptance Criteria & Verification
- [ ] Model achieves $\ge 95\%$ accuracy on test split across digits `0`–`9`.
- [ ] Sequential digits `['9', '8', '2', '0']` auto-commit as `"9820"` in the sentence buffer.
- [ ] PyTest benchmark suite asserts discrimination between `1` vs `D` (index extended) and `2` vs `V` (finger spread).
