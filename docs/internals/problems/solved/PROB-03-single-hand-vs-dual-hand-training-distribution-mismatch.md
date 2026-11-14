---
id: PROB-03
title: "Single-Hand vs Dual-Hand Training Distribution Mismatch"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: Critical
category: Training Data & Augmentation
---

# PROB-03: Single-Hand vs Dual-Hand Training Distribution Mismatch

## 1. Problem Description
When signing single-handed letters in live webcam mode (where only the right hand is in camera view), the model's accuracy on letters like `D`, `E`, `M`, `N`, and `X` previously suffered because the network had only been trained on dual-handed vectors where the resting non-signing hand occupied slots `0:63`.

## 2. Root Cause Analysis
In `data/processed/isl_master_clean_126d.npz`:
1. Over $94\%$ of all original training samples were recorded with both hands visible in frame. Even for one-handed signs, the signer kept their non-signing left hand resting in front of their chest or lap within MediaPipe's field of view.
2. When a live user drops their non-signing hand below frame, slots `0:63` are zero-padded (`0.0`).
3. Because the MLP weights had never seen an all-zero input in `0:63` during training, the first linear layer activations fell completely out of distribution.

## 3. Implementation Plan & Proposed Fix
1. In `scripts/generate_multisigner_dataset.py`:
   - Added Single-Hand Independent Hand Dropout during dataset generation:
     - For single-handed classes (`A`, `B`, `C`, `D`, `E`, `I`, `L`, `M`, `N`, `O`, `U`, `V`, `W`, `X`, `Y`):
       - $50\%$ probability: drop Left Hand (`vector[0:63] = 0.0`).
       - $15\%$ probability: drop Right Hand (`vector[63:126] = 0.0`) for left-handed signers.
2. In `training/train_isl_dualhand_mlp.py`:
   - Updated `ISLDualHandDataset` to apply dynamic hand dropout (`hand_dropout_prob = 0.35`) on incoming batches during training.

## 4. Acceptance Criteria & Verification
- [x] Model dataset contains explicit single-hand training samples with `vector[0:63] == 0.0`.
- [x] PyTest test `test_prob03_hand_dropout_in_dualhand_dataset` in `tests/test_live_bugfixes.py` passes 100%.
