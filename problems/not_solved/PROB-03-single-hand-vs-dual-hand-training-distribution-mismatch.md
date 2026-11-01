---
id: PROB-03
title: "Single-Hand vs Dual-Hand Training Distribution Mismatch"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: Critical
category: Training Data & Augmentation
---

# PROB-03: Single-Hand vs Dual-Hand Training Distribution Mismatch

## 1. Problem Description
When signing single-handed letters in live webcam mode (where only the right hand is in camera view), the model's accuracy on letters like `D`, `E`, `M`, `N`, and `X` collapses to $0.0\%$, producing random wrong classifications or dead zones.

## 2. Root Cause Analysis
In the training dataset `isl_master_clean_126d.npz`:
1. Over $94\%$ of all training samples were recorded with both hands visible in frame. Even for one-handed signs, the signer kept their non-signing left hand resting in front of their chest or lap within MediaPipe's field of view.
2. As a consequence, the first 63 dimensions (`0:63`, left hand) in almost all training vectors contain valid landmark coordinates $(x,y,z)$.
3. In live webcam inference, when a user drops their left hand below camera frame, `recognize.py` correctly fills the missing Left Hand slots `0:63` with zeros (`0.0`).
4. Because the MLP weights connected to inputs `0:63` have never seen an all-zero input during training, the first linear layer activations fall completely out of distribution, causing the entire network prediction to collapse.

## 3. Implementation Plan & Proposed Fix
1. In `scripts/train_realsign_mlp.py` and `scripts/generate_multisigner_dataset.py`:
   - Implement **Independent Hand Dropout** (Bernoulli $p=0.5$ for single-handed classes):
     - For single-handed letters (`A`, `B`, `C`, `D`, `E`, `I`, `L`, `M`, `N`, `O`, `U`, `V`, `W`, `X`, `Y`):
       - Randomly zero out Left Hand landmarks (`vector[0:63] = 0.0`) in $50\%$ of training samples.
       - Randomly zero out Right Hand landmarks (`vector[63:126] = 0.0`) in $20\%$ of training samples (to support left-handed signers).
2. Add input layer batch normalization / LayerNorm to stabilize activation scale when one hand is absent.

## 4. Acceptance Criteria & Verification
- [ ] Retrained model achieves $> 85\%$ accuracy when tested on pure single-hand inputs with Left Hand set to zero (`feat[0:63] = 0`).
- [ ] Letters `D`, `E`, `M`, `N`, `X` achieve $> 80\%$ recall on single-hand test inputs.
