---
id: PROB-08
title: "Absence of Negative / NO_SIGN Class in Alphabet Model"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Model Architecture & Classification Head
---

# PROB-08: Absence of Negative / NO_SIGN Class in Alphabet Model

## 1. Problem Description
When a user is simply resting their hands, gesturing casually, or adjusting their camera, the system constantly produces confident letter predictions from 'A' to 'Z' rather than staying silent.

## 2. Root Cause Analysis
The RealSign MLP classifier outputs a 26-element logit vector corresponding strictly to indices $0\dots 25$ (`'A'` through `'Z'`).
$$\text{Softmax}(\vec{z}) = \left[\frac{e^{z_0}}{\sum_j e^{z_j}}, \dots, \frac{e^{z_{25}}}{\sum_j e^{z_j}}\right]$$
Because probabilities sum to $1.0$, even a completely arbitrary resting posture or casual hand movement will result in the highest logit being normalized to some value $> 0.10$. If the confidence threshold is met or no explicit thresholding is used, random letters are emitted as true signs.

## 3. Implementation Plan & Proposed Fix
1. Extend the alphabet label space to 27 classes:
   - Index 26: `_NO_SIGN_` / `IDLE`.
2. Generate synthetic and empirical negative samples:
   - Relaxed resting hand postures (hands at side, hands on keyboard, face touching, open neutral palm).
   - Incomplete transition shapes.
3. Train the classifier to output high probability on class 26 during neutral intervals, allowing the recognition pipeline to discard non-signing frames gracefully.

## 4. Acceptance Criteria & Verification
- [ ] Relaxed resting hands in camera view trigger `_NO_SIGN_` or empty prediction $> 90\%$ of the time.
- [ ] Letter accuracy on valid active signs remains $\ge 90\%$.

