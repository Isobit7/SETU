---
id: PROB-06
title: "Missing Explicit Joint Flexion Angles & Fingertip Spread"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Feature Engineering & Invariants
---

# PROB-06: Missing Explicit Joint Flexion Angles & Fingertip Spread

## 1. Problem Description
Certain phonetically close ISL letter pairs (such as `M` vs `N` vs `T`, or `U` vs `V` vs `W`) differ primarily by:
- Number of folded vs extended fingers over the thumb.
- Angular spread between adjacent extended fingertips ($V$ spread vs $U$ touching).
The 126-D feature representation forces the neural network to learn complex trigonometry from scratch on noisy Cartesian coordinates, causing frequent confusion between these letters.

## 2. Root Cause Analysis
- Cartesian coordinates $(x_i, y_i, z_i)$ are sensitive to viewpoint angle and camera perspective distortion.
- Joint angles (flexion at MCP, PIP, DIP) are rotation-invariant and viewpoint-invariant scalar quantities:
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right)$$
- When joint angles and inter-fingertip spread distances are absent from heuristic arbitration, ambiguous signs fail to separate cleanly.

## 3. Implementation Plan & Proposed Fix
1. In `training/features/geometry.py`:
   - Verified and tested `compute_joint_angles_15` (15 anatomical flexion angles per hand in $[0, 1]$ invariant to rigid transformations).
   - Verified and tested `compute_finger_spread_angles` (4 angular spreads between adjacent fingertip rays normalized by $\pi$).
   - Verified heuristic arbitration in `evaluate_alphabet_heuristics` differentiating 'U' (fingers touching) from 'V' (fingers spread).

## 4. Acceptance Criteria & Verification
- [x] Joint flexion angles return values in range $[0, 1]$ invariant to 3D hand rotation and translation.
- [x] Discrimination between 'U' (fingers closed together) and 'V' (fingers spread apart) achieves clean separation with spread angle margin $> 2.5\times$.
- [x] PyTest test `test_compute_finger_spread_angles_and_uv_separation` in `tests/test_invariant_features_and_segmentation.py` passes 100%.
