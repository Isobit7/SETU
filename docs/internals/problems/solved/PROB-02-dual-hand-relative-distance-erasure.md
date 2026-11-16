---
id: PROB-02
title: "Dual-Hand Relative Distance Erasure During Normalization"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Geometry & Canonical Alignment
---

# PROB-02: Dual-Hand Relative Distance Erasure During Normalization

## 1. Problem Description
ISL two-hand letters (such as 'H', 'P', 'R', 'K') require distinguishing whether the two hands are touching each other, spaced 5 cm apart, or spaced 30 cm apart. Under the previous independent normalization pipeline, both wrists were centered to $(0,0,0)$, completely erasing relative position and separation distance.

## 2. Root Cause Analysis
In `training/features/geometry.py`:
During canonical 126-D feature extraction:
- Left hand wrist (landmark 0) is translated to $(0,0,0)$ and scaled by its own palm length.
- Right hand wrist (landmark 0) is translated independently to $(0,0,0)$ and scaled by its own palm length.
Because each hand was independently zero-centered:
$$\vec{w}_{\text{left}} = (0,0,0), \quad \vec{w}_{\text{right}} = (0,0,0)$$
The Euclidean distance between wrists in the normalized representation was always $0.0$, regardless of whether the user held hands together or wide apart.

## 3. Implementation Plan & Proposed Fix
1. In `training/features/geometry.py`:
   - Implemented `compute_inter_hand_relative_geometry(left_hand, right_hand)`:
     - `wrist_distance`: Scale-invariant Euclidean distance between wrists ($\|\vec{w}_{\text{rh}} - \vec{w}_{\text{lh}}\| / s_{\text{ref}}$).
     - `min_fingertip_distance`: Minimum pairwise distance between any right fingertip ($4, 8, 12, 16, 20$) and all 21 left-hand landmarks.
     - `are_hands_touching`: Boolean physical contact detector ($\le 0.12$).
     - `relative_offset`: 3D displacement vector $(\vec{w}_{\text{rh}} - \vec{w}_{\text{lh}}) / s_{\text{ref}}$.
2. Preserves strict 126-D dimensionality in `extract_dual_hand_126d` while making rich relative spatial channels accessible to heuristic arbitration and sequence decoders.

## 4. Acceptance Criteria & Verification
- [x] Relative inter-wrist distance varies proportionally with real world hand distance ($> 17\times$ variation across contact vs detached poses).
- [x] Two-handed letters that touch ('P', 'H', 'K') have distinct distance profiles from detached dual signs (`are_hands_touching == True` vs `False`).
- [x] Geometry tests pass without breaking 126-D dimensionality constraints (`test_inter_hand_relative_geometry` in `tests/test_invariant_features_and_segmentation.py` passes 100%).
