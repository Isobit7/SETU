---
id: PROB-06
title: "Missing Explicit Joint Flexion Angles & Fingertip Spread"
stage: not_solved
status: not_solved
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
The current 126-D feature representation forces the neural network to learn complex trigonometry from scratch on noisy Cartesian coordinates, causing frequent confusion between these letters.

## 2. Root Cause Analysis
- Cartesian coordinates $(x_i, y_i, z_i)$ are sensitive to viewpoint angle and camera perspective distortion.
- Joint angles (flexion at MCP, PIP, DIP) are rotation-invariant and viewpoint-invariant scalar quantities:
  $$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right)$$
- Currently, neither joint angles nor inter-fingertip spread distances ($\|\vec{p}_{\text{index}} - \vec{p}_{\text{middle}}\|$, etc.) are explicitly computed in `backend/app/kinetics/geometry.py`.

## 3. Implementation Plan & Proposed Fix
1. In `backend/app/kinetics/geometry.py`:
   - Compute the 15 anatomical flexion angles per hand:
     - 3 angles per digit (MCP, PIP, DIP).
   - Compute the 4 inter-fingertip spread ratios:
     - Thumb-Index, Index-Middle, Middle-Ring, Ring-Pinky.
   - Append these rotation-invariant geometric features or use them in an enhanced feature extractor to disambiguate subtle finger-count and spread differences.

## 4. Acceptance Criteria & Verification
- [ ] Joint flexion angles return values in range $[0, \pi]$ invariant to 3D hand rotation.
- [ ] Discrimination between 'U' (fingers closed together) and 'V' (fingers spread apart) achieves $> 95\%$ test accuracy on synthetic and real benchmarks.
