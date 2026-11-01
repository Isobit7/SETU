---
id: PROB-02
title: "Dual-Hand Relative Distance Erasure During Normalization"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: High
category: Geometry & Canonical Alignment
---

# PROB-02: Dual-Hand Relative Distance Erasure During Normalization

## 1. Problem Description
ISL two-hand letters (such as 'H', 'P', 'R', 'K') require distinguishing whether the two hands are touching each other, spaced 5 cm apart, or spaced 30 cm apart. Under the current normalization pipeline, all dual-hand relative position and separation distance is completely erased.

## 2. Root Cause Analysis
In `backend/app/kinetics/geometry.py`:
During canonical 126-D feature extraction:
- Left hand wrist (landmark 0) is translated to $(0,0,0)$ and scaled by its own palm length.
- Right hand wrist (landmark 0) is translated independently to $(0,0,0)$ and scaled by its own palm length.
Because each hand is independently zero-centered:
$$\vec{w}_{\text{left}} = (0,0,0), \quad \vec{w}_{\text{right}} = (0,0,0)$$
The Euclidean distance between wrists in the normalized representation is always $0.0$, regardless of whether the user holds hands together or wide apart. Two completely distinct signs with identical hand shapes but different relative distances look identical to the classifier.

## 3. Implementation Plan & Proposed Fix
1. In `geometry.py`:
   - Compute global inter-wrist displacement vector:
     $$\Delta \vec{w} = \frac{\vec{w}_{\text{right}} - \vec{w}_{\text{left}}}{s_{\text{dominant}}}$$
     normalized by the dominant hand scale factor $s$.
   - Retain relative wrist offset by centering only the dominant (or left) hand at origin, while translating the secondary hand by its true scaled offset $\Delta \vec{w}$.
2. Store inter-hand minimum distance, fingertip-to-palm contact points, and inter-wrist distance as explicit relative spatial channels.

## 4. Acceptance Criteria & Verification
- [ ] Relative inter-wrist distance varies proportionally with real world hand distance.
- [ ] Two-handed letters that touch ('P', 'H', 'K') have distinct distance profiles from detached dual signs.
- [ ] Geometry tests pass without breaking 126-D dimensionality constraints.

