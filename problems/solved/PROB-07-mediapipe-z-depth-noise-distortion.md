---
id: PROB-07
title: "MediaPipe Z-Depth Noise Distorting 3D Planar Alignment"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Kinetics & Signal Filtering
---

# PROB-07: MediaPipe Z-Depth Noise Distorting 3D Planar Alignment

## 1. Problem Description
MediaPipe returns estimated monocular depth $z$, which exhibits high noise variance (signal-to-noise ratio is up to $4\times$ worse than $x, y$ coordinates). When performing 3D out-of-plane rotation during canonical alignment, this $z$-depth noise causes hands to randomly pitch and yaw by $15^\circ\text{--}20^\circ$, destabilizing static gestures.

## 2. Root Cause Analysis
In `training/features/geometry.py` and `training/extract_realsign_63d_features.py`:
During 3D alignment, the out-of-plane tilt angle is computed via:
$$\phi = \arctan2(z_9, -y_9)$$
Because $z$ is inferred by MediaPipe from relative landmark scale rather than a depth sensor, slight lighting variations or finger shadows induce jitter of $\pm 0.05$ on $z_9$.
Since $-y_9 \approx 0.15\text{--}0.25$, an error of $\pm 0.05$ produces an angle swing of:
$$\Delta \phi \approx \arctan\left(\frac{0.05}{0.20}\right) \approx 14^\circ$$
Applying a $14^\circ$ out-of-plane 3D rotation alters the apparent $(x,y)$ 2D projections of all 21 landmarks, degrading classifier stability.

## 3. Implementation Plan & Proposed Fix
1. In `training/features/geometry.py` and `training/extract_realsign_63d_features.py`:
   - Applied a continuous soft deadzone ($0.08$ radians $\approx 4.6^\circ$) and $0.50\times$ dampening factor to out-of-plane rotation:
     $$\phi = \begin{cases} 0.0, & \text{if } |\text{raw\_phi}| \le 0.08 \\ \text{sign}(\text{raw\_phi}) \cdot (|\text{raw\_phi}| - 0.08) \cdot 0.50, & \text{otherwise} \end{cases}$$
   - This ensures small depth sensor noise is completely zeroed out while genuine deliberate tilt gestures are smoothed smoothly without discontinuity.

## 4. Acceptance Criteria & Verification
- [x] Landmark variance on static held hand is eliminated under small monocular depth noise ($|\text{diff}| < 1e-6$).
- [x] Canonical landmark constraints ($L_0 = (0,0,0)$ and $L_9 = (0,-1,0)$) remain strictly enforced.
- [x] PyTest test `test_prob07_z_depth_noise_deadzone` in `tests/test_live_bugfixes.py` and all 12 tests in `test_invariant_features_and_segmentation.py` pass 100%.
