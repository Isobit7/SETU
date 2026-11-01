---
id: PROB-07
title: "MediaPipe Z-Depth Noise Distorting 3D Planar Alignment"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Kinetics & Signal Filtering
---

# PROB-07: MediaPipe Z-Depth Noise Distorting 3D Planar Alignment

## 1. Problem Description
MediaPipe returns estimated monocular depth $z$, which exhibits high noise variance (signal-to-noise ratio is up to $4\times$ worse than $x, y$ coordinates). When performing 3D out-of-plane rotation during canonical alignment, this $z$-depth noise causes hands to randomly pitch and yaw by $15^\circ\text{--}20^\circ$, destabilizing static gestures.

## 2. Root Cause Analysis
In `backend/app/kinetics/geometry.py`:
During 3D alignment, the out-of-plane tilt angle is computed via:
$$\phi = \arctan2(z_9, -y_9)$$
Because $z$ is inferred by MediaPipe from relative landmark scale rather than a depth sensor, slight lighting variations or finger shadows induce jitter of $\pm 0.05$ on $z_9$.
Since $-y_9 \approx 0.15\text{--}0.25$, an error of $\pm 0.05$ produces an angle swing of:
$$\Delta \phi \approx \arctan\left(\frac{0.05}{0.20}\right) \approx 14^\circ$$
Applying a $14^\circ$ out-of-plane 3D rotation alters the apparent $(x,y)$ 2D projections of all 21 landmarks, degrading classifier stability.

## 3. Implementation Plan & Proposed Fix
1. In `geometry.py`:
   - Apply a soft shrinkage / dampening factor to out-of-plane rotation:
     $$\phi_{\text{damped}} = \text{sign}(\phi) \cdot \max(0, |\phi| - \phi_{\text{deadzone}}) \cdot \gamma_z$$
     where $\gamma_z = 0.5$ dampens out-of-plane rotation noise unless there is clear deliberate hand tilting.
   - Alternatively, perform 2D in-plane canonical rotation ($\theta = \arctan2(x_9, -y_9)$) when $z$-confidence is below threshold.

## 4. Acceptance Criteria & Verification
- [ ] Landmark variance on static held hand is reduced by $\ge 40\%$.
- [ ] 3D planar rotation does not oscillate wildly on consecutive webcam frames under monocular RGB.
