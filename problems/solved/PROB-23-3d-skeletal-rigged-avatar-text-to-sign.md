---
id: PROB-23
title: "True 3D Skeletal Rigged Avatar for Text-to-Sign"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:41
severity: High
category: Frontend & 3D Graphics
---

# PROB-23: True 3D Skeletal Rigged Avatar for Text-to-Sign

## 1. Problem Description
`frontend/src/components/TextToSignWorkspace.tsx` currently renders an avatar using a 2D HTML Canvas element with flat circles and lines representing hands and arms. It only supports ~30 hardcoded 2D static poses. When converting longer phrases or complex sentences, the 2D stick figure lacks realistic depth perception, cannot show 3D rotational hand orientations, and jerks abruptly between poses.

## 2. Root Cause Analysis
- 2D Canvas lines lack bone hierarchy, depth testing ($z$-buffer), and joint inverse kinematics.
- Hand signs in ISL heavily rely on palm orientation (facing inward, outward, upward, or downward), which is impossible to represent accurately on a flat 2D canvas without 3D geometry.

## 3. Implementation Plan & Proposed Architecture
1. **Integrate Three.js / WebGL Humanoid Rig**:
   - Use an open-source GLTF 3D rigged character (e.g. standard Mixamo/ReadyPlayerMe humanoid skeleton with 21 finger bones per hand).
2. **Animation Keyframe Interpolation Engine**:
   - Instead of snapping between frames, use spherical linear interpolation (SLERP) for bone quaternions and cubic Hermite splines for positions.
   - Introduce smooth co-articulation (e.g., hand begins anticipating the next sign trajectory before the previous sign completes).
3. **ISL Gesture Animation Library**:
   - Map ISL vocabulary tokens to 3D bone keyframes rather than 2D pixel coordinates.

## 4. Acceptance Criteria & Verification
- [ ] 3D avatar renders at steady 60 FPS in browser with WebGL acceleration.
- [ ] Palm orientations (e.g., palm facing forward for `B` vs facing chest for `C`) are clearly visible in 3D perspective.
- [ ] Transitions between consecutive signs exhibit smooth, non-jerky interpolation.
