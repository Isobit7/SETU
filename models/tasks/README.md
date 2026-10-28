# Google MediaPipe Vision Task Bundles

This directory contains on-device vision task bundles from Google AI Edge used by SETU for real-time body, face, and hand landmark extraction.

---

## 1. Task Bundle Specifications

| Bundle Name | Primary Purpose | Landmark Outputs | Typical Latency |
| :--- | :--- | :--- | :--- |
| **`holistic_landmarker.task`** | Full-body posture & dual hands | 33 Pose + 21 Left + 21 Right + 468 Face | ~12–16 ms (CPU) |
| **`hand_landmarker.task`** | Dedicated dual-hand tracking | 21 3D Coordinates per detected hand | ~4–7 ms (CPU) |

---

## 2. Detailed Component Breakdown

### A. Holistic Landmarker (`holistic_landmarker.task`)
- **Origin:** Google AI Edge ([MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/holistic_landmarker))
- **File Size:** ~13 MB
- **Coordinate Space:** Normalized image space $[0.0, 1.0]$ with relative metric depth $z$.
- **Extraction Protocol:**
  - **Pose:** 33 points (shoulders, elbows, wrists, hips).
  - **Dual Hands:** $2 \times 21$ skeletal points (wrist, CMC, MCP, PIP, DIP, tip).
  - **Face Mesh:** 468 facial contour landmarks for optional head/lip motion tracking.
- **Git Tracking:** Ignored by Git via `.gitignore` to prevent repository bloat.

### B. Hand Landmarker (`hand_landmarker.task`)
- **Origin:** Google AI Edge ([MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker))
- **File Size:** ~9 MB
- **Coordinate Space:** Screen-normalized $(x, y, z)$ coordinates with handedness score (Left vs. Right).
- **Primary Use:** Powers the RealSign 26-class alphabet model with ultra-low latency finger-spelling detection.

---

## 3. Verification & Download Instructions

If task bundles are not present in a clean clone, download them using Python or curl:

```powershell
# Download Holistic Landmarker bundle
curl -o models/tasks/holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

# Download Hand Landmarker bundle
curl -o models/tasks/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```
