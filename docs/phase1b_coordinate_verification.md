# SETU — Phase 1B: Coordinate Systems, Handedness & Geometric Verification

**Document Version:** 1.0.0  
**Status:** Approved Technical Reference  
**Scope:** MediaPipe coordinate representations, aspect-ratio impact, handedness mapping, and inter-hand contact invariants.

---

## 1. MediaPipe Coordinate Spaces

Google MediaPipe Tasks (`HolisticLandmarker` and `HandLandmarker`) provides 3D landmark coordinates $(x, y, z)$ for detected hands:

1. **Normalized Image Coordinates ($x, y$):**
   - $x \in [0.0, 1.0]$: Horizontal position normalized by image width $W$.
   - $y \in [0.0, 1.0]$: Vertical position normalized by image height $H$.
   - The coordinate $(0.0, 0.0)$ corresponds to the top-left corner of the camera frame; $(1.0, 1.0)$ corresponds to the bottom-right corner.

2. **Relative Depth Coordinate ($z$):**
   - $z$ represents landmark depth relative to the wrist (landmark 0).
   - Smaller $z$ values indicate points closer to the camera; larger values indicate points further back.
   - MediaPipe scales $z$ roughly at the same magnitude as $x$, but $z$ is inferred monocularly from 2D RGB imagery and has higher statistical noise ($\approx 3\times$ to $5\times$) than $(x, y)$.

---

## 2. Aspect Ratio & Anisotropic Distance Analysis

### The Anisotropy Issue
Standard webcam resolutions (e.g., $640 \times 480$) have a 4:3 aspect ratio.
- A displacement of $\Delta x = 0.10$ in normalized units equals $0.10 \times 640 = 64$ pixels.
- A displacement of $\Delta y = 0.10$ in normalized units equals $0.10 \times 480 = 48$ pixels.

If Euclidean distance is calculated naively in normalized $[0, 1]$ coordinates without reference scaling:
$$d_{\text{naive}} = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$$
A physical 64-pixel vertical distance would register as $\Delta y = 0.133$, producing artificial distance elongation along the vertical axis.

### SETU's Two-Stage Geometric Normalization

#### Stage A: Single-Hand Translation & Scale Normalization (Slots [0:73] and [73:146])
To prevent aspect-ratio stretching and distance variations from corrupting sign recognition, each individual hand is transformed into local anatomical coordinates:
1. **Wrist Translation:** $\mathbf{p}_{\text{trans}}^{(i)} = \mathbf{p}^{(i)} - \mathbf{p}^{(0)}$ (shifting wrist to $(0, 0, 0)$).
2. **Palm Span Scaling:** The anatomical palm span $S$ is defined as the distance between the wrist ($i=0$) and the middle MCP knuckle ($i=9$):
   $$S = \|\mathbf{p}^{(9)} - \mathbf{p}^{(0)}\|_2$$
   $$\mathbf{p}_{\text{norm}}^{(i)} = \frac{\mathbf{p}_{\text{trans}}^{(i)}}{\max(S, \epsilon)} \quad (\epsilon = 10^{-6})$$
This guarantees that single-hand finger angles and distances are invariant to:
- Distance of the signer from the camera (depth scaling).
- Signer hand size differences (child vs. adult).
- Global camera translation across the frame.

#### Stage B: Inter-Hand Relational Features (Slots [148:158])
For inter-hand features, using wrist-centered coordinates would erase inter-hand distance because both wrists would sit at $(0, 0, 0)$. Therefore:
- The delta vectors $\Delta \mathbf{w} = \mathbf{w}_R - \mathbf{w}_L$ and $\Delta \mathbf{i} = \mathbf{p}_R^{(8)} - \mathbf{p}_L^{(8)}$ are computed in the shared uncentered camera coordinate space.
- The smooth Gaussian contact score $C$ is computed as:
  $$C = \exp\left(-\frac{\min(d_{\text{wrist}}, d_{\text{index}}, d_{\text{thumb}})^2}{2\sigma^2}\right) \quad (\sigma = 0.05)$$
  - When hands make physical contact, $d \approx 0.0 \implies C \approx 1.0$.
  - When hands are separated by $> 0.15$ normalized units, $C \approx 0.0$.
  - Because contact detection occurs when $d \to 0$, aspect-ratio anisotropy near contact is virtually zero.

---

## 3. Handedness & Camera Mirroring Contract

```
+-----------------------------------------------------------------------------------+
|                            SETU CAMERA & HANDEDNESS FLOW                          |
+-----------------------------------------------------------------------------------+
| 1. Physical Signer Facing Webcam:                                                 |
|    - User's Right Hand is anatomically RIGHT.                                     |
|    - Appears on LEFT of raw image sensor (x < 0.5).                               |
|                                                                                   |
| 2. Video Capture (setu.vision.camera.CameraReader):                              |
|    - raw_bgr_frame: Unaltered sensor feed -> Passed to MediaPipe Tasks.           |
|    - preview_bgr_frame: cv2.flip(frame, 1) -> Displayed to User (Mirror UX).      |
|                                                                                   |
| 3. Keypoint Detection (setu.vision.landmarks.HolisticLandmarkerWrapper):          |
|    - MediaPipe analyzes raw_bgr_frame.                                            |
|    - Accurately classifies User Right Hand as right_hand_landmarks.                |
|    - Accurately classifies User Left Hand as left_hand_landmarks.                  |
|                                                                                   |
| 4. Slotting Assignment (setu.features.extractor.FeatureExtractor158):             |
|    - Slot 0 (Indices 0:73):   User Right Hand (Dominant). Presence Flag: 146.     |
|    - Slot 1 (Indices 73:146):  User Left Hand (Support).  Presence Flag: 147.     |
+-----------------------------------------------------------------------------------+
```

### Safety Rules
1. **Never pass a mirrored frame to MediaPipe Tasks:** Flipping the frame horizontally before landmark detection inverts MediaPipe's internal handedness classification, causing the right hand to be mislabeled as left.
2. **Mirror preview only for UI:** Horizontal flipping must be strictly isolated to the display pipeline for human user comfort.
