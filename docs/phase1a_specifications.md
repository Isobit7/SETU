# SETU — Phase 1A: Feature Specification, Data Sources & Collection Protocol

---

## 1. Feature Representation Contract (158 Dimensions)

### 1.1 Architectural Rationale
The primary failure mode identified in Phase 0 was that raw pixel coordinates $[x, y] \in [0, 1]$ drift when signers change distance, chair height, or camera angle. Furthermore, isolated single-hand normalization loses the spatial relationship between two hands (which is essential for Indian Sign Language).

SETU solves this with a **hybrid 158-feature schema**:
1. **Wrist-Centered & Span-Normalized Coordinates (126 features):** Hands are invariant to translation and camera distance.
2. **Intrinsic Finger Geometry (20 features):** 5 fingertip-to-wrist normalized distances + 5 finger PIP bend angles per hand.
3. **Explicit Presence Flags (2 features):** Distinguishes one-handed vs. two-handed gestures without ambiguity.
4. **Inter-Hand Relational Features (10 features):** Preserves relative hand displacement, distance, and physical contact.

### 1.2 Immutable Slice Layout Table
| Feature Range | Slice Notation | Count | Description | Normalization / Units |
| :--- | :--- | :---: | :--- | :--- |
| `[0:63]` | `RIGHT_HAND_COORDS_SLICE` | 63 | Right Hand 21 landmarks $\times (x, y, z)$ | Centered at wrist (0); divided by palm span |
| `[63:68]` | `RIGHT_HAND_DISTANCES_SLICE` | 5 | Right Hand fingertip-to-wrist Euclidean distances | Normalized by palm span |
| `[68:73]` | `RIGHT_HAND_ANGLES_SLICE` | 5 | Right Hand finger PIP flexion bend angles | In radians, normalized to $[0.0, 1.0]$ ($\theta / \pi$) |
| `[73:136]` | `LEFT_HAND_COORDS_SLICE` | 63 | Left Hand 21 landmarks $\times (x, y, z)$ | Centered at wrist (0); divided by palm span |
| `[136:141]` | `LEFT_HAND_DISTANCES_SLICE` | 5 | Left Hand fingertip-to-wrist Euclidean distances | Normalized by palm span |
| `[141:146]` | `LEFT_HAND_ANGLES_SLICE` | 5 | Left Hand finger PIP flexion bend angles | In radians, normalized to $[0.0, 1.0]$ ($\theta / \pi$) |
| `[146:148]` | `PRESENCE_FLAGS_SLICE` | 2 | Hand presence flags: `[is_right_present, is_left_present]` | Binary: `0.0` (missing) or `1.0` (detected) |
| `[148:151]` | `INTER_WRIST_VECTOR_SLICE` | 3 | Relative wrist displacement vector: $\text{Wrist}_R - \text{Wrist}_L$ | Normalized camera space $(\Delta x, \Delta y, \Delta z)$ |
| `[151:152]` | `INTER_WRIST_DISTANCE_INDEX` | 1 | Euclidean distance between wrists: $\|\vec{\Delta}_{\text{wrist}}\|_2$ | Scalar Euclidean distance in $[0, \sqrt{3}]$ |
| `[152:155]` | `INTER_INDEX_VECTOR_SLICE` | 3 | Relative index tip displacement: $\text{Index}_R - \text{Index}_L$ | Normalized camera space $(\Delta x, \Delta y, \Delta z)$ |
| `[155:156]` | `INTER_INDEX_DISTANCE_INDEX` | 1 | Euclidean distance between index fingertips: $\|\vec{\Delta}_{\text{index}}\|_2$ | Scalar Euclidean distance |
| `[156:157]` | `INTER_THUMB_DISTANCE_INDEX` | 1 | Euclidean distance between thumb tips: $\|\vec{\Delta}_{\text{thumb}}\|_2$ | Scalar Euclidean distance |
| `[157:158]` | `CONTACT_SCORE_INDEX` | 1 | Smooth contact indicator: $\exp(-\min(d_i, d_w, d_t)^2 / (2\sigma^2))$ | Smooth scalar in $[0.0, 1.0]$ ($\sigma = 0.05$) |
| **Total** | | **158** | Complete Feature Dimension | All values float32, zero-padded when hand missing |

---

## 2. Authentic ISL Data Sources & License Audit

### 2.1 Standard Indian Sign Language Structure
Unlike American Sign Language (ASL), which uses a single hand for the entire A–Z manual alphabet, **Indian Sign Language (ISL)**, standardized by the **Indian Sign Language Research and Training Centre (ISLRTC)**, is predominantly **two-handed**:
- **Two-Handed Letters:** A, B, D, E, F, G, H, J, K, M, N, P, Q, R, S, T, W, X, Y, Z.
- **One-Handed Letters:** C, I, L, O, U, V (can be formed with a single dominant hand, but frequently supported by the non-dominant hand in formal signing).
- **Rule:** Models trained on ASL datasets (like Kaggle ASL Alphabet) fail completely when presented with genuine ISL two-handed signs. SETU must only train on authentic ISL data.

### 2.2 Audited Public Datasets for Phase 1
| Dataset Name | Source / Institution | Content & Scope | License | Accessibility & URL |
| :--- | :--- | :--- | :--- | :--- |
| **AI4Bharat INCLUDE** | IIT Madras / AI4Bharat | 263 dynamic sign words, 4,287 videos, 15 categories | **CC-BY 4.0** | Zenodo Record [4010759](https://zenodo.org/records/4010759) |
| **MaitreeVaria ISL** | Open Source Community | 26 static ISL alphabets (A–Z) + 10 numbers (0–9) | **MIT License** | [GitHub: MaitreeVaria](https://github.com/MaitreeVaria/Indian-Sign-Language-Detection) |
| **ISL-CSLTR** | IIT Guwahati / IIIT | Isolated static ISL signs & numbers | Academic Research | IITG Repository |
| **Mendeley ISL Dataset** | Rastogi et al. (Mendeley Data) | Static ISL alphabet & digit collection | **CC-BY 4.0** | Mendeley Data Portal |

### 2.3 Supported Target Label Set for SETU MVP
1. **Static Alphabet (29 Classes):**
   - Letters: `A`, `B`, `C`, `D`, `E`, `F`, `G`, `H`, `I`, `J`, `K`, `L`, `M`, `N`, `O`, `P`, `Q`, `R`, `S`, `T`, `U`, `V`, `W`, `X`, `Y`, `Z` (26 classes).
   - Control Tokens: `SPACE`, `DELETE` (2 classes).
   - Neutral State: `NO_SIGN` (1 class).
2. **Dynamic Greetings & Words (10 Classes):**
   - Greetings: `hello`, `thank you`, `good morning`, `good afternoon`, `good evening`, `good night`, `how are you`, `alright`, `pleased` (9 classes).
   - Neutral State: `NO_SIGN` (1 class).

---

## 3. Dataset Collection & Preprocessing Protocol

### 3.1 Directory Layout
```
c:\MINI PROJ-SETU\
├── data/
│   ├── raw/
│   │   ├── static/
│   │   │   └── <signer_id>_<session_id>/
│   │   │       ├── A/ (e.g. frame_001.jpg, ...)
│   │   │       └── NO_SIGN/
│   │   └── dynamic/
│   │       └── <signer_id>_<session_id>/
│   │           ├── hello/ (e.g. clip_001.mp4, ...)
│   │           └── NO_SIGN/
│   ├── processed/
│   │   ├── static/
│   │   │   ├── train_features_158.npz
│   │   │   └── val_features_158.npz
│   │   └── dynamic/
│   │       ├── train_sequences_30x158.npz
│   │       └── val_sequences_30x158.npz
│   └── metadata/
│       ├── manifest.json
│       └── signers.json
```

### 3.2 Metadata Schema (`manifest.json`)
Every recorded sample or session must be indexed with strict environmental metadata:
```json
{
  "sample_id": "dyn_signer01_sess01_hello_001",
  "signer_id": "signer_01",
  "session_id": "sess_20261016_01",
  "timestamp": "2026-10-16T10:30:00+05:30",
  "label": "hello",
  "is_dynamic": true,
  "frame_count": 30,
  "fps": 15,
  "environment": {
    "lighting": "indoor_natural",
    "background": "plain_wall",
    "distance_meters": 0.8,
    "camera_resolution": [640, 480],
    "dominant_hand": "right"
  },
  "verified_by_human": true
}
```

### 3.3 The Explicit `NO_SIGN` (Negative / Idle) Class Strategy
To solve the false positive failure observed in Phase 0, `NO_SIGN` is treated as a first-class citizen in the training data:
- **Allocation:** At least **15% to 20%** of total training frames and video sequences must belong to `NO_SIGN`.
- **4 Mandatory Sub-Scenarios:**
  1. *Stationary Idle:* Signer seated or standing with hands resting on table or lap (motion energy $\approx 0$).
  2. *Sign Transitions:* Frames where hands enter the frame from below or return to resting position after signing.
  3. *Everyday Non-Sign Incidental Motion:* Scratching head, adjusting eyeglasses, resting chin on palm, stretching fingers, typing.
  4. *Background Noise / Empty Frame:* Video feed with person in background or empty scene.

---

## 4. Pipeline Execution Contract

```
[Raw Camera Video / Image]
         │
         ▼
[MediaPipe Tasks HolisticLandmarker]
         │
         ▼  (pose_landmarks, left_hand_landmarks, right_hand_landmarks)
[setu.features.FeatureExtractor158]
         │
         ▼
[158-Dimensional Feature Vector]
         │
         ├─────────────────────────────┬─────────────────────────────┐
         ▼                             ▼                             ▼
[Static Mode: Single Frame]  [Dynamic Mode: 30-Frame FIFO]  [State Machine / Debounce]
         │                             │                             │
         ▼                             ▼                             ▼
[Static Classifier (A-Z)]     [LSTM Dynamic Classifier]       [Composed Word Output]
```
