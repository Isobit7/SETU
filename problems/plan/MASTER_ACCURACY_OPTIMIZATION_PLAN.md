# SETU Master Accuracy Optimization & Problem Lifecycle Plan

This directory structure governs the iterative resolution of the 10 core bottlenecks causing low live-inference accuracy in SETU Indian Sign Language (ISL) recognition.

---

## 1. Directory Structure & Lifecycle Flow

The `problems/` directory is partitioned into four explicit operational stages:

```
problems/
├── plan/          # New architectural proposals, redesign roadmaps, and unconfirmed RFCs
├── not_solved/    # Approved, validated problem cards awaiting engineering implementation
├── in_progress/   # Currently active problem being coded, modified, or tuned
├── solved/        # Verified solutions with automated tests passing and documentation
└── manage.py      # Automated lifecycle manager CLI
```

### Transition Lifecycle:
$$\text{plan/} \xrightarrow{\text{Approve/Triage}} \text{not\_solved/} \xrightarrow{\text{Start Work}} \text{in\_progress/} \xrightarrow{\text{Verify \& Test}} \text{solved/}$$

Every problem starts as a structured card in `plan/` or `not_solved/`, transitions to `in_progress/` as soon as code files are modified, and finally moves to `solved/` only when unit tests pass and live validation benchmarks confirm the fix without regressions.

---

## 2. Problem Manager CLI (`problems/manage.py`)

All movements and tracking can be performed automatically using the standard Python CLI:

| Command | Action |
| :--- | :--- |
| `python problems/manage.py list` | Displays the live ASCII status board across all 4 stages |
| `python problems/manage.py move <ID> <stage>` | Transitions card `<ID>` to `<stage>` and updates frontmatter metadata |
| `python problems/manage.py new <ID> <slug> "<Title>"` | Scaffolds a new problem card template in `plan/` |
| `python problems/manage.py test` | Runs self-contained assertions verifying directory state and transition logic |

---

## 3. The 10 Core Accuracy Bottlenecks

| ID | Problem Title | Severity | Impact Zone | Root Cause Summary |
| :--- | :--- | :--- | :--- | :--- |
| **PROB-01** | Heuristic Slicing Bug / False 'O' Hijack | Critical | `realsign.py` | Slices zero-padded left hand during single-hand signs, calculating zero distance and forcing $+2.0$ logit boost for 'O' (94.7% confidence hijack). |
| **PROB-02** | Dual-Hand Spatial Distance Erasure | High | `geometry.py` | Normalizes both hands independently to $(0,0,0)$, destroying inter-wrist Euclidean distance and rendering relative hand position invisible. |
| **PROB-03** | Single-Hand vs Dual-Hand Training Mismatch | Critical | Dataset / Training | Training data had resting hand in slots `[0:63]`. Real single-hand signs zero out LH, causing 0% accuracy collapse on `D, E, M, N, X`. |
| **PROB-04** | Landmark 9 Canonical Coordinate Drift | Medium | `generate_multisigner_dataset.py` | Synthetic 3D rotations tilted Landmark 9 away from canonical $(0, -1, 0)$ vector ($std = 0.418$), while inference strictly clamps it to $(0, -1, 0)$. |
| **PROB-05** | MediaPipe Handedness Flapping in Mirror Mode | High | `recognize.py` | MediaPipe flips 'Left'/'Right' labels on alternating frames, causing sagittal reflection ($x \to -x$) and erratic multi-letter flickering. |
| **PROB-06** | Missing Explicit Joint Flexion Angles & Spread | Medium | Feature Extraction | 126-D vector has only raw $(x,y,z)$ coordinates, forcing the MLP to infer non-linear joint angles and inter-finger spread from scratch. |
| **PROB-07** | MediaPipe Z-Depth Noise Distortion | Medium | `geometry.py` | Noisy MediaPipe depth values cause out-of-plane rotation $\phi = \arctan2(z_9, -y_9)$ to tilt landmarks by $15^\circ\text{--}20^\circ$. |
| **PROB-08** | Absence of Negative / `NO_SIGN` Class | High | Model & Dataset | 26-class alphabet model has no background/neutral class, forcing resting hands and idle frames into random false-positive letters. |
| **PROB-09** | Inference Running on Transition Movement Frames | High | `recognize.py` | Alphabet classification runs on every frame regardless of kinetic velocity, producing garbage predictions during hand movement between signs. |
| **PROB-10** | Lack of Diverse Anthropometric Signer Morphology | Medium | Synthetic Data | Multi-signer generator only varied rotation and scale, omitting physiological hand variations (e.g. finger length vs palm width ratios). |

---

## 4. Phased Execution Roadmap

### Phase 1: Immediate High-Impact Runtime Fixes (PROB-01 & PROB-09)
- **PROB-01**: Fix `realsign.py` heuristic slicing so it selects the active hand and checks whether hand energy is $> 0$ before applying any boost.
- **PROB-09**: Gate alphabet inference in `recognize.py` by kinetic energy threshold ($E_k < \text{threshold}$) so transitional motion frames are skipped until the hand stabilizes into a steady sign.

### Phase 2: Stabilization & Coordinate Robustness (PROB-05 & PROB-07)
- **PROB-05**: Implement a hysteresis buffer for MediaPipe handedness tags to eliminate single-frame flipping and mirror mode inversion.
- **PROB-07**: Dampen the $z$-axis rotation angle during 3D canonical alignment when $z$-variance exceeds sensor noise thresholds.

### Phase 3: Relative Spatial Geometry & Feature Augmentation (PROB-02 & PROB-06)
- **PROB-02**: Preserve relative inter-wrist displacement in dual-hand features while maintaining scale invariance.
- **PROB-06**: Compute explicit joint flexion angles (MCP-PIP-DIP) and fingertip spread distances as invariant auxiliary features.

### Phase 4: Training Distribution Alignment & Negative Class Ingestion (PROB-03, PROB-04, PROB-08, PROB-10)
- **PROB-04 & PROB-10**: Re-generate multi-signer dataset with strictly aligned canonical vectors and variable anthropometric proportions.
- **PROB-03 & PROB-08**: Retrain RealSign MLP with 50% independent hand dropout and a 27th `NO_SIGN` class to eliminate false resting-hand collapses.
