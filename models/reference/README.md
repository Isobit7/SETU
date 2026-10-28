# Phase 0 Reference Pretrained Models (Quarantined)

This directory contains external historical reference artifacts evaluated during Phase 0 proof-of-concept testing.

> ⚠️ **STRICT IP QUARANTINE**: The reference model in this directory is kept under strict legal quarantine and is excluded from SETU production runtimes and distributions. For complete legal audit findings, refer to [**`docs/aju22_license_resolution.md`**](file:///c:/MINI%20PROJ-SETU/docs/aju22_license_resolution.md).

---

## 1. Reference Artifact: `isl_model.h5`

| Attribute | Specification |
| :--- | :--- |
| **Origin Repository** | `aju22/Real-Time-ISL-Translation` (GitHub) |
| **Model Architecture** | Sequential LSTM: LSTM(64) $\to$ LSTM(128) $\to$ LSTM(64) $\to$ Dense(64) $\to$ Dense(32) $\to$ Dense(11) |
| **Input Shape** | `(1, 30, 150)` — Temporal sequence of 30 frames with 150 legacy raw coordinates |
| **Target Vocabulary (11)** | `Hello`, `I am`, `Deaf`, `and`, `Mute`, `Father`, `Mother`, `Happy`, `Sad`, `Food`, `Water` |
| **License / IP Status** | **Unlicensed / All Rights Reserved** (GitHub API confirmed `"license": null`) |
| **Operational Status** | **Strict Quarantine (Read-Only)** — Not packaged or invoked in production. |

---

## 2. Clean-Room Engineering Policy

1. **Zero Upstream Code:** No code, scripts, or normalization functions from `aju22` exist in SETU.
2. **Independent Primary Data:** SETU trains its canonical models directly on primary, legally licensed datasets:
   - **RealSign ISL Alphabet:** Open Research dual-hand dataset (26 classes).
   - **AI4Bharat INCLUDE:** CC BY 4.0 International open dataset (IIT Madras).
3. **No Feature Contamination:** SETU uses a mathematically invariant 126-D / 158-D normalized coordinate space rather than the unnormalized 150-D legacy space.
