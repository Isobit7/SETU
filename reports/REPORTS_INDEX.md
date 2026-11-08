# Master Reports & Problems Index
## Project SETU: Evaluation Reports, Benchmarks & Problem Resolution Catalog

*A chronological index of all quantitative model evaluation reports and technical problem resolution audits.*

---

### Part 1: Quantitative Evaluation Reports (Timestamped: 2026-11-06)

| Report File | Benchmark Dataset | Date Generated | Classes | Sample Count | Top-1 Accuracy | Top-3 Accuracy | Macro F1 |
|---|---|---|---|---|---|---|---|
| [`static_isl_alphabet_evaluation_report.json`](file:///c:/MINI%20PROJ-SETU/reports/static_isl_alphabet_evaluation_report.json) | Static ISL Alphabet (A–Z) | 2026-11-06 | 26 | 7,545 | **92.00%** | **97.33%** | 0.918 |
| [`static_isl_numerals_evaluation_report.json`](file:///c:/MINI%20PROJ-SETU/reports/static_isl_numerals_evaluation_report.json) | Static ISL Digits (0–9) | 2026-11-06 | 10 | 2,841 | **94.08%** | **98.95%** | 0.939 |
| [`static_isl_hindi_vowels_evaluation_report.json`](file:///c:/MINI%20PROJ-SETU/reports/static_isl_hindi_vowels_evaluation_report.json) | Static ISL Hindi Vowels (अ to ऐ) | 2026-11-06 | 8 | 2,082 | **94.71%** | **97.60%** | 0.947 |
| [`amrita_slr_evaluation_report.json`](file:///c:/MINI%20PROJ-SETU/reports/amrita_slr_evaluation_report.json) | Amrita Continuous SLR Phrases | 2026-11-06 | 49 | 978 | **94.20%** | **98.10%** | 0.938 |

---

### Part 2: Detailed Model Confusion & Evaluation Matrices

All full confusion matrices and per-class precision/recall distributions are located in [`reports/model_matrices/`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/):
- **Alphabet MLP Matrix:** [`realsign_26class_mlp_matrix.md`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/realsign_26class_mlp_matrix.md) — Documents confusion pairs (e.g. M vs N, U vs V) and normalization mitigation.
- **Multi-Signer Robustness Matrix:** [`realsign_multisigner_matrix.md`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/realsign_multisigner_matrix.md) — Tests cross-signer generalization across 12 unseen test subjects.
- **Continuous BiLSTM Matrix:** [`amrita_isl_phrase_bilstm_matrix.md`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/amrita_isl_phrase_bilstm_matrix.md) — Measures temporal sequence transitions across 49 conversational phrases.
- **Hindi Vowels Matrix:** [`isl_hindi_vowels_mlp_matrix.md`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/isl_hindi_vowels_mlp_matrix.md) — Evaluates vowel hand poses with minor confusion between अं and ऋ.
- **Numerals Matrix:** [`isl_numerals_mlp_matrix.md`](file:///c:/MINI%20PROJ-SETU/reports/model_matrices/isl_numerals_mlp_matrix.md) — Analyzes digit finger extensions (0 through 9).

---

### Part 3: Solved Problems Catalog (PROB-01 through PROB-25)

The project systematically documented and resolved 25 major technical engineering hurdles located in [`problems/solved/`](file:///c:/MINI%20PROJ-SETU/problems/solved/):

#### Computer Vision & Spatial Normalization (Problems 01–07, 10, 16)
- **`PROB-01`**: **Heuristic Slicing O-Hijack** — Single-hand circular detector falsely triggered Letter 'O' during neutral hand relaxation. Resolved with dynamic resting hand thresholding.
- **`PROB-02`**: **Dual-Hand Relative Distance Erasure** — Processing left and right hands independently lost the critical spatial contact required for ISL letters B, D, P, T, Q. Resolved by explicitly appending 21 inter-hand Euclidean distance scalars.
- **`PROB-03`**: **Single-Hand vs Dual-Hand Distribution Mismatch** — Zero-padding non-dominant hand features distorted single-hand distribution. Resolved with dual-head routing.
- **`PROB-04`**: **Landmark 9 Canonical Coordinate Drift** — Wrist normalization anchor shifted under hand rotation. Resolved with bone-axis Gram-Schmidt alignment.
- **`PROB-05`**: **MediaPipe Handedness Flapping in Mirror Mode** — Selfie camera mirroring caused left/right hand flipping at camera borders. Resolved with spatial hysteresis latching.
- **`PROB-06`**: **Missing Joint Flexion Angles & Spread** — Cartesian $(x,y,z)$ coordinates alone were sensitive to hand size. Resolved with invariant dot-product joint angle features.
- **`PROB-07`**: **MediaPipe Z-Depth Noise Distortion** — Monocular depth estimates fluctuated wildly under low light. Resolved with 2D-dominant projection weights.
- **`PROB-10`**: **Anthropometric Signer Morphology Variance** — High test error on signers with short fingers. Resolved with multi-signer bone length normalization.
- **`PROB-16`**: **Holistic One-Euro Filter** — High-frequency landmark jitter during stationary holds. Resolved with 1-Euro adaptive low-pass filter.

#### Temporal Recognition & Debouncing (Problems 08, 09, 11)
- **`PROB-08`**: **Absence of No-Sign Neutral Class** — Model forced spurious classifications when hands rested. Resolved by introducing explicit synthetic neutral negative training vectors.
- **`PROB-09`**: **Inference on Transitional Movement Frames** — Fast movements between signs triggered spurious intermediate classifications. Resolved with kinetic velocity gating (suppressing inference while hands are in fast motion).
- **`PROB-11`**: **Multi-Signer Weights Wiring** — Dynamic model registry routing between single-signer and multi-signer ONNX models.

#### Frontend Engineering & Memory Leaks (Problems 12–15, 21, 22)
- **`PROB-12`**: **Frontend Canvas PNG Trap** — `toDataURL('image/png')` caused high CPU usage and frame drops. Replaced with compressed JPEG (`toBlob('image/jpeg', 0.6)`) reducing frame serialization latency from 28ms to 3ms.
- **`PROB-13`**: **AudioContext Memory Leak** — Text-to-Speech audio synthesizer created unclosed Web Audio contexts, degrading browser memory. Resolved with single persistent audio context.
- **`PROB-14`**: **Windows Tempfile Handle Leak** — Backend file upload handlers threw `PermissionError: [WinError 32]` on Windows. Resolved with explicit context managers.
- **`PROB-15`**: **Hardcoded Localhost URLs** — Resolved with environment-aware dynamic origin discovery.
- **`PROB-21`**: **Client-Side ONNX Edge Inference** — In-browser ONNX runtime implementation for offline zero-latency alphabet recognition.
- **`PROB-22`**: **Two-Way Split-Screen Conversation Mode** — Layout architecture separating deaf signer video stream and hearing recipient text/speech output.

#### NLP, Translation & Modern Interaction (Problems 17–20, 23–25)
- **`PROB-17`**: **Multilingual Indic Translation Engine** — SOV grammar restructuring converting raw ISL gloss into fluent Hindi, Tamil, Telugu, and Kannada.
- **`PROB-18`**: **Numbers 0–9 and Currency Signs** — Added dedicated numeral detection pipeline.
- **`PROB-19`**: **Signer Personal Calibration Wizard** — User calibration step calculating personalized bone scale ratios.
- **`PROB-20`**: **Active Non-Manual Markers (Facial Grammar)** — Integrated facial landmarks for questions and affirmative nod detection.
- **`PROB-23`**: **3D Skeletal Rigged Avatar (Text-to-Sign)** — 3D anatomical hand visualizer rendering correct ISL hand geometries.
- **`PROB-24`**: **Domain Vocabulary Packs (Medical & Banking)** — Specialized vocabularies for hospital triage and banking transactions.
- **`PROB-25`**: **Active Learning Self-Improvement Pipeline** — Low-confidence edge case logging for targeted dataset retraining.

---

### Part 4: Open Watchpoints & Future Enhancements

| Item | Area | Current Status | Next Milestone |
|---|---|---|---|
| **Facial Expression Depth** | Vision | 2D Facial mesh markers active (head nod, eyebrows) | Train 3D blendshape classifier for complex emotions |
| **Low-Light Night Mode** | Camera | Contrast enhancement active | Adaptive exposure normalization in WebGL shader |
| **Expanded Regional Sign Dialects** | NLP | Standard Northern/Southern ISL covered | Collect community dialect samples (Mumbai vs Kerala ISL) |

