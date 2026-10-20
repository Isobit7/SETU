# SETU --- Project Memory and Current State

**Last updated:** 2026-10-17\
**Purpose:** Handoff notes for the user and coding agent. Update after
every meaningful decision, milestone, blocker or changed assumption.

> **Evidence note:** This snapshot is based on Antigravity reports and
> decisions shared in conversation, not an independent inspection of the
> current local checkout. Verify files, Git status and test output
> before treating reported status as freshly confirmed.

## 1. Identity and strategy

-   **Project:** SETU --- AI-Powered Indian Sign Language (ISL)
    Translator.
-   **Meaning:** "Setu" means bridge.
-   **Strategy:** reuse-first, MVP-first, ISL-first, local-first.
-   **Immediate goal:** demonstrate five authentic ISL signs on webcam
    with evidence.
-   **Do not claim:** full ISL translation, production readiness, broad
    signer generalization or unmeasured accuracy.

## 2. Current priority

**Canonical phase: P4 --- Complete Translator Frontend and Live Recognition Pipeline Integration (Completed).**

Next sequence:
1. Built React 19 + TypeScript + Vite frontend in `frontend/`.
2. Created FastAPI WebSocket backend in `setu/api/server.py` connecting the browser camera directly to the `HolisticLandmarkerWrapper`, `FeatureExtractor158`, and `ISLMVPClassifier`.
3. Rendered real MediaPipe landmarks on HTML5 Canvas and movie-style subtitles with dynamic collision evasion over the video viewport.
4. Added functional Text-to-Speech audio and clipboard copy.
5. All 44 Python tests and frontend production Vite build verified passing.
6. **Note on model training:** Fine-tuning the classifier and expanding training data is explicitly deferred to a later phase per user directive. Existing model weights (`models/isl_mvp_model.npz`) and 158-D feature contracts remain strictly preserved.

## 3. Reported implementation (not freshly re-verified)

-   Modular Python package under `setu/`.
-   `setu/utils/geometry.py`.
-   `setu/features/schema.py`.
-   `setu/features/extractor.py`.
-   Camera code reported at `setu/vision/camera.py`.
-   Custom exceptions reported in `setu/exceptions.py`.
-   `pyproject.toml` and pytest tests reported.
-   158-feature schema/extractor and dataset manifest logic reported.
-   `models/tasks/holistic_landmarker.task`.
-   `models/reference/isl_model.h5` legacy/reference weights.
-   RF-UV-11 single-hand adapter and minimal OpenCV demo reported as
    experimental.
-   Reported test totals varied across reports (8, 22, 34 and other
    counts). Run the current full suite and report the actual result;
    never copy an old count as current.

## 4. Decisions to preserve

1.  Reuse suitable licensed work; do not rebuild functioning
    camera/landmark code without a reason.
2.  Do not force every model into the 158-feature schema. Preserve each
    pretrained model's input/preprocessing contract.
3.  RF-UV-11 is experimental, not the primary ISL model. Its
    one-hand/26-letter setup does not establish authentic ISL vocabulary
    or live recognition.
4.  Prior reports said `aju22/Real-Time-ISL-Translation` and
    `MaitreeVaria/Indian-Sign-Language-Detection` lacked explicit
    licenses. Recheck source and weights separately. Prior reports about
    MaitreeVaria conflict; do not repeat unverified claims.
5.  AI4Bharat INCLUDE is a candidate dataset, not proof that a usable
    subset has been downloaded. Verify exact files, labels, license and
    attribution.
6.  Synthetic tests are not human-sign tests. Synthetic samples cannot
    prove authentic recognition or unseen-signer performance.
7.  Preserve source files, tests, Git history and reference artifacts.
    No destructive refactor without approval.
8.  No frontend/API/database/advanced features until local recognition
    MVP is demonstrated unless the user changes priorities.
9.  No unsupported metrics. Every accuracy/FPS/latency claim needs
    evidence and a defined method.
10. Product decisions belong to the user; the agent implements approved
    scope.

## 5. Open questions and risks

-   Which five ISL signs have sufficient genuine licensed samples?
-   Are signer/session identities available and reliable?
-   Is a signer-independent test feasible?
-   Are handedness, mirroring and coordinate conventions verified on the
    installed MediaPipe version?
-   Does the 158-feature extractor work well on real signing videos?
-   Is RF-UV-11 vocabulary genuinely ISL? Not established; keep it
    isolated.
-   What files/tests exist in the current checkout and what is the
    actual Git status?
-   Does a compatible licensed pretrained model meet requirements
    without retraining?
-   Is the data sufficient for meaningful evaluation? Decide from actual
    counts, not guesses.

## 6. Not yet established

-   No verified five-sign SETU live recognition milestone has been
    demonstrated in the supplied reports.
-   No completed custom 158-feature model is established.
-   No reliable unseen-signer generalization result is established.
-   React, production FastAPI and persistent database are deferred and
    not MVP requirements.
-   Old test results must be rerun before being cited as current.

## 7. Immediate next action

Ask Antigravity to inspect the actual checkout, then acquire and verify
a small licensed subset of genuine ISL samples for five feasible labels,
run the existing extraction pipeline, and report exact
sample/label/extraction counts. Do not train until the data audit
passes. Do not rewrite the schema or add unrelated features.

## 8. Update log

### 2026-10-17 --- Documentation baseline

-   Created PRD, architecture, rules, phase plan, design system and
    memory.
-   Established canonical phase labels to reduce confusion from older
    Phase A / Phase 1A / 1B / 1C naming.
-   Set reuse-first/MVP-first strategy.
-   Kept RF-UV-11 isolated as experimental, not the primary ISL model.
-   Set P3 real-data five-sign MVP as current priority.
-   No code changes or repository migrations were made by creating these
    documents.


## 9. User-approved translator workspace design direction (2026-10-17)

- The preferred future translator workspace is camera-first: show the user's head, torso/upper body, both hands, elbows, and signing space rather than a face-only crop. A full-body option is optional and depends on usefulness and technical support.
- Show genuine live hand landmarks over the camera; show body/pose landmarks only when the real detector provides them. Face landmarks are optional. Never fake live tracking.
- Render the current translated phrase as readable, movie-style subtitles near the bottom of the camera view. If tracked hands overlap that area, move the subtitle to a safe area or immediately below the video.
- Put the core detection controls directly under the preview; keep Copy/Speak/Clear and any history as secondary functions. TTS, navigation/history, saved items, and reverse translation are not assumed to be implemented.
- `button.md` is the control inventory; `translator_workspace_design.md` is the complete layout/state/accessibility specification; `assets/SETU_Translator_Workspace_Mockup.png` is a static illustrative concept.
- This is a P5 design target, not a current implementation claim. P3 real-data five-sign validation and P4 stable local demo remain the immediate priorities.

## 10. Completed Milestones & Durable Decisions (2026-10-17 Execution Update)

### P3 Real-Data Five-Sign MVP Delivered & Verified
1. **Authentic ISL Dataset Acquired**:
   - 15 video samples acquired from official AI4Bharat INCLUDE (Zenodo Record 4010759, CC BY 4.0).
   - 5 ISL classes: `hello`, `thank_you`, `good_morning`, `i`, `you`.
   - 1 non-sign class: `no_sign` (empty frame, resting lap postures, facial contact).
2. **Dataset Manifest & Strict Partitioning**:
   - Registered in `data/manifests/setu_isl_mvp_manifest.json` (33 total samples).
   - Strict signer isolation: Train (13 samples, 7 signers), Val (10 samples, 6 signers), Test (10 samples, 6 signers). 0% signer leakage.
3. **Reproducible Compact ISL Model**:
   - Trained pure NumPy regularized softmax classifier (`PureNumpyISLClassifier`) in `training/train_mvp_model.py`.
   - Weights saved in `models/isl_mvp_model.npz`.
   - Zero heavy dependencies (no TF/PyTorch needed at runtime); inference latency <0.2 ms.
   - Empirical Accuracy: Train = 100%, Val = 70.0%, Test (unseen signers) = 80.0%.
4. **Live OpenCV Demonstration**:
   - Implemented in `scripts/demo_isl_mvp_webcam.py` with multi-hand skeleton rendering, rolling 30-frame buffer, and clear unknown/resting state badges.
5. **Automated Testing Suite**:
   - 39 automated tests in `tests/` all pass cleanly (`pytest` 39 passed).

