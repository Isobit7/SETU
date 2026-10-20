# SETU --- Product Requirements Document (PRD)

**Status:** Initial baseline; provisional decisions must be verified
against the repository and real data.\
**Last updated:** 2026-10-17\
**Product/project name:** SETU --- AI-Powered Indian Sign Language (ISL)
Translator

## 1. One-line description

SETU is a local-first prototype that uses a camera and computer vision
to recognize a carefully verified subset of Indian Sign Language (ISL)
signs and display the predicted meaning as text. Speech output and
broader translation are future possibilities, not current MVP promises.

## 2. Product vision

Build a useful, demonstrable communication aid by extending suitable,
legally reusable work wherever possible. Deliver a small,
evidence-backed recognition workflow first; expand only after the
current milestone works.

SETU is a student prototype, not a replacement for a qualified
interpreter or a fluent translator for unrestricted ISL conversation.

## 3. Target users

-   **Primary evaluation users:** the student developer and project
    evaluators.
-   **Intended users:** ISL signers and hearing people who need help
    understanding the limited set of supported signs.
-   **Secondary users:** students and developers learning computer
    vision and sign recognition.

The MVP is not yet a user-validated accessibility product. Testing with
Deaf or hard-of-hearing participants must be voluntary and
consent-based.

## 4. Problem

People unfamiliar with ISL can struggle to understand signs. Recognition
is difficult because signs can depend on hand shape, movement, location,
orientation, two-hand relationships, facial expression, context and
regional variation. Training on the wrong sign language or incorrect
labels can produce confidently wrong results.

SETU first targets a narrower problem: recognize a small, explicitly
documented set of authentic ISL signs under defined test conditions and
show the result clearly.

## 5. Product principles

1.  **Reuse first:** prefer appropriate, maintained, legally reusable
    repositories, models, datasets and training code.
2.  **Evidence before claims:** distinguish code implementation,
    software tests, model metrics and successful human sign recognition.
3.  **ISL first:** do not substitute ASL data or labels for ISL.
4.  **Small milestones:** demonstrate the core flow before adding a
    dashboard, database, API server or advanced features.
5.  **Preserve work:** avoid destructive refactors; retain Git history,
    tests and reference artifacts.
6.  **Honest uncertainty:** unsupported or low-confidence input must not
    become confident text.
7.  **Privacy by default:** process frames locally where practical; do
    not record or upload video without a clear purpose and consent.

## 6. MVP must-haves

-   Capture a live camera feed using existing camera code.
-   Detect/extract hand landmarks using the existing MediaPipe
    integration.
-   Use one documented model whose vocabulary and input format match the
    selected data.
-   Recognize **five authentic ISL signs selected after inspecting real
    licensed data**. The labels are not fixed until data and vocabulary
    are verified.
-   Display the predicted sign/text and a score whose meaning is
    documented.
-   Handle absent hands, low-confidence predictions, unsupported inputs,
    camera failures and model errors.
-   Define `NO_SIGN`/unknown behavior using appropriate examples;
    synthetic negatives alone are insufficient evidence.
-   Provide a minimal local OpenCV demo; React is not required for this
    milestone.
-   Record data provenance, label mappings, splits, model configuration
    and evaluation.
-   Run regression tests and provide an evidence-based report.
-   Follow source-data and model licenses and attribution requirements.

### MVP acceptance criteria

1.  Five labels have traceable ISL references and usable genuine
    samples.
2.  Data quality, class counts, corrupted files and extraction failures
    are reported.
3.  Train/validation/test splits avoid sample leakage; use
    signer-independent splits when reliable signer IDs and enough
    signers exist. Otherwise disclose the limitation and use a
    defensible held-out split.
4.  The model can be run reproducibly and its method is documented.
5.  Report per-class precision/recall/F1, macro-F1 and confusion matrix
    where data supports them; accuracy alone is insufficient.
6.  The live demo displays predictions and abstains/reports uncertainty
    when appropriate.
7.  Controlled webcam trials are reported separately from offline
    metrics.
8.  Regression tests pass and the milestone is checkpointed in Git.

**Proposed target, not a current result:** aim for macro-F1 ≥ 0.80 on a
defensible held-out set if the sample size supports meaningful
measurement. If not, report the measured result and limitations rather
than claiming generalization.

## 7. Nice-to-have later

-   Broader ISL alphabet and isolated-sign coverage.
-   Simple React/Vite interface with camera, result, confidence and
    output area.
-   FastAPI/WebSocket integration after the local loop works.
-   Optional text-to-speech.
-   Sentence composition, editable text, cooldown/repetition handling.
-   Separate static and dynamic models if evidence warrants them.
-   Performance optimization, offline packaging and usability
    improvements.

### User-approved future workspace direction (deferred to P5)

The eventual translator workspace should prioritize a large live upper-body camera preview, genuine hand-landmark overlay (body/pose landmarks when available), and a high-contrast movie-style translation subtitle near the bottom of the same video. The subtitle must move when tracked hands would be obscured. Keep camera controls directly below the preview and secondary translation actions such as Copy, Speak (only when implemented), and Clear current output in a supporting panel. Responsive mobile layout and explicit permission/no-detection/uncertainty/error states are required for that future interface. The complete control-by-control specification is in `button.md` and `translator_workspace_design.md`.

This approved visual direction does **not** change the current priority: verify the real-data five-sign recognition path and stable local demo before implementing the web workspace. The mockup is illustrative, not evidence of a working translator or valid prediction.

## 8. Out of scope now

-   Full/fluent ISL-to-English translation or every ISL sign/dialect.
-   Treating ASL as ISL.
-   Text-to-sign/avatar generation or reverse translation.
-   Facial-emotion interpretation, object detection or general scene
    understanding.
-   Dashboard, accounts, cloud hosting, multi-user database or
    analytics.
-   Automatic recording/upload of camera footage.
-   Advanced language models or sentence prediction before recognition
    works.
-   Multiple custom models before a baseline is demonstrated.
-   Unmeasured accuracy, FPS, latency or user-benefit claims.

## 9. Success measures

  -----------------------------------------------------------------------
  Measure                             Evidence required
  ----------------------------------- -----------------------------------
  Vocabulary validity                 Source and notes for each label

  Data quality                        File/class/signer/session counts,
                                      corrupt files and extraction
                                      failures

  Leakage control                     Split membership and signer overlap
                                      check where metadata permits

  Offline recognition                 Per-class precision/recall/F1,
                                      macro-F1 and confusion matrix

  False activations                   Defined tests using genuine
                                      resting, transition and non-signing
                                      examples

  Live usability                      Repeated controlled attempts per
                                      sign; correct/incorrect/abstained
                                      counts

  Runtime                             Hardware, resolution, warm-up and
                                      end-to-end measurement method

  Reproducibility                     Commands, versions and actual
                                      outcomes

  Maintainability                     Tests, configuration, clear modules
                                      and Git checkpoint
  -----------------------------------------------------------------------

## 10. Constraints and assumptions

-   The reported working directory is `C:\\MINI PROJ-SETU`; verify the
    actual path before commands.
-   Python, OpenCV, MediaPipe and NumPy are part of the reported
    foundation; exact installed versions must be inspected.
-   A custom 158-feature extractor and tests are reported as
    implemented, but are not automatically compatible with every
    pretrained model.
-   RF-UV-11 is an experimental single-hand model, not an accepted
    primary ISL model.
-   Dataset access, license scope, signer metadata and labels must be
    verified for selected assets.
-   Frontend, API, database and hosting remain deferred until the core
    MVP works.

## 11. Product decision rule

When choosing between a new feature and validating the current
milestone, prioritize validation. Do not add a feature unless required
by acceptance criteria or explicitly approved by the user.
