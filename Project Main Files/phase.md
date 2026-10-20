# SETU --- Phased Delivery Plan

**Status:** Canonical forward plan. Older reports used Phase A / Phase
1A / 1B / 1C inconsistently.\
**Last updated:** 2026-10-17

## 1. Delivery strategy

Build small, demonstrable milestones. The goal is a working recognition
path using verified ISL data and reusable components---not the maximum
number of features or architecture documents.

**Gate rule:** A phase is complete only when exit criteria have
evidence. Passing unit tests alone does not prove sign recognition.

## 2. Phase overview

  -----------------------------------------------------------------------------
  Phase             Name              Status based on      Outcome
                                      supplied reports     
  ----------------- ----------------- -------------------- --------------------
  P0                Preserve and      Reported complete;   Safe baseline and
                    audit             verify actual Git    inventory
                                      state                

  P1                Camera and        Reported             Camera-to-landmark
                    landmark          implemented/tested   path
                    foundation                             

  P2                Reuse feasibility RF-UV-11 adapter     Reuse decision based
                    experiment        reported; not        on evidence
                                      accepted as primary  
                                      ISL model            

  P3                Real-data         COMPLETED;           Real data (AI4Bharat INCLUDE)
                    five-sign MVP     verified with        -> pure NumPy model
                                      evidence (80% test)  -> live OpenCV predictions

  P4                Translator        COMPLETED            React 19 + TS frontend,
                    Frontend & API                         FastAPI WebSocket backend,
                    Integration                            real landmarks canvas,
                                                           movie-style subtitles

  **P5**            **Model Tuning    **Deferred**         Retraining, dataset
                    & Vocabulary                           scaling, sentence
                    Expansion**                            composition
  -----------------------------------------------------------------------------

## 3. P0 --- Preserve and audit

**Work:** inspect actual tree and `git status`; protect
source/tests/data/models/licenses; verify safe Git checkpoint; inventory
dependencies and run commands.

**Exit:** branch and uncommitted changes documented; work recoverable;
no audit-time deletions; main modules, tests and assets inventoried.

**Status note:** prior reports describe Git checkpoints and an audit.
Verify the current checkout before assuming it is clean.

## 4. P1 --- Camera and landmark foundation

**Work:** reuse OpenCV camera and MediaPipe detector; check camera
failure/shutdown, absent hands and invalid results; verify physical
left/right mapping and preview mirroring on the installed API; separate
extraction from UI.

**Exit:** camera opens/closes cleanly; zero/one/two-hand cases are
checked where supported; errors are clear; regression tests pass;
real-camera check is recorded.

**Status note:** earlier throughput/latency figures were reported, but
new feature extraction still needs real human-sign testing.

## 5. P2 --- Reuse feasibility experiment

**Work:** test candidate repositories/models in isolation; verify code,
weights, data, labels, preprocessing, license and dependencies
separately; keep unverified models isolated.

**Exit:** reuse matrix distinguishes integrated/tested/studied/rejected;
a candidate decision has evidence; no unlicensed assets copied; ISL
vocabulary is not inferred from repository name.

**Reported outcome:** RF-UV-11 has an experimental single-hand adapter
and automated tests, but authentic ISL vocabulary and real recognition
performance are not established. Keep it isolated.

## 6. P3 --- Real-data five-sign MVP (current priority)

### Step A --- Select data and vocabulary

Choose five signs only after inspecting real, usable, appropriately
licensed samples. Verify labels against traceable ISL references. Record
source, license, class counts, signer/session metadata and limitations.
Keep synthetic data separate.

### Step B --- Extract and validate

Run the existing MediaPipe and feature pipeline on actual files.
Validate shapes, finite values, tracking rate, labels and extraction
failures. Verify hand mapping and coordinate conventions. Preserve raw
files.

### Step C --- Validate or train one baseline

Prefer a compatible licensed pretrained model if it genuinely covers the
task. Otherwise train the smallest suitable model. Do not force
incompatible models into the 158-feature schema. Keep splits separate
and signer-independent when feasible. Record config, seed, versions and
artifact hash.

### Step D --- Evaluate

Report per-class precision/recall/F1, macro-F1 and confusion matrix
where data supports them. Test genuine non-sign/resting/transition
examples. Document rejection behavior. Separate offline metrics from
live trials.

### Step E --- Demonstrate

Create a minimal OpenCV demo showing camera, predicted label,
score/status and unknown/no-sign behavior. Perform repeated controlled
attempts per sign and record mistakes.

**P3 exit criteria** - Five real, traceable ISL labels and usable
samples. - Verified feature/model input contract. - Reproducible
inference/training. - Honest offline evaluation and split limitations. -
Live webcam predictions demonstrated; errors recorded. - Relevant tests
pass, Git checkpoint created, `memory.md` updated.

Do not proceed to a full React interface before P3 passes.

## 7. P4 --- Stable local application

Improve lifecycle, error messages, model-loading diagnostics and
shutdown. Add rejection/smoothing only where tests show need. Improve
repetition behavior only after observing failure modes.

**Exit:** repeated runs, missing model, camera failure and no-hand
states work safely; false activations and latency measured with a
documented method.

## 8. P5 --- API and web interface (deferred)

Only after the local recognition interface is stable: define a minimal
versioned FastAPI contract; add React/Vite for agreed requirements; use
WebSockets only if needed; add database only for an approved persistence
use case.


### Approved translator workspace design target (deferred)

When P5 begins, use `translator_workspace_design.md`, `button.md`, and `assets/SETU_Translator_Workspace_Mockup.png` as the agreed starting point. The workspace should use a live upper-body camera preview; genuine real-time hand landmarks (body/pose when supported); a movie-style subtitle band near the bottom of the preview that repositions if it obscures hands; controls immediately below the camera; and clear permission, loading, paused, no-hands, uncertainty, and error states. History, saved items, reverse translation, and speech are only built when the actual functionality is approved and working.

This does not promote P5 above the current P3/P4 work. Do not start the web implementation merely because these design documents now exist.

**Exit:** typed contract, no duplicated recognition logic,
connection/API/model-not-ready errors handled, local demo remains
usable, privacy documented.

## 9. P6 --- Vocabulary expansion (deferred)

Add classes only with traceable data and evaluation examples. Separate
static/dynamic models only if data/error analysis justifies it. Recheck
class balance and prior-class regressions.

**Exit:** each class has provenance and adequate test cases; class-wise
results and relevant conditions are reported.

## 10. P7 --- Advanced features (deferred)

Potential future items: TTS, sentence composition/correction, broader
alphabet, dynamic signs, text-to-sign/avatar, additional facial/body
cues, offline packaging or personalization. Each needs a user need,
approved scope, test plan and privacy review.

## 11. Scope-change gate

Before changing model architecture or adding a feature, state: which
acceptance criterion requires it; why existing components cannot satisfy
it; files affected; tests; compatibility/licensing impact; and whether
approval is needed. If not needed for the current phase, defer it.
