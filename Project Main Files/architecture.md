# SETU --- Architecture

**Status:** Baseline architecture. Clearly distinguishes reported
components from future plans.\
**Last updated:** 2026-10-17

## 1. Objective

Keep the first recognition path small, testable and reuse-first. Do not
force a pretrained model into SETU's custom feature format. Do not build
the web stack before the local recognition demo works.

## 2. Reported stack (verify against actual checkout)

  ---------------------------------------------------------------------------------
  Layer                   Reported choice                   Status
  ----------------------- --------------------------------- -----------------------
  Language                Python                            Existing project

  Camera                  OpenCV                            Camera pipeline
                                                            reported

  Landmarks               MediaPipe Tasks / Holistic task   Reported present;
                          bundle                            verify API/version

  Features                NumPy custom 158-feature          Reported implemented
                          extractor                         and synthetic-tested

  Tests                   pytest                            Reported test suites

  Config/errors           Python configuration and custom   Reported; verify files
                          exceptions                        

  Experimental model      RF-UV-11 adapter/model            Isolated experiment;
                                                            not primary ISL model

  Reference weights       `models/reference/isl_model.h5`   Legacy/reference only;
                                                            compatibility unproven

  Frontend                React/Vite                        Future

  API                     FastAPI, optional WebSockets      Future; only after
                                                            local recognition works

  Database                SQLite if later needed            Not required for MVP

  Hosting                 Local-first on Windows            Initial target
  ---------------------------------------------------------------------------------

Do not assume a dependency or model is installed merely because it
appears in a plan. Check actual configuration and environment.

## 3. MVP flow

``` text
Webcam → OpenCV camera reader → MediaPipe landmarks
      → validate result → model-specific preprocessing adapter
      → selected model → confidence/rejection policy
      → optional smoothing (only if needed) → OpenCV preview
```

If no hand or invalid landmarks are present, return a known
status/unknown outcome rather than inventing a label. Raw frames for
inference and mirrored preview frames must be kept distinct where
needed.

## 4. Model contracts

### SETU custom features

The reported schema has 158 values per frame and is described in
`docs/phase1a_specifications.md`; inspect the real schema before relying
on exact indices or mathematical claims. - Static recognition may use
one frame or documented temporal aggregation. - Dynamic recognition uses
an ordered sequence, e.g. `(T, 158)`. - A batch may be
`(batch_size, T, 158)`. - `T=30` is an example from prior plans, not a
universal constant. - Training and inference must use identical
preprocessing and missing-hand behavior.

### Reused pretrained models

Preserve each model's original input shape, landmark order, coordinate
scaling, normalization, output order and label mapping. Record
repository URL, exact revision, model hash, license evidence and
dependencies. Keep unverified models out of the default recognition
path.

## 5. Model selection

-   Static isolated shapes: a small MLP/SVM or compatible licensed
    pretrained model may suffice.
-   Motion-dependent signs: consider GRU/LSTM/TCN only when suitable
    temporal data exists.
-   Two-hand signs require a representation that preserves relevant
    inter-hand information; a single-hand model cannot directly encode
    it.
-   Prefer pretrained inference when vocabulary, license, preprocessing
    and real-sign performance are verified.
-   Do not build a multi-model router until a real requirement is
    demonstrated.

## 6. Intended repository structure

This is a logical target, not proof that every folder exists. Do not
move files merely to match this diagram; inspect, migrate safely, update
imports and run tests first.

``` text
MINI PROJ-SETU/
├── .gitignore
├── pyproject.toml
├── README.md
├── .env.example
├── docs/
│   ├── PRD.md
│   ├── architecture.md
│   ├── rules.md
│   ├── phase.md
│   ├── design.md
│   └── memory.md
├── setu/                 # existing Python package
│   ├── vision/           # camera and landmark detection
│   ├── features/         # schema and extraction
│   ├── utils/            # geometry and helpers
│   └── exceptions.py
├── training/             # manifests, training and evaluation
├── tests/
├── models/
│   ├── tasks/
│   ├── pretrained/       # licensed, validated assets
│   └── reference/        # legacy/reference assets
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
├── scripts/
├── backend/              # future; not MVP blocker
└── frontend/             # future; not MVP blocker
```

## 7. Module responsibilities

  -----------------------------------------------------------------------
  Module                  Owns                    Does not own
  ----------------------- ----------------------- -----------------------
  Camera reader           Open/release camera,    Inference or UI state
                          frames, camera errors   

  Landmark detector       Structured              Training or persistence
                          landmarks/handedness    

  Feature extractor       Deterministic           Camera/file I/O
                          landmark-to-feature     
                          transformation          

  Model adapter           Model-specific          Arbitrary downloads or
                          preprocessing,          silent label changes
                          inference and label     
                          mapping                 

  Recognition service     Coordinate detector,    Rendering GUI or
                          model and rejection     dataset acquisition
                          policy                  

  Demo/UI                 Preview, result, status Duplicated feature math
                          and controls            

  Dataset manifest        Provenance, label,      Invented
                          signer/session if       identities/license
                          known, split, hashes    metadata

  Training/evaluation     Splits, training,       Overwriting raw data
                          metrics, artifact       
                          metadata                
  -----------------------------------------------------------------------

## 8. Future API (do not implement yet)

Potential versioned FastAPI endpoints after MVP acceptance: -
`GET /health` --- process health. - `GET /api/v1/recognition/status` ---
model readiness/version. - `WS /api/v1/recognition/ws` --- optional live
result stream if frontend needs it. -
`POST /api/v1/recognition/sessions` --- optional metadata only if
session persistence is approved.

Use typed schemas, bounded inputs, timeouts and structured errors. Do
not transmit raw video to a server by default. Do not duplicate
recognition logic in the transport layer.

## 9. Data manifest and artifacts

Record where available: sample ID/path, label, original URL and asset
identifier, license and attribution, verification date, SHA-256, file
type/duration/resolution, signer/session only if genuinely supplied,
real/synthetic/augmented status, extractor/schema version, frame count,
rejection reason and split. Never fabricate provenance. Mark synthetic
data explicitly.

## 10. Future database (not required for MVP)

If persistence is approved later: - `recognition_sessions`: ID,
start/end time, model version, optional anonymous test metadata. -
`recognition_events`: ID, session ID, timestamp, predicted label, score,
accepted/rejected, optional correction. - `model_registry`: model
ID/version, source, license, input schema, label version, artifact hash,
creation time.

Do not store raw video or identifiable biometric data by default. This
is a proposal, not an existing database.

## 11. External components

-   **AI4Bharat INCLUDE:** candidate dataset; verify exact files, labels
    and applicable license/attribution before use.
-   **AI4Bharat OpenHands:** code/research candidate; verify
    checkpoints, compatibility and current terms.
-   **RF-UV-11 ISL Recognition MediaPipe:** experimental single-hand
    candidate; ISL vocabulary and performance are not established.
-   **aju22 / MaitreeVaria repositories:** supplied audits reported no
    explicit license; do not copy code/weights without confirmed
    permission. Prior reports about MaitreeVaria conflict, so inspect
    the actual repository before repeating claims.

No third-party hosted service is required for the MVP. Any new service
needs justification and approval.

## 12. Error behavior

-   Camera failure: actionable message, retry if possible, release
    resources.
-   No/invalid landmarks: typed status; no fabricated prediction.
-   Wrong shape or NaN/Inf: reject and log diagnostic.
-   Missing/incompatible model: clear startup error.
-   Low/ambiguous score: unknown/"Not sure", not forced output.
-   Corrupt data: record exclusion reason.
-   Future API errors: structured codes without secrets or stack traces
    exposed to users.

## 13. Architecture decision rule

Use the smallest architecture that serves the current milestone. New
layers, services, tables and frameworks require a concrete need, a test
and a clear benefit.
