# SETU Foundation Documents — Updated Design Edition

This archive preserves the original SETU foundation documents and adds the user-requested translator-workspace design direction and button-by-button control guide.

## Added files

- `button.md` — purpose and behavior for every recommended button/control, disabled/error states, status indicators, and a phase-aware release checklist.
- `translator_workspace_design.md` — detailed camera framing, movie-style subtitle placement, real landmark overlay, layout, state handling, accessibility, privacy, responsiveness, and verification-first build order.
- `assets/SETU_Translator_Workspace_Mockup.png` — visual workspace concept; suitable for viewing/sharing.
- `assets/SETU_Translator_Workspace_Mockup.svg` — vector version of the concept board.
- `assets/SETU_Camera_Frame_Illustration.png` — cropped illustrative camera frame used by the concept board.

## Core design decision

The live camera preview is the main focus. Frame the signer from the head through the torso, with both hands, elbows, and signing space visible. Show detected hand landmarks over the camera; add body/pose landmarks when supported by the detector. Place translated text in a high-contrast, movie-style subtitle band at the bottom of the preview. If that band would hide tracked hands, reposition it or place it immediately below the video. Keep controls below the preview and Copy/Speak/Clear as secondary actions.

## Important phase boundary

The workspace is a **future P5 web-interface design target**, not a claim that the existing application already has a web UI, landmark overlay, speech output, history, or a validated translation. The original plan remains MVP-first: validate real licensed ISL samples and the five-sign local recognition flow before building the larger web interface. The mockup contains illustrative pixels/text only; runtime tracking/predictions must come from the real detector and model.

## Existing documents preserved

`PRD.md`, `architecture.md`, `rules.md`, `phase.md`, `design.md`, and `memory.md` remain in the archive. The PRD, phase plan, design guidance, and project memory have small additions to reflect the requested future UI while preserving the current P3/P4 priorities.

## Authoritative Documentation Directory & Navigation

This directory (`Project Main Files/`) is the canonical central reference for the SETU project:

1. [README.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/README.md) — (This file) Overview and document navigation.
2. [PRD.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PRD.md) — Product requirements, problem statement, user personas, and target metrics.
3. [architecture.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/architecture.md) — System architecture, module boundaries, data contracts, and pipeline flow.
4. [design.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/design.md) — Design principles, accessibility guidelines, typography, and color tokens.
5. [translator_workspace_design.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/translator_workspace_design.md) — Camera framing, subtitle positioning, landmark overlays, and workspace layouts.
6. [button.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/button.md) — Control inventory, interaction specs, state transitions, and phase availability.
7. [memory.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/memory.md) — Project memory, durable decisions, handoff notes, and verified status.
8. [phase.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/phase.md) — Phased delivery plan (P0–P7) with strict entry/exit criteria.
9. [rules.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/rules.md) — Mandatory engineering rules and AI agent operating constraints.
10. [PROJECT_STRUCTURE.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PROJECT_STRUCTURE.md) — Full repository layout, folder responsibilities, and structural rules.
11. [KEY_FILES_INDEX.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/KEY_FILES_INDEX.md) — Fast-lookup index for all source files, models, scripts, and tests.
12. [PROJECT_STATUS.md](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/PROJECT_STATUS.md) — Current implementation health, verified features, and next tasks.

