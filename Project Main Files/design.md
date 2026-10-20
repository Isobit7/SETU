# SETU --- Design System and User Experience

**Status:** Design direction for MVP and future UI. Web UI is deferred
until the local recognition milestone works.\
**Last updated:** 2026-10-17

## 1. Personality

SETU should feel **calm, clear, respectful, accessible and dependable**.
It is a communication aid, not a flashy AI showcase. Make the result and
uncertainty easy to understand. Avoid clutter, decorative motion and
unnecessary technical terms in the main view.

## 2. Priorities

1.  Recognition result is the main focus.
2.  Camera/model status is understandable.
3.  "Not sure" is a valid outcome.
4.  Contrast/readability beat decoration.
5.  Do not use color alone to convey state.
6.  The first deliverable is an OpenCV demo. This document does not
    imply a web UI already exists.

## 3. Palette

  Token                 Hex         Use
  --------------------- ----------- -----------------------------
  Primary / SETU teal   `#0F766E`   Main actions and identity
  Primary dark          `#115E59`   Hover/pressed states
  Secondary blue        `#2563EB`   Links and secondary actions
  Accent amber          `#F59E0B`   Warning/uncertainty
  Background            `#F8FAFC`   Light page background
  Surface               `#FFFFFF`   Panels/cards
  Main text             `#0F172A`   Primary content
  Secondary text        `#475569`   Supporting labels
  Border                `#CBD5E1`   Dividers and outlines
  Success               `#15803D`   Paired with label/icon
  Error                 `#B91C1C`   Paired with explanation
  Dark background       `#0B1220`   Optional camera workspace
  Dark surface          `#111827`   Dark panels

Check contrast for actual sizes and states before implementation; not
every pairing is guaranteed accessible.

## 4. Typography

-   Preferred font: Inter; fallback: system UI sans-serif.
-   Main result: 28--36 px desktop, strong weight.
-   Section heading: 20--24 px.
-   Body: 16 px or larger where practical.
-   Supporting labels: 13--14 px; essential status must remain readable.
-   Use sentence case, concise wording and tabular numerals for metrics
    where supported.
-   Avoid decorative display fonts and long all-caps labels.

## 5. Shape and spacing

-   Use an 8 px spacing rhythm: 8, 16, 24, 32 px.
-   Panels/cards: 12--16 px radius; controls: 8--12 px.
-   Use subtle borders and rare/light shadows.
-   Make keyboard focus visible and controls clearly labeled.
-   Keep camera, result and status regions consistently spaced.

## 6. MVP screen

The local demo should show only: 1. **Camera preview** --- dominant
area; landmarks may be drawn for debugging. 2. **Prediction** --- large
recognized sign/text. 3. **Status** --- `Ready`, `Hand not detected`,
`Not sure`, `Camera error` or `Model unavailable`. 4. **Score** ---
shown with a clear label; do not call it a calibrated probability unless
calibration is verified. 5. **Controls** --- quit/start/stop, and clear
output only if implemented. 6. **Debug details** --- optional and
separate from user-facing output.

No dashboard, login, profile, history, analytics or settings page in the
MVP.

## 7. Future web layout (deferred)

When React is approved: simple header/status; camera and result side by
side on wide screens and stacked on narrow screens; editable output area
only after sentence composition exists; short supported-vocabulary and
privacy notes. Keep navigation shallow; no sidebar unless workflows
justify it.

## Approved future translator workspace direction (P5; not the current MVP)

The user has selected a camera-first workspace concept for the eventual web interface:

- Make the live camera preview the largest and most important region.
- Frame the head, shoulders, torso, both hands, elbows, and signing space; avoid a face-only crop. Offer a full-body option only if it is technically useful and supported.
- Draw genuine real-time hand landmarks over the camera. Add body/pose landmarks when available from the chosen detector; face landmarks are optional.
- Place the current translation in a readable movie-style subtitle band at the bottom of the camera view. Keep it shallow and move it to a nearby safe position or just below the video whenever detected hands overlap the subtitle region.
- Keep controls below the preview and a secondary panel for the current translation, Copy, Speak (only after TTS is implemented), Clear Current Translation, and optional history.
- Show explicit camera/model states including permission needed, loading, detecting, paused, no hands detected, low light, uncertain result, and error.
- Use a calm light workspace, deep-navy navigation, blue primary actions, restrained success/warning/error colours, readable text, and responsive layouts.
- Keep history, Saved, Settings, accounts, reverse translation, and audio as optional/later unless the actual project phase and working functionality justify them.

This is a **design target**, not a claim that a web interface, speech synthesis, history, or landmark rendering is currently implemented. The current P3 real-data recognition priority and P4 local demo are unchanged. See `translator_workspace_design.md`, `button.md`, and `assets/SETU_Translator_Workspace_Mockup.png`.

## 8. Interaction states

Define default, hover, focus, pressed, disabled, loading, ready,
warning/uncertain and error states. Do not let camera/model
initialization appear frozen; show status and a recovery action where
possible.

## 9. Recognition UX

-   Label output clearly, e.g. `Predicted sign: WATER`.
-   Low score: `Not sure — please repeat`.
-   No hand: do not leave stale output appearing newly recognized.
-   Add cooldown/smoothing only after repeated output problems are
    observed.
-   Clear accumulated text if sentence/text accumulation is later added.
-   State the model's limited vocabulary.
-   Smoothing must not conceal recognition errors during evaluation.

## 10. Accessibility and respect

Use plain language and readable text. Do not frame Deaf people as
needing to be "fixed." Explain system limits without blaming signers.
Avoid flashing animations and unnecessary audio. Future web controls
must be keyboard accessible. Make camera usage visible and explain local
processing. Never record/upload signing videos without consent.

## 11. Design decision rule

Do not polish the UI before the recognition loop works. Every proposed
visual feature must serve a current user need and milestone.
