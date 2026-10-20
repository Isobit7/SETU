# SETU Translator Workspace — Design Specification

## 1. The main design decision

Build the translator around a **large live camera preview**. The user should be able to see their upper body and both hands while signing, see the real detector's hand/body landmarks over the video, and read each result in a short, movie-style subtitle band near the bottom of that same camera view.

The subtitle is the immediate feedback. A small translation panel beside or below the preview can repeat the current text and provide Copy, Speak, Clear, and history actions, but it must not replace the requested on-video subtitle.

## 2. Camera framing

Use a wide, adjustable framing target that includes:

- head and shoulders;
- torso down to at least the waist when practical;
- both hands and elbows through the full signing space;
- enough margin around the arms that hand movement does not leave the frame.

Do not crop to a face-only view. The default should be a wide upper-body shot for clear hand and body tracking. Offer a full-body framing option if the camera/model and use case can support it; provide a simple on-screen prompt such as `Step back until both hands and your upper body are visible`.

## 3. Subtitle placement — most important visual detail

- Default to **subtitles ON**.
- Render the translated phrase in a centered, high-contrast, semi-transparent dark strip at the bottom edge of the camera preview, like film subtitles.
- Use large readable text, ideally one or two lines; animate only subtly when the text changes.
- Keep the band shallow so it does not occupy the user's hands/torso area unnecessarily.
- If actual hand landmarks enter or overlap the caption safe area, move the subtitle to another safe band or temporarily place it in a strip immediately below the video. Do not cover the sign just to preserve a fixed position.
- Make caption visibility and text size adjustable. Caption controls must not clear the result.
- On mobile, keep the same visual relationship: camera first, subtitle at the lower edge, then controls and persistent translation details below.

## 4. Landmark overlay

Overlay real-time detected landmark points and connecting lines directly on the camera image. Suggested visual encoding:

- left hand: one bright, distinct colour;
- right hand: a second distinct colour;
- body/pose: a third colour;
- face: optional separate colour only if useful and supported.

Show a small legend using both colour and text labels. The user asked to see landmarks because this makes detection visible and reassuring. Do not use decorative/hard-coded landmarks when the model is not tracking; show `Model loading`, `No hands detected`, `Low light`, or an honest error state instead. Hand landmarks are the priority; body/pose landmarks are included if the chosen detector exposes them. Face landmarks are optional, not required for the first version.

## 5. Recommended desktop layout

1. **Left navigation:** SETU brand; Translator (selected); History; Saved; Settings; Help & Support. Keep it narrow and collapsible.
2. **Top control bar:** translation direction, sign-language selection (ISL initially), output language (for example English), and accessibility settings.
3. **Primary camera area:** the largest component on screen, with live/paused status, framing guide, real landmark overlay, full-screen action, and the caption band.
4. **Under-camera controls:** Start Detection; Pause/Resume; Stop Detection; Reset; Full Screen; Landmark controls. Group destructive and non-destructive actions clearly.
5. **Translation details panel:** current text, confidence only if actually available, Copy, Speak/Play Audio, Clear Current Translation, and optionally Save.
6. **Recent translations:** short current-session history for review. On smaller windows, this may move below the camera instead of beside it.
7. **Bottom status strip:** camera and detection status plus small tips for framing and lighting.

The preview should have the strongest visual emphasis. Avoid a crowded dashboard or putting the translation entirely in the sidebar.

## 6. Recommended visual direction

- **Style:** calm, modern, inclusive, trustworthy; functional before decorative.
- **Base:** soft neutral/light background and deep navy navigation.
- **Primary accent:** clear blue for the main action and active navigation.
- **Tracking feedback:** distinct green/cyan/orange/purple colours for landmark layers, each with a legend and labels.
- **Text:** large and readable; avoid thin text over video.
- **Controls:** rounded, consistent buttons with both icons and text for primary actions.
- **States:** explicit Live, Paused, No Hands Detected, Low Light, Camera Permission Needed, and Error states.
- **Motion:** subtle landmark/caption updates; respect reduced-motion preferences.

The graphic in `assets/SETU_Translator_Workspace_Mockup.png` is a custom wireframe that demonstrates the intended caption location and landmark placement. The additional concept boards are references, not a pixel-perfect commitment.

## 7. Interaction flow

1. User opens Translator.
2. If camera access is not granted, explain why it is needed and offer Allow Camera / Not Now.
3. Show a short framing tip and whether the camera is ready.
4. User presses Start Detection.
5. Show live video, genuine landmarks, and a visible `Detecting` status.
6. When the supported model produces a recognized result, update the subtitle; keep the latest result in the adjacent details panel if shown.
7. User can Pause/Resume, Stop, Reset, Copy, Speak, hide/show landmarks, change caption size, or enter fullscreen.
8. If hands are not detected, show a helpful instruction rather than a blank screen or invented result.
9. Stopping detection releases the camera. History should only be cleared by a separate, explicit action.

## 8. Detection and error states

Design and implement the following states explicitly:

- No camera permission: explain the purpose and provide browser-specific recovery guidance.
- Camera loading / model loading: progress state, no fake tracking points.
- Camera ready but not detecting: show framing/lighting tips.
- Detecting: show current camera status and genuine landmark overlay.
- Paused: keep the last result but label it paused; do not suggest new signs are being processed.
- Low confidence or ambiguous result: label as uncertain when model confidence or equivalent diagnostics are actually available.
- Low lighting: explain how to improve lighting.
- Camera unavailable/model failure: show error details and Retry.
- Camera stopped: remove active indicator and release the stream.

## 9. Accessibility and privacy

- Keyboard-operable controls and visible focus rings.
- Accessible names for icon-only controls; descriptive text for camera and detection statuses.
- Adequate text/background contrast and adjustable caption size.
- Do not rely on colour alone to explain landmark type or error severity.
- Respect reduced motion.
- Make camera-active status obvious, and stop the camera on Stop Detection or workspace exit.
- Do not claim local-only processing, zero storage, or end-to-end privacy unless the actual architecture guarantees it. Explain what is processed/stored in accurate terms.
- Never imply that a single isolated hand pose means the system understands every sentence. Describe only the language, vocabulary, and translation direction supported by the current model.

## 10. Responsive behavior

**Desktop:** camera on the left/center, translation details to the right, navigation at the left edge.

**Tablet:** camera remains first; translation panel may drop below it; controls wrap cleanly.

**Mobile:** compact/collapsible navigation; camera first; subtitle within preview; large Start/Pause/Stop controls directly below; transcript/history after controls. Avoid tiny control clusters or horizontal scrolling.

## 11. Build sequence & Implementation Status

1. [x] **Static & Responsive Layout**: Built in `frontend/` using React 19 + TypeScript + Vite, deep navy sidebar, and light workspace.
2. [x] **Camera & Hardware Controls**: `getUserMedia` integration with Start, Pause, Resume, Stop (stream track termination), and Reset.
3. [x] **Real Landmark Detector Integration**: MediaPipe Tasks `HolisticLandmarkerWrapper` streaming normalized coordinates over WebSocket, drawn to HTML5 canvas (Emerald = Left Hand, Amber = Right Hand, Sky Blue = Pose, Violet = Face).
4. [x] **Movie-Style Subtitle Placement**: Centered translucent dark band over camera bottom with automatic collision evasion (repositions up when hands enter the lower frame).
5. [x] **ISL Model Output Binding**: Connected directly to `ISLMVPClassifier` recognizing 5 authentic ISL signs (`hello`, `thank_you`, `good_morning`, `i`, `you`) + `no_sign` resting state.
6. [x] **Audio & Action Controls**: Functional Web Speech API text-to-speech (`speechSynthesis`), clipboard copy, and session history list.
7. [x] **Verification**: 44/44 Python tests passing and frontend Vite production build passing. Model retraining and vocabulary fine-tuning deferred to next phase.
