# SETU Translator Workspace — Button Guide

**Purpose:** This guide defines every important button/control for the SETU sign-language translator workspace, including its expected behaviour, disabled states, and safety/accessibility expectations. It is a product/UI specification, not proof that the underlying camera or translation model already implements these capabilities.

## 1. Core interaction rule

The camera preview is the primary focus. The user should be able to start signing, see real hand/body landmarks over the live image, and read the latest translation as a movie-style subtitle at the bottom of that same preview. The subtitle must not consistently cover the hands. If the user's tracked hands enter the subtitle's safe area, move the subtitle to a nearby safe position or show it in a caption strip immediately below the preview.

## 2. Main camera and detection controls

| Button / control | Required behaviour | State and safety rules |
|---|---|---|
| **Start Detection** | Requests camera permission if needed, opens the chosen camera, initializes the available landmark/recognition pipeline, and begins updating the live result. | Show clear `Requesting camera`, `Camera ready`, `Detecting`, or error status. Never pretend detection is running if the camera/model failed. |
| **Pause** | Pauses inference and new translation updates while leaving the preview available so the user can reposition. Keep the last caption but mark it as paused/stale. | Change the button to **Resume**. Make it visually obvious that new signing is not being translated. |
| **Resume** | Restarts inference without requiring the user to re-grant permission when the stream is still active. | If the stream has ended, guide the user to Start Detection instead. |
| **Stop Detection** | Stops recognition, releases the camera stream, and ends live translation. | Camera indicator must visibly turn off. Keep prior history unless the user separately clears it. |
| **Reset** | Resets transient tracking/smoothing state and clears the current live result/subtitle so the user can start a fresh phrase. | Do **not** silently delete saved/session history. If reset would discard an unsaved manual edit, ask first. |
| **Switch Camera** | Switches between front/rear or another available camera. | Display only when another usable camera exists. Stop/restart cleanly and show an error if switching fails. |
| **Full Screen** | Expands the camera preview and caption area to use the screen. | Keep Stop/Pause, caption, and status accessible. `Esc` should exit fullscreen. |
| **Camera Settings** | Opens available device/resolution/framing options. | Do not offer settings the browser/device cannot control. |
| **Framing Guide** | Shows an outline or prompt that helps the user fit head, torso, both hands, elbows, and signing space inside the frame. | On by default for first-time users; allow it to be hidden. Avoid forcing a face-only crop. |

## 3. Landmark visibility controls

| Button / control | Required behaviour | Default / notes |
|---|---|---|
| **Landmarks On/Off** | Shows or hides the real-time landmark overlay drawn on top of the camera feed. | **On by default** because visible tracking is a core SETU design requirement. |
| **Hand Landmarks** | Toggles detected left/right hand points and connecting skeleton lines. | On by default. Label left and right hands clearly. |
| **Body/Pose Landmarks** | Toggles detected body/upper-body pose points and connections. | On by default when the selected model provides them. |
| **Face Landmarks** | Toggles optional face points if the model provides them and they are useful to the supported sign language. | Optional and off by default unless genuinely required. Explain why it is used. |
| **Landmark Legend** | Displays which colours identify left hand, right hand, pose, and optional face points. | Use both colour and text/shape labels; never rely on colour alone. |

**Important:** Landmarks must come from the real detector/model output. Do not permanently draw decorative or hard-coded dots and present them as real detection. During loading, missing hands, poor lighting, or model failure, show an honest status message instead.

## 4. Subtitle and translation-output controls

| Button / control | Required behaviour | State and safety rules |
|---|---|---|
| **Subtitle On/Off** | Shows/hides the movie-style caption over the lower edge of the live camera preview. | On by default. Hiding the subtitle must not erase the translation result. |
| **Subtitle Text Size** | Changes subtitle font size. | Provide readable sizes; do not require pinch/zoom. |
| **Copy** | Copies the current translated text to the clipboard. | Announce success or failure accessibly; do not silently fail. |
| **Speak / Play Audio** | Reads the current translated text aloud using text-to-speech when supported. | Disable if there is no text or speech synthesis is unavailable. |
| **Stop Speaking** | Stops the current audio playback. | Only shown/enabled while audio is playing. |
| **Clear Current Translation** | Clears the current subtitle/output only. | Must not erase conversation history or saved translations. If the same result is in active use, update the UI immediately. |
| **Save / Bookmark** | Saves the current translation to the user's Saved area or session collection. | Show whether it is saved; do not imply cloud sync if only stored locally. |
| **Edit / Correct Text** *(optional)* | Lets a user correct a mistranscription if that workflow is included. | Label the result as user-corrected if analytics/training later depends on provenance. Do not silently treat edits as verified model output. |
| **Translation Details** *(optional)* | Shows model confidence or other diagnostics when available. | Do not show made-up confidence percentages. A low-confidence result should be labelled as uncertain. |

## 5. Top-bar selectors

| Control | Required behaviour | Notes |
|---|---|---|
| **Translation Direction** | Selects the supported direction, initially **ISL → Text/Speech**. | Show a second direction such as Text/Speech → Sign only after that direction actually works. Future/unsupported modes should be marked `Coming later` or disabled, not presented as complete. |
| **Language Selector** | Selects a supported sign language or spoken/text output language. | Default to Indian Sign Language (ISL) for the first target market, while clearly separating sign language from output language. Only list implemented languages. |
| **Accessibility Settings** | Opens text size, contrast, motion, caption and keyboard-accessibility options. | Keep changes immediately visible and persistent for the current user/session when possible. |
| **Theme Toggle** *(optional)* | Switches light/dark themes. | Only include if both themes are implemented and contrast remains readable. |
| **Help / Instructions** | Shows a short guide for camera positioning, lighting, landmark overlays, status messages, and current model limitations. | Use plain language and explain how to grant camera permission safely. |

## 6. Navigation controls

| Button / control | Required behaviour |
|---|---|
| **SETU Logo / Home** | Opens the app's home or landing page. If the translator is the default home, it may return to the translator. |
| **Translator** | Opens the live translator workspace; show it as the selected navigation item when active. |
| **History / Chat History** | Shows recent translations with time, translation direction, and text, when history is supported. |
| **Saved** | Shows translations the user has intentionally bookmarked. |
| **Settings** | Opens camera, accessibility, privacy, language, and app preferences that are actually available. |
| **Help & Support** | Provides use instructions, troubleshooting, supported-language details, privacy explanation, and known limitations. |
| **Collapse / Expand Sidebar** | Collapses navigation on smaller screens or when more room is needed for the camera. | 

## 7. History-row actions

| Button / control | Required behaviour |
|---|---|
| **Open Translation** | Restores a past translation for review without falsely resuming a past camera session. |
| **Copy History Item** | Copies the selected historical translation. |
| **Play History Item** | Reads a historical translation aloud when text-to-speech is supported. |
| **Remove Saved Item** | Removes one saved/bookmarked item. Ask for confirmation only when recovery is not available. |
| **Clear History** | Clears the current user's translation history. Use a confirmation step and explain what will be deleted. Keep this separate from Reset and Clear Current Translation. |

## 8. Camera-permission and error-state actions

| Button / control | Required behaviour |
|---|---|
| **Allow Camera** | Requests browser camera permission in response to an intentional user action and explains why it is needed. |
| **Not Now** | Closes the permission prompt without repeatedly re-prompting. Explain that camera translation cannot start until access is granted. |
| **Retry Camera** | Retries opening the camera after the user fixes permission/device issues. |
| **Retry Detection** | Reinitializes the detection pipeline after a recoverable model/runtime error. |
| **How to Fix** | Opens practical tips for low light, missing hands, blocked camera, unsupported browser, or unavailable model. |

## 9. Status indicators (not buttons)

Use persistent text labels plus visual indicators for:

- `Camera off`, `Requesting permission`, `Camera ready`, `Camera active`
- `Detecting`, `Paused`, `No hands detected`, `Low confidence`, `Low light`
- `Landmarks active`, `Model loading`, `Detection error`
- `Microphone active` only if microphone input is genuinely being used

Status colours may help (green = ready, amber = attention, red = error, purple/blue = processing), but always include an icon or text label. Do not rely on colour alone.

## 10. Release checklist and phase alignment

**Important:** these lists describe a target user interface, not proof of implementation. The current project plan prioritizes the real-data five-sign model and a stable local OpenCV demo. The full web workspace below remains a later P5 design target unless the project owner explicitly changes the phase order.

### Current local recognition MVP (P3/P4) — keep this small

- [ ] Camera opens/closes safely and status is explicit
- [ ] Real hand landmarks are shown when the detector returns them
- [ ] A verified supported sign produces a labelled prediction
- [ ] No-hand, unknown/uncertain, camera failure, and model failure states are honest
- [ ] Start/stop/quit and clear-current-output controls work as actually implemented
- [ ] Tests and real-camera evidence are recorded

### Future translator workspace (P5) — implement after its phase is approved

- [ ] Start Detection, Pause, Resume, Stop Detection, Reset
- [ ] Camera permission / retry state
- [ ] Live camera preview with upper-body framing guidance
- [ ] Genuine real-time hand landmarks; body/pose overlay when supported by the chosen detector
- [ ] Movie-style subtitle at the bottom of the video, with overlap-aware repositioning
- [ ] Subtitle On/Off and readable text-size controls
- [ ] Copy and Clear Current Translation; Speak/Play Audio only after text-to-speech is implemented
- [ ] Translation direction/language selectors for implemented modes only
- [ ] Camera/detection status plus clear no-detection/error states
- [ ] Full Screen and responsive desktop/tablet/mobile layouts
- [ ] Accessible focus states, keyboard control, contrast, labels, and text scaling

### Optional / later, once the underlying product supports them

- [ ] Full-body framing mode
- [ ] Front/rear camera switching when available
- [ ] Separate face landmarks, only if the detector exposes useful points and the signing use case needs them
- [ ] Saved translations, persistent history, edit/correction, and export/share
- [ ] Text/voice → sign output or avatar
- [ ] Extra languages, account/profile, themes, and advanced model diagnostics

## 11. Non-negotiable product rules

1. Never display fake live tracking, fake confidence, or a fake successful translation.
2. Do not hide the user's hands behind controls or subtitles; move the caption when tracking shows an overlap.
3. Keep **Reset**, **Clear Current Translation**, and **Clear History** as three different actions.
4. Show when the camera is active; release it when Stop Detection is pressed or the workspace is closed.
5. Keep unsupported capabilities visibly marked as unavailable, beta, or future work.
