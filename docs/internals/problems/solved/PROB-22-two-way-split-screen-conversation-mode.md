---
id: PROB-22
title: "Two-Way Split-Screen Conversational Calling Mode"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:30
severity: High
category: UI/UX & Interaction
---

# PROB-22: Two-Way Split-Screen Conversational Calling Mode

## 1. Problem Description
Currently, the SETU user interface isolates sign recognition into the "Translator" tab and sign synthesis into the "Text to Sign" tab. In a live face-to-face dialogue between a Deaf person and a Hearing person (e.g. at a hospital consultation, bank counter, or retail counter), toggling between disparate screens disrupts the flow of conversation.

## 2. Root Cause Analysis
- The frontend views were developed as modular standalone components (`CameraWorkspace.tsx` and `TextToSignWorkspace.tsx`) rather than a unified bidirectional communication session.
- Audio input (Speech-to-Text via Web Speech API) is not wired to directly trigger sign generation in real time.

## 3. Implementation Plan & Proposed Architecture
1. **Build `frontend/src/components/ConversationStudio.tsx`**:
   - **Top / Signer Pane:** Live camera stream with MediaPipe hand overlay, real-time ISL recognition, auto-accumulated sentence buffer, and text-to-speech speaker output.
   - **Bottom / Voice Pane:** One-touch microphone listening button (Speech-to-Text in English/Hindi/Tamil) that captures spoken words and immediately passes them to the sign avatar.
2. **Interactive Dialogue Bubble Feed**:
   - Visual chat transcript with sender tags:
     - 🤟 **Signer (ISL $\to$ Speech):** "I need to see the doctor."
     - 🎙️ **Speaker (Voice $\to$ Sign):** "Please wait in Room 3."
   - Export dialogue transcript as PDF / text for medical and legal records.
3. **Turn-Taking Visual Indicators**:
   - Animated visual cues showing who is currently signing vs speaking to prevent conversational collision.

## 4. Acceptance Criteria & Verification
- [x] Speech spoken into microphone directly translates into sign avatar animation without leaving the page.
- [x] Signer signing in front of camera outputs spoken audio to computer speakers.
- [x] Dialogue history records both sides chronologically with timestamps.
