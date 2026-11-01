---
id: PROB-20
title: "Active Non-Manual Markers for Grammatical Punctuation"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:41
severity: High
category: Vision & Linguistics
---

# PROB-20: Active Non-Manual Markers for Grammatical Punctuation

## 1. Problem Description
In Indian Sign Language (ISL) and all natural sign languages, facial expressions and head movements (Non-Manual Markers, or NMMs) convey core grammatical structure: questions, negation, conditionals, and emotional tone. Although `extract_facial_expression_features` exists in `training/feature_extractor.py`, facial keypoints are completely discarded in live recognition and disabled in `App.tsx`. As a result, the system cannot distinguish between a statement and a question, or an affirmative statement and a negated one, without manual button clicks.

## 2. Root Cause Analysis
1. MediaPipe FaceMesh keypoints were historically disabled in the frontend to save network bandwidth and avoid lag.
2. In `backend/app/websocket/recognize.py`, face landmarks are not forwarded into the classification or NLP accumulator pipeline.
3. Facial blendshapes (`browInnerUp`, `headYaw`, `mouthSmile`) are not mapped to grammatical tokens in `SentenceAccumulator`.

## 3. Implementation Plan & Proposed Architecture
1. **Lightweight Facial Blendshape Extraction**:
   - Extract 3 critical geometric ratios from MediaPipe FaceMesh (computationally cheap, negligible overhead):
     - $\Delta y_{\text{eyebrow}}$: Normalized distance between eyebrows and eyes (detects raised eyebrows $\to$ Question).
     - $\Delta x_{\text{head}}$: Head yaw angular velocity (detects head shake $\to$ Negation).
     - $\Delta y_{\text{head}}$: Head pitch (detects nod $\to$ Affirmation).
2. **Grammar Conditioning in `SentenceAccumulator`**:
   - When $\Delta y_{\text{eyebrow}} > 0.045$ at sentence completion: automatically transform sentence into interrogative form (`?`).
   - When head shaking is sustained during a verb sign: automatically insert negation marker (`NOT`).
3. **Disambiguation Feedback in HUD**:
   - Show subtle icon indicator: "❓ Question detected from expression" or "❌ Negation detected".

## 4. Acceptance Criteria & Verification
- [x] Raised eyebrows during signing `['YOU', 'COME']` automatically produces `"Will you come?"` instead of `"You come."`.
- [x] Sustained head shaking during `['WANT']` produces `"I do not want."`.
- [x] Unit tests in `tests/test_non_manual_markers.py` verify blendshape threshold triggers.
