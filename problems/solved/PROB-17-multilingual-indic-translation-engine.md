---
id: PROB-17
title: "Multilingual Indic Language Translation Engine"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:22
severity: High
category: NLP & Localization
---

# PROB-17: Multilingual Indic Language Translation Engine

## 1. Problem Description
The SETU platform currently translates Indian Sign Language (ISL) glosses exclusively into English (`syntax_transformer.py` and `neural_translator.py`). Although the frontend voice picker (`useTTS.ts`) lists Indic regional language codes (`hi-IN`, `ta-IN`, `te-IN`, `kn-IN`), selecting Hindi or Tamil causes the browser to speak English text with an unnatural Indian accent (e.g., reading "I would like water" in Hindi TTS rather than translating to "मुझे पानी चाहिए"). The vast majority of deaf users and their families in India require direct translation to Indic languages.

## 2. Root Cause Analysis
1. `backend/app/services/nlp/syntax_transformer.py` only implements English grammatical transformations (SOV -> SVO, Wh-fronting, copula insertion `is/am/are`).
2. There is no Indic grammar mapper or bilingual lookup table.
3. Interestingly, ISL and Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu) share the same **Subject-Object-Verb (SOV)** and postpositional sentence structures. Translating ISL glosses into Hindi is linguistically more direct and less error-prone than English transformation.

## 3. Implementation Plan & Proposed Architecture
1. **Build `backend/app/services/nlp/indic_transformer.py`**:
   - Direct SOV-to-Indic mapping for Hindi (`hi`), Tamil (`ta`), Telugu (`te`), and Kannada (`kn`).
   - Postposition markers: `I` + `WANT` -> `मुझे चाहिए`, `HE` + `GO` -> `वह जा रहा है`.
   - Question marker transformations: `YOU NAME WHAT` -> `आपका नाम क्या है?`.
2. **Expose Target Language Parameter in Accumulator & WebSocket**:
   - Add `target_language: str = "en"` to `SentenceAccumulator.translate_sentence()` and WebSocket payload.
   - When user selects `hi-IN` on frontend, backend returns both `translated_text` (English) and `translated_indic` (Hindi/regional script).
3. **Frontend Integration in `TranslationPanel.tsx` & `useTTS.ts`**:
   - Render localized script directly in subtitles and speak native Indic text with appropriate regional voice.

## 4. Acceptance Criteria & Verification
- [x] ISL glosses `['ME', 'WATER', 'WANT']` translate directly to `"मुझे पानी चाहिए"` when Hindi is active.
- [x] Question glosses `['YOU', 'LIVE', 'WHERE']` translate to `"आप कहाँ रहते हैं?"`.
- [x] Unit tests in `tests/test_indic_translation.py` assert translation accuracy across top 50 ISL phrases.
- [x] Full PyTest test suite remains 100% green.
