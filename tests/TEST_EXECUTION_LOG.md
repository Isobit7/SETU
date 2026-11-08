# SETU — Automated Test Execution Report

> **Execution Date & Time:** 2026-11-08 21:01:58 IST  
> **Overall Status:** **PASSED (100% Green)**  
> **Total Scenarios Evaluated:** 250 tests across 31 test suites  
> **Passed:** 250 | **Failed:** 0 | **Skipped:** 0  
> **Total Execution Duration:** 120.78 seconds  

---

## 1. Scenario-by-Scenario Verification Summary

| # | Test Scenario / Feature Tested | Result | Duration | Details & Functionality |
| :-: | :--- | :---: | :---: | :--- |
| 1 | **Full System State Transitions & Edge Case Handling**<br>_test_all_states_and_scenarios.py_ | **✓ PASS** | 11705.5 ms | Verifies camera lifecycle, pause, resume, reset, and client state consistency (12/12 checks passed) |
| 2 | **REST API Health & Diagnostics Endpoints**<br>_test_api.py_ | **✓ PASS** | 5707.4 ms | Checks /api/health, /api/vocabulary, and server telemetry responses (15/15 checks passed) |
| 3 | **FastAPI Modular Lifecycle & CORS Security**<br>_test_backend_app_modular.py_ | **✓ PASS** | 41.9 ms | Ensures clean startup, model pre-loading, and CORS header restrictions (6/6 checks passed) |
| 4 | **Camera Feed & MediaPipe Vision Pipeline**<br>_test_camera_and_holistic_vision.py_ | **✓ PASS** | 368.1 ms | Tests OpenCV frame decoding, aspect ratio normalization, and landmarker inputs (4/4 checks passed) |
| 5 | **Configuration Integrity & Structured Error Hierarchy**<br>_test_config_exceptions.py_ | **✓ PASS** | 2.8 ms | Validates settings.py invariants, missing weight handling, and error types (4/4 checks passed) |
| 6 | **Personal Sign Recording & Dynamic Calibration Wizard**<br>_test_custom_sign_pipeline.py_ | **✓ PASS** | 4315.8 ms | Tests 5-take custom gesture recording and personal rest-elevation calibration (5/5 checks passed) |
| 7 | **Numerals (0-9, ₹) & Domain Vocabulary Packs**<br>_test_digits_and_domain_packs.py_ | **✓ PASS** | 1847.5 ms | Tests medical, emergency, and banking vocabulary packs with numbers (5/5 checks passed) |
| 8 | **Multi-Dimensional Feature Extractors (126D / 158D / 176D)**<br>_test_feature_extraction_and_manifest_pipeline.py_ | **✓ PASS** | 1090.6 ms | Verifies bimanual palm normalization, joint angles, and body anchors (4/4 checks passed) |
| 9 | **158-D Spatial Feature Schema Invariants**<br>_test_feature_schema.py_ | **✓ PASS** | 12.4 ms | Validates feature ordering, Euclidean distance symmetry, and tensor shapes (14/14 checks passed) |
| 10 | **AI4Bharat INCLUDE 263-Class Large Vocabulary Model**<br>_test_include_model.py_ | **✓ PASS** | 12916.7 ms | Tests continuous temporal window sliding, top-5 candidates, and softmax output (11/11 checks passed) |
| 11 | **Indic Multilingual Translation (Kannada, Telugu, Hindi, Tamil)**<br>_test_indic_translation_and_calibration.py_ | **✓ PASS** | 495.8 ms | Verifies canonical idioms (e.g. I WATER NEED -> ನನಗೆ ನೀರು ಬೇಕು / నాకు నీళ్లు కావాలి) (7/7 checks passed) |
| 12 | **Scale & Rotation Invariant Spatial Representations**<br>_test_invariant_features_and_segmentation.py_ | **✓ PASS** | 20.4 ms | Checks user distance scaling, tilt invariance, and kinetic energy boundaries (14/14 checks passed) |
| 13 | **Conversational ISL MVP 6-Class Phrase Classifier**<br>_test_isl_mvp_model.py_ | **✓ PASS** | 55.2 ms | Tests global spatial pooling, Ridge classification, and probability bounds (11/11 checks passed) |
| 14 | **Dynamic ISL Continuous Sequence BiLSTM**<br>_test_isl_sequence_lstm_classifier.py_ | **✓ PASS** | 18.6 ms | Tests 30-frame temporal classification, class confidence, and CTC blank collapse (9/9 checks passed) |
| 15 | **Kinetic Energy Movement & Transit Segmentation**<br>_test_kinetic_segmenter.py_ | **✓ PASS** | 3.3 ms | Verifies static vs dynamic motion classification and transition suppression (3/3 checks passed) |
| 16 | **Live Production Stability & Sentence Buffer Protection**<br>_test_live_bugfixes.py_ | **✓ PASS** | 10707.9 ms | Tests hand drop buffer persistence, shake candidate cancellation, and 60 FPS speed (14/14 checks passed) |
| 17 | **RealSign Multi-Signer Model Accuracy Verification**<br>_test_model_accuracy_benchmarks.py_ | **✓ PASS** | 77.3 ms | Verifies 92.00% benchmark accuracy across all 26 alphabet classes (A-Z) (8/8 checks passed) |
| 18 | **Layer-by-Layer Weights & Biases Inspector**<br>_test_model_evaluation_matrices.py_ | **✓ PASS** | 9767.1 ms | Validates parameter statistics, weight tensors, and training metadata (4/4 checks passed) |
| 19 | **On-Device Qwen2.5-0.5B Neural Gloss Translator**<br>_test_neural_translator.py_ | **✓ PASS** | 2.0 ms | Verifies zero-shot ISL gloss to fluent English grammar synthesis (3/3 checks passed) |
| 20 | **Sentence Accumulator & Word Timing Buffers**<br>_test_nlp_pipeline.py_ | **✓ PASS** | 6730.6 ms | Tests letter buffering, word timeouts, and backspace operations (13/13 checks passed) |
| 21 | **Non-Manual Markers (NMMs) & Head Intent Gestures**<br>_test_nmm_avatar_and_active_learning.py_ | **✓ PASS** | 1069.1 ms | Tests eyebrow elevation (question) and head nod/shake detection gates (7/7 checks passed) |
| 22 | **One-Euro Adaptive Speed Landmark Smoothing Filter**<br>_test_one_euro_filter.py_ | **✓ PASS** | 5.5 ms | Verifies jitter suppression during steady poses and zero lag in motion (3/3 checks passed) |
| 23 | **ONNX Graph Export & Lightweight Edge Inference**<br>_test_onnx_edge_and_conversation.py_ | **✓ PASS** | 174.6 ms | Validates ONNX Runtime CPU/GPU execution speed (< 25 microseconds) (3/3 checks passed) |
| 24 | **Real-World Dataset Integrity & Manifest Consistency**<br>_test_real_data_readiness.py_ | **✓ PASS** | 391.6 ms | Checks landmark data distributions, absence of NaNs, and class balance (5/5 checks passed) |
| 25 | **RealSign 26-Class Bimanual MLP Classifier**<br>_test_realsign_classifier.py_ | **✓ PASS** | 3359.1 ms | Tests palm normalization, heuristic disambiguation, and left-hand mirroring (15/15 checks passed) |
| 26 | **Roboflow & Ultralytics Secondary Adapters**<br>_test_rf_uv11_adapter.py_ | **✓ PASS** | 43.1 ms | Verifies secondary fallback classifier compatibility and schema consistency (7/7 checks passed) |
| 27 | **Norvig Statistical Spell Corrector & Lexicon Bounds**<br>_test_spell_corrector_and_nlp.py_ | **✓ PASS** | 1161.1 ms | Tests auto-correction of fingerspelled typing typos against ISL dictionary (7/7 checks passed) |
| 28 | **ISL SOV to English SVO Grammar Syntax Transformer**<br>_test_syntax_transformer.py_ | **✓ PASS** | 3.3 ms | Tests word re-ordering, copula insertion (is/am/are), and pronoun normalization (5/5 checks passed) |
| 29 | **Dataset Preprocessing & Augmentation Pipelines**<br>_test_training_pipeline_integrity.py_ | **✓ PASS** | 634.2 ms | Validates train/validation/test split isolation and data augmentations (5/5 checks passed) |
| 30 | **User Error Recovery & Extreme Edge Cases**<br>_test_user_behavior_and_errors.py_ | **✓ PASS** | 1957.6 ms | Tests sudden camera disconnects, extreme dark frames, and empty streams (11/11 checks passed) |
| 31 | **Rapid Signing Stutter & Co-Articulation Buffering**<br>_test_user_inputs_and_edge_cases.py_ | **✓ PASS** | 1263.7 ms | Tests continuous fingerspelling, duplicate rejection, and double letter hold (16/16 checks passed) |

---

## 2. Key System Capabilities Verified

- **Sentence Buffer Persistence:** Confirmed `sentence_tokens` are never deleted when hands drop.
- **MediaPipe Head Intent Gating:** Confirmed Head Nod locks candidates instantly and Head Shake cancels active staged letters without touching committed words.
- **Alphabet Mode 60 FPS Speed:** Confirmed `HandLandmarker` runs isolated in alphabet mode with <15ms latency and zero Holistic interference.
- **Real-Time Multilingual Indic Translation:** Verified `I WATER NEED` / `WATER NEED` translates immediately into Kannada (`ನನಗೆ ನೀರು ಬೇಕು.`), Telugu (`నాకు నీళ్లు కావాలి.`), Hindi (`मुझे पानी चाहिए।`), and Tamil (`எனக்கு தண்ணீர் வேண்டும்.`).
- **Regional Voice TTS Routing:** Verified `Speak [S]` routes regional scripts to native browser voices (`kn-IN`, `te-IN`, `ta-IN`, `hi-IN`).

---
*Report auto-generated by SETU Test Runner on 2026-11-08 21:01:58*