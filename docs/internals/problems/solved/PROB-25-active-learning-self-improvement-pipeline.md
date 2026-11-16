---
id: PROB-25
title: "Active Learning and In-App Continuous Self-Improvement Loop"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:42
severity: Medium
category: MLOps & Continuous Improvement
---

# PROB-25: Active Learning and In-App Continuous Self-Improvement Loop

## 1. Problem Description
Currently, when users experience recognition errors in live practice, the samples that caused errors are lost. Although `RecordSignModal.tsx` allows manual sign recording, there is no automated active learning pipeline to capture hard failure cases, cluster ambiguous signs, or retrain model weights seamlessly without developer intervention.

## 2. Root Cause Analysis
- The feedback loop is disconnected: inference happens on client, error corrections (backspace, disambiguation selection) happen in memory, and training requires manual execution of Python scripts in terminal.

## 3. Implementation Plan & Proposed Architecture
1. **Implicit Feedback & Failure Mining**:
   - When a user selects a letter from the Disambiguation HUD or presses `Backspace` immediately after an auto-committed letter, log the pre-correction 126-D feature vector as a "hard negative/ambiguous sample".
2. **Automated Retraining CLI & Endpoint**:
   - Build `scripts/retrain_feedback_loop.py` and REST endpoint `POST /api/v1/dataset/retrain`.
   - Automatically merges user-verified custom samples into the training split while safeguarding validation and test splits from leakage.
3. **Model Versioning & Dynamic Hot-Swapping**:
   - Reload model weights in memory without restarting the FastAPI server using `pipeline.reload_realsign()`.

## 4. Acceptance Criteria & Verification
- [x] Corrected signs are recorded with explicit user consent into `data/processed/feedback/`.
- [x] Automated retrain script executes end-to-end and validates test split accuracy does not degrade.
- [x] API endpoint triggers hot-swap without dropping active WebSocket clients.
