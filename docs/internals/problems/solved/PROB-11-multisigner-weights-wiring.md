---
id: PROB-11
title: "Multi-Signer 42k Weights Default Wiring in Pipeline Manager"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01 12:17
severity: High
category: Model Weights & Pipeline Wiring
---

# PROB-11: Multi-Signer 42k Weights Default Wiring in Pipeline Manager

## 1. Problem Description
Although `realsign_multisigner_mlp.pth` was trained on 42,867 spatial-augmented samples across multiple signers (achieving 97.4% test accuracy) and properly registered in `models/registry.py`, both the live WebSocket recognition pipeline and the REST `/api/predict` endpoint were still defaulting to the older 1,200-sample single-signer checkpoint (`realsign_26class_mlp.pth`). Furthermore, if the frontend requested `"realsign_multisigner"`, the WebSocket model dispatcher ignored it.

## 2. Root Cause Analysis
1. In `backend/app/services/pipeline_manager.py` (line 78):
   ```python
   self.realsign_model_path = Path("models/weights/realsign_26class_mlp.pth")
   ```
2. In `backend/app/services/realsign_classifier_service.py` (lines 138–140):
   ```python
   self.candidate_paths = [
       self.project_root / "models" / "weights" / "realsign_26class_mlp.pth",
   ]
   ```
3. In `backend/app/websocket/recognize.py` (line 335):
   ```python
   if req_model in ("realsign", "isl_mvp", "isl_sequence_lstm", "include_stream"):
   ```
   Selecting `"realsign_multisigner"` was not accepted by the dispatcher and did not switch to the multi-signer weights.

## 3. Resolution & Code Changes
1. **`pipeline_manager.py`**:
   - Updated initialization to dynamically check for `models/weights/realsign_multisigner_mlp.pth` first, defaulting to it when available.
2. **`realsign_classifier_service.py`**:
   - Added `realsign_multisigner_mlp.pth` as the primary entry in `candidate_paths`.
3. **`backend/app/websocket/recognize.py`**:
   - Updated `set_model` command handler to accept `"realsign_multisigner"` and `"realsign_alphabet"`, routing seamlessly to the active alphabet engine.
4. **`backend/app/main.py`**:
   - Added `"realsign_multisigner_mlp"` to `HealthResponse.active_classifiers`.

## 4. Acceptance Criteria & Verification
- [x] Server startup logs confirm `realsign_multisigner_mlp.pth` is loaded.
- [x] Switching model to `"realsign_multisigner"` over WebSocket succeeds.
- [x] All unit tests pass with zero regressions.
