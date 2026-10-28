# Sathwik Documentation Hub — SETU Indian Sign Language Translator

Welcome to the **Sathwik** project documentation directory. This folder contains the complete architectural breakdowns, problem diagnostics, model audits, and improvement plans for the SETU codebase.

---

## 📌 Single Canonical Master Document

[**`SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md) **(Unified Master Edition)**
This single canonical document consolidates the entire engineering audit into one authoritative reference:
- **Executive Summary & End-to-End Execution Flow:** Camera &rarr; Landmarks &rarr; Normalization &rarr; Inference &rarr; Speech (with ASCII & Mermaid diagrams).
- **Mathematical Invariance Engineering:** Translation (wrist-centering) and scale (palm-span) invariance proofs.
- **Exhaustive 228-File Repository Inventory:** Complete coverage of all 228 files across `backend/`, `frontend/`, `models/`, `training/`, `data/`, `config/`, `scripts/`, `docs/`, and `tests/`.
- **Present Codebase Behavior vs. Actual Intended Scope:** Comprehensive "Vision Gap" analysis comparing what SETU does today vs. its true production mission.
- **Exhaustive ML Model Deep-Dive & Comparative Matrix:** RealSign MLP, ISL MVP Ridge, Sequence BiLSTM, AI4Bharat INCLUDE, and YOLOv11.
- **Complete 13-Endpoint API & WebSocket Encyclopedia:** Full request/response JSON schemas, wire protocols, and frontend callers.
- **Coupling & Fragility Analysis:** 63D/158D/164D/328D contracts, handedness mirroring, backpressure, and race condition prevention.
- **Working vs. Defective Feature Audit:** Concrete step-by-step code diffs to fix Vite proxy, history bookmarks, and custom recording.
- **Inaccuracies, Bottlenecks & Algorithmic Solutions:** Two-handed ISL, adaptive energy filtering, and CPU offloading.
- **Retraining Strategy, Required Datasets Catalog & Target Architectures:** ST-GCN, CTC-BiLSTM, and Conformer models.
- **Proposed New Features & Actionable 5-Phase Roadmap:** Phases 1 to 5.

---

## 📂 Supplementary Guides

1. [**`problems/HOW_IT_CAN_BE_SOLVED.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/problems/HOW_IT_CAN_BE_SOLVED.md)
   - Deep algorithmic analysis of confidence floor clamping, streak lockouts, and coordination pitfalls with code diff solutions.

2. **`commands/`**: Operational commands and workflow references.
3. **`skill-groups/`**: Engineering domain skill groupings.
