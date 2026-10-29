# Sathwik Documentation Hub — SETU Indian Sign Language Translator

Welcome to the **Sathwik** project documentation directory. This folder contains the complete architectural breakdowns, problem diagnostics, model audits, and improvement plans for the SETU codebase.

---

## 📌 Primary Reference Documents

1. [**`OPTIMISE.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/OPTIMISE.md) **(Codebase Modernization & Neural Optimization Blueprint)**
   - **System Audit**: Comprehensive analysis of why traditional `if-else` rules, hardcoded timers, and static dictionaries fail on real signers.
   - **Modern Neural Solutions**: Replacing `CONVERSATIONAL_BIGRAMS` with Qwen2.5-0.5B Trie-constrained SLM, replacing refractory timer `0.45s` with continuous CTC decoding, and replacing CLAHE with Zero-DCE.
   - **Mathematical Formulations & Architecture Flowcharts**: Exact equations, loss functions, and code before/after transformations.
   - **5-Phase Implementation Plan**: Concrete milestones, latency budgets, and engineering schedule.

2. [**`SETU_WORKING_AND_FORMULAS_GUIDE.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/SETU_WORKING_AND_FORMULAS_GUIDE.md) **(Complete System Working & Mathematical Formulas Guide)**
   - **End-to-End Visual Architecture**: Mermaid flowchart of the 7-stage processing pipeline.
   - **Mathematical Formulas**: Exact equations for CLAHE contrast boost, Euclidean framing & distance evaluation, wrist translation centering, palm scale normalization, 6-D facial ratios, and 12-D body anchors.
   - **ML & DL Engines**: In-depth review of RealSign 26-Class MLP, ISL Sequence BiLSTM, and AI4Bharat INCLUDE (263 signs).
   - **Qwen2.5-0.5B Neural Seq2Seq Translation**: How the on-device Small Language Model on CUDA transforms arbitrary sign tokens and spelled letters into fluent, grammatically correct English sentences in ~350 ms.
   - **Actionable Improvement Ideas**: Step-by-step suggestions for improving model accuracy, finger jitter filtering, and future enhancements.

3. [**`SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md) **(Checkpoint 28 Master Hardened Edition)**
   - Complete 236-file repository inventory, 15-endpoint API encyclopedia, deep scientific diagnostic audit, coupling analysis, and 5-phase engineering roadmap.

4. [**`reports/evaluation_matrices/README.md`**](file:///c:/MINI%20PROJ-SETU/reports/evaluation_matrices/README.md) **(Automated Model Evaluation Matrices & Parameter Catalog)**
   - Layer-by-layer weights, biases, parameter statistics, training hyperparameters (learning rate, optimizer, epochs), and live test accuracy for all models in the project with dynamic auto-updating.

---

## 📂 Supplementary Guides

1. [**`problems/HOW_IT_CAN_BE_SOLVED.md`**](file:///c:/MINI%20PROJ-SETU/Sathwik/problems/HOW_IT_CAN_BE_SOLVED.md)
   - Deep algorithmic analysis of confidence floor clamping, streak lockouts, and coordination pitfalls with code diff solutions.

2. **`commands/`**: Operational commands and workflow references.
3. **`skill-groups/`**: Engineering domain skill groupings.
