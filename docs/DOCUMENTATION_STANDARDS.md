# SETU Documentation Standards Guide

## 1. Overview and Purpose

This document establishes the official engineering documentation standards for the **SETU (C2)** Indian Sign Language (ISL) recognition system. All project documentation—including architecture manuals, developer onboarding, API contracts, research reports, and user guides—must adhere to these standards to ensure consistency, technical precision, and readability across all previewers and platforms (GitHub, VS Code, IDE viewers, and web browsers).

---

## 2. Core Principles

1. **Preserve Meaning and Technical Depth:**
   - Never truncate, oversimplify, or remove technical parameters, mathematical definitions, error bounds, or API parameters for the sake of brevity.
   - Refactor explanations to improve clarity, not to reduce detail.
2. **Audience-Specific Framing:**
   - **User Guides & Conceptual Manuals:** Focus on what it is, why it is needed, how it works, practical step-by-step walkthroughs, and troubleshooting.
   - **Developer & Architecture Docs:** Detail component boundaries, data structures, state machines, latencies, threads, and contracts.
   - **API Specs:** Explicitly define HTTP methods, endpoints, request/response JSON schemas, error codes, and WebSocket packet payloads.
3. **Verified Facts Only:**
   - Statements must match active code, configuration, and passing test suites. Never document hypothetical or unverified behavior as existing. Clearly mark planned features with `[Status: Planned]`.

---

## 3. Layout and Table Width Constraints

### The 4-Column Table Limit
Tables rendered in Markdown previewers and GitHub frequently stretch off-screen or wrap awkwardly when they exceed 4 columns.

- **Rule 1:** Markdown tables must contain a maximum of **3 to 4 columns**.
- **Rule 2:** Never use inline HTML `<br>` tags to cram multiline paragraphs or lists into table cells.
- **Rule 3:** For complex entities with 5 or more properties (e.g., dataset specs, model architectures, network endpoints), format them as **Vertical Specification Cards** using 2-column Key-Value tables or structured definition lists.

#### Example: Prohibited Table Pattern
```markdown
<!-- PROHIBITED: 7 columns with <br> tags causes horizontal overflow -->
| Model | Vocab | Architecture | Feature Vector | Input Shape | Accuracy | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RealSign RF | 26 | Random Forest<br>100 trees | 126-D | (1, 126) | 99.4% | ~2.1ms |
```

#### Example: Recommended Standard (Overview Table + Detailed Cards)
```markdown
<!-- Standard Overview Table (Max 4 Columns) -->
| Model Key | Vocabulary | Architecture | Primary Metric |
| :--- | :--- | :--- | :--- |
| `realsign_rf` | 26 (A–Z) | Random Forest (100 Trees) | 99.4% Test Accuracy |
| `include_lstm` | 5 signs | 2-Layer Bidirectional LSTM | 95.8% Val Accuracy |

<!-- Detailed Vertical Specification Card -->
#### Model: `realsign_rf` (RealSign ISL Alphabet Classifier)

| Parameter | Specification |
| :--- | :--- |
| **Model Key** | `realsign_rf` |
| **Target Vocabulary** | 26 ISL Letters (A–Z) |
| **Input Shape** | 126-D Dual-Hand Feature Vector `(1, 126)` |
| **Normalization** | Wrist-relative subtraction + palm-width scaling |
| **Inference Engine** | Scikit-learn Random Forest (100 estimators) |
| **Inference Latency** | ~2.1 ms CPU execution time |
| **Checkpoint Path** | `models/pretrained/realsign_rf_acc99.joblib` |
```

---

## 4. Code Blocks, Inline Formatting, and Escapes

1. **Inline Code:** Always wrap file names, directory paths, variables, class names, functions, and CLI flags in single backticks.
   - Example: `CameraWorkspace.tsx`, `--reload`, `src/core/`
   - **Prohibited:** Visible escape characters like `\`CameraWorkspace.tsx\`` or `**\*1. Frame Capture\***`.
2. **Fenced Code Blocks:** Always specify the syntax highlighting language (e.g., `bash`, `python`, `typescript`, `json`, `mermaid`).
3. **Literal Examples:** When documenting regular expressions, shell paths, or escape sequences, do not double-escape characters unless explicitly required by the syntax.

---

## 5. Diagramming Conventions (Mermaid)

- Use **Mermaid.js** syntax for architectural graphs, sequence interactions, and state machines.
- Node labels with special characters (parentheses, brackets, hyphens) must be enclosed in double quotes: `id["Process (Mode A)"]`.
- Every diagram must be accompanied by explanatory narrative text defining input triggers, state transitions, and failure modes.
- Avoid cluttered graphs. Prefer directional flows (`TD` for top-down hierarchies, `LR` for data pipelines).

---

## 6. Mathematical Notation and Equations

- Format mathematical formulas using standard LaTeX syntax enclosed in single dollar signs (`$x$`) for inline expressions or double dollar signs (`$$...$$`) for standalone display equations.
- When explaining coordinate normalization:
  $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}_i - \mathbf{p}_{\text{wrist}}}{\|\mathbf{p}_{\text{mcp\_middle}} - \mathbf{p}_{\text{wrist}}\|_2}$$
- Ensure all variable symbols are defined directly below the equation.

---

## 7. Cross-Referencing and Hyperlinks

- Use relative repository links to connect related documents.
- Always verify that target files exist before linking.
- Standard cross-links:
  - Architecture overview $\to$ `docs/SETU_DEEP_ARCHITECTURE_GUIDE.md`
  - Active living records $\to$ `records/DATASETS_AND_MODELS.md`, `records/ARCHITECTURE.md`, `records/STRUCTURE.md`
  - User and conceptual guides $\to$ `docs/user_understanding/`
  - Standards and audit reports $\to$ `docs/DOCUMENTATION_STANDARDS.md`, `docs/DOCUMENTATION_AUDIT_REPORT.md`
