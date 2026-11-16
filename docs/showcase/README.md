# Project SETU — Interactive Offline Documentation Hub

> **Owner:** Sathwik (3rd Year B.Tech AI & ML Student)  
> **Directory:** `Sathwik/`  
> **Purpose:** Centralized, 100% offline-first interactive technical documentation, architecture walkthrough, model registry, and Viva defense preparation suite for **Project SETU (Indian Sign Language Translator)**.

---

## 🚀 Quick Access: Open the Documentation in Your Browser

Double-click or open [`index.html`](index.html) in any web browser (Chrome, Edge, Firefox, Safari) to launch the master visual documentation hub. **No internet connection, web server, or Node/Python process is required.**

```powershell
# Open directly in Windows default browser
Start-Process Sathwik\index.html
```

---

## 📚 Master Documentation Suite Structure

| Document | Focus Area & Content |
| :--- | :--- |
| **[`index.html`](index.html)** | **Central Portal & Overview:** Interactive entry point with high-level system architecture diagram, quick stats, feature matrix, and navigation cards. |
| **[`01_Project_Overview.html`](01_Project_Overview.html)** | **Mission & Authentic ISL:** Linguistic realities of Indian Sign Language (ISL vs. ASL, two-handed alphabet requirement, non-manual markers), target users, scope, and technology stack. |
| **[`02_APIs_and_Backend.html`](02_APIs_and_Backend.html)** | **Backend Architecture & APIs:** Fast-path REST endpoints (16 route handlers vs. 21 URL paths), real-time WebSocket protocol (`/ws/recognize`), Pydantic schemas, and filesystem persistence. |
| **[`03_AI_Models_and_ML.html`](03_AI_Models_and_ML.html)** | **AI/ML Models & Spatial Geometry:** Master model registry, RealSign 26-class MLP (94.99% test acc), 126-D bimanual landmark normalization math, neuro-symbolic tie-breakers, 328-D BiLSTM, YOLO-World v2, and Qwen2.5-0.5B causal decoder LLM. |
| **[`04_End_to_End_Workflow.html`](04_End_to_End_Workflow.html)** | **Execution Lifecycle & Pipeline Flow:** The complete 10-step frame-to-speech pipeline, 3-state temporal debouncing FSM, kinetic energy segmentation, accumulator timeouts, and Web Speech API TTS. |
| **[`05_Codebase_and_File_Index.html`](05_Codebase_and_File_Index.html)** | **File Index, Developer Guide & Viva Defense:** Complete annotated repository file directory, developer quick start, Google Colab cloud training setup, and **25 comprehensive Technical Viva & Exam Defense Q&As**. |

---

## 🎨 Local Assets & Diagrams (`Sathwik/assets/`)

- **`assets/styles/documentation.css`**: Responsive design system supporting dark and light themes, responsive breakpoints (768px, 1024px), print styles, and styled code blocks.
- **`assets/scripts/documentation.js`**: Client-side interactive features: instant search filtering across topics, dark/light mode toggle with `localStorage` persistence, and one-click code copy buttons.
- **`assets/diagrams/`**:
  - `architecture_overview.svg`: Scalable vector system architecture.
  - `pipeline_flow.svg`: Step-by-step frame processing dataflow.
  - `feature_schemas.svg`: Visual breakdown of landmark dimensions (63-D, 126-D, 158-D, 164-D, 328-D).

---

## 📦 Historical & Legacy Archive

The previous raw draft markdown files have been safely preserved in:
`docs/archive/legacy_markdown_docs.zip`
