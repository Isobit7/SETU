# SETU Conceptual Architecture & User Manuals

This directory houses accessible, human-friendly conceptual documentation for teammates, developers, evaluators, and end users seeking to understand the **SETU** Indian Sign Language (ISL) recognition system.

Each document follows our progressive disclosure format:
1. **What it is** (Plain English summary)
2. **Why it is needed** (Real-world motivation and user challenges)
3. **How it works** (Algorithmic and structural flow)
4. **How it is used in SETU** (Concrete engineering implementation)
5. **Practical Walkthrough & Diagrams** (Visual representations and tables)
6. **Key Constraints & Limitations** (Edge cases and technical boundaries)

---

## Master Manual Directory

| Manual | Primary Audience | Core Focus |
| :--- | :--- | :--- |
| [**`PROJECT_OVERVIEW.md`**](PROJECT_OVERVIEW.md) | Beginners / General Audience | Mission statement, social impact, technology overview, and feature matrix |
| [**`ARCHITECTURE_AND_PIPELINE.md`**](ARCHITECTURE_AND_PIPELINE.md) | Teammates / Developers | Five-stage data pipeline from camera capture to speech synthesis |
| [**`MODEL_AND_DATASET_SPEC.md`**](MODEL_AND_DATASET_SPEC.md) | ML Engineers / Researchers | Detailed specifications for all 5 ML models and authentic datasets |
| [**`END_TO_END_FLOW.md`**](END_TO_END_FLOW.md) | System Architects / Evaluators | Lifecycle of a single video frame with sequence and timing diagrams |
| [**`VISUAL_DASHBOARD.md`**](VISUAL_DASHBOARD.md) | Frontend Developers / Designers | UI layout, color tokens, skeletal rendering specs, and component cards |
| [**`ROADMAP_AND_SELF_AUDIT.md`**](ROADMAP_AND_SELF_AUDIT.md) | Project Leads / Evaluators | Quality audit checklist, verification metrics, and future roadmap |

---

## Quick Navigation Rules
- **For New Developers:** Start with [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md), then read [`ARCHITECTURE_AND_PIPELINE.md`](ARCHITECTURE_AND_PIPELINE.md).
- **For ML & Computer Vision Specialists:** Jump directly to [`MODEL_AND_DATASET_SPEC.md`](MODEL_AND_DATASET_SPEC.md).
- **For Frontend & UX Contributors:** Review [`VISUAL_DASHBOARD.md`](VISUAL_DASHBOARD.md) and [`END_TO_END_FLOW.md`](END_TO_END_FLOW.md).
