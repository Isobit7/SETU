# SETU Architecture & Flow Diagrams

> **Audience:** 3rd Year B.Tech AI & ML Student  
> **Mission:** Visual architecture blueprints for the SETU project written in standardized, GitHub-compatible Mermaid (`.mmd`) format.

---

## Diagram Index

1. **[architecture_overview.mmd](source/architecture_overview.mmd)**
   * Complete high-level architecture diagram showing client, network channel, FastAPI server, computer vision layers, and neural models.
2. **[websocket_data_flow.mmd](source/websocket_data_flow.mmd)**
   * Sequence diagram tracing binary frame streaming, protocol upgrade, OpenCV decoding, and JSON prediction delivery.
3. **[neural_pipeline_flow.mmd](source/neural_pipeline_flow.mmd)**
   * Detailed data flow diagram tracing raw $640 \times 480$ pixels through MediaPipe landmark extraction, coordinate normalization, PyTorch RealSign MLP, ISL Sequence BiLSTM, YOLO-World v2, and Qwen SLM syntax translation.

---

## How to Render and View These Diagrams
* **Inside GitHub / VS Code / Markdown Viewers:** Mermaid diagrams render automatically inside standard markdown code blocks tagged with ````mermaid ... ````.
* **Mermaid Live Editor:** Copy any `.mmd` file contents and paste them into [https://mermaid.live](https://mermaid.live) to generate high-resolution PNG or SVG exports for your project presentations and viva slides!

