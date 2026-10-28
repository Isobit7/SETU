# SETU — System Diagnostic Audits & Scientific Evaluation Registry

This directory archives dated evaluation reports, stress tests, telemetry logs, and implementation roadmaps for the SETU Indian Sign Language translation workstation.

---

## Registry of Diagnostic Records

| Date | Document Name | Format | Focus / Deliverable | Status |
| :---: | :--- | :---: | :--- | :---: |
| **2026-10-28** | [`2026-10-28_setu_master_deep_system_diagnostic_audit.md`](file:///c:/MINI%20PROJ-SETU/reports/diagnostics/2026-10-28_setu_master_deep_system_diagnostic_audit.md) | Markdown | Deep diagnostic audit covering Frontend, Backend API/WS, Model Weights Stability, Overfitting analysis, Hallucination stress tests, Precision/Recall/F1, and Latency benchmarks. | **100% HEALTHY** |
| **2026-10-28** | [`2026-10-28_setu_deep_system_diagnostic_telemetry.json`](file:///c:/MINI%20PROJ-SETU/reports/diagnostics/2026-10-28_setu_deep_system_diagnostic_telemetry.json) | JSON | Machine-readable quantitative telemetry generated directly by `scripts/deep_system_diagnostic.py`. | **VERIFIED** |
| **2026-10-28** | [`2026-10-28_setu_master_enhancement_plan.md`](file:///c:/MINI%20PROJ-SETU/reports/diagnostics/2026-10-28_setu_master_enhancement_plan.md) | Markdown | Master enhancement roadmap covering Show-and-Sign COCO object detection, Hand-lift Disambiguation HUD, CLAHE low-light enhancer, and continuous capture retraining. | **EXECUTED** |

---

## Diagnostic Reproduction

To re-run the end-to-end diagnostic suite at any time and generate fresh evaluation numbers:

```powershell
.venv\Scripts\python.exe scripts/deep_system_diagnostic.py
```

