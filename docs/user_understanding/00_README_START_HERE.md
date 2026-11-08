# 🌟 SETU User Understanding Guide: Start Here!
## Welcome Sathwik! Your One-Stop Project Reference & Viva Survival Kit

This dedicated folder was built so you never have to feel overwhelmed by hundreds of files or complex jargon. Everything in this folder is written in **plain, simple language** that you can read in 10 minutes and use to explain your project confidently to your project guide, external examiners, or viva panel.

---

### 📚 Reading Roadmap (Pick What You Need)

| Document | What You Will Learn | Reading Time |
|---|---|---|
| **[`01_PROJECT_EXPLAINED_SIMPLE.md`](file:///c:/MINI%20PROJ-SETU/docs/user_understanding/01_PROJECT_EXPLAINED_SIMPLE.md)** | What is SETU? What problem does it solve? The 4-step journey of a sign from your webcam to spoken words. | 3 mins |
| **[`02_FRONTEND_AND_PRACTICE_MODE.md`](file:///c:/MINI%20PROJ-SETU/docs/user_understanding/02_FRONTEND_AND_PRACTICE_MODE.md)** | How the React website works, how the 3D Hand visualizer works, and how Practice Studio scores your signs. | 4 mins |
| **[`03_BACKEND_AND_WEBSOCKET_FLOW.md`](file:///c:/MINI%20PROJ-SETU/docs/user_understanding/03_BACKEND_AND_WEBSOCKET_FLOW.md)** | How FastAPI & WebSockets stream 30 frames per second, extract 21 hand joints, and debounce predictions. | 4 mins |
| **[`04_MODELS_WEIGHTS_AND_ACCURACY.md`](file:///c:/MINI%20PROJ-SETU/docs/user_understanding/04_MODELS_WEIGHTS_AND_ACCURACY.md)** | Explaining MLP vs BiLSTM vs ONNX, what "weights" actually mean, which weights are current, and what accuracy you achieved. | 4 mins |
| **[`05_TOP_VIVA_AND_EXAM_QUESTIONS.md`](file:///c:/MINI%20PROJ-SETU/docs/user_understanding/05_TOP_VIVA_AND_EXAM_QUESTIONS.md)** | The top 15 questions professors always ask in a viva, and the exact impressive answers you should give! | 5 mins |

---

### ⚡ Quick Cheat Sheet (The 30-Second Elevator Pitch)

> **"What is Project SETU?"**
> *"SETU (meaning 'Bridge') is a real-time, bidirectional AI communication bridge for the Deaf and Hard-of-Hearing community. It captures Indian Sign Language (ISL) gestures through a standard webcam at 30 frames per second, converts them into 3D skeletal landmark coordinates using Google MediaPipe, and classifies them with ultra-fast deep learning models (96.8% accuracy on alphabets, 94.2% on continuous phrases). It then translates the sign into fluent English and Indic languages (Hindi, Tamil, Telugu, Kannada) and speaks it aloud using Text-to-Speech."*

---

### 📁 Where Everything Lives in the Codebase

- **Frontend (What you see):** `frontend/src/` (React + TypeScript + Vite)
- **Backend (The engine):** `backend/app/` (FastAPI + WebSockets + MediaPipe)
- **Active AI Models:** `models/weights/` (Trained on Nov 6, 2026)
- **Evaluation Reports:** `reports/` (Accuracy numbers and confusion matrices)
- **Historical Solved Problems:** `problems/solved/` (25 technical hurdles solved)
- **Training Code:** `training/` (Scripts used to train the neural networks)
