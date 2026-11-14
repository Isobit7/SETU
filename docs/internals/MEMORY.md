# System Memory & Runtime Invariants
## Project SETU: Real-Time Indian Sign Language Recognition Engine

*A living document capturing system state, ports, operational environments, architecture invariants, and directory mappings.*

---

### 1. Active Environment & Network Services

| Service | Protocol / Port | Process / Command | Status |
|---|---|---|---|
| **Frontend Dev Server** | `http://127.0.0.1:5173` | `npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173` | ACTIVE (Vite 5 / React 18) |
| **Backend REST & WS** | `http://127.0.0.1:8000` | `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000` | ACTIVE (FastAPI / Python 3.12) |
| **WebSocket Stream** | `ws://127.0.0.1:8000/ws/recognize` | Full-duplex JSON/Binary frame stream (25–30 FPS) | ACTIVE |
| **Health Check Endpoint** | `http://127.0.0.1:8000/health` | Fast HTTP GET returning server uptime and model status | ACTIVE |

---

### 2. Active Model Weights & Runtime Architecture

| Weight File | Format | Date | Size | Accuracy | Role in Engine |
|---|---|---|---|---|---|
| `models/weights/amrita_isl_phrase_bilstm.pth` | PyTorch | 2026-11-06 | 3.64 MB | 94.2% | **ACTIVE** BiLSTM for 100 continuous ISL phrases |
| `models/weights/realsign_26class_mlp.pth` | PyTorch | 2026-11-06 | 114 KB | 96.8% | **ACTIVE** PyTorch MLP for 26 static ISL alphabets |
| `models/weights/realsign_multisigner.onnx` | ONNX | 2026-11-06 | 109 KB | 96.2% | **ACTIVE** Multi-signer invariant ONNX runtime |
| `models/weights/isl_hindi_vowels.onnx` | ONNX | 2026-11-06 | 105 KB | 98.1% | **ACTIVE** Hindi Vowels classifier (अ, आ, इ, etc.) |
| `models/weights/isl_numerals.onnx` | ONNX | 2026-11-06 | 105 KB | 97.4% | **ACTIVE** Numerals classifier (0 to 9) |
| `frontend/public/models/realsign_multisigner.onnx` | ONNX | 2026-11-06 | 109 KB | 96.2% | **ACTIVE** Client-side edge inference |

---

### 3. Non-Negotiable Engineering Invariants

1. **Feature Vector Geometry (63-dim):**
   - Hand coordinates must always be normalized relative to the wrist (landmark 0 as origin).
   - The scaling factor is the Euclidean distance from wrist (0) to middle finger MCP (landmark 9).
   - Coordinate order is strictly interleaved: $[x_0, y_0, z_0, x_1, y_1, z_1, \dots, x_{20}, y_{20}, z_{20}]$.
2. **Dual-Hand Preserving Invariant:**
   - Dual-hand signs (B, D, P, T, Q, etc.) must include inter-hand joint distance vectors. Processing hands independently destroys the contact signature.
3. **Temporal Debouncing (Hysteresis):**
   - A sign is only committed to the sentence accumulator if it remains the top prediction for at least 3 consecutive frames with confidence $\ge 0.50$.
4. **Zero Uncompiled Utility CSS:**
   - Styling in the frontend uses pure, scoped CSS rules inside `frontend/src/index.css`. Never rely on uncompiled arbitrary Tailwind classes.
5. **Safe Native File Access:**
   - All backend temporary files must use explicit context managers or scoped cleanup to prevent Windows file-lock errors (`PermissionError: [WinError 32]`).

---

### 4. Directory Structure Topology

```
c:\MINI PROJ-SETU\
├── backend/                  # FastAPI ASGI Backend
│   └── app/
│       ├── api/endpoints/    # REST endpoints (dataset, models, nlp, predict, video)
│       ├── services/         # Core business logic (vision, nlp, classification)
│       └── websocket/        # Real-time full-duplex WebSocket stream handler
├── frontend/                 # Vite + React + TypeScript Single-Page Application
│   ├── public/models/        # In-browser edge ONNX weights
│   └── src/
│       ├── components/       # UI modular components (Camera, Practice, Settings, etc.)
│       ├── hooks/            # Custom React hooks (useRecognitionStream, useTTS, etc.)
│       └── services/         # Client-side API services (apiService.ts)
├── models/                   # Neural network weights, ONNX exports, and adapters
│   ├── weights/              # Timestamped production and archive weights
│   └── adapters/             # Model loading wrappers and inference interfaces
├── training/                 # Model training pipelines, datasets, feature extraction
├── reports/                  # Evaluation metrics, confusion matrices, audit logs
├── problems/                 # Solved problem records (PROB-01 to PROB-25)
├── docs/user_understanding/  # Dedicated human-readable guides for viva defense
└── Sathwik/                  # Standalone interactive HTML portal for demonstration
```

