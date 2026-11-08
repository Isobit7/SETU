# Backend API Endpoints Directory (`backend/app/api/`)
## RESTful API Services for Project SETU

This directory houses all backend REST endpoints mounted on FastAPI (`/api/*`). Each file serves a specific domain:

---

### Endpoint Catalog

| Endpoint File | Mounted Route Prefix | Responsibility | Client Consumer |
|---|---|---|---|
| [`endpoints/predict.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/predict.py) | `/api/v1/predict`, `/api/predict` | Single-frame HTTP POST inference on uploaded images. | Testing, external integrations, legacy clients. |
| [`endpoints/models.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/models.py) | `/api/v1/models`, `/api/models` | Queries active neural models, switches active weights, reports model memory footprints. | Frontend Settings Modal (`SettingsModal.tsx`), health monitoring. |
| [`endpoints/nlp.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/nlp.py) | `/api/v1/nlp`, `/api/nlp` | Sentence expansion, spelling correction, next-sign prediction, and Indic language translation (Hindi, Tamil, Telugu, Kannada). | Frontend Translation Panel (`TranslationPanel.tsx`) and Sentence Builder. |
| [`endpoints/vocabulary.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/vocabulary.py) | `/api/v1/vocabulary`, `/api/vocabulary` | Returns lists of supported ISL signs, category mappings (Alphabet, Phrases, Digits), and Domain Packs (Medical, Banking). | Vocabulary Browser Modal (`VocabularyModal.tsx`), Practice Studio. |
| [`endpoints/dataset.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/dataset.py) | `/api/v1/dataset`, `/api/dataset` | Handles custom sign sample recording, saving landmark coordinates to disk for active learning. | Record Custom Sign Modal (`RecordSignModal.tsx`). |
| [`endpoints/video.py`](file:///c:/MINI%20PROJ-SETU/backend/app/api/endpoints/video.py) | `/api/v1/video`, `/api/video` | Offline batch video processing. Accepts MP4/AVI uploads and returns full timestamped subtitles. | Offline Video Subtitle Translator (`IncludeVideoModal.tsx`). |

---

### Distinction: Backend REST vs Real-Time WebSocket vs Frontend Services

1. **Real-Time Streaming:** The live webcam feed does **NOT** use these REST endpoints. It connects directly to the high-throughput WebSocket at `ws://127.0.0.1:8000/ws/recognize` (implemented in [`backend/app/websocket/recognize.py`](file:///c:/MINI%20PROJ-SETU/backend/app/websocket/recognize.py)) to maintain 30 FPS with <32ms latency.
2. **REST Endpoints:** These endpoints handle infrequent configuration actions (fetching sign lists, changing language, uploading videos, or requesting spelling suggestions).
3. **Frontend API Client:** All client-side HTTP calls to these endpoints are centralized in [`frontend/src/services/apiService.ts`](file:///c:/MINI%20PROJ-SETU/frontend/src/services/apiService.ts).

