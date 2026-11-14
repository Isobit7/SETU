# SETU Codebase Architecture & Technical Inventory

> **Audited Repository:** `c:\MINI PROJ-SETU`  
> **Course Target:** 3rd Year B.Tech AI & ML Student  
> **Purpose:** Comprehensive source-code verified map of all technologies, subsystems, files, data shapes, and communication mechanisms in SETU.

---

## 1. Verified Technology Stack

| Layer | Technology | Version / Spec | Responsibility in SETU | Verified Source Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **Backend Runtime** | Python | 3.12.x (Virtualenv) | Primary server environment | `.\.venv\Scripts\python.exe` |
| **Web Server Framework** | FastAPI | 0.115.x | High-performance REST endpoints & WebSocket handler | `backend/app/main.py` |
| **ASGI Server** | Uvicorn | 0.32.x | Asynchronous event-loop server with auto-reload | `backend/app/main.py#L177-L179` |
| **Deep Learning Framework** | PyTorch | 2.6.0+cu124 | GPU-accelerated tensor operations & neural inference | `models/adapters/realsign.py` |
| **Vision & Landmarks** | MediaPipe | 0.10.21 | 21-point 3D hand landmarks & holistic landmarks | `backend/app/services/vision/landmarks.py` |
| **Image Processing** | OpenCV (cv2) | 4.10.x | Frame decoding, resizing, BGR/RGB conversion | `backend/app/services/vision/camera.py` |
| **Object Grounding** | YOLO-World v2 | Ultralytics 8.3.x (`yolov8s-worldv2.pt`) | Zero-shot open-vocabulary held object detector | `backend/app/services/vision/object_detector.py` |
| **Syntax SLM** | Qwen2.5-0.5B-Instruct | Hugging Face Transformers | ISL gloss-to-English natural syntax reconstruction | `models/weights/qwen_0.5b/` |
| **Frontend Framework** | React | 19.x | Component lifecycle, responsive UI, live HUD | `frontend/package.json` |
| **Frontend Language** | TypeScript | 5.7.x | Strict type safety for data packets & UI state | `frontend/src/types.ts` |
| **Frontend Build Tool** | Vite | 6.x | Fast dev server, ES module hot-reloading | `frontend/vite.config.ts` |
| **Speech Engine** | Web Speech API | Native Browser W3C | Zero-dependency text-to-speech audio synthesis | `frontend/src/hooks/useTTS.ts` |

---

## 2. Directory & Component Responsibilities

```
c:\MINI PROJ-SETU\
├── backend\                       # Server-side application logic
│   ├── app\
│   │   ├── main.py                # FastAPI entry point, lifespan, routers, CORS
│   │   ├── api\endpoints\         # REST API route handlers
│   │   │   ├── predict.py         # Static image & landmark prediction endpoints
│   │   │   ├── models.py          # Active model catalog and metadata
│   │   │   ├── video.py           # Video file upload & sequence inference
│   │   │   ├── dataset.py         # Dataset manifest and sample inspection
│   │   │   └── vocabulary.py      # Supported ISL phrase dictionary
│   │   ├── services\              # Business logic & inference services
│   │   │   ├── classifier.py      # Base classification abstraction
│   │   │   ├── pipeline_manager.py# Master singleton orchestrating vision + models
│   │   │   ├── realsign_classifier_service.py # RealSign model manager
│   │   │   ├── nlp\               # Natural Language Processing & Fusion
│   │   │   │   ├── accumulator.py # Temporal token accumulation & CTC blank logic
│   │   │   │   ├── gloss_parser.py# Gloss normalization & token cleaning
│   │   │   │   ├── multimodal_fuser.py # Fuses held objects + gestures into English
│   │   │   │   ├── neural_translator.py# Qwen2.5-0.5B syntax reconstructor
│   │   │   │   └── predictive.py  # N-gram & vocabulary completion engine
│   │   │   └── vision\            # Computer Vision & Landmark Extraction
│   │   │       ├── camera.py      # Frame preprocessors & color space converters
│   │   │       ├── enhancer.py    # Adaptive CLAHE low-light frame enhancer
│   │   │       ├── landmarks.py   # MediaPipe holistic & hand landmarker wrapper
│   │   │       └── object_detector.py # YOLO-World v2 Show-and-Sign detector
│   │   └── websocket\
│   │       └── recognize.py       # 30 FPS full-duplex binary frame streaming loop
├── frontend\                      # Client-side web application
│   ├── src\
│   │   ├── App.tsx                # Master container, state manager, layout
│   │   ├── main.tsx               # React root renderer
│   │   ├── types.ts               # Shared TypeScript interfaces (RecognitionPayload)
│   │   ├── components\            # Modular React UI components
│   │   │   ├── CameraWorkspace.tsx# Video viewport, canvas HUD, center prediction
│   │   │   ├── CameraControls.tsx # Stream control buttons, thresholds, models
│   │   │   ├── TranslationPanel.tsx# Live sentence buffer, speech trigger, history
│   │   │   ├── TopNavbar.tsx      # Header, connection pill, navigation tabs
│   │   │   ├── VisualFramingGuide.tsx# User positioning guide (neutral prompts)
│   │   │   └── AlphabetModal.tsx  # Fingerspelling reference sheet
│   │   ├── hooks\                 # Custom React state hooks
│   │   │   ├── useRecognitionStream.ts # WebSocket stream lifecycle & binary dispatch
│   │   │   ├── useSentenceBuilder.ts   # Debounces tokens into sentence buffers
│   │   │   └── useTTS.ts          # Browser audio voice synthesiser
│   │   └── utils\
│   │       ├── landmarksDrawer.ts # High-speed batched canvas renderer (60 FPS)
│   │       └── framingEvaluator.ts# Geometry checks for posture & distance
├── models\                        # Neural architectures, adapters & registries
│   ├── adapters\                  # Unified adapter interface for all models
│   │   ├── realsign.py            # RealSignMLP (126-D input, 26 alphabet classes)
│   │   ├── sequence_lstm.py       # ISLSequenceLSTM (328-D input, 7 phrase classes)
│   │   ├── include.py             # AI4Bharat INCLUDE BiLSTM (263 classes)
│   │   └── mvp.py                 # 5-sign Ridge classifier baseline
│   ├── weights\                   # Trained binary weights
│   │   ├── realsign_26class_mlp.pth # PyTorch state dict for alphabets
│   │   ├── isl_sequence_lstm.pth    # PyTorch state dict for sequence gestures
│   │   ├── yolov8s-worldv2.pt       # YOLO-World open-vocab weights
│   │   └── qwen_0.5b\             # Local Qwen2.5 weights and tokenizer
├── tests\                         # 164 automated pytest tests
├── Start_SETU.bat                 # One-click Windows startup script
└── README.md                      # Project documentation and setup guide
```

---

## 3. Execution Entry Points

### 1. Unified Launcher
* **File:** [`Start_SETU.bat`](file:///c:/MINI%20PROJ-SETU/Start_SETU.bat)
* **Mechanism:** Double-click batch script that initializes `python.exe -m uvicorn backend.app.main:app --port 8000 --reload` in one terminal window, `npm run dev -- --host 127.0.0.1` in a second window, waits 3 seconds, and automatically opens `http://localhost:5173` in the default browser.

### 2. Backend Entry Point
* **File:** [`backend/app/main.py`](file:///c:/MINI%20PROJ-SETU/backend/app/main.py)
* **Function:** `lifespan(app: FastAPI)`
* **Startup Sequence:**
  1. `classifier_service.load_model()` loads the PyTorch `RealSignMLP` weights onto CUDA/CPU.
  2. `pipeline.load()` loads MediaPipe Holistic landmarker, warms up YOLO-World, initializes the `NeuralGlossTranslator` (Qwen SLM), and prepares the `ISLSequenceLSTMClassifier`.
  3. Uvicorn starts listening on `http://0.0.0.0:8000`.

### 3. Frontend Entry Point
* **File:** [`frontend/src/main.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/main.tsx)
* **Sequence:** Mounts [`App.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/App.tsx) into the DOM root (`#root`). Initializes `useRecognitionStream`, mounts `CameraWorkspace`, and awaits user camera activation.

---

## 4. End-to-End Major Data Flows

```mermaid
flowchart TD
    User([User in front of Webcam]) -->|Video Frames 640x480| Camera[Browser navigator.mediaDevices]
    Camera -->|toBlob JPEG| Hook[useRecognitionStream.ts]
    Hook -->|Binary ArrayBuffer ws.send| WS[ws://127.0.0.1:8000/api/v1/ws/recognize]
    
    subgraph Backend [FastAPI Asynchronous Engine]
        WS -->|cv2.imdecode| CV[OpenCV Frame Decoding]
        CV -->|RGB Frame| MP[MediaPipe Holistic Landmarker]
        MP -->|21 Left + 21 Right Hand (x,y,z)| Norm[Coordinate Normalization & Invariant Features]
        
        Norm -->|126-D Vector| MLP[PyTorch RealSignMLP Alphabet]
        Norm -->|328-D Temporal Window| LSTM[PyTorch ISL Sequence Attention BiLSTM]
        CV -.->|Async Background Task| YOLO[YOLO-World v2 Held Object Grounding]
        
        MLP --> Accum[Temporal Accumulator & CTC Blank Filter]
        LSTM --> Accum
        YOLO --> Fuser[Multimodal Fuser]
        Accum --> Fuser
        Fuser --> Qwen[Qwen2.5-0.5B Neural Translator]
    end
    
    Qwen -->|JSON RecognitionPayload| Hook
    Hook -->|State Update| ReactUI[CameraWorkspace HUD & Subtitle Center Badge]
    Hook -->|Committed Tokens| TTS[Web Speech API Audio Playback]
```

---

## 5. Verified AI/ML Models & Input/Output Shapes

### Model 1: RealSign Alphabet Classifier (`RealSignMLP`)
* **Architecture:** Multi-Layer Perceptron (Dense Neural Network)
* **Weights File:** `models/weights/realsign_26class_mlp.pth`
* **Input Features:** `126` float32 numbers (Dual hand coordinates: 21 landmarks $\times$ 3 coordinates $(x, y, z) \times 2 = 126$, origin-anchored to the wrist and scaled by palm span).
* **Network Layers:**
  $$\text{Linear}(126 \to 128) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Linear}(128 \to 64) \to \text{BatchNorm1d} \to \text{ReLU} \to \text{Linear}(64 \to 26)$$
* **Output:** 26 logits scaled by temperature $T = 0.75$, followed by Softmax probabilities across English alphabet classes (`A` through `Z`).

### Model 2: ISL Dynamic Sequence Gesture Classifier (`ISLSequenceLSTM`)
* **Architecture:** 2-Layer Bidirectional LSTM with Multi-Head Self-Attention Pooling
* **Weights File:** `models/weights/isl_sequence_lstm.pth`
* **Input Features:** `328` float32 features per timestep across a temporal sliding window of 30 frames (Shape: $[B, 30, 328]$). Features include dual hand joints, shoulder/elbow body anchors, and facial eyebrow/mouth openness ratios.
* **Network Layers:**
  $$\text{LayerNorm}(328) \to \text{BiLSTM}(328, 128, \text{num\_layers}=2) \to \text{SelfAttention}(256 \to 64 \to 1) \to \text{Dense}(256 \to 256 \to 64 \to 7)$$
* **Output:** 7 dynamic ISL phrase classes: `['hello', 'thank_you', 'yes', 'no', 'help', 'namaste', 'please']`.

### Model 3: Multimodal Object Grounding (`YOLO-World v2`)
* **Architecture:** Open-Vocabulary Vision-Language Object Detector (`yolov8s-worldv2.pt`, ~13M parameters)
* **Classes:** 13 physical handheld object categories (`cell phone`, `water bottle`, `coffee cup`, `mug`, `book`, `pen`, `scissors`, `keys`, `wallet`, `laptop`, `apple`, `banana`, `orange`).
* **Input:** RGB image frame ($384 \times 288$ or $640 \times 480$).
* **Output:** Bounding boxes $[y_{min}, x_{min}, y_{max}, x_{max}]$, confidence scores, and `is_held` flag computed by fingertip knuckle proximity.

### Model 4: Syntax Reconstruction SLM (`Qwen2.5-0.5B-Instruct`)
* **Architecture:** Transformer-based Small Language Model (0.5 Billion parameters)
* **Weights Path:** `models/weights/qwen_0.5b/`
* **Input:** String of raw sign glosses (e.g., `"HELLO YOU NAME WHAT"`).
* **Output:** Coherent Indian English grammatical sentence (e.g., `"Hello, what is your name?"`).

---

## 6. Verification Methods & Test Suite

* **Automated Unit & Integration Tests:** 164 tests located in `tests/`.
  * Verified Command: `.\.venv\Scripts\pytest.exe tests/ -q`
  * Result: **164 passed in 81.10s (0 failures, 100% pass rate)**.
* **Coverage Scope:**
  * Endpoint health and status (`test_api.py`)
  * Feature extraction geometry and invariance (`test_feature_schema.py`)
  * RealSign classification accuracy and temperature scaling (`test_realsign_classifier.py`)
  * ISL Sequence LSTM state tracking (`test_isl_sequence_lstm_classifier.py`)
  * AI4Bharat INCLUDE inference adapter (`test_include_model.py`)
  * Low-light frame adaptation (`test_camera_and_holistic_vision.py`)
  * Qwen neural translator generation (`test_neural_translator.py`)
* **Frontend Verification:**
  * Build check: `npm --prefix frontend run build` (Compiles with 0 errors via `tsc -b && vite build`).

---

## 7. Observed Facts vs. Areas for Ongoing Exploration

1. **Observed Fact:** The core real-time recognition pipeline does not use external cloud APIs; it runs 100% locally on your machine with CUDA GPU acceleration.
2. **Observed Fact:** The RealSign model classifies 26 static fingerspelling letters from spatial hand landmarks.
3. **Observed Fact:** The ISL Sequence LSTM classifies temporal dynamic signs over time from a sliding sequence buffer.
4. **Distinction:** YOLO-World detects *objects*, while MediaPipe detects *hands and body posture*. They are distinct computer vision subsystems fused by the `MultimodalFuser`.
5. **Exploration Note:** In pure Alphabet fingerspelling mode, object detection is bypassed to maximize frames per second and prevent wearable accessories (like fitness bands) from producing false object triggers.
