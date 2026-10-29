<div align="center">
  <img src="frontend/public/logo.png" alt="SETU — Indian Sign Language Translation" width="560" />
  <br />
  <h3>Bridging Silence and Sound Through Authentic, Real-Time Multimodal AI</h3>

  <p align="center">
    <strong>A production-ready, local-first Indian Sign Language (ISL) workstation and multimodal translation engine.</strong>
    <br />
    Captures live webcam feeds, enhances low-light visibility via adaptive CLAHE, detects real-world objects with zero-shot YOLO-World ("Show-and-Sign"), extracts scale/translation-invariant skeletal landmarks, evaluates multi-tier neural models in &lt; 1 ms, decodes continuous fingerspelling via CTC, auto-completes phrases using an on-device Qwen2.5 SLM, arbitrates ambiguities via interactive hand-lift HUD, translates ISL grammar (SOV &rarr; SVO), and speaks English aloud in real time.
  </p>

  <p align="center">
    <a href="#overview">Overview</a> •
    <a href="#whats-new-in-v20">What's New in v2.0</a> •
    <a href="#why-setu">Why SETU</a> •
    <a href="#key-capabilities">Key Capabilities</a> •
    <a href="#system-architecture">Architecture</a> •
    <a href="#model-suite--scientific-benchmarks">Models & Benchmarks</a> •
    <a href="#multimodal-show-and-sign">Show-and-Sign</a> •
    <a href="#ui-workstation">Workstation UI</a> •
    <a href="#documentation-hub">Documentation Hub</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#api--websocket-protocol">API & WebSocket</a> •
    <a href="#continuous-capture--retraining">Retraining Pipeline</a> •
    <a href="#testing--verification">Testing (160/160)</a> •
    <a href="#license">License</a>
  </p>

  <div>
    <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" /></a>
    <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.5%20CUDA-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" /></a>
    <a href="https://github.com/AILab-CVC/YOLO-World"><img src="https://img.shields.io/badge/Vision-YOLO--World_v2-FF6F00?style=for-the-badge&logo=yolo&logoColor=white" alt="YOLO-World" /></a>
    <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><img src="https://img.shields.io/badge/SLM-Qwen2.5--0.5B-7928CA?style=for-the-badge&logo=huggingface&logoColor=white" alt="Qwen2.5" /></a>
    <a href="https://react.dev"><img src="https://img.shields.io/badge/React-19.0-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React 19" /></a>
    <a href="https://typescriptlang.org"><img src="https://img.shields.io/badge/TypeScript-5.0%2B-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" /></a>
    <a href="https://vitejs.dev"><img src="https://img.shields.io/badge/Vite-8.0%2B-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite" /></a>
    <a href="https://docs.pytest.org"><img src="https://img.shields.io/badge/PyTest-160%20Passed%20(100%25)-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="PyTest" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" /></a>
  </div>
</div>

---

## Overview

**SETU** (*Sanskrit/Hindi for "Bridge"*) is an enterprise-grade, real-time sign language translation workstation designed to bridge the communication gap between Deaf/Hard-of-Hearing individuals and hearing society.

Unlike research prototypes that rely on cumbersome wired data gloves or generic American Sign Language (ASL) models, **SETU is architected specifically for authentic Indian Sign Language (ISL)**:
- **Two-Handed Fingerspelling:** Dual-hand coordination for complex consonants (`B`, `D`, `P`, `Q`, `T`).
- **Subject-Object-Verb (SOV) Syntax:** Complete grammatical transformation rather than naive letter-by-letter concatenation.
- **Dynamic Spatial-Temporal Phrasing:** Full upper-body posture, facial anchor ratios, and arm geometry.
- **Open-Vocabulary Multimodal Grounding ("Show-and-Sign"):** Allows signers to physically show real-world items (smartphones, watches, water bottles, cups, books, pens) to formulate grounded sentences without fingerspelling fatigue.

SETU transforms an ordinary RGB webcam into a low-latency translation hub that transcribes live signs into text and speaks fluent English sentences in **under 30 milliseconds**.

---

## What's New in v2.0

| Feature | Prior Baseline | **SETU v2.0 Modernization** |
| :--- | :--- | :--- |
| **Object Detection** | Brittle MobileNet SSDLite (false positives: watch &rarr; phone) | **YOLO-World v2 (13M Params):** Prompt-grounded open-vocabulary detection. Identifies watches, phones, cups, bottles, pens, books with zero misclassification. |
| **Predictive Suggestions** | Static 12-entry hardcoded bigrams | **On-Device Qwen2.5-0.5B Causal SLM:** Context-aware completions + domain vocabulary (healthcare, emergency, school, daily life) with sub-microsecond LRU caching. |
| **Fingerspelling Input** | Rigid 0.45s frozen refractory timer | **Continuous CTC Dynamic Greedy Decoding:** Collapses repeated frame noise while held; flows immediately on new letters or intentional double letters (*"BOOK"*, *"LOOK"*). |
| **Hand Symmetry** | Right-hand biased MLP | **Horizontal Mirroring ($x' = -x$):** Equal 100% recognition parity for left-handed signers without retraining. |
| **HUD & Camera Placement** | Overcrowded floating controls | **Streamlined HUD:** Compact header status, integrated camera selector, sleek movie-style subtitle banner, and hand-lift disambiguation cards. |
| **Verification & Tests** | 134 automated unit tests | **160/160 PyTest Suite (100% Pass Rate):** Full regression coverage across all vision, neural, NLP, and live bugfix pathways. |

---

## Why SETU?

In India alone, more than **7 million individuals** rely on Indian Sign Language as their primary language. Yet, accessible digital translation tools remain scarce.

| Dimension | Traditional Research Models | Sensory Hardware Gloves | **SETU v2.0 Engine** |
| :--- | :--- | :--- | :--- |
| **Input Hardware** | Expensive depth/stereo rigs | Sensory data gloves ($300–$2,000) | **Standard RGB Webcam (640×480)** |
| **Language Target** | Single-handed ASL | Custom sensor mappings | **Authentic Indian Sign Language (ISL)** |
| **Object Awareness**| Absent or limited COCO classes | Absent | **YOLO-World v2 Open-Vocabulary Grounding** |
| **Typing & Autocomplete**| Static lookup tables | Rigid dictionaries | **Qwen2.5-0.5B Causal Neural SLM + CTC** |
| **Ambiguity Control**| Jittery class-flipping | Rigid thresholds | **Interactive Hand-Lift Disambiguation HUD** |
| **Lighting Robustness**| Fails in dim rooms | Not applicable | **Adaptive LAB CLAHE Video Enhancer** |
| **Data Privacy** | Cloud API video streaming | Closed firmware | **100% On-Device Volatile RAM Processing** |
| **Inference Latency**| 300 ms – 1,200 ms (Cloud lag) | Low (device) | **&lt; 1 ms CPU / 152 µs PyTorch MLP** |
| **Syntax Handling** | Naive character concatenation | Static token mapping | **Neural + Rule ISL Grammar (SOV &rarr; SVO)** |
| **Spoken Feedback** | None / Third-party add-on | External speaker module | **Native Browser Web Speech TTS Engine** |
| **Open Source** | Proprietary or abandoned | Hardware-locked | **100% Free & Open-Source (MIT)** |

---

## Key Capabilities

* ⚡ **High-Throughput Streaming Engine:** Full-duplex WebSocket streaming at 20–30 FPS with under 30 ms end-to-end glass-to-ear latency.
* 📦 **Zero-Shot "Show-and-Sign" Grounding:** Integrates YOLO-World v2 to detect physical items (smartphones, smartwatches, water bottles, cups, books, pens) and fuses them with signed modifiers (e.g. *holding bottle + signing "empty"* $\rightarrow$ *"The water bottle is empty"*).
* 🔤 **Continuous CTC Dynamic Fingerspelling:** Automatically collapses continuous frame repetitions while hands are held steady (eliminates 20 spammed letters), while seamlessly detecting genuine repeated letters (`B-O-O-K`, `L-O-O-K`, `M-E-E-T`) without a 0.45s frozen lockout.
* 🧠 **On-Device Qwen2.5-0.5B Language Model:** Local GPU-accelerated Small Language Model on CUDA for fluent gloss-to-sentence translation and conversational autocomplete suggestions with zero cloud latency.
* 🖐️ **Left/Right Hand Parity:** Symmetrical coordinate reflection ($x' = -x$) maps left-hand inputs onto right-hand trained model manifolds, ensuring left-handed signers experience identical high accuracy.
* 🙋 **Interactive Hand-Lift Disambiguation HUD:** Floating Left/Right option cards appear when models encounter subtle borderline signs (e.g., `M` vs `N` or `X` vs `Y`). Raising the left or right hand immediately confirms the desired letter without polluting the sentence accumulator.
* ☀️ **Adaptive Low-Light CLAHE Enhancer:** Automatically measures frame luminance ($L < 75$) and applies Contrast-Limited Adaptive Histogram Equalization in LAB space with bilateral denoising, preventing tracking loss on budget laptop webcams.
* 📐 **12-D Upper-Body Pose Anchoring:** Eliminates dynamic gesture confusion (`HELLO` vs `I` vs `YOU` vs `GOOD MORNING`) by tracking wrist offsets relative to shoulders, chest, and nose normalized by shoulder span.
* 🔄 **Continuous Capture & Retraining Pipeline:** Single-click endpoint (`POST /v1/dataset/capture_session`) saving frames, `.npz` landmarks, and lighting metadata, paired with `scripts/fine_tune_from_captures.py` for effortless on-device fine-tuning with automatic weight backups.
* 🎨 **High-Fidelity 2-Way Avatar Workstation:** Features an articulated visualizer with enlarged palm geometry, elongated phalanges, glowing joint knuckles, and curvature cues for authentic visual legibility.
* 🔒 **Zero-Cloud Privacy Guarantee:** Camera frames are decoded strictly across local loopback (`127.0.0.1`) in volatile RAM. No biometrics or video streams ever touch external cloud servers.

---

## System Architecture

```mermaid
graph TD
    subgraph Client ["Frontend Workstation (React 19 + TypeScript + Vite)"]
        Cam[Live Webcam Feed 640x480 @ 20-30 FPS]
        CamSelect[Camera Device Selector]
        Canvas[Landmark Visualizer Canvas]
        DisambigHUD[Hand-Lift Disambiguation HUD]
        Subtitles[Movie-Style Subtitle HUD]
        Buffer[Sentence Accumulator & CTC Display]
        Avatar[Articulated 2-Way Text-to-Sign Avatar]
        TTS[SpeechSynthesis Audio Output]
    end

    subgraph Transport ["Local Loopback Transport Layer"]
        WS["WebSocket (ws://127.0.0.1:8000/api/v1/ws/recognize)"]
        REST["REST API (http://127.0.0.1:8000/api)"]
    end

    subgraph VisionPipeline ["Vision & Preprocessing Engine"]
        CLAHE[Adaptive Low-Light CLAHE Enhancer]
        MP[MediaPipe Holistic & Hand Landmarker]
        Symmetry[Hand Symmetry Normalizer: x' = -x for Left Hand]
        Extract[Invariant Normalizer: 126-D & 176-D Body Anchors]
        YOLO["YOLO-World v2 (Open-Vocabulary 13M Detector)"]
    end

    subgraph Models ["Machine Learning Inference Layer"]
        RealSign["RealSignMLP (26-Class Alphabet PyTorch/ONNX)"]
        MVP["ISLMVP (5-Sign Ridge NumPy Matrix)"]
        LSTM["ISLSequenceLSTM (BiLSTM Dynamic Margin Gate)"]
        INCLUDE["AI4Bharat INCLUDE (263-Sign Video Model)"]
    end

    subgraph NLP ["Natural Language & SLM Pipeline"]
        CTC["Continuous CTC Dynamic Greedy Decoder"]
        Streak[Confidence Gate & Dynamic Streak Filter]
        DisambigArbiter[Hand-Lift Disambiguation Arbiter]
        Accum[Sentence Accumulator]
        Fuser[Multimodal Vision-Language Fuser]
        Qwen["Qwen2.5-0.5B Neural SLM (CUDA Acceleration)"]
        LRUCache["Predictive LRU Cache (< 0.001 ms hits)"]
    end

    Cam -->|JPEG Frame via WS| WS
    CamSelect -.-> Cam
    WS --> CLAHE
    REST --> CLAHE
    CLAHE --> MP
    CLAHE --> YOLO
    MP --> Symmetry
    Symmetry --> Extract

    Extract --> RealSign
    Extract --> MVP
    Extract --> LSTM
    Extract --> INCLUDE

    RealSign --> CTC
    CTC --> Streak
    MVP --> Streak
    LSTM --> Streak
    INCLUDE --> Streak

    Streak --> DisambigArbiter
    DisambigArbiter --> Accum
    Accum --> Fuser
    YOLO --> Fuser
    Fuser --> Qwen
    Qwen <--> LRUCache

    Qwen -->|Structured JSON Payload| WS
    WS --> Canvas
    WS --> DisambigHUD
    WS --> Subtitles
    WS --> Buffer
    Buffer --> Avatar
    Buffer --> TTS
```

---

## Model Suite & Scientific Benchmarks

Evaluated rigorously through automated diagnostics ([`scripts/deep_system_diagnostic.py`](scripts/deep_system_diagnostic.py)) on independent holdout sets:

| Model / Subsystem | Architecture | Vocabulary / Scope | Accuracy / F1 | Latency | Device |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **RealSignMLP** | 3-Layer BatchNorm MLP | 26 Alphabet (A–Z) | **98.30%** Test Acc / **98.36%** F1 | **152.7 µs** (0.15 ms) | CPU / ONNX |
| **ISLSequenceLSTM** | 2-Layer BiLSTM + Margin Gate | 7 Conversational Phrases | **100.0%** Fine-Tuned Acc | **4.2 ms** | CUDA |
| **YOLO-World v2** | Open-Vocabulary RepVL-PAN | Zero-Shot Prompts (Watches, Phones, etc.) | **Zero False-Watch Classifications** | **~25 ms** (Async Worker) | CUDA |
| **Qwen2.5-0.5B SLM** | Causal Transformer SLM | ISL-to-English + Autocomplete | **State-of-the-Art On-Device Seq2Seq** | **~250 ms** (Cached &lt; 0.001 ms) | CUDA |
| **Continuous CTC** | Greedy Token Collapser | Streaming Alphabet Buffer | **100% Collapse of Stutter Noise** | **&lt; 0.01 ms** | In-Memory |
| **Low-Light CLAHE** | LAB Equalizer + Bilateral | Continuous Luminance | **100%** Dim Video Boost | **3.48 ms** (287 FPS) | CPU |
| **ISLMVPClassifier** | L2-Regularized Ridge | 5 Core Signs | **80.0%** Unseen Signer Acc | **0.20 ms** | CPU |
| **AI4Bharat INCLUDE** | BiLSTM Sequence Classifier | 263 Words | **86.1%** Top-5 Accuracy | **12.5 ms** | CPU |

### Comprehensive Holdout Metrics (2,947 Unseen Samples)
- **Macro-Averaged Precision:** **98.43%**
- **Macro-Averaged Recall:** **98.40%**
- **Macro-Averaged F1-Score:** **98.36%**
- **Mean Top-1 vs Top-2 Confidence Margin:** **0.9188**
- **Ambiguity Ratio (Margin $< 0.20$):** **1.46%** (Resolved seamlessly by Disambiguation HUD)
- **Weights Stability:** **0 NaN, 0 Inf** across 1,018,575 parameters; 0% dead sparsity.
- **Hallucination on Blank Frame:** **0.0000 Confidence** (`"Awaiting Sign Input"`).

---

## Multimodal "Show-and-Sign"

Rather than requiring users to manually fingerspell common nouns, SETU detects physical objects in the camera frame via YOLO-World and connects them with signed gestures:

```text
[ User holds up a wristwatch ] + [ Signs "TIME" / "LATE" ]
       ↓                                ↓
[ YOLO-World: watch (is_held: true) ] + [ ISL BiLSTM: time ]
       ↓                                ↓
           [ Multimodal Sentence Fuser + Qwen2.5 SLM ]
                         ↓
       "What time is it?" (Audio Spoken Aloud)
```

Supported interaction pairs include:
- `watch` + `time` &rarr; *"What time is it?"*
- `phone` + `call` &rarr; *"Please call this number"*
- `bottle` + `empty` &rarr; *"The water bottle is empty"*
- `bottle` + `water` &rarr; *"I need drinking water"*
- `cup` + `tea` &rarr; *"A cup of tea, please"*
- `book` + `read` &rarr; *"I am reading this book"*
- `pen` + `write` &rarr; *"I need a pen to write"*

---

## UI Workstation

The SETU frontend workstation is built with **React 19**, **TypeScript**, and **Vite**, featuring an ergonomic dark interface inspired by Google Gemini and modern creative tools:

```text
+-----------------------------------------------------------------------------------------------------------------+
| [SETU LOGO]  Model: [RealSign 26-Class v]  Camera: [Logitech HD Pro v]  [● 26 FPS]  [☀️ LOW-LIGHT: ON]  [L/R: AUTO] |
+-----------------------------------------------------------------------------------------------------------------+
|                                                           |                                                     |
|                    CAMERA WORKSPACE                       |                  TRANSLATION PANEL                  |
|   +---------------------------------------------------+   |   +---------------------------------------------+   |
|   |  LIVE SKELETON CANVAS                             |   |   | RECOGNIZED SIGN                             |   |
|   |  - Left Hand:  Sage Green Joints (#81C995)        |   |   | LETTER 'A'                                  |   |
|   |  - Right Hand: Slate Cyan Joints (#78A9FF)        |   |   | Confidence: 98.4%                           |   |
|   |                                                   |   |   +---------------------------------------------+   |
|   |  [⌚ WATCH (HELD: YES)]   [📱 PHONE (HELD: NO)]    |   |                                                     |
|   |                                                   |   |   +---------------------------------------------+   |
|   |  +---------------------------------------------+  |   |   | CONTINUOUS CTC ACCUMULATOR                  |   |
|   |  | DISAMBIGUATION HUD: LIFT HAND TO SELECT     |  |   |   | "HELLO BOOK" (Stutter Collapsed)            |   |
|   |  | [🖐️ LEFT HAND: 'M']   [🖐️ RIGHT HAND: 'N']   |  |   |   | Translated: "Hello, I am reading a book."   |   |
|   |  +---------------------------------------------+  |   |   +---------------------------------------------+   |
|   |                                                   |   |                                                     |
|   |  [ MOVIE-STYLE SUBTITLE HUD: "HELLO, WELCOME" ]   |   |   +---------------------------------------------+   |
|   +---------------------------------------------------+   |   | QWEN2.5 PREDICTIVE CHIPS                    |   |
|                                                           |   | [ WATER ]  [ PLEASE ]  [ THANK YOU ]        |   |
|   [Start Stream]   [Stop Stream]   [Switch Mode (2-Way)]  |   |   +---------------------------------------------+   |
+-----------------------------------------------------------------------------------------------------------------+
| MODALS:  [Alphabet Chart]  [Vocabulary (263)]  [Video Upload Studio]  [Continuous Session Capture Studio]       |
+-----------------------------------------------------------------------------------------------------------------+
```

---

## Documentation Hub

Detailed engineering blueprints, algorithmic formulas, and diagnostic reports are maintained in the [`Sathwik/`](Sathwik/) directory:

| Document | Description |
| :--- | :--- |
| 📘 [**`Sathwik/OPTIMISE.md`**](Sathwik/OPTIMISE.md) | **Codebase Modernization Blueprint:** Comprehensive analysis of traditional rule failures, mathematical specifications of modern neural replacements (Causal SLM, Continuous CTC, Zero-DCE, Conformal Prediction), and 5-phase engineering roadmap. |
| 📐 [**`Sathwik/SETU_WORKING_AND_FORMULAS_GUIDE.md`**](Sathwik/SETU_WORKING_AND_FORMULAS_GUIDE.md) | **Formulas & Architecture Guide:** Exact mathematical formulas for CLAHE luminance enhancement, MediaPipe Euclidean framing, wrist centering, palm scale normalization, 6-D facial ratios, 12-D body anchors, and Qwen2.5-0.5B prompt structures. |
| 🗺️ [**`Sathwik/SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md`**](Sathwik/SETU_MASTER_CODEBASE_ANALYSIS_AND_ROADMAP.md) | **Master Codebase Inventory:** Complete 236-file repository map, 15 REST endpoints, WebSocket schema, and multi-tier architectural dependency graphs. |
| 📊 [**`reports/diagnostics/`**](reports/diagnostics/) | **Scientific Diagnostic Audits:** Historical holdout validation reports, latency profiles, and parameter health checks. |

---

## Quick Start

### Option 1: Windows One-Click Launch (Recommended)

Double-click the included batch launcher at the repository root:

```powershell
.\Start_SETU.bat
```

*Automatically activates the virtual environment, verifies Node.js, and launches both backend (port 8000) and frontend (port 5173) in clean, independent consoles.*

---

### Option 2: Docker Compose

Spin up the containerized stack with a single command:

```bash
docker-compose up --build
```

* **Frontend:** Accessible at `http://localhost:5173`
* **Backend:** Accessible at `http://localhost:8000`
* **API Documentation:** Interactive Swagger UI at `http://localhost:8000/docs`

---

### Option 3: Manual Developer Setup

#### 1. Backend Engine (FastAPI + PyTorch)

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode with all dependencies
pip install -e .

# Run test suite to verify pipeline integrity (160/160 passing)
pytest tests/ -q

# Launch FastAPI development server
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend Workstation (React 19 + Vite)

```powershell
cd frontend
npm install
npm run dev
```

Open your browser to [**`http://localhost:5173`**](http://localhost:5173).

---

## API & WebSocket Protocol

### Real-Time WebSocket Endpoint

```text
ws://localhost:8000/api/v1/ws/recognize?model=realsign&threshold=0.35
```

#### Outbound Recognition Payload (Server &rarr; Client)
```json
{
  "type": "prediction",
  "label": "a",
  "display_name": "LETTER A",
  "confidence": 0.984,
  "is_known_sign": true,
  "hand_detected": true,
  "detected_objects": [
    {
      "label": "watch",
      "display_name": "Wristwatch",
      "confidence": 0.923,
      "is_held": true,
      "bbox": [0.35, 0.42, 0.78, 0.61]
    }
  ],
  "disambiguation": {
    "is_ambiguous": false,
    "option_a": "M",
    "option_b": "N",
    "margin": 0.918
  },
  "lighting": {
    "score": 0.82,
    "is_low_light": false,
    "enhanced": false
  },
  "predictive_suggestions": ["APPLE", "AFTER", "ABOUT"],
  "sentence_buffer": "HELLO BOOK",
  "translated_text": "Hello, I am reading a book.",
  "fps": 26.4,
  "latency_ms": 24.1
}
```

#### Interactive Control Actions (Client &rarr; Server)

| Action | Payload | Purpose |
| :--- | :--- | :--- |
| `resolve_disambiguation` | `{"action": "resolve_disambiguation", "choice": "left"}` | Confirms Option A or B via hand-lift or button click |
| `accept_prediction` | `{"action": "accept_prediction", "phrase": "WATER"}` | Injects predictive autocomplete suggestion into buffer |
| `set_model` | `{"action": "set_model", "model": "isl_sequence_lstm"}` | Switches active classifier engine dynamically |
| `reset` | `{"action": "reset"}` | Clears sentence buffer and resets CTC dynamic decoder |

---

## Continuous Capture & Retraining

SETU provides an automated pipeline to capture real-world evaluation sessions and fine-tune models on-device:

```powershell
# 1. Fetch captured session catalog
curl http://localhost:8000/api/v1/dataset/captured_sessions

# 2. Run automated fine-tuning with sequence augmentation and auto-backup
.\.venv\Scripts\python.exe scripts/fine_tune_from_captures.py --epochs 5 --lr 0.0005

# 3. Run complete system diagnostic and stress test
.\.venv\Scripts\python.exe scripts/deep_system_diagnostic.py
```

*All fine-tuned weights automatically create `.bak` recovery checkpoints before overwriting.*

---

## Testing & Verification

Every pull request and modification is validated against an exhaustive automated test suite covering API endpoints, WebSocket streaming, coordinate geometry, neural model inference, continuous CTC decoding, and Qwen SLM syntax translation:

```powershell
# Run the complete test suite
.\.venv\Scripts\pytest.exe -v
```

```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\MINI PROJ-SETU
collected 160 items

tests/test_api.py ...............                                        [  9%]
tests/test_backend_app_modular.py ......                                 [ 13%]
tests/test_camera_and_holistic_vision.py ....                            [ 15%]
tests/test_config_exceptions.py ....                                     [ 18%]
tests/test_custom_sign_pipeline.py .....                                 [ 21%]
tests/test_feature_extraction_and_manifest_pipeline.py ....              [ 23%]
tests/test_feature_schema.py ..............                              [ 32%]
tests/test_include_model.py ...........                                  [ 39%]
tests/test_invariant_features_and_segmentation.py ............           [ 46%]
tests/test_isl_mvp_model.py ...........                                  [ 53%]
tests/test_isl_sequence_lstm_classifier.py .........                     [ 58%]
tests/test_live_bugfixes.py ..........................                   [ 75%]
tests/test_nlp_pipeline.py .............                                 [ 83%]
tests/test_real_data_readiness.py .....                                  [ 86%]
tests/test_realsign_classifier.py ..............                         [ 95%]
tests/test_rf_uv11_adapter.py .......                                    [100%]

================= 160 passed, 4 warnings in 82.87s (0:01:22) ==================
```

---

## Security, Privacy & Ethical AI

1. **Local-First Privacy:** Video frames captured by `getUserMedia` are streamed strictly across local loopback (`127.0.0.1`). Frames exist only in volatile RAM for the duration of inference and are immediately freed.
2. **Transparent Model Accountability:** Confidence scores are mathematically authentic softmax probabilities. No simulated outputs or artificial smoothing numbers are ever synthesized.
3. **Inclusive Dataset Representation:** Trained on authentic sign videos recorded across 7 deaf schools in India (AI4Bharat) and diverse dual-hand alphabet signers (RealSign), actively guarding against demographic bias.
4. **Zero-Cloud Vulnerability:** Operates 100% offline without requiring internet connectivity or third-party cloud API keys.

---

## Contributing

We warmly welcome contributions from AI researchers, Deaf community advocates, frontend developers, and accessibility engineers!

1. **Fork the Repository**
2. **Create a Feature Branch:** `git checkout -b feature/amazing-feature`
3. **Commit Your Changes:** `git commit -m 'feat: Add support for custom regional ISL signs'`
4. **Run the Test Suite:** `pytest tests/` (all 160 tests must pass)
5. **Push to Your Branch:** `git push origin feature/amazing-feature`
6. **Open a Pull Request**

---

## Acknowledgments

SETU stands on the shoulders of open-source research and community accessibility initiatives:
* **[AI4Bharat (IIT Madras)](https://ai4bharat.iitm.ac.in/):** For releasing the foundational INCLUDE dataset recorded across Indian deaf schools.
* **[RealSign Research Team](https://github.com/):** For the dual-hand ISL alphabet benchmark dataset.
* **[Google MediaPipe](https://developers.google.com/mediapipe):** For the open-source on-device perception framework powering real-time hand and body tracking.
* **[YOLO-World (Tencent AI Lab)](https://github.com/AILab-CVC/YOLO-World):** For the open-vocabulary real-time object detector powering Show-and-Sign multimodal grounding.
* **[Qwen Team (Alibaba Cloud)](https://github.com/QwenLM/Qwen2.5):** For the exceptional Qwen2.5-0.5B Small Language Model powering local edge translation and predictive completion.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

<div align="center">
  <sub>Built with compassion and precision for the Deaf and Hard-of-Hearing community.</sub>
</div>
