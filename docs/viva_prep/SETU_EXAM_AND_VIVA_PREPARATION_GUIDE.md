# SETU EXAM, VIVA & TROUBLESHOOTING HANDBOOK
## Comprehensive Technical Defense, System Diagnostics & Engineering Glossary

> **Audience:** 3rd Year B.Tech Artificial Intelligence & Machine Learning Student  
> **Course:** Project SETU Viva Voce, Academic Examination & Technical Placement Mastery  
> **Repository:** `c:\MINI PROJ-SETU`

---

# TABLE OF CONTENTS
1. [Viva Voce Defense Questions & Model Answers](#1-viva-voce-defense-questions--model-answers)
   - [Category 1: System Architecture & Web Engineering](#category-1-system-architecture--web-engineering)
   - [Category 2: Computer Vision & Spatial Geometry](#category-2-computer-vision--spatial-geometry)
   - [Category 3: Deep Learning Architectures](#category-3-deep-learning-architectures)
   - [Category 4: Multimodal Grounding & Natural Language](#category-4-multimodal-grounding--natural-language)
   - [Category 5: Production Engineering & Latency Optimization](#category-5-production-engineering--latency-optimization)
2. [Systematic Troubleshooting & Diagnostic Procedures](#2-systematic-troubleshooting--diagnostic-procedures)
   - [1. Backend Connection Refused (WinError 10061)](#1-backend-connection-refused-winerror-10061)
   - [2. PyTorch CUDA Out-of-Memory (OOM) or Fallback](#2-pytorch-cuda-out-of-memory-oom-or-fallback)
   - [3. Browser Webcam Lag & Canvas Stutter](#3-browser-webcam-lag--canvas-stutter)
   - [4. False Object Detections (Smartwatch / Wristbands)](#4-false-object-detections-smartwatch--wristbands)
   - [5. Repeated Character Spamming (e.g. "HHHHH")](#5-repeated-character-spamming-eg-hhhhh)
3. [Plain-English Engineering Glossary](#3-plain-english-engineering-glossary)
4. [Self-Assessment Mastery Checklist](#4-self-assessment-mastery-checklist)

---

# 1. Viva Voce Defense Questions & Model Answers

### Category 1: System Architecture & Web Engineering

#### Q1: Why did you use WebSockets instead of standard HTTP REST for live camera recognition?
* **Examiner Intent:** Testing if you understand network protocols and latency trade-offs.
* **Model Answer:**
  > *"Standard HTTP is a request-response protocol over TCP. Every time an HTTP request is made, it incurs the overhead of TCP handshakes, TLS renegotiation, and HTTP headers. At 30 frames per second, sending 30 separate HTTP POST requests per second causes network queuing and introduces 80–150 ms of avoidable latency.
  > 
  > Instead, we use a single, persistent, full-duplex WebSocket connection (`ws://localhost:8000/api/v1/ws/recognize`). Once the handshake is complete, we stream raw binary JPEG ArrayBuffers directly to the server. The server processes each frame and streams back lightweight JSON prediction packets in under 15 ms, delivering a real-time, zero-lag experience."*

#### Q2: What is the purpose of FastAPI's `lifespan` context manager?
* **Examiner Intent:** Checking if you know how server lifecycle and deep learning weight management work.
* **Model Answer:**
  > *"In deep learning APIs, loading multi-megabyte PyTorch models, allocating CUDA memory, and warming up GPU kernels takes 2 to 4 seconds. If models were loaded inside a route handler, the first user would experience a massive timeout.
  > 
  > FastAPI's `lifespan` manager runs pre-initialization logic before the server starts accepting network traffic. In our implementation (`backend/app/main.py`), `lifespan` loads the RealSign MLP onto CUDA, warms up YOLO-World, and initializes the Qwen tokenizer. By the time the server reports healthy, all models are resident in VRAM and ready for sub-5ms inference."*

---

### Category 2: Computer Vision & Spatial Geometry

#### Q3: Why don't you feed raw camera images directly into a Convolutional Neural Network (CNN)?
* **Examiner Intent:** Testing if you understand feature engineering vs end-to-end deep learning.
* **Model Answer:**
  > *"Training an end-to-end 2D CNN (like ResNet-50) directly on raw webcam pixels requires massive datasets of diverse signers, rooms, skin tones, and lighting conditions. Even then, a change in room background or clothing can cause the CNN to misclassify. Furthermore, running a 25-million parameter CNN at 30 FPS consumes significant GPU power and incurs 30–50 ms of latency.
  > 
  > Instead, we adopted a two-stage decoupled pipeline: MediaPipe first extracts 21 anatomical 3D hand landmarks, and our PyTorch RealSign MLP classifies the geometric coordinates. This provides complete background invariance, eliminates skin-tone bias, reduces model size to just 150 KB, and achieves an inference latency of under 3 milliseconds."*

#### Q4: How does SETU handle users standing at different distances from the camera?
* **Examiner Intent:** Testing your mathematical understanding of scale invariance.
* **Model Answer:**
  > *"Raw pixel coordinates vary drastically depending on whether a user stands close or far from the camera. To make recognition invariant to distance, we apply two mathematical transformations:
  > 1. **Origin Anchoring (Translation Invariance):** We subtract the wrist landmark coordinates $(x_0, y_0, z_0)$ from all 21 hand landmarks, placing the origin $(0, 0, 0)$ at the wrist.
  > 2. **Palm Span Scaling (Scale Invariance):** We compute the Euclidean distance $d$ between the wrist and the middle finger MCP knuckle joint. We divide all coordinates by $d$.
  > As a result, whether your hand appears large or small in the video frame, the normalized coordinate values remain identical."*

---

### Category 3: Deep Learning Architectures

#### Q5: Explain the difference between your RealSign MLP and your ISL Sequence LSTM.
* **Examiner Intent:** Verifying that you understand spatial vs temporal modeling.
* **Model Answer:**
  > *"RealSign MLP is a static spatial classifier. It takes a single snapshot of hand landmarks (a 126-D vector) to classify fingerspelled letters (A to Z), where the static shape of the fingers completely defines the symbol.
  > 
  > In contrast, dynamic ISL signs (like 'HELLO' or 'NAMASTE') depend on movement and trajectory over time. A single frame cannot show the direction of a hand wave. For dynamic signs, our ISL Sequence Attention BiLSTM processes a rolling temporal sequence of 30 frames (a $[30, 328]$ tensor). The bidirectional LSTM captures trajectory in both directions, and a self-attention layer pools the most expressive peak frames of the gesture."*

#### Q6: What is the mathematical role of Self-Attention pooling in your Sequence LSTM?
* **Examiner Intent:** Deep testing of sequence architecture nuances.
* **Model Answer:**
  > *"Instead of naively taking the final frame ($h_{30}$) or averaging all frames uniformly (which dilutes informative motion with lead-in and rest movements), the self-attention network computes dynamic importance weights $\alpha_t$ across all 30 frames:
  > $$e_t = \mathbf{v}^T \tanh(\mathbf{W}_a \mathbf{h}_t), \quad \alpha_t = \frac{\exp(e_t)}{\sum_k \exp(e_k)}, \quad \mathbf{c} = \sum_{t=1}^{30} \alpha_t \mathbf{h}_t$$
  > This statistically locates and weights the expressive peak of the sign while assigning near-zero weight to beginning and ending hand drops."*

---

### Category 4: Multimodal Grounding & Natural Language

#### Q7: What is Show-and-Sign, and how does YOLO-World participate?
* **Examiner Intent:** Assessing your understanding of multimodal AI.
* **Model Answer:**
  > *"In real-world sign communication, signers often hold up a physical object (like a water bottle) and sign an action (like 'EMPTY' or 'WANT').
  > 
  > In SETU, YOLO-World v2 runs in an asynchronous background thread to detect handheld items. We compute proximity between the object bounding box and the user's fingertips to ensure it is actively held (while strictly excluding wristbands). The `MultimodalFuser` then merges the held object and the sign gesture to produce a complete sentence, such as 'The bottle is empty.' When full sentences are formed, our local Qwen2.5-0.5B small language model reconstructs natural English syntax."*

#### Q8: Why run Qwen2.5-0.5B locally on CUDA rather than calling the OpenAI GPT-4 API?
* **Examiner Intent:** Testing production edge vs cloud economics and privacy knowledge.
* **Model Answer:**
  > *"A real-time assistive translation system must be instant, cost-free, and private. Cloud API calls introduce 800–1500 ms of network latency, require ongoing paid token subscriptions, and transmit personal camera data across the public internet. By hosting Qwen2.5-0.5B locally in half-precision (FP16), it consumes under 1 GB of GPU VRAM, runs completely offline with zero API bills, guarantees 100% user privacy, and generates translations in just 35 milliseconds."*

---

### Category 5: Production Engineering & Latency Optimization

#### Q9: What caused UI lag in early versions, and how did you achieve 60 FPS?
* **Examiner Intent:** Testing frontend performance profiling skills.
* **Model Answer:**
  > *"Profiling revealed that the PyTorch backend was executing in 2 ms, but the browser was lagging due to 84 individual `ctx.stroke()` and `ctx.fill()` draw calls on the HTML5 canvas. Each call forced a pipeline flush to the GPU, causing 2,520 flushes per second. We implemented path batching in `landmarksDrawer.ts`: all 21 bone line segments are buffered into a single continuous path and stroked once, and all 21 joint dots are filled once. This reduced draw calls from 84 to 2, accelerating canvas render time from 14 ms to 0.4 ms (60 FPS)."*

#### Q10: How does CTC debouncing eliminate repeated character spam?
* **Examiner Intent:** Testing continuous temporal token classification knowledge.
* **Model Answer:**
  > *"At 30 FPS, holding sign 'A' for 1 second produces 30 consecutive predictions. We adapted Connectionist Temporal Classification (CTC) from speech recognition: we treat the physical hand drop or resting state (`no_sign`) as the CTC blank token ($\epsilon$). The accumulator enforces a 3-frame commit streak to reject transition noise, then locks out duplicate emissions until the user drops their hands for > 200 ms, correctly enabling intentional double letters ('LL' in 'HELLO')."*

---

# 2. Systematic Troubleshooting & Diagnostic Procedures

### 1. Backend Connection Refused (`WinError 10061`)
* **Symptom:** Terminal shows `urllib.error.URLError: <urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>`.
* **Root Cause:** Uvicorn is not running on port 8000.
* **Remedy:** Launch the backend via:
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
  ```

### 2. PyTorch CUDA Out-of-Memory (OOM) or Fallback
* **Symptom:** `RuntimeError: CUDA out of memory` or `health` endpoint returns `"device": "cpu"`.
* **Root Cause:** Multiple background processes are consuming GPU VRAM, or PyTorch is running in FP32.
* **Remedy:**
  1. Check VRAM usage: `nvidia-smi`.
  2. Verify PyTorch CUDA build:
     ```powershell
     .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
     ```
  3. Ensure Qwen runs in half-precision (`torch.float16`).

### 3. Browser Webcam Lag & Canvas Stutter
* **Symptom:** Video stream displays at 10–15 FPS; skeletal overlay lags behind hand movements.
* **Root Cause:** Canvas draw calls are not batched, or React state (`useState`) is updating on every frame.
* **Remedy:** Ensure `landmarksDrawer.ts` uses batched paths and `useRecognitionStream.ts` stores high-frequency stream metadata in `useRef` rather than `useState`.

### 4. False Object Detections (Smartwatch / Wristbands)
* **Symptom:** User wears a watch or bracelet and the system claims they are *"holding a watch"*.
* **Root Cause:** Proximity check included landmark 0 (the wrist joint).
* **Remedy:** Verify that `backend/app/services/vision/object_detector.py` skips landmark 0:
  ```python
  if idx % 21 == 0: continue # Strictly ignore wrist joint
  ```

### 5. Repeated Character Spamming (e.g. "HHHHH")
* **Symptom:** Letters repeat uncontrollably at 30 FPS.
* **Root Cause:** `repeat_locked` flag was not set after token commit.
* **Remedy:** Ensure `backend/app/services/nlp/accumulator.py` sets `repeat_locked = True` on commit, and resets only when `sign in ("no_sign", "resting")`.

---

# 3. Plain-English Engineering Glossary

- **ASGI (Asynchronous Server Gateway Interface):** The modern Python specification for asynchronous web servers (Uvicorn/FastAPI) supporting concurrent long-lived WebSockets.
- **Attention Pooling:** A neural mechanism that calculates dynamic softmax weights across time, aggregating a 30-frame sequence into a single context vector focused on the sign's expressive peak.
- **BatchNorm:** Normalization applied across all samples in a batch for a single feature.
- **CTC (Connectionist Temporal Classification):** An algorithmic framework for sequence modeling that collapses repeated outputs unless separated by a blank token ($\epsilon$).
- **Euclidean Norm ($L_2$ Distance):** The straight-line physical distance between two 3D spatial points: $\sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$.
- **Hysteresis:** Requiring a condition to persist steadily for several frames (e.g., 3 consecutive frames) before taking an action, filtering out transient transition jitter.
- **LayerNorm:** Normalization applied across all features for a single sample independently, providing numerical stability for recurrent neural networks.
- **Logits:** Raw, unnormalized real-valued output scores produced by the final linear layer of a neural network before Softmax.
- **MediaPipe:** Google's lightweight open-source framework for detecting 21 3D hand anatomical keypoints at 60+ FPS on consumer hardware.
- **Multimodal Grounding:** Combining visual object presence (held objects) with gestural sign language to interpret rich conversational meaning.
- **Palm Span:** The rigid physical distance between the wrist (Joint 0) and the middle knuckle (Joint 9), used as a constant scale divisor.
- **Scale Invariance:** The mathematical property where coordinates remain identical regardless of whether a hand is close to or far from the webcam.
- **SLM (Small Language Model):** A compact Transformer language model (e.g. 0.5B parameters) engineered to execute locally on consumer edge hardware.
- **Temperature Scaling ($T$):** Dividing logits by a temperature constant before Softmax. Values of $T < 1.0$ sharpen probabilities and eliminate prediction flickering.
- **Translation Invariance:** The mathematical property where coordinates remain identical regardless of whether a hand is in the top-left or bottom-right of the video screen.

---

# 4. Self-Assessment Mastery Checklist

Before concluding your preparation, verify that you can explain and perform each task:
- [ ] Run `.\.venv\Scripts\pytest.exe tests/ -q` and confirm all 164 tests pass.
- [ ] Explain the difference between `.py` source and `.pyc` bytecode.
- [ ] Explain why pure Python blocks on CPU but PyTorch runs at native CUDA speeds.
- [ ] Trace a WebSocket binary JPEG frame from browser canvas to server OpenCV decoding.
- [ ] Write the mathematical formulas for wrist centering and palm-span scaling.
- [ ] Explain why the middle MCP knuckle (Joint 9) is used for palm span rather than fingertips.
- [ ] State the architecture of RealSign MLP and why $T=0.75$ temperature scaling is used.
- [ ] Draw the ISL Sequence BiLSTM architecture with its 4 subsystems on a whiteboard.
- [ ] Explain why landmark 0 (wrist) is excluded from YOLO-World `is_held` checks.
- [ ] Explain how local Qwen2.5-0.5B translates ISL SOV syntax to English SVO grammar.
- [ ] Explain how HTML5 Canvas path batching eliminated UI lag (reducing draw calls from 84 to 2).
- [ ] Describe how CTC debouncing prevents letter spam while permitting double letters.

---
*End of SETU Exam, Viva & Troubleshooting Handbook*

