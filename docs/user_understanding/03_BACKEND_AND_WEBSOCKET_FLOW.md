# 03. Backend, WebSocket Pipeline & Vision Processing Explained

---

### Backend Architecture Overview

The backend is built with **FastAPI**, a modern, high-performance Python 3.12 web framework running on the **Uvicorn** ASGI server.

```
backend/app/
├── main.py                     # ASGI entrypoint, CORS configuration, route registration
├── websocket/
│   └── recognize.py           # Real-time full-duplex WebSocket stream handler
├── services/
│   ├── vision/
│   │   ├── landmarks.py       # Google MediaPipe Holistic landmark extraction & normalization
│   │   ├── one_euro_filter.py # Adaptive jitter smoothing filter
│   │   └── kinetic_segmenter.py # Detects when hands are moving vs resting
│   ├── nlp/
│   │   ├── accumulator.py     # Debounces raw signs into coherent sentences
│   │   └── indic_transformer.py # SOV grammar restructuring for Indian languages
│   └── realsign_classifier_service.py # Direct inference engine for static alphabet
└── api/endpoints/              # Supplementary REST endpoints (health, dataset, vocabulary)
```

---

### The Real-Time Frame Journey (Step-by-Step)

```
[Webcam Canvas (Frontend)]
   |  Captures video frame, compresses to JPEG (quality: 0.6)
   |  Sends via WebSocket (ws://127.0.0.1:8000/ws/recognize)
   v
[WebSocket Ingestion (backend/app/websocket/recognize.py)]
   |  Receives base64 string or binary payload
   |  Decodes to NumPy array via OpenCV (cv2.imdecode)
   v
[MediaPipe Extraction (backend/app/services/vision/landmarks.py)]
   |  Runs MediaPipe Holistic
   |  Extracts 21 joints for each hand (X, Y, Z coordinates)
   v
[1-Euro Filter Smoothing]
   |  Removes high-frequency jitter caused by webcam noise
   v
[Canonical Normalization]
   |  Translates wrist (Landmark 0) to (0, 0, 0)
   |  Divides all coordinates by distance from wrist to middle knuckle
   |  Result: Invariant to how close or far the user sits from camera!
   v
[Neural Model Inference]
   |  Feeds 63-dim vector into RealSign MLP or ONNX model
   |  Returns probabilities for all 26 classes (e.g. Letter 'C': 98.2%)
   v
[Hysteresis Debouncing]
   |  Sign must be stable for 3 consecutive frames (0.1s)
   |  Avoids spurious flickers while moving hands
   v
[Sentence Accumulator & Indic Translation]
   |  Appends letters/words into grammatically ordered sentence
   v
[WebSocket Payload Return]
   Sends JSON back to browser: { label: "C", confidence: 0.98, fps: 29.4, bbox: [...] }
```

---

### Key Vision Concepts Explained

#### 1. What are the 21 Hand Landmarks?
MediaPipe numbers each joint from 0 to 20:
- **0:** Wrist (base of the hand)
- **1, 2, 3, 4:** Thumb (CMC joint $\to$ MCP joint $\to$ IP joint $\to$ Thumb tip)
- **5, 6, 7, 8:** Index finger (MCP knuckle $\to$ PIP $\to$ DIP $\to$ Index tip)
- **9, 10, 11, 12:** Middle finger (Knuckle $\to$ PIP $\to$ DIP $\to$ Middle tip)
- **13, 14, 15, 16:** Ring finger (Knuckle $\to$ PIP $\to$ DIP $\to$ Ring tip)
- **17, 18, 19, 20:** Pinky finger (Knuckle $\to$ PIP $\to$ DIP $\to$ Pinky tip)

#### 2. What is Canonical Normalization?
If you hold your hand 30cm away from the camera, your fingers look huge. If you stand 2 meters away, your hand looks tiny.
- If we fed raw pixel coordinates into the neural network, it would fail whenever you moved closer or further away.
- **Our Solution:** We shift the wrist $(x_0, y_0, z_0)$ to $(0, 0, 0)$. Then we measure the distance from the wrist (joint 0) to the middle knuckle (joint 9), and divide every joint coordinate by this distance.
- Now, no matter where you sit, the hand vector is always scaled between $[-1.0, +1.0]$.

#### 3. What is the 1-Euro Adaptive Filter?
Webcams produce subtle landmark shaking (jitter). If you apply a standard heavy blur or smoothing filter, the hand feels "laggy" when you move quickly.
- The **1-Euro Filter** has an *adaptive cutoff frequency*:
  - When your hand is moving fast $\to$ cutoff increases $\to$ zero lag.
  - When your hand stops and holds a sign $\to$ cutoff decreases $\to$ zero jitter.

#### 4. What is Hysteresis Debouncing?
When transitioning between Letter 'A' and Letter 'C', your fingers temporarily pass through random intermediate shapes.
- Without debouncing, the screen would flash letters like 'E' or 'S' for 1 frame.
- **Our Debouncer:** Requires the neural network to output the same sign for at least **3 consecutive frames** with confidence $\ge 50\%$ before committing it.
