# SETU — Complete Working, Architectural Formulas & System Guide
*The Comprehensive Engineering, Mathematical & Algorithmic Blueprint of the SETU Indian Sign Language Translator*

---

## 📋 Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Visual End-to-End Pipeline Diagram](#2-visual-end-to-end-pipeline-diagram)
3. [Vision Preprocessing & Lighting Enhancement](#3-vision-preprocessing--lighting-enhancement)
   - [Adaptive CLAHE Low-Light Enhancer](#adaptive-clahe-low-light-enhancer)
   - [Real-Time Framing, Distance & Positioning Evaluator](#real-time-framing-distance--positioning-evaluator)
4. [MediaPipe Landmark Extraction & Coordinate Geometries](#4-mediapipe-landmark-extraction--coordinate-geometries)
   - [Anatomical Keypoint Breakdown (543 Holistic vs 42 Hand Points)](#anatomical-keypoint-breakdown)
   - [Mathematical Invariance Formulas (Translation, Scale, Orientation)](#mathematical-invariance-formulas)
   - [176-D Feature Vector Assembly (Hands, Face Ratios, Body Anchors)](#176-d-feature-vector-assembly)
5. [Show-and-Sign Multimodal Object Detection](#5-show-and-sign-multimodal-object-detection)
6. [Machine Learning & Deep Learning Inference Engines](#6-machine-learning--deep-learning-inference-engines)
   - [RealSign 26-Class Alphabet PyTorch Model (Fingerspelling)](#realsign-26-class-alphabet-pytorch-model)
   - [Dynamic ISL Sequence LSTM (Continuous Gestures)](#dynamic-isl-sequence-lstm)
   - [AI4Bharat INCLUDE Pretrained Engine (263 Signs)](#ai4bharat-include-pretrained-engine)
7. [Temporal Stabilization, Streak Filtering & Disambiguation](#7-temporal-stabilization-streak-filtering--disambiguation)
8. [Sentence Accumulation, Word Buffers & CTC Decoding](#8-sentence-accumulation-word-buffers--ctc-decoding)
9. [Neural Seq2Seq Translation Engine (Qwen2.5-0.5B on GPU)](#9-neural-seq2seq-translation-engine-qwen25-05b-on-gpu)
10. [Client-Server WebSocket Protocol & Latency Breakdown](#10-client-server-websocket-protocol--latency-breakdown)
11. [How You Can Improve Each Component (Actionable Ideas)](#11-how-you-can-improve-each-component-actionable-ideas)

---

## 1. Executive Summary & High-Level Architecture

SETU is an edge-native, real-time Indian Sign Language (ISL) bidirectional communication platform. It translates continuous two-handed signing, static alphabet fingerspelling, physical objects held by the signer, and facial grammar into fluent, natural English text and synthesized speech.

The system operates across a **hybrid client-server architecture**:
- **Frontend (React 19 + TypeScript + Vite)**: Runs a 60 FPS zero-allocation canvas viewfinder, renders skeleton landmark meshes, evaluates client-side distance and positioning in $<0.05$ ms, manages browser-native Text-to-Speech (TTS), and provides interactive disambiguation prompts.
- **Backend (FastAPI + PyTorch + MediaPipe + Transformers on CUDA)**: Ingests compressed binary JPEG frames over WebSocket, performs adaptive luminance enhancement, extracts 543 skeletal landmarks, aligns them with affine translation and scale invariance, runs specialized classification models, fuses physical object nouns, and uses a local **Qwen2.5-0.5B-Instruct** neural sequence-to-sequence model to generate fluent English sentences.

---

## 2. Visual End-to-End Pipeline Diagram

```mermaid
flowchart TD
    A["Webcam Video Stream\n(60 FPS Browser Capture)"] --> B["Framing & Distance Evaluator\n(Euclidean Shoulder & Headroom Math)"]
    B --> C["Binary JPEG WebSocket Transmission\n(ws://localhost:8000/api/v1/ws/recognize)"]
    
    subgraph Vision_Preprocessing["1. Vision Preprocessing & Enhancement"]
        C --> D["Frame Backpressure Drain Guard"]
        D --> E["Adaptive CLAHE Low-Light Enhancer\n(LAB Color Space & Bilateral Filter)"]
    end

    subgraph Landmark_Extraction["2. Landmark Extraction & Normalization"]
        E --> F["MediaPipe Holistic & Hand Landmarkers\n(543 Keypoints: Hands, Pose, Face)"]
        F --> G["Wrist Translation Centering\n(p' = p - p_wrist)"]
        G --> H["Palm Span Scale Normalization\n(p'' = p' / max_distance)"]
        H --> I["176-D Feature Assembly\n(158D Hand/Arm + 6D Face + 12D Body Anchors)"]
    end

    subgraph Object_Grounded_Perception["3. Multimodal Object Perception"]
        E -->|Every 10 frames| J["COCO 80+ Object Detector\n(Detects Bottle, Cup, Phone, Book)"]
    end

    subgraph Inference_Engines["4. ML & Deep Learning Inference"]
        I --> K{Active Model Selector}
        K -->|Alphabet A-Z| L["RealSign 26-Class MLP\n(126-D Hand Landmark Tensor)"]
        K -->|Dynamic Gestures| M["ISL Sequence BiLSTM\n(30 Frames x 176-D Features)"]
        K -->|Comprehensive Lexicon| N["AI4Bharat INCLUDE Engine\n(263 Vocabulary Signs)"]
    end

    subgraph Temporal_Stabilization["5. Filtering & Ambiguity Arbiter"]
        L & M & N --> O["Confidence Hysteresis Gate\n(T_enter = 0.55, T_exit = 0.40)"]
        O --> P["4-Frame Streak Debouncer"]
        P --> Q{"Ambiguity Margin Check\nP(Top1) - P(Top2) <= 0.12"}
        Q -->|Ambiguous| R["Emit Touchless Disambiguation Prompt"]
        Q -->|Confident| S["Committed Sign Token"]
    end

    subgraph NLP_Sentence_Generation["6. NLP Sentence Accumulation & Neural Seq2Seq"]
        S --> T["SentenceAccumulator\n(CTC Collapse + 1.2s Word Timeout)"]
        J --> U["Show-and-Sign Multimodal Fuser\n(Noun Object + Gesture Verb)"]
        T & U --> V{"Exact Canonical Idiom?\n(e.g., ME WATER DRINK)"}
        V -->|Yes| W["ISLGlossParser\n(Direct Canonical English: 'I want water')"]
        V -->|No (Open Vocabulary)| X["Qwen2.5-0.5B-Instruct Neural Seq2Seq\n(Runs on CUDA in ~350ms)"]
    end

    subgraph Output_HUD["7. Output & Speech Synthesis"]
        W & X --> Y["Formatted English Sentence Output"]
        Y --> Z["WebSocket JSON Response"]
        Z --> AA["Frontend Subtitle Display"]
        Z --> AB["Web Speech Text-to-Speech (TTS)"]
    end
```

---

## 3. Vision Preprocessing & Lighting Enhancement

### Adaptive CLAHE Low-Light Enhancer
*Implemented in:* [`backend/app/services/vision/enhancer.py`](file:///c:/MINI%20PROJ-SETU/backend/app/services/vision/enhancer.py)

Real webcams suffer from sensor grain, dim indoor backlighting, and uneven shadows. Standard thresholding fails because brightness is non-uniform. SETU applies **Contrast Limited Adaptive Histogram Equalization (CLAHE)** on the lightness channel of the LAB color space.

#### Mathematical Formulation:
1. **Color Space Transformation:**
   $$\text{BGR} \xrightarrow{\text{CIE } L^*a^*b^*} (L, a, b)$$
   where $L \in [0, 255]$ represents perceptual luminance, and $a, b$ represent chromatic color-opponent dimensions.

2. **Mean Luminance Calculation:**
   $$\mu_L = \frac{1}{H \times W} \sum_{x=1}^{W} \sum_{y=1}^{H} L(x, y)$$

3. **Normalized Lighting Quality Score:**
   $$S_{\text{light}} = \text{round}\left(\min\left(\frac{\mu_L}{128.0}, 1.0\right), 3\right)$$
   - $S_{\text{light}} \in [0.0, 0.38)$: Low light condition $\implies$ triggers on-screen warning: *"Low light detected — please turn on a light facing you"*.
   - $S_{\text{light}} \in [0.38, 1.0]$: Optimal working illumination.

4. **Adaptive Contrast Boost:**
   If $\mu_L < 75.0$, CLAHE is applied with clipping threshold $\beta = 2.5$ over an $8 \times 8$ grid of contextual regions:
   $$L_{\text{clahe}} = \text{CLAHE}(L; \text{clip\_limit}=2.5, \text{grid}=(8, 8))$$
   
5. **Sensor Denoising (Bilateral Filter):**
   To remove webcam thermal grain amplified by contrast boosting without blurring finger edges:
   $$L_{\text{enhanced}}(x, y) = \frac{1}{W_p} \sum_{x_i, y_i} L_{\text{clahe}}(x_i, y_i) \cdot \exp\left(-\frac{\|(x, y) - (x_i, y_i)\|^2}{2\sigma_s^2}\right) \cdot \exp\left(-\frac{|L_{\text{clahe}}(x, y) - L_{\text{clahe}}(x_i, y_i)|^2}{2\sigma_r^2}\right)$$
   where diameter $d = 5$, spatial sigma $\sigma_s = 35$, range sigma $\sigma_r = 35$.

---

### Real-Time Framing, Distance & Positioning Evaluator
*Implemented in:* [`frontend/src/utils/framingEvaluator.ts`](file:///c:/MINI%20PROJ-SETU/frontend/src/utils/framingEvaluator.ts) and [`backend/app/websocket/recognize.py`](file:///c:/MINI%20PROJ-SETU/backend/app/websocket/recognize.py)

Users cannot intuitively judge physical distances like "50–70 cm". SETU evaluates upper-body keypoints to guide the user in real time.

#### Mathematical Formulas & Distance Rules:
1. **Shoulder Span (Euclidean Distance Metric):**
   Let $p_{11} = (x_{11}, y_{11})$ be the left shoulder, and $p_{12} = (x_{12}, y_{12})$ be the right shoulder in normalized $[0, 1]$ coordinate space:
   $$w_{\text{shoulder}} = \sqrt{(x_{12} - x_{11})^2 + (y_{12} - y_{11})^2}$$

   - **Too Far Away ($w_{\text{shoulder}} < 0.14$):**
     $$\text{Prompt: } \textbf{"Step Closer — Upper body too small"}$$
   - **Too Close ($w_{\text{shoulder}} > 0.64$):**
     $$\text{Prompt: } \textbf{"Move Back — Too close to camera"}$$
   - **Optimal Distance ($0.14 \le w_{\text{shoulder}} \le 0.64$):**
     Ideal physical distance of 50–70 cm from the lens.

2. **Horizontal Centering Check:**
   Let $c_x = \frac{x_{11} + x_{12}}{2}$ be the torso midpoint:
   - If $c_x < 0.28$: Signer is drifted left $\implies$ Prompt: **`"Center Yourself — Shift Right"`** (adjusted for mirrored feed).
   - If $c_x > 0.72$: Signer is drifted right $\implies$ Prompt: **`"Center Yourself — Shift Left"`**.

3. **Headroom & Vertical Framing Check:**
   Let $y_{\text{nose}}$ be the vertical coordinate of the nose (index 0):
   - If $y_{\text{nose}} < 0.03$: Head cut off at top of screen $\implies$ Prompt: **`"Tilt Camera Down / Lower Head"`**.
   - If $y_{\text{nose}} > 0.45$: Camera mounted too high $\implies$ Prompt: **`"Raise Camera"`**.

4. **Optimal State:**
   When distance, centering, headroom, and hands are all within bounds:
   $$\text{HUD Chip: } \textbf{"Framing Optimal — Ready to sign"} \quad (\text{Emerald Green Indicator})$$

---

## 4. MediaPipe Landmark Extraction & Coordinate Geometries

### Anatomical Keypoint Breakdown
SETU extracts up to **543 skeletal landmarks** across the body:

```
                      (0) Nose
                     /        \
          (11) Left Shoulder   (12) Right Shoulder
                 |                   |
            (13) Left Elbow     (14) Right Elbow
                 |                   |
            (15) Left Wrist     (16) Right Wrist
                 |                   |
       [21 Left Hand Joints]   [21 Right Hand Joints]
```

- **Pose Landmarks (33 points):** Track torso position, shoulders, elbows, wrists, and hips.
- **Left Hand Landmarks (21 points):** Wrist (0), Thumb (1–4), Index (5–8), Middle (9–12), Ring (13–16), Pinky (17–20).
- **Right Hand Landmarks (21 points):** Symmetric 21 joints.
- **Face Landmarks (468 points):** Selectively filtered for lip contours, eye pupils, and eyebrow crests to measure non-manual grammatical facial expressions.

---

### Mathematical Invariance Formulas

Raw webcam coordinates are sensitive to where the person stands in front of the lens. If the signer moves to the left corner of their room, raw coordinates change completely even if they make the exact same hand shape.

SETU eliminates this problem by making coordinates **Translation-Invariant** and **Scale-Invariant**:

```
 [Raw Camera Coordinates] ──► [Wrist-Centered Translation] ──► [Scale-Normalized Unit Span]
    p_i = (x_i, y_i, z_i)         p'_i = p_i - p_wrist             p''_i = p'_i / max_span
```

#### 1. Translation Invariance (Wrist-Centering):
For all 21 hand joints $i \in \{0, 1, \dots, 20\}$, subtract the wrist coordinate $p_0 = (x_0, y_0, z_0)$:
$$p_i' = p_i - p_0 = \begin{bmatrix} x_i - x_0 \\ y_i - y_0 \\ z_i - z_0 \end{bmatrix}$$
After this step, the wrist is always at $(0, 0, 0)$, regardless of where the hand is in the video frame.

#### 2. Scale Invariance (Palm Span Normalization):
Signers with larger hands or people sitting closer to the webcam produce larger coordinate spans. We compute the maximum Euclidean distance from the wrist to any fingertip:
$$s_{\text{max}} = \max_{i \in \{0, \dots, 20\}} \|p_i'\|_2 = \max_{i} \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2 + (z_i - z_0)^2}$$

The scale-normalized coordinate $\hat{p}_i$ is:
$$\hat{p}_i = \frac{p_i'}{s_{\text{max}} + \epsilon}, \quad \text{where } \epsilon = 10^{-7}$$
Now, all hand coordinates strictly reside in a normalized unit sphere: $\|\hat{p}_i\|_2 \le 1.0$.

---

### 176-D Feature Vector Assembly
*Implemented in:* [`training/feature_extractor.py`](file:///c:/MINI%20PROJ-SETU/training/feature_extractor.py) and [`backend/app/websocket/recognize.py`](file:///c:/MINI%20PROJ-SETU/backend/app/websocket/recognize.py)

To classify continuous two-handed signs, SETU concatenates hands, facial expressions, and torso reference points into an authoritative **176-dimensional feature vector**:

$$\mathbf{v}_{176} = [\mathbf{f}_{\text{hands\_arms}}^{158}, \mathbf{f}_{\text{facial\_ratios}}^{6}, \mathbf{f}_{\text{body\_anchors}}^{12}]$$

#### Component 1: 158-D Hand & Arm Vector ($\mathbf{f}^{158}$)
- Right hand 21 joints $\times$ 3 $(x, y, z) = 63$ dimensions.
- Left hand 21 joints $\times$ 3 $(x, y, z) = 63$ dimensions.
- Upper body arms (shoulders, elbows, wrists) = 32 dimensions.
- Sum: $63 + 63 + 32 = 158$ dimensions.

#### Component 2: 6-D Facial Emotion Ratios ($\mathbf{f}^{6}$)
Sign language grammar uses eyebrow raises for questions and mouth shapes for intensity:
1. **Left Eyebrow Raise:**
   $$r_1 = \frac{y_{\text{brow\_left}} - y_{\text{eye\_left\_top}}}{d_{\text{interocular}}}$$
2. **Right Eyebrow Raise:**
   $$r_2 = \frac{y_{\text{brow\_right}} - y_{\text{eye\_right\_top}}}{d_{\text{interocular}}}$$
3. **Mouth Vertical Openness:**
   $$r_3 = \frac{|y_{\text{lower\_lip}} - y_{\text{upper\_lip}}|}{w_{\text{mouth}}}$$
4. **Mouth Horizontal Aspect Ratio:**
   $$r_4 = \frac{w_{\text{mouth}}}{d_{\text{interocular}}}$$
5. **Left Lip Corner Curvature (Smile/Frown):**
   $$r_5 = \frac{y_{\text{lip\_center}} - y_{\text{left\_corner}}}{w_{\text{mouth}}}$$
6. **Right Lip Corner Curvature:**
   $$r_6 = \frac{y_{\text{lip\_center}} - y_{\text{right\_corner}}}{w_{\text{mouth}}}$$

#### Component 3: 12-D Body Anchor Distances ($\mathbf{f}^{12}$)
Many signs share identical handshapes but differ by where they are signed on the body (e.g. signing near the forehead means *FATHER*, signing near the chin means *MOTHER*). We compute Euclidean distances from each hand's wrist to 6 anatomical body anchors:
$$\mathbf{a}_k = \|p_{\text{wrist}} - p_{\text{anchor}_k}\|_2, \quad k \in \{1, \dots, 6\}$$
Anchors:
1. Nose (Head center)
2. Left Shoulder
3. Right Shoulder
4. Mouth Center
5. Left Hip
6. Right Hip

$6 \text{ distances} \times 2 \text{ hands (Left + Right)} = 12 \text{ dimensions}$.

---

## 5. Show-and-Sign Multimodal Object Detection
*Implemented in:* [`backend/app/services/vision/object_detector.py`](file:///c:/MINI%20PROJ-SETU/backend/app/services/vision/object_detector.py) and [`backend/app/services/nlp/multimodal_fuser.py`](file:///c:/MINI%20PROJ-SETU/backend/app/services/nlp/multimodal_fuser.py)

In natural everyday sign conversations, signers frequently hold an object (like a water bottle, medicine box, cup, or phone) rather than spelling out its full noun.

SETU runs a COCO 80+ class object detector asynchronously every 10 frames:
1. **Object Detection:** Detects bounding box and class:
   $$\text{Detection} = (c_{\text{obj}}, \text{bbox}, \text{confidence})$$
2. **Show-and-Sign Fusion Rule:**
   When the user signs a verb (e.g., `DRINK`, `EMPTY`, `GIVE`, `CALL`), the `MultimodalFuser` combines the detected physical object with the gesture:
   - Object: `bottle` + Sign: `empty` $\implies$ **"The bottle is empty."**
   - Object: `bottle` + Sign: `water` $\implies$ **"I need drinking water."**
   - Object: `phone` + Sign: `help` $\implies$ **"Please call emergency help."**
   - Object: `cup` + Sign: `tea` $\implies$ **"I would like a cup of tea."**

---

## 6. Machine Learning & Deep Learning Inference Engines

SETU contains three specialized neural network classifiers:

### RealSign 26-Class Alphabet PyTorch Model
*Implemented in:* [`models/adapters/realsign.py`](file:///c:/MINI%20PROJ-SETU/models/adapters/realsign.py)
*Weights:* `models/weights/realsign_26class_mlp.pth`

- **Task:** 26-class English alphabet fingerspelling ($A$ through $Z$).
- **Input:** 126-D tensor ($21 \text{ landmarks} \times 3 \text{ coordinates} \times 2 \text{ hands}$).
- **Architecture:** 3-Layer Deep Multi-Layer Perceptron (MLP):
  $$\mathbf{x} \in \mathbb{R}^{126} \xrightarrow{\text{Linear}(126 \to 256)} \text{BatchNorm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.2)$$
  $$\xrightarrow{\text{Linear}(256 \to 128)} \text{BatchNorm} \xrightarrow{\text{ReLU}} \text{Dropout}(0.2)$$
  $$\xrightarrow{\text{Linear}(128 \to 26)} \mathbf{z} \in \mathbb{R}^{26}$$
- **Probability Output:**
  $$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{j=1}^{26} e^{z_j}}$$
- **Latency:** ~2 to 5 ms per frame.

---

### Dynamic ISL Sequence LSTM
*Implemented in:* [`models/adapters/sequence_lstm.py`](file:///c:/MINI%20PROJ-SETU/models/adapters/sequence_lstm.py)
*Weights:* `models/weights/isl_sequence_lstm.pth`

- **Task:** Temporal continuous signs (`hello`, `thank_you`, `good_morning`, `i`, `you`, `water`, `food`).
- **Input:** Sliding temporal window of $T = 30$ frames $\times 176$ dimensions:
  $$\mathbf{X} \in \mathbb{R}^{30 \times 176}$$
- **Architecture:** 2-Layer Bidirectional LSTM with Self-Attention:
  $$\mathbf{h}_t = \text{BiLSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}), \quad \mathbf{h}_t \in \mathbb{R}^{256}$$
  $$\alpha_t = \frac{\exp(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_t))}{\sum_{k=1}^{30} \exp(\mathbf{w}^T \tanh(\mathbf{W}_a \mathbf{h}_k))}$$
  $$\mathbf{c} = \sum_{t=1}^{30} \alpha_t \mathbf{h}_t \xrightarrow{\text{Linear}(256 \to C)} \text{Softmax}$$
- **Latency:** ~12 to 18 ms per sequence window.

---

### AI4Bharat INCLUDE Pretrained Engine
*Implemented in:* [`models/adapters/include.py`](file:///c:/MINI%20PROJ-SETU/models/adapters/include.py)

- **Task:** Comprehensive Indian Sign Language vocabulary (263 signs).
- **Source:** AI4Bharat National ISL Benchmark Dataset.
- **Inference Mode:** Streaming circular buffer over upper-body landmark sequences.

---

## 7. Temporal Stabilization, Streak Filtering & Disambiguation

Webcam landmark tracking fluctuates from frame to frame due to lighting flicker and finger micro-movements. If an ML model emits raw predictions on every frame, the text flickers erratically (*"A... A... S... A... S"*).

SETU solves this with a **Dual-Stage Stabilization Filter**:

### 1. Confidence Hysteresis
Instead of a single fixed threshold, SETU uses an **entry threshold** ($T_{\text{enter}} = 0.55$) and an **exit threshold** ($T_{\text{exit}} = 0.40$):
- A sign is only recognized when confidence rises above $55\%$.
- Once active, it remains accepted until confidence drops below $40\%$.
This eliminates jitter at threshold boundaries.

### 2. 4-Frame Streak Debouncer
Let $\hat{y}_t = \arg\max_c P(y_t = c)$. The streak counter $S_t$ updates as:
$$S_t = \begin{cases} S_{t-1} + 1 & \text{if } \hat{y}_t = \hat{y}_{t-1} \\ 1 & \text{otherwise} \end{cases}$$
A token is only committed to the word accumulator if:
$$S_t \ge 4 \quad \text{AND} \quad P(\hat{y}_t) \ge T_{\text{confidence}}$$
This requires the signer to hold a hand shape for at least 4 consecutive frames (~66 ms at 60 FPS), filtering out transient movements between signs.

---

### 3. Touchless Disambiguation Arbiter
*Implemented in:* [`backend/app/websocket/recognize.py`](file:///c:/MINI%20PROJ-SETU/backend/app/websocket/recognize.py)

Certain signs in Indian Sign Language share nearly identical hand shapes from a single webcam angle (e.g. *'M'* vs *'N'*, or *'A'* vs *'S'*).

When the confidence difference between the top-2 predictions is small:
$$\Delta P = P(\text{Top}_1) - P(\text{Top}_2) \le 0.12$$
SETU activates the **Ambiguity Arbiter**:
- It pauses automatic letter commitment.
- Displays interactive choice chips on screen: `[ Choice A: 'M' ]` vs `[ Choice B: 'N' ]`.
- The user can select their intended sign, or lift their hand slightly to confirm without touching their mouse.

---

## 8. Sentence Accumulation, Word Buffers & CTC Decoding
*Implemented in:* [`backend/app/services/nlp/accumulator.py`](file:///c:/MINI%20PROJ-SETU/backend/app/services/nlp/accumulator.py)

To assemble individual spelled letters and gesture words into sentences, `SentenceAccumulator` uses two timeout clocks:

```
  Letter Input:  S ──► E ──► T ──► U
                              │
                    [1.2s Idle Pause]
                              ▼
                        Committed Word: "SETU"
                              │
                    [2.5s Idle Pause]
                              ▼
            Trigger Sentence Translation & Text-to-Speech
```

1. **Word Timeout Buffer ($\tau_{\text{word}} = 1.2\text{ s}$):**
   - As you fingerspell letters (`S`, `E`, `T`, `U`), they accumulate in `current_word`.
   - When no new letter is detected for 1.2 seconds, `current_word` is pushed into `sentence_tokens` as a complete word, and `current_word` is cleared.

2. **Sentence Timeout Buffer ($\tau_{\text{sentence}} = 2.5\text{ s}$):**
   - When the signer finishes signing and drops their hands to rest for 2.5 seconds, the accumulated sentence tokens are dispatched to the translation engine.

3. **CTC Greedy Collapsing:**
   For continuous gesture streams, Connectionist Temporal Classification (CTC) collapses repeated frame emissions:
   $$\text{Collapse}("no\_sign", "HELLO", "HELLO", "no\_sign", "WATER") \implies ["HELLO", "WATER"]$$

---

## 9. Neural Seq2Seq Translation Engine (Qwen2.5-0.5B on GPU)
*Implemented in:* [`backend/app/services/nlp/neural_translator.py`](file:///c:/MINI%20PROJ-SETU/backend/app/services/nlp/neural_translator.py)
*Weights:* `models/weights/qwen_0.5b/model.safetensors`

### Why Rule Dictionaries Fail
Hardcoded rules like `("ME", "WATER", "DRINK"): "I want water"` only work for the exact sentences written in the dictionary. If a user signs:
`["DOCTOR", "HOSPITAL", "COME", "HURRY"]`
a rule dictionary fails completely and outputs broken pidgin: *"Doctor hospital come hurry"*.

### The Neural Solution: Qwen2.5-0.5B-Instruct
SETU uses an on-device Small Language Model (SLM) running on the local **NVIDIA RTX 3050 GPU (CUDA)**:
- **Model Size:** 490 Million parameters (FP16 weights = 988 MB).
- **VRAM Footprint:** ~1.1 GB (leaves $>5$ GB free for MediaPipe and PyTorch).
- **Inference Speed:** ~350 ms per sentence.

#### Conditioning Prompt Architecture:
```python
messages = [
    {
        "role": "system",
        "content": (
            "You are an Indian Sign Language (ISL) to fluent English translator. "
            "Convert raw sign words into one natural, fluent, grammatically correct English sentence. "
            "Output ONLY the English sentence without extra text or quotation marks."
        ),
    },
    {"role": "user", "content": "ISL: ME WATER DRINK"},
    {"role": "assistant", "content": "I want to drink water."},
    {"role": "user", "content": "ISL: MOTHER MEDICINE BUY NEED"},
    {"role": "assistant", "content": "My mother needs to buy medicine."},
    {"role": "user", "content": "ISL: MY NAME S A T H W I K"},
    {"role": "assistant", "content": "My name is Sathwik."},
    {"role": "user", "content": f"ISL: {raw_accumulated_tokens}"},
]
```

#### Deterministic Decoding Formula:
The model generates tokens using greedy decoding (`do_sample=False`):
$$y_t = \arg\max_w P(w \mid y_{<t}, X_{\text{ISL}})$$
This produces stable, reproducible grammar without random hallucinations.

---

## 10. Client-Server WebSocket Protocol & Latency Breakdown

### WebSocket Message Payloads
- **Endpoint:** `ws://localhost:8000/api/v1/ws/recognize`
- **Client Frame Transmission:** Sends binary JPEG bytes:
  ```json
  [Binary JPEG Image Bytes (640x480, quality 0.82)]
  ```
- **Client Control Commands:** Sends text JSON:
  ```json
  {"action": "set_model", "model": "realsign"}
  {"action": "clear_sentence"}
  {"action": "backspace"}
  {"action": "resolve_disambiguation", "choice": "M"}
  ```
- **Server Response Payload:**
  ```json
  {
    "type": "prediction",
    "label": "hello",
    "confidence": 0.942,
    "landmarks": {
      "left_hand": [...],
      "right_hand": [...],
      "pose": [...]
    },
    "guidance": "Position good — Ready to translate",
    "lighting": {
      "score": 0.82,
      "was_enhanced": false
    },
    "sentence_buffer": "HELLO",
    "translated_text": "Hello, nice to meet you.",
    "latency_ms": 28.4
  }
  ```

### Real-Time Latency Budget ($< 45\text{ ms}$ Total)
```
┌────────────────────────┬─────────────┬────────────────────────────────┐
│ Stage                  │ Latency     │ Implementation                 │
├────────────────────────┼─────────────┼────────────────────────────────┤
│ 1. JPEG Decode (C++)   │ 1.5 - 3 ms  │ OpenCV cv2.imdecode            │
│ 2. CLAHE Light Check   │ 1.0 - 2 ms  │ OpenCV cv2.cvtColor + mean     │
│ 3. MediaPipe Landmarks │ 15 - 22 ms  │ GPU-accelerated MediaPipe Task │
│ 4. Affine Normalization│ 0.2 - 0.5 ms│ NumPy vector math              │
│ 5. MLP Inference       │ 2.0 - 4 ms  │ PyTorch CUDA FP16              │
│ 6. JSON Serialization  │ 0.8 - 1.5 ms│ orjson / standard json         │
├────────────────────────┼─────────────┼────────────────────────────────┤
│ Total Per-Frame Loop   │ ~24 - 32 ms │ 30 - 40 FPS sustained stream   │
└────────────────────────┴─────────────┴────────────────────────────────┘
```

---

## 11. How You Can Improve Each Component (Actionable Ideas)

Now that you have the complete working blueprint of SETU, here are the highest-impact improvements you can build next:

### 1. Vision & Landmark Improvements:
- **OneEuroFilter for Finger Jitter:**
  Implement a `OneEuroFilter` (adaptive low-pass filter) on the 21 finger coordinates. It keeps the hand steady during slow signing while eliminating delay during fast movements.
- **Dynamic Crop Region of Interest (ROI):**
  Instead of feeding the full $640 \times 480$ frame to MediaPipe every time, crop a $256 \times 256$ box around the detected hands from the previous frame. This drops MediaPipe latency from 20 ms to ~8 ms!

### 2. Model Accuracy & Retraining:
- **Spatial-Temporal Graph Convolutional Network (ST-GCN):**
  The current sequence model uses an LSTM. Upgrading to an ST-GCN treats hand bones as edges on a graph and finger joints as nodes. ST-GCN achieves $>95\%$ accuracy on Indian Sign Language benchmarks.
- **Dataset Augmentation:**
  Use the built-in dataset augmentation pipeline (`scripts/record_custom_sign.py`) to record signs under multiple camera angles (tilt $\pm 15^\circ$) and lighting conditions to prevent overfitting.

### 3. Sentence & NLP Improvements:
- **Hands-Down Rest Pose Trigger:**
  Instead of waiting for the 2.5-second idle timer, detect when both wrists drop below the hips ($y_{\text{wrist}} > y_{\text{hip}}$). The instant the user drops their hands, trigger the Qwen sentence translation immediately!
- **Fuzzy Levenshtein Auto-Correction:**
  If a user fingerspells a word with one wrong letter (e.g. `"W-A-T-R"` instead of `"WATER"`), an automatic minimum-edit-distance corrector can fix the spelling before feeding it to Qwen.

### 4. Capstone Unified Auto-Engine:
- Once individual models achieve high confidence, connect the velocity detector to route automatically between fingerspelling and dynamic gestures without any dropdown menus.
