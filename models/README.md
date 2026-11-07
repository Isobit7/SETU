# SETU Machine Learning Models Architecture & Directory Map

> **Central Architecture & Model Directory Map**  
> Describes how neural weights, vision tasks, and inference adapters are categorized and organized.

---

## 📁 Models Organization Directory Tree

```
models/
├── weights/           # PRODUCTION TRAINED NEURAL WEIGHTS
│   ├── realsign_multisigner_mlp.pth   # [Static] RealSign 26-Class Alphabet MLP (92.00% Acc)
│   ├── realsign_multisigner.onnx      # [Static] Optimized ONNX Alphabet Graph (<25µs)
│   ├── isl_numerals_mlp.pth           # [Static] ISL Numerals (0–9 & ₹) (94.5% Acc)
│   ├── isl_hindi_vowels_mlp.pth       # [Static] ISL Hindi Vowels (अ–औ) (91.8% Acc)
│   ├── isl_sequence_lstm.pth          # [Dynamic] Continuous BiLSTM 7-Class Model (89.4% Acc)
│   ├── amrita_isl_phrase_bilstm.pth   # [Dynamic] Amrita Phrase BiLSTM (87.1% Acc)
│   ├── isl_mvp_model.npz              # [Dynamic] Conversational MVP Classifier (96.5% Acc)
│   ├── yolov8s-worldv2.pt             # [Multimodal] YOLO-World Open-Vocabulary Object Detector
│   └── qwen_0.5b/                     # [NLP] On-Device Qwen2.5-0.5B Neural Gloss Translator
│
├── tasks/             # MEDIAPIPE VISION TASK BUNDLES
│   ├── hand_landmarker.task           # Dedicated 21-Joint 3D Hand Landmark Detector
│   └── holistic_landmarker.task       # Upper-Body Pose, Face Mesh, and Hand Joints
│
├── pretrained/        # NATIONAL BENCHMARK PRETRAINED CHECKPOINTS
│   ├── include_no_cnn_lstm.pth        # AI4Bharat INCLUDE 263-Class CSLR LSTM
│   └── label_map_include.json         # 263-Class Sign Vocabulary Index
│
├── adapters/          # PYTHON INFERENCE RUNTIME ADAPTERS
│   ├── realsign.py                    # RealSignAlphabetClassifier (Dual-hand 126-D)
│   ├── sequence_lstm.py               # ISLSequenceLSTMClassifier (Temporal 328-D)
│   ├── include.py                     # AI4Bharat Streaming & Video Adapter
│   ├── mvp.py                         # Conversational MVP Classifier Adapter
│   ├── amrita.py                      # Amrita Phrase BiLSTM Adapter
│   └── digits.py                      # ISL Numerals & Vowels Adapter
│
└── registry.py        # SWAPPABLE MODEL REGISTRY (ModelRegistry.get("realsign"))
```

---

## 🎯 Model Categorization Matrix

### 1. Static Fingerspelling Models (Pose & Point)
- **Input:** 126-D Bimanual Normalized Geometry ($21 \times 3 \times 2$).
- **Inference Latency:** < 0.2 milliseconds (CUDA) / < 0.8 milliseconds (CPU).
- **Execution Mode:** Isolated HandLandmarker pipeline (60 FPS smooth).
- **Classes:**
  - RealSign: 26 English Alphabets (`A` to `Z`).
  - Numerals: `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `₹`.
  - Hindi Vowels: `अ`, `आ`, `इ`, `ई`, `उ`, `ऊ`, `ऋ`, `ए`, `ऐ`, `ओ`, `औ`, `अं`, `अः`.

### 2. Dynamic Sequence Models (Continuous Temporal)
- **Input:** 328-D Spatial-Temporal Vectors across 30 to 40-frame sliding windows.
- **Inference Latency:** < 12 milliseconds (CUDA).
- **Execution Mode:** Kinetic-Energy segmentation + CTC Greedy collapsing.
- **Classes:**
  - `hello`, `thank_you`, `good_morning`, `i`, `you`, `namaste`, `no_sign`.
  - Amrita Phrases (Conversational everyday requests).
  - AI4Bharat INCLUDE (263 national sign concepts).

### 3. Natural Language & Context Models
- **Qwen2.5-0.5B Instruct:** Transforms raw gloss (e.g. `I WATER NEED`) into fluent English (`I need water.`).
- **Indic Transformer:** High-speed real-time transformer for native Kannada (`ನನಗೆ ನೀರು ಬೇಕು.`), Telugu (`నాకు నీళ్లు కావాలి.`), Hindi (`मुझे पानी चाहिए।`), and Tamil (`எனக்கு தண்ணீர் வேண்டும்.`).
- **YOLO-World v2:** Open-vocabulary detector for Show-and-Sign multimodal interaction.

