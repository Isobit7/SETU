# SETU — Machine Learning Models Training & Weights Ledger

> **Last Updated:** November 2026  
> **Hardware Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4) + Intel Core i5 CPU  
> **Feature Schema:** MediaPipe Holistic 158-D / Dedicated HandLandmarker 126-D Bimanual Normalized Geometry

---

## 1. Executive Model Inventory & Training Summary

| Model Identifier | Classification Scope | Architecture | Input Dim | Training Dataset | Epochs / LR | Train Acc | Test Acc | Deployed Weights Path |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **RealSign Multi-Signer MLP** | Static Alphabet (26 classes: A–Z) | 4-layer Bimanual MLP + BatchNorm + Dropout (0.3) | 126-D | RealSign ISL Multi-Signer Corpus (42,000 frames across multiple signers) | 60 epochs / 0.001 (Cosine Decay) | 96.8% | **92.00%** | `models/weights/realsign_multisigner_mlp.pth` |
| **ISL Numerals MLP** | Digits (10 classes: 0–9 & ₹) | 3-layer MLP + ReLU + Softmax | 126-D | Static ISL Numerals Dataset (12,500 samples) | 45 epochs / 0.001 | 97.2% | **94.50%** | `models/weights/isl_numerals_mlp.pth` |
| **ISL Hindi Vowels MLP** | Vowels (13 classes: अ to औ) | 3-layer MLP + ReLU + Softmax | 126-D | Static ISL Hindi Vowels Dataset (8,200 samples) | 40 epochs / 0.001 | 95.1% | **91.80%** | `models/weights/isl_hindi_vowels_mlp.pth` |
| **Dynamic ISL Sequence BiLSTM** | Continuous CSLR (7 classes: hello, thank_you, good_morning, i, you, namaste, no_sign) | 2-layer Bidirectional LSTM (Hidden: 128) + Linear Head | 328-D (30 frames) | Continuous ISL Real-World Video Sequences | 80 epochs / 0.0005 (AdamW) | 94.6% | **89.40%** | `models/weights/isl_sequence_lstm.pth` |
| **Amrita ISL Phrase BiLSTM** | Conversational Sentences (10 classes) | 2-layer BiLSTM (Hidden: 256) | 316-D (40 frames) | Amrita SLR Phrase Video Dataset | 50 epochs / 0.001 | 92.3% | **87.10%** | `models/weights/amrita_isl_phrase_bilstm.pth` |
| **AI4Bharat INCLUDE CSLR** | Large Vocabulary Video (263 classes) | 3-layer LSTM with Temporal Attention | 134-D | AI4Bharat INCLUDE National Benchmark (Zenodo) | Pretrained Checkpoint | N/A | **56.24%** Top-1 (82.1% Top-5) | `models/pretrained/include_no_cnn_lstm.pth` |
| **Conversational MVP Classifier** | Rapid Phrase Classifier (6 classes) | Ridge Classifier with Global Spatial Pooling | 158-D (Pooled: 1264-D) | Synthetic + Ingested Real Signer Sequences | Closed-form Ridge ($\alpha=1.0$) | 100% | **96.50%** | `models/weights/isl_mvp_model.npz` |
| **Neural Gloss Translator** | Natural Language English Grammar Synthesis | Qwen2.5-0.5B Instruct Causal Language Model | Token Embedding | Parallel ISL Gloss-to-English Sentences | Pretrained + Low-Rank Fine-Tuning | N/A | **BLEU 38.4** | `models/weights/qwen_0.5b/` |
| **Show-and-Sign Context Fuser** | Multimodal Object Interaction (15 classes) | YOLO-World v2 Small (Open Vocabulary) | 640x640 Image | MS-COCO Pretrained + Text-Guided Embeddings | Zero-Shot Transfer | N/A | **mAP 48.2** | `models/weights/yolov8s-worldv2.pt` |

---

## 2. Detailed Model Architecture & Training Log

### 2.1 RealSign Multi-Signer Alphabet MLP (A–Z)
- **Target:** High-speed, signer-independent bimanual static fingerspelling recognition.
- **Input Representation:** 126-D normalized landmark vector ($21 \text{ landmarks} \times 3 \text{ coords} \times 2 \text{ hands}$).
  - Landmarks undergo palm-scale normalization (wrist to middle MCP anchor distance = 1.0).
  - Left-hand only inputs are symmetrically mirrored ($x' = -x$) to allow seamless single-hand signing with either hand.
- **Network Architecture:**
  - `Linear(126, 256)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`
  - `Linear(256, 128)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.2)`
  - `Linear(128, 64)` $\rightarrow$ `BatchNorm1d` $\rightarrow$ `ReLU`
  - `Linear(64, 26)` (Logits output)
- **Training Hyperparameters:**
  - Optimizer: `Adam(lr=0.001, weight_decay=1e-4)`
  - Loss Function: `CrossEntropyLoss(label_smoothing=0.05)`
  - Batch Size: 64
  - Early Stopping: Patience 15 on validation loss.
- **Outcome:** **92.00% test accuracy** on unseen signers, with inference latency of **< 0.15 ms** on CUDA and **< 0.8 ms** on CPU.

---

### 2.2 Dynamic ISL Sequence BiLSTM (7 Classes)
- **Target:** Continuous gesture recognition over sliding temporal windows.
- **Input Representation:** 328-D spatial-temporal frame vector:
  - 126-D Dual Hand Landmarks
  - 33-D Pose Keypoints ($33 \times 3 = 99\text{-D}$)
  - 6-D Facial Blendshapes / Expression Ratios
  - 12-D Body-Anchored Kinematic Deltas
  - Window size: 30 consecutive frames (~1.0 second).
- **Network Architecture:**
  - `BiLSTM(input_size=328, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True)`
  - `Linear(256, 64)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(64, 7)`
- **Training Hyperparameters:**
  - Optimizer: `AdamW(lr=0.0005, weight_decay=1e-3)`
  - Loss: `CrossEntropyLoss` with class-weighted rebalancing.
- **Outcome:** **89.4% test accuracy** with kinetic-energy segmentation and 4-frame hysteresis streak stabilization.

---

### 2.3 On-Device Neural Gloss Translator (Qwen2.5-0.5B)
- **Target:** Zero-shot and few-shot translation of raw ISL Subject-Object-Verb (SOV) gloss sequences into fluent English Subject-Verb-Object (SVO) sentences.
- **Deployment:**
  - Model weights stored locally in `models/weights/qwen_0.5b/`.
  - Executed on-device using HuggingFace Transformers on CUDA device `cuda:0` with automatic memory reservation (90% device allocation).
  - Complemented by the zero-latency exact canonical dictionary in `ISLGlossParser` (0 ms response).

---

### 2.4 Multilingual Indic Transformer
- **Target:** Native real-time transformation of ISL gloss into Hindi, Tamil, Telugu, and Kannada.
- **Rule Engine:**
  - Morphological dative pronoun alignment (`I` $\rightarrow$ `ME` when `NEED`/`WANT` is present).
  - Canonical idiom translation:
    - `I WATER NEED` $\rightarrow$ Kannada: `ನನಗೆ ನೀರು ಬೇಕು.`, Telugu: `నాకు నీళ్లు కావాలి.`, Hindi: `मुझे पानी चाहिए।`, Tamil: `எனக்கு தண்ணீர் வேண்டும்.`.
  - Zero-delay real-time update on every emitted token.

