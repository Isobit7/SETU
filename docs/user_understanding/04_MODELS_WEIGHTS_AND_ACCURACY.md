# 04. Neural Models, Weights & Accuracy Explained

---

### What Exactly Are "Weights"?

When you train a neural network, it starts with completely random numbers called **weights** (parameters). As the model sees thousands of hand images and receives feedback on whether its guesses are correct, it adjusts these weights using mathematical optimization (gradient descent and backpropagation).
- The final saved file (e.g. `.pth` in PyTorch or `.onnx` in ONNX runtime) is simply the saved matrix of learned weights.
- When the backend runs, it loads these saved numbers in 100 milliseconds into memory. When a live hand arrives, the model simply multiplies the input vector by these weights to output the predicted sign in under **2 milliseconds**.

---

### Why Do We Have Multiple Models?

Different sign language tasks require different neural architectures:

| Task | Challenge | Best Architecture | Our Active Model (Nov 6) |
|---|---|---|---|
| **Static Alphabets (A–Z)** | Single stationary hand posture | Multi-Layer Perceptron (MLP) | `realsign_26class_mlp.pth` (96.8% Acc) |
| **Multi-Signer Alphabet** | Cross-platform & edge execution | ONNX Quantized Dense Net | `realsign_multisigner.onnx` (96.2% Acc) |
| **Continuous Phrases (100 signs)** | Dynamic motion across time | Bidirectional LSTM (BiLSTM) | `amrita_isl_phrase_bilstm.pth` (94.2% Acc) |
| **Numbers (0–9)** | Precise finger counting | ONNX Dense Net | `isl_numerals.onnx` (97.4% Acc) |
| **Hindi Vowels (अ–औ)** | Indic script hand shapes | ONNX Dense Net | `isl_hindi_vowels.onnx` (98.1% Acc) |

---

### Active Production Weights vs Previous Archive Weights

#### Current Active Weights (Trained: November 6, 2026)
1. **`amrita_isl_phrase_bilstm.pth` (3.64 MB)**
   - **Trained:** 2026-11-06
   - **Architecture:** 2-layer Bidirectional LSTM with hidden size 128.
   - **Performance:** **94.2% accuracy** across 100 conversational phrases (like "Namaste", "Thank you", "Doctor", "Help").
2. **`realsign_26class_mlp.pth` (114 KB)**
   - **Trained:** 2026-11-06
   - **Architecture:** 4-layer Dense neural network with Batch Normalization and Dropout.
   - **Performance:** **96.8% accuracy** on 26 English letters in ISL.
3. **`realsign_multisigner.onnx` (109 KB)**
   - **Trained:** 2026-11-06
   - **Performance:** **96.2% accuracy** across 12 diverse test signers.
4. **`isl_hindi_vowels.onnx` (105 KB)**
   - **Trained:** 2026-11-06
   - **Performance:** **98.1% accuracy** on Hindi vowel signs.
5. **`isl_numerals.onnx` (105 KB)**
   - **Trained:** 2026-11-06
   - **Performance:** **97.4% accuracy** on numbers 0 through 9.

#### Previous Historical Weights (Archive)
- **`realsign_26class_mlp.pth.bak` (2026-10-30):** Single-signer model (88.4% accuracy). Archived because it overfit to one person's hand dimensions.
- **`isl_sequence_lstm.pth` (2026-10-28):** Early 1-layer unidirectional LSTM (82.1% accuracy). Archived because it lacked bidirectional context and inter-hand distance vectors.
- **`hf_isl_26class_mlp.pth` (2026-10-26):** Trained on uncleaned HuggingFace web data (76.5% accuracy). Archived due to noisy z-depth coordinates.
- **`isl_mvp_model.npz` (2026-10-26):** Initial proof-of-concept linear model (68.0% accuracy).

---

### The Two Halves of the Brain: Vision vs NLP

Students often get confused about what Computer Vision does versus what Natural Language Processing (NLP) does in this project:

```
+------------------------------------+------------------------------------+
|       COMPUTER VISION (The Eyes)   |          NLP (The Brain)           |
+------------------------------------+------------------------------------+
| - MediaPipe Landmark Extraction    | - Spelling Correction & Levenshtein|
| - 63-dim Canonical Normalization   | - Predictive Word Completion       |
| - RealSign MLP & BiLSTM Inference  | - ISL Gloss to English SOV Grammar |
| - Answers: "What hand shape is this?| - Indic Translation (Hindi/Tamil)  |
|   (e.g., Output: [C, A, T])        | - Answers: "What sentence is this? |
|                                    |   (e.g., Output: "The cat is here")|
+------------------------------------+------------------------------------+
```
- **Computer Vision** converts physical hand gestures into raw sign tokens.
- **NLP** takes those raw tokens, fixes any typo or missed letter, applies grammar rules, and translates them into natural Indian languages.

