# SETU — Official Indian Sign Language (ISL) Master Sign Catalog & Dictionary

Welcome to the **SETU Sign Dictionary & Gestural Reference Library**. 
This directory contains the definitive, production-verified catalog of all sign language categories, finger postures, hand gestures, and dynamic phrases supported and recognized by the SETU AI Workstation.

---

## 📚 Master Catalog Index

| Category | Class Count | Model Engine | Modality | Document Link |
|---|:---:|---|---|---|
| **Dual-Hand Alphabet (A–Z)** | **26 Signs** | `realsign_multisigner` (PyTorch CUDA & ONNX) | Static 3D Hand Landmarks (126-D) | [01_ALPHABET_A_TO_Z.md](01_ALPHABET_A_TO_Z.md) |
| **ISL Numerals (0–9)** | **10 Signs** | `isl_numerals` (PyTorch MLP & ONNX) | Static Dual-Hand Landmarks (126-D) | [02_NUMERALS_0_TO_9.md](02_NUMERALS_0_TO_9.md) |
| **Indic Hindi Vowels (अ–ऐ)** | **8 Signs** | `isl_hindi_vowels` (PyTorch MLP & ONNX) | Static Dual-Hand Landmarks (126-D) | [03_HINDI_VOWELS.md](03_HINDI_VOWELS.md) |
| **Conversational Phrases** | **49 Phrases** | `amrita_isl_phrase_bilstm` & `isl_sequence_lstm` | Continuous Spatio-Temporal Video (316-D) | [04_CONVERSATIONAL_PHRASES.md](04_CONVERSATIONAL_PHRASES.md) |
| **Large-Vocabulary Signs** | **263 Concepts** | `ai4bharat_include_263sign` (Deep BiLSTM) | 134-D Holistic Body/Hands Keypoints | [05_LARGE_VOCABULARY_INCLUDE_263.md](05_LARGE_VOCABULARY_INCLUDE_263.md) |
| **Signing Best Practices** | Guidelines | System-Wide | Webcam Framing, Lighting, Timing | [06_SIGNING_GUIDELINES_FOR_MAX_ACCURACY.md](06_SIGNING_GUIDELINES_FOR_MAX_ACCURACY.md) |

---

## 🏛️ The Distinction: Alphabet (Fingerspelling) vs. Whole Words (Phrases)

A common question from users and researchers is:
> *"If I sign letter 'A', how does the workstation know I am showing letter 'A' rather than a whole word like 'HELLO' or 'APPLE'?"*

SETU solves this through a **two-tier multimodal pipeline**:

```
                              [Live Video Stream]
                                       |
                +----------------------+----------------------+
                |                                             |
     [Static Hand Framing]                          [Dynamic Body Motion]
                |                                             |
       MediaPipe HandLM (15ms)                      MediaPipe Holistic (35ms)
                |                                             |
     Wrist-Normalized 126-D Vector                Spatio-Temporal Sequence (T, 316-D)
                |                                             |
   RealSign 26-Class Alphabet MLP                 Amrita BiLSTM Phrase Model
                |                                             |
     Emits: Single Letter ('A')                  Emits: Dynamic Word ('HELLO')
                |                                             |
                +----------------------+----------------------+
                                       |
                        [SentenceAccumulator Buffer]
                                       |
                     - Letters: Accumulated into Words (H-E-L-P -> "HELP")
                     - Words: Emitted as Whole Tokens Immediately
                                       |
                        [ISL Phonological Spell-Corrector]
                                       |
                        [Multilingual Translation Engine]
                         (English SVO + Hindi / Tamil / Telugu / Kannada)
```

1. **Static Fingerspelling:**
   When you hold an individual handshape (like `A`, `B`, `C`), the **RealSign Alphabet Classifier** activates. It outputs a decisive single character with high confidence (e.g., `Letter A: 98.8%`).
2. **Dynamic Whole Words:**
   When you perform whole-arm or torso motion trajectories (like waving for `HELLO`, placing a flat hand to the chin for `THANK YOU`, or tapping the chest for `HELP`), the **BiLSTM Temporal Sequence Model** activates, recognizing the entire concept in one stroke.
3. **Fingerspelled Word Synthesis:**
   If you spell an uncommon word or proper name (e.g. `S-A-T-H-W-I-K`), the **SentenceAccumulator** buffers each letter. After an idle pause (1.2 seconds), it commits the fingerspelled word and applies the ISL phonological spell-corrector to fix any rapid-signing typos.

---

## 📊 Summary of Production Verification Scores

All models in this catalog have been verified against strictly unseen test sets (80/10/10 split):
- **Alphabet (26 Letters):** **92.00% Top-1**, **97.33% Top-3**
- **Numerals (0–9):** **94.08% Top-1**, **98.95% Top-3**
- **Hindi Vowels (8 Vowels):** **94.71% Top-1**, **97.60% Top-3**
- **Conversational Phrases (49 Phrases):** **78.57% Top-1**, **90.82% Top-3**, **94.90% Top-5**
