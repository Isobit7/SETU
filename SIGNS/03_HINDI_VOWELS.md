# SETU — Indian Sign Language Indic Hindi Vowels Catalog (हिंदी स्वर)

Indian Sign Language includes standardized sign formations for the core vowels (Swar) of Indic scripts (Devanagari/Hindi). These enable native vernacular fingerspelling without requiring phonetic transcription into English.

SETU includes a dedicated, verified Indic Hindi vowel classifier (`models/weights/isl_hindi_vowels_mlp.pth` and `isl_hindi_vowels.onnx`) achieving **94.71% Top-1 Accuracy** and **97.60% Top-3 Accuracy** on unseen test partitions.

---

## 🪷 The 8 Indic Vowel Signs (Swar)

| Vowel | Devanagari | Phonetic | Hand Configuration | Dominant & Base Hand Posture |
|:---:|:---:|:---:|---|---|
| **1** | **अ** | /a/ | Dual-Hand | Right index touches base of left palm; thumb extended upright. |
| **2** | **अं** | /am/ (Anusvara) | Dual-Hand + Dot | Handshape for 'अ' followed by right index tapping dot above left hand. |
| **3** | **आ** | /aa/ | Dual-Hand | Right hand open with extended thumb, placed parallel to left hand with an additional vertical stroke. |
| **4** | **इ** | /i/ (Short I) | Single/Dual | Right pinky extended vertically upward while thumb touches base knuckle. |
| **5** | **ई** | /ee/ (Long I) | Dual-Hand | Pinky extended upward with right index adding upper hook arc over left palm. |
| **6** | **ऋ** | /ri/ | Dual-Hand | Three fingers curved over left palm forming stylized "Ri" ligature. |
| **7** | **ए** | /e/ | Dual-Hand | Right index finger touches tip of left index finger. |
| **8** | **ऐ** | /ai/ | Dual-Hand | Right index and middle fingers touch tip of left index finger simultaneously. |

---

## 🌐 Multilingual Integration in SETU

When a signer spells words in Hindi vowels:
- The **RealSign Indic Classifier** decodes native Devanagari Unicode characters directly (`अ`, `आ`, `इ`, etc.).
- The **SentenceAccumulator** buffers the Devanagari characters into words.
- The **IndicTransformer** (`backend/app/services/nlp/indic_transformer.py`) maps the text into grammatically aligned Hindi, Telugu, Tamil, and Kannada translations.
