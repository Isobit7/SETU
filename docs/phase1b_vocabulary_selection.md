# SETU — Phase 1B: Authentic ISL Vocabulary Selection & Dataset Specification

**Document Version:** 1.0.0  
**Status:** Approved Reference  
**Standards Body:** Indian Sign Language Research and Training Centre (ISLRTC)  
**Primary Open Sources:** AI4Bharat INCLUDE (Zenodo 4010759) & Mendeley Data ISL Corpora

---

## 1. Scope & MVP Target Vocabulary

For an achievable, highly reliable college mini-project implementation, an initial vocabulary of **10 authentic Indian Sign Language (ISL) word signs + 1 NO_SIGN background class** (11 total classes) is established.

Every selected sign corresponds to genuine, standardized gestures documented by the **Indian Sign Language Research and Training Centre (ISLRTC)**, avoiding the historical hackathon failure of mixing American Sign Language (ASL) fingerspelling with Indian signs.

| Class ID | Label Name | Sign Type | Authentic ISL Gesture Description (ISLRTC Standard) | Source Dataset & Category |
| :---: | :--- | :---: | :--- | :--- |
| **0** | `hello` | 1-Hand | Open dominant palm facing forward, slight outward lateral greeting movement. | INCLUDE: `Greetings/48. Hello/` |
| **1** | `thank_you` | 1-Hand | Flat dominant hand moves gently from chin forward towards recipient with slight nod. | INCLUDE: `Greetings/86. Thank you/` |
| **2** | `help` | 2-Hand | Dominant closed fist rests on open flat non-dominant palm, both lifting upward together. | INCLUDE: `Society/38. Help/` |
| **3** | `water` | 1-Hand | Three fingers extended ('W' handform) tapping near the mouth or chin. | INCLUDE: `Home/72. Water/` |
| **4** | `father` | 1-Hand | Extended index and middle fingers tapping side of forehead. | INCLUDE: `People/14. Father/` |
| **5** | `mother` | 1-Hand | Open flat hand or thumb/index touching side of chin. | INCLUDE: `People/47. Mother/` |
| **6** | `yes` | 1-Hand | Dominant fist mimics nodding head (vertical wrist bend up and down). | INCLUDE: `Adjectives/118. Yes/` |
| **7** | `no` | 1-Hand / 2-Hand | Dominant index and middle finger snap to thumb, or flat hand waving laterally. | INCLUDE: `Adjectives/119. No/` |
| **8** | `home` | 2-Hand | Both flat hands touch at fingertips forming a triangular roof shape. | INCLUDE: `Home/40. Home/` |
| **9** | `work` | 2-Hand | Dominant wrist gently taps top of non-dominant fist repeatedly. | INCLUDE: `Jobs/18. Work/` & Mendeley `work.zip` |
| **10** | `no_sign` | Background | Hands resting in lap, on table, typing, adjusting face/glasses, or frame empty. | SETU Controlled Generator & Baseline Feed |

---

## 2. Dataset Verification & Provenance

### Primary Dataset: AI4Bharat INCLUDE
- **Creators:** Advaith Sridhar, R. G. Santhosh, Dr. M. Anand Kumar (AI4Bharat / SSN College of Engineering, ACM Multimedia 2020).
- **Official Repository:** [Zenodo Record 4010759](https://zenodo.org/records/4010759)
- **Hugging Face Mirror:** [`ai4bharat/INCLUDE`](https://huggingface.co/datasets/ai4bharat/INCLUDE)
- **Code Repository:** [`github.com/AI4Bharat/INCLUDE`](https://github.com/AI4Bharat/INCLUDE) (MIT License)
- **Dataset License:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Participants:** 7 deaf student signers from St. Louis School for the Deaf, Adyar, Chennai.
- **Recording Spec:** Full HD (1920x1080) at 30 FPS, MOV / MP4 formats.

### Secondary Static Dataset: Mendeley Data ISL
- **Repository:** Mendeley Data (DOI: `10.17632/s6kgb6r3ss.2` and `10.17632/w7fgy7jvs8.3`).
- **License:** **CC BY 4.0** (Open Access).
- **Format:** RGB image sequences recorded across multiple subjects in real-world environments.

---

## 3. Signer-Stratified Splitting & Leakage Prevention

A critical failure point in prior sign recognition projects was randomly shuffling frames across splits, which caused:
1. **Frame Leakage:** Consecutive video frames from the same performance appearing in both training and test sets.
2. **Signer Overfitting:** The neural network memorizing clothing, background, or facial features of signers rather than the actual sign kinematics.

### SETU Splitting Policy
1. **Signer-Level Isolation:**
   - Signers $S_1, S_2, S_3, S_4, S_5$ $\implies$ **Training Set** (~70%)
   - Signer $S_6$ $\implies$ **Validation Set** (~15%)
   - Signer $S_7$ $\implies$ **Test Set** (~15%)
2. **Evaluation Guarantee:** The model is evaluated exclusively on signers it has never encountered during training or hyperparameter tuning.
