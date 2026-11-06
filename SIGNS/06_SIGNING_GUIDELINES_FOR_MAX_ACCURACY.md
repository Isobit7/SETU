# SETU — Practical Signing Guidelines for Maximum Recognition Accuracy

To achieve **95%+ real-time accuracy** on standard laptop webcams, signers should follow these gestural and spatial guidelines.

---

## 📷 1. Camera Distance & Torso Framing

* **Optimal Distance:** Sit approximately **1.5 to 2.5 feet (45–75 cm)** directly in front of the laptop webcam.
* **Vertical Framing (Waist-Up):** 
  - Ensure the webcam captures from the **top of your head down to your mid-torso / stomach**.
  - Dual-handed consonants in ISL (`B`, `D`, `P`, `Q`, `T`, `M`, `N`) require both hands to be completely within the camera frame simultaneously. If one hand is partially cropped at the bottom edge, MediaPipe landmark confidence drops precipitously.

---

## 💡 2. Lighting & Background Environment

* **Front-Facing Diffuse Light:** Ensure soft light illuminates your face and hands from the front (or slightly from the sides).
* **Avoid Strong Backlight:** Do **NOT** sit with a bright window, lamp, or spotlight directly behind you. Strong backlighting causes camera auto-exposure to underexpose hands into dark silhouettes, destroying landmark joint detection.
* **Background Contrast:** A neutral, non-cluttered wall provides the cleanest edge contrast for finger contour segmentation.

---

## ⏱️ 3. Timing: Hold Duration vs. Fluid Signing

* **Static Letters & Numbers (0.2s–0.3s Hold):**
  - When fingerspelling words, hold each letter configuration steady for **approximately 200–300 milliseconds**.
  - The internal Exponential Moving Average (EMA) filter ($S_t = 0.4 P_t + 0.6 S_{t-1}$) requires ~3–4 consecutive stable frames to eliminate false transitional noise and lock onto the decisive sign.
* **Continuous Co-articulation (CTC Greedy Decoder):**
  - You do **not** need to drop your hands to your lap between letters.
  - When spelling `"BOOK"`, hold `"O"` for 0.8s; the CTC decoder automatically handles double letters without spamming duplicate characters.

---

## ✋ 4. Distinguishing Letters from Whole Words

| You Want To Sign... | How To Do It | Pipeline Mechanism |
|---|---|---|
| **A Specific Letter (e.g. 'A')** | Form the handshape and hold it steady in front of your chest for 0.3s. | The **RealSign Alphabet Classifier** emits `Letter A` with high confidence. |
| **A Whole Phrase (e.g. 'HELLO')** | Perform the complete waving motion at temple level in one continuous stroke. | The **BiLSTM Phrase Model** detects kinetic motion onset, follows trajectory, and emits `HELLO`. |
| **A Fingerspelled Word (e.g. 'WATER')** | Fingerspell `W` &rarr; `A` &rarr; `T` &rarr; `E` &rarr; `R`, then pause hands for 1.2s. | The **SentenceAccumulator** buffers the letters, auto-corrects typos, and emits `"WATER"`. |

---

## 🔄 5. Left-Handed vs. Right-Handed Signers (Mirror Mode)

* The dominant hand in standard ISL training data is the **Right Hand** (making contact with the base Left Hand).
* If you are a naturally left-handed signer:
  - Toggle **Mirror Mode** on the frontend HUD.
  - SETU's internal symmetry normalizer will invert horizontal landmark coordinates ($x \to -x$), allowing natural left-handed signing with zero accuracy degradation.

---

## 🛑 6. The Neutral Rest Pose (Between Sentences)

* Between distinct sentences or phrases, rest your hands down near your lap or lower torso.
* This returns the kinetic energy indicator to **Idle** ($E_k < 0.02$) and prompts the SentenceAccumulator to finalize sentence boundary translation into English and Indic languages.
