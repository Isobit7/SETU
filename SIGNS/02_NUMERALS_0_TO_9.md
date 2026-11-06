# SETU — Indian Sign Language Numerals Catalog (0 to 9)

In Indian Sign Language (ISL), numbers are expressed through standardized hand configurations with specific finger extension counts, thumb positions, and palm orientations.

SETU includes a dedicated, verified numeral classifier (`models/weights/isl_numerals_mlp.pth` and `isl_numerals.onnx`) achieving **94.08% Top-1 Accuracy** and **98.95% Top-3 Accuracy** on unseen test partitions.

---

## 🔢 Digits 0 to 9 Sign Formations & Anatomical Mechanics

| Digit | Hand | Active Fingers | Posture & Finger Mechanics | Key Distinguishing Feature |
|:---:|:---:|---|---|---|
| **0** | Single | All fingers curled | All 5 fingers curled with fingertips touching thumb tip, forming a complete circle/oval. | Enclosed zero loop; palm facing forward. |
| **1** | Single | Index | Index finger extended upright vertically; thumb folds over curled middle, ring, and pinky. | Single vertical finger ($d_{\text{index}} > 0.85$, others $< 0.35$). |
| **2** | Single | Index, Middle | Index and Middle fingers extended vertically; thumb holds ring and pinky against palm. | Two vertical fingers separated or together. |
| **3** | Single | Thumb, Index, Middle | Thumb, Index, and Middle fingers extended spread; ring and pinky folded into palm. | 3 active extended digits including thumb. |
| **4** | Single | Index, Middle, Ring, Pinky | Four main fingers extended upright; thumb tucked tightly across the lower palm. | 4 fingers extended; thumb retracted across palm. |
| **5** | Single | All 5 Fingers | All five fingers fully extended and spread outward away from palm center. | Full open hand; maximum palm surface area. |
| **6** | Single | Index, Middle, Ring, Pinky | Thumb tip touches Pinky tip; Index, Middle, and Ring fingers held extended upward. | Thumb-to-pinky contact loop; 3 fingers upright. |
| **7** | Single | Index, Middle, Pinky | Thumb tip touches Ring fingertip; Index, Middle, and Pinky fingers held extended upward. | Thumb-to-ring contact loop. |
| **8** | Single | Index, Ring, Pinky | Thumb tip touches Middle fingertip; Index, Ring, and Pinky fingers held extended upward. | Thumb-to-middle contact loop. |
| **9** | Single | Middle, Ring, Pinky | Thumb tip touches Index fingertip; Middle, Ring, and Pinky fingers held extended upward. | Thumb-to-index contact loop (similar to "OK" gesture). |

---

## 💰 Multi-Digit Numbers & Currency Integration

SETU's real-time NLP tokenizer and SentenceAccumulator automatically format streaming digit tokens into formatted numbers and currency:
- **Consecutive Digits:** Streaming `["5", "0", "0"]` &rarr; Formatted as `"500"`.
- **Currency Signs:** When signed in conjunction with the ISL "RUPEES" sign, `["RUPEES", "5", "0", "0"]` &rarr; Formatted as `"₹500"`.
- **Time Indicators:** `["TIME", "5", "3", "0"]` &rarr; Formatted as `"5:30"`.
