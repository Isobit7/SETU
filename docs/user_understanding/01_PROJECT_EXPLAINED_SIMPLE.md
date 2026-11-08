# 01. Project SETU: Explained in Simple Words

---

### What Problem Does SETU Solve?

In India, over **18 million people** are deaf or hard of hearing. They primarily communicate using **Indian Sign Language (ISL)**. However, over 99% of hearing people do not know sign language. This creates a massive communication barrier in hospitals, banks, police stations, schools, and workplaces.

Most existing projects only translate simple American Sign Language (ASL). But **Indian Sign Language is fundamentally different**:
1. ISL is predominantly **two-handed** (for letters like B, D, P, T, Q and most everyday words).
2. ISL uses **Subject-Object-Verb (SOV)** grammar, whereas English uses Subject-Verb-Object (SVO).
3. ISL relies heavily on facial expressions (called *non-manual markers*) to indicate questions, negation, and emphasis.

**SETU** acts as a real-time digital interpreter that allows a deaf person to sign naturally to a laptop webcam, immediately converting their gestures into readable text and synthesized voice in multiple Indian languages.

---

### How SETU Works in 4 Simple Steps

```
[Step 1: Webcam Capture]
User stands in front of any normal laptop webcam. 
Frames are captured at 30 FPS.
       ↓
[Step 2: Skeletal Extraction (MediaPipe)]
Instead of sending heavy video to the AI, MediaPipe finds 21 joints on each hand
and 33 joints on the body. We turn the video into lightweight (x, y, z) numbers!
       ↓
[Step 3: Deep Neural Classification]
Our custom deep neural networks (MLP & BiLSTM) analyze the coordinates:
- "Is this the letter 'C'?"
- "Is this the phrase 'Thank you'?"
       ↓
[Step 4: Grammar Translation & Speech]
The detected signs are arranged into proper grammatical sentences 
(English, Hindi, Tamil, Telugu, Kannada) and spoken aloud via Web Speech Text-to-Speech!
```

---

### Key Architectural Strengths

- **Ultra-Fast (Real-Time):** Round-trip latency is only **32 milliseconds**. When you make a sign, the result appears almost instantly.
- **Privacy-Safe:** No video is ever stored on the server. Only mathematical joint coordinates are processed in memory and discarded.
- **Works on Normal Laptops:** Because we use skeletal landmarks rather than raw RGB pixels, you do not need an expensive supercomputer. It runs smoothly on budget GPUs (like NVIDIA RTX 3050) and standard laptop CPUs.
- **Edge ONNX Capable:** Even if you lose internet connection, the basic alphabet recognition can run completely inside your browser using WebAssembly/ONNX!
