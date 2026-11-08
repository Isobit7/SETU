# 02. Frontend, Practice Studio & 3D Hand Model Explained

---

### Frontend Architecture Overview

The frontend is a single-page web application built with **React 18**, **TypeScript**, and **Vite**. It is designed to be responsive, dark-themed, and responsive to live camera streams without freezing the user interface.

```
frontend/src/
├── App.tsx                        # Master layout & state coordinator
├── components/
│   ├── PracticeStudio.tsx         # Interactive practice & learning mode
│   ├── Practice3DHandViewer.tsx   # 3D Skeletal perspective hand visualizer
│   ├── SettingsModal.tsx          # Clean 4-tab settings modal (Apple/Linear style)
│   ├── CameraWorkspace.tsx        # Video canvas, framing guide & overlay
│   ├── TranslationPanel.tsx       # Live transcript & Indic language output
│   └── TopNavbar.tsx              # Clean top navigation with status indicators
├── hooks/
│   ├── useRecognitionStream.ts    # WebSocket connection & 30 FPS frame capture
│   ├── useSentenceBuilder.ts      # Combines signs into grammatically correct sentences
│   └── useTTS.ts                  # Multi-lingual voice synthesis
└── services/
    └── apiService.ts              # REST client for backend health, models & vocabulary
```

---

### How Practice Studio Works

The **Practice Studio** allows anyone to learn and practice Indian Sign Language with real-time AI scoring.

1. **Target Selection:** The user selects a letter (A through Z) or a conversational phrase.
2. **Visual 3D Guide:** The user sees a 3D skeletal hand model demonstrating the exact hand curve and finger placement.
3. **Live Camera Verification:**
   - The user forms the sign in front of their webcam.
   - The AI classifier checks if the detected sign matches the target sign with at least 45% confidence.
4. **1-Second Hold Timer:**
   - Anyone can accidentally flash a hand shape for 0.05 seconds. To ensure the user has genuinely mastered the sign, a dynamic progress bar fills up as the user holds the sign steady for **1.0 second**.
5. **Streak & Audio Feedback:**
   - Once held for 1 second, a pleasant chime plays, Text-to-Speech announces "Correct!", the streak counter increments (+1), and the studio automatically advances to the next sign.

---

### How the 3D Skeletal Hand Model Works

In [`Practice3DHandViewer.tsx`](file:///c:/MINI%20PROJ-SETU/frontend/src/components/Practice3DHandViewer.tsx):
- Instead of using a static, flat 2D photograph that only shows one angle, we built an authentic **3D skeletal perspective hand model** directly on an HTML5 `<canvas>`.
- **True 3D Spatial Geometry:**
  - Every joint is defined by 3D coordinates $(X, Y, Z)$.
  - For example, for the letter **'C'**, the thumb curves downward and arches forward, while the four fingers arch upward and curve over the top, forming an open cylindrical arch in 3D space.
- **Interactive Controls:**
  - **Mouse Drag / Touch Drag:** You can click and drag the hand in any direction to rotate it in 3D (yaw and pitch rotation).
  - **Angle Preset Buttons:**
    - `3/4 View`: Best perspective showing both the palm and finger depth.
    - `Side Profile`: Shows the curvature of the fingers and thumb arch.
    - `Front Palm`: Direct view facing the signer's palm.
    - `Signer POV`: Shows what the hand looks like looking down from your own eyes.
- **Depth Shading:**
  - Joints closer to the camera are rendered larger and brighter, while joints further back are smaller and shaded with subtle perspective scaling.

---

### The Redesigned Settings Modal

Following modern enterprise standards (similar to Apple System Preferences and Google Meet):
- **Tab 1: Video & Camera:** Select webcam hardware, resolution (720p vs 1080p), flip/mirror mode, and guide overlays.
- **Tab 2: Recognition:** Adjust AI confidence threshold, toggle dual-hand mode, and select between Cloud Deep Model or Browser ONNX Edge Model.
- **Tab 3: Speech & Audio:** Choose Text-to-Speech voice, speech speed, volume, and automatic pronunciation upon sign detection.
- **Tab 4: Captions & Display:** Configure subtitle font size, caption persistence duration, and high-contrast dark theme mode.

