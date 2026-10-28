# SETU Frontend UI Workstation

## 1. Overview and Architecture

The **SETU Frontend Workstation** is a real-time, bidirectional accessibility dashboard built with **React 19**, **TypeScript**, and **Vite**. It connects to the FastAPI backend via low-latency WebSockets to provide continuous Indian Sign Language (ISL) recognition, real-time skeletal visualization, grammatical translation, and dual-directional communication for Deaf and hearing users.

```
                      +-----------------------------+
                      |   HTML5 Camera Feed         |
                      |   (640x480 @ 30 FPS)        |
                      +--------------+--------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|  useRecognitionStream (Custom Reactive Hook)                            |
|  - Frame downsampling (15-20 FPS)                                       |
|  - JPEG Base64 encoding                                                 |
|  - Full-duplex WebSocket: ws://localhost:8000/api/v1/ws/recognize       |
+------------------------------------+------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+-------------------------------+           +-------------------------------+
|  CameraWorkspace              |           |  TranslationPanel             |
|  - HTML5 <video> Stream       |           |  - Recognized Sign & Confidence|
|  - 2D Canvas Skeletal Overlay |           |  - SVO Sentence Accumulator   |
|  - Movie-Style Subtitle HUD   |           |  - Web Speech API (TTS Engine)|
+-------------------------------+           +-------------------------------+
```

---

## 2. Key Technical Stack

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **Framework** | React 19 (`react`, `react-dom`) | Modern component lifecycle and reactive state |
| **Language** | TypeScript 5.8 | Strict compile-time typing across all schemas |
| **Bundler** | Vite 6 | Lightning-fast HMR and optimized production bundle |
| **Styling** | Tailwind CSS v4 & Lucide Icons | Ergonomic dark workspace theme (Deep Slate / Dark Charcoal) |
| **Vision Overlay** | HTML5 2D Canvas API | High-performance 60 FPS skeletal landmark rendering |
| **Speech** | Web Speech API (`SpeechSynthesis`) | Instant browser-native auditory playback for hearing users |

---

## 3. Directory and Component Structure

```text
frontend/src/
├── components/
│   ├── CameraWorkspace.tsx          # Video viewport, canvas overlays, and HUD subtitles
│   ├── TranslationPanel.tsx         # Sign display, confidence bar, accumulator, TTS
│   ├── Navbar.tsx                   # Model selector, direction toggle, FPS monitor
│   ├── AlphabetChartModal.tsx       # 26-class ISL finger-spelling visual reference
│   ├── VocabularyModal.tsx          # Searchable 263-class INCLUDE dictionary
│   ├── VideoUploadModal.tsx         # File drag-and-drop for offline video recognition
│   └── GestureCalibrationModal.tsx  # Interactive webcam custom sign recorder (5-10 takes)
│
├── hooks/
│   ├── useRecognitionStream.ts      # WebSocket connection, reconnection, frame dispatch
│   ├── useSentenceBuilder.ts        # Streak thresholding, word buffering, backspace
│   └── useTTS.ts                    # Speech synthesis trigger with speech queue management
│
├── services/
│   └── apiService.ts                # Centralized typed HTTP client for /api/v1 endpoints
│
├── utils/
│   └── landmarkDrawing.ts           # Skeletal bone connectives, joint circles, color maps
│
├── types/
│   └── index.ts                     # TypeScript interfaces for WebSocket payloads & models
│
├── App.tsx                          # Root workspace layout and modal orchestration
└── main.tsx                         # React entrypoint with StrictMode
```

---

## 4. Skeletal Landmark Drawing Specifications

To ensure high visual contrast across varying camera lighting environments, skeletal overlays are drawn directly onto an HTML5 2D canvas aligned to the camera feed:

| Skeletal Entity | Joint Color | Bone Connective Style | Radius / Width |
| :--- | :--- | :--- | :--- |
| **Left Hand** | Sage Green (`#81C995`) | Solid Line (`rgba(129, 201, 149, 0.7)`) | 4px Joint / 2px Bone |
| **Right Hand** | Slate Cyan (`#78A9FF`) | Solid Line (`rgba(120, 169, 255, 0.7)`) | 4px Joint / 2px Bone |
| **Pose / Arms** | Slate Grey (`#9AA0A6`) | Dashed Line (`rgba(154, 160, 166, 0.5)`) | 3px Joint / 1.5px Bone |

---

## 5. Getting Started & Development

### Prerequisites
- **Node.js:** v18.0.0 or higher
- **npm:** v9.0.0 or higher
- Active SETU FastAPI backend running on `http://localhost:8000`

### Installation and Launch

```bash
# 1. Navigate to the frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start local development server (with HMR)
npm run dev
```

The application will be accessible at [**`http://localhost:5173`**](http://localhost:5173).

### Production Build

```bash
# Validate TypeScript types and bundle production assets
npm run build

# Preview the production build locally
npm run preview
```

---

## 6. Accessibility & Performance Benchmarks

1. **High Refresh Rate:** Video and canvas processing operates asynchronously from WebSocket network dispatch, ensuring the local viewfinder maintains **60 FPS** without stuttering.
2. **Network Bandwidth Optimization:** Frames are compressed to `image/jpeg` at 0.65 quality before transmission, consuming less than **1.2 Mbps** of network bandwidth at 18 FPS.
3. **Screen Reader Compatibility:** Critical prediction state changes in `TranslationPanel` utilize `aria-live="polite"` regions so assistive technology users receive real-time updates.
