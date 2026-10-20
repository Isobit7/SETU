# SETU — Project Structure & Beginner-Friendly Navigation Guide

**Welcome to SETU!**  
If you are learning while building, this guide is designed specifically for you. You do not need to memorize complex file trees or guess what files do. Everything in SETU has a specific, clear purpose and a designated home.

---

## 1. Quick "Where Do I Look?" Reference Table

When you want to work on a specific part of SETU, use this cheat sheet to jump directly to the right folder:

| What you want to do | Which folder to open | Key files to look at |
| :--- | :--- | :--- |
| 📖 **Read requirements, rules, architecture & design** | [`Project Main Files/`](file:///c:/MINI%20PROJ-SETU/Project%20Main%20Files/) | `README.md`, `PRD.md`, `architecture.md`, `rules.md` |
| 📹 **Work with the camera or live video feed** | [`setu/vision/`](file:///c:/MINI%20PROJ-SETU/setu/vision/) | `camera.py` |
| 🖐️ **Work with MediaPipe hand & body landmarks** | [`setu/vision/`](file:///c:/MINI%20PROJ-SETU/setu/vision/) | `landmarks.py` |
| 📐 **See how landmarks turn into 158 numbers** | [`setu/features/`](file:///c:/MINI%20PROJ-SETU/setu/features/) | `schema.py`, `extractor.py` |
| 🧠 **See the sign recognition AI model & predictions** | [`setu/models/`](file:///c:/MINI%20PROJ-SETU/setu/models/) | `isl_mvp_classifier.py` |
| ⚙️ **Change camera resolution, FPS, or thresholds** | [`configs/`](file:///c:/MINI%20PROJ-SETU/configs/) | `settings.py`, `.env.example` |
| 🎥 **Look at training videos or processed numbers** | [`data/`](file:///c:/MINI%20PROJ-SETU/data/) | `data/raw/include/`, `data/manifests/` |
| 🏋️ **Train or re-train the AI model** | [`training/`](file:///c:/MINI%20PROJ-SETU/training/) | `train_mvp_model.py` |
| 🚀 **Run the live webcam recognition program** | [`scripts/`](file:///c:/MINI%20PROJ-SETU/scripts/) | `demo_isl_mvp_webcam.py` |
| 🧪 **Run or add automated tests** | [`tests/`](file:///c:/MINI%20PROJ-SETU/tests/) | `test_isl_mvp_model.py`, `test_vision.py` |
| 💥 **Investigate a crash, bug, or error message** | [`setu/exceptions.py`](file:///c:/MINI%20PROJ-SETU/setu/exceptions.py) | Check `SETUError` classes and `tests/` |

---

## 2. The Complete Project Directory Tree

Below is the complete repository layout. Notice that the names are standard, clean, and follow standard Python and machine learning conventions:

```
c:\MINI PROJ-SETU\
│
├── Project Main Files/            # 📚 The central reference hub (All documentation & rules)
│   ├── README.md                  # Quick navigation map for all documentation
│   ├── PRD.md                     # Product Requirements: What we are building & why
│   ├── architecture.md            # System Architecture: How all the parts fit together
│   ├── design.md                  # Design Principles: Visual guidelines and color tokens
│   ├── translator_workspace_design.md # Future Web Workspace: Video framing & subtitle layout
│   ├── button.md                  # Button Inventory: What every button will do
│   ├── memory.md                  # Project Memory: Log of major decisions and milestones
│   ├── phase.md                   # Phased Roadmap: P0 (Audit) through P7 (Advanced)
│   ├── rules.md                   # Development Rules: Quality standards and rules for AI
│   ├── PROJECT_STRUCTURE.md       # (This file) Plain-English structure & navigation guide
│   ├── KEY_FILES_INDEX.md         # Direct clickable links to every important source file
│   ├── PROJECT_STATUS.md          # Current working features, test results & blockers
│   └── assets/                    # Concept pictures and diagrams
│
├── setu/                          # 🐍 Core Python application package (the engine)
│   ├── __init__.py                # Package root
│   ├── api/                       # 🌐 FastAPI WebSocket & REST endpoints (Web integration)
│   │   ├── __init__.py
│   │   └── server.py              # Real-time WebSocket server connecting web client to vision & model
│   ├── exceptions.py              # Custom error types (CameraError, ModelArtifactError, etc.)
│   ├── features/                  # Feature Engineering (turning body landmarks into numbers)
│   │   ├── __init__.py
│   │   ├── extractor.py           # FeatureExtractor158: does the math to extract 158 values
│   │   └── schema.py              # The dictionary defining what each of the 158 numbers means
│   ├── models/                    # AI Model Inference Code (runs inside the app)
│   │   ├── __init__.py
│   │   ├── isl_mvp_classifier.py  # The primary 5-sign ISL model (<0.2ms, pure NumPy)
│   │   └── rf_uv11_adapter.py     # Experimental single-hand model kept isolated for reference
│   ├── utils/                     # Reusable Math Helpers
│   │   ├── __init__.py
│   │   └── geometry.py            # Angles, distances, and centering coordinates
│   └── vision/                    # Seeing the World (Camera & MediaPipe)
│       ├── __init__.py
│       ├── camera.py              # CameraReader: safely opens, mirrors, and reads your webcam
│       └── landmarks.py           # HolisticLandmarkerWrapper: runs MediaPipe to find hands & body
│
├── frontend/                      # 💻 React + TypeScript + Vite Web Application
│   ├── index.html                 # Main web page entry
│   ├── package.json               # Frontend dependencies (React 19, Lucide icons, Vite)
│   └── src/
│       ├── App.tsx                # Main translator workspace page
│       ├── index.css              # Custom styling with dark navy sidebar & movie subtitles
│       ├── types.ts               # TypeScript contracts for recognition, landmarks & state
│       ├── components/            # Reusable UI widgets
│       │   ├── CameraWorkspace.tsx # Live video, real landmarks canvas, movie subtitle band
│       │   ├── CameraControls.tsx  # Start, Pause, Stop, Reset, Landmark toggle buttons
│       │   ├── TranslationPanel.tsx # Output card, Copy, Speech audio, session history
│       │   ├── Sidebar.tsx         # Deep navy navigation bar with SETU branding
│       │   ├── TopBar.tsx          # Language direction, backend connection pill
│       │   ├── StatusStrip.tsx     # Hardware status, framing guidance, latency counter
│       │   ├── VocabularyModal.tsx # Dialog explaining 5 supported ISL signs & tips
│       │   └── SettingsModal.tsx   # Subtitle size, landmark layers, sensitivity slider
│       ├── hooks/
│       │   └── useRecognitionStream.ts # WebSocket stream hook with auto-reconnect & frame capture
│       └── utils/
│           └── landmarksDrawer.ts  # HTML5 Canvas real-time landmark & skeleton bone renderer
│
├── configs/                       # ⚙️ Application Settings
│   ├── __init__.py
│   └── settings.py                # Default settings (camera resolution, confidence threshold)
│
├── data/                          # 📁 Datasets and Processed Features
│   ├── manifests/                 # JSON files tracking which videos belong to which signer & split
│   │   └── setu_isl_mvp_manifest.json
│   ├── processed/                 # .npz files containing extracted numbers ready for AI training
│   └── raw/                       # Original, untouched source videos
│       ├── include/               # Authentic Indian Sign Language videos (AI4Bharat INCLUDE)
│       └── real_no_sign/          # Real webcam videos of resting hands and idle behavior
│
├── models/                        # 💾 Stored AI Model Weight Files (binary files)
│   ├── isl_mvp_model.npz          # The trained weights for our 5-sign ISL classifier
│   ├── pretrained/                # Reference weights from external models (like RF-UV-11)
│   ├── reference/                 # Legacy model weights
│   └── tasks/                     # MediaPipe model bundles (holistic_landmarker.task)
│
├── scripts/                       # 💻 Runnable Programs & Demonstrations (Launch these!)
│   ├── demo_isl_mvp_webcam.py     # ✨ THE MAIN DEMO: Runs real-time ISL recognition on your webcam
│   ├── demo_rf_uv11_webcam.py     # Reference demo for single-hand alphabet model
│   ├── download_include_subset.py # Helper to download authentic ISL videos from Zenodo
│   ├── process_include_subset.py  # Helper to process downloaded videos into .npz feature files
│   ├── run_controlled_alphabet_test.py # Automated testing script for single-hand model
│   └── webcam_physical_verification.py # Helper to verify camera capture works
│
├── tests/                         # 🧪 Automated Test Suite (Guarantees nothing breaks)
│   ├── test_config_exceptions.py  # Tests settings and error handling
│   ├── test_feature_schema.py     # Tests that 158-feature math is accurate and bug-free
│   ├── test_isl_mvp_model.py      # Tests the 5-sign ISL model on real and empty inputs
│   ├── test_pipeline.py           # Tests video processing and uniform resampling
│   ├── test_real_data_readiness.py# Tests that dataset has no signer overlap between train & test
│   ├── test_rf_uv11_adapter.py    # Tests isolated RF-UV-11 model
│   └── test_vision.py             # Tests camera opening and MediaPipe landmarker
│
├── training/                      # 🏋️ Training Code (Used when creating or retraining models)
│   ├── __init__.py
│   ├── dataset_manifest.py        # Logic for creating and validating dataset records
│   ├── extract_features.py        # Pipeline to convert raw video files into 158-D sequences
│   └── train_mvp_model.py         # Trains the pure NumPy classifier and reports accuracy
│
├── docs/                          # 📝 Historical milestone notes and research reports
├── .env.example                   # 🔒 Safe template for setting environment variables
├── pyproject.toml                 # 📦 Python project settings and list of dependencies
└── README.md                      # 🏁 Main project welcome and setup instructions
```

---

## 3. Detailed Plain-English Explanation of Important Folders

### `Project Main Files/`
* **What it is:** The official bookshelf of the project.
* **Why it matters:** Contains all rules, design guidelines, requirements, and historical decisions.
* **Beginner Tip:** Read `README.md` in this folder first! It links to all the other documents. Never put Python code or data files in this folder—it is for documentation only.

### `setu/`
* **What it is:** The core Python package. When you write `import setu`, Python looks inside here.
* **Why it matters:** Contains the actual reusable translation engine.
* **Subfolders:**
  * `setu/vision/`: Code that talks to hardware (cameras) and computer vision models (MediaPipe).
  * `setu/features/`: Code that turns visual hand landmarks (x, y, z dots) into a standard vector of 158 numbers.
  * `setu/models/`: Code that takes those numbers and predicts which sign was performed.
  * `setu/utils/`: Helper math functions like calculating angles between fingers and distances.

### `configs/`
* **What it is:** The control center for settings.
* **Why it matters:** Instead of hardcoding numbers like `width = 640` or `confidence = 0.5` inside five different files, we define them once in `configs/settings.py`. You can also override them using environment variables.

### `data/`
* **What it is:** Where training videos and numbers live.
* **Subfolders:**
  * `data/raw/`: Original, unmodified video files (MP4/MOV). We never edit raw files directly.
  * `data/processed/`: The numbers extracted from those videos (`.npz` files). These are fast to load during training.
  * `data/manifests/`: JSON spreadsheets that list every sample, its true label, who signed it, and whether it belongs to the training, validation, or test group.

### `models/`
* **What it is:** The model storage vault.
* **Why it matters:** Holds the saved binary weights (the "brain") produced after training. Code never lives here; only model weights and task bundles (`.npz`, `.h5`, `.task`).

### `scripts/`
* **What it is:** Your toolbox of runnable CLI programs.
* **Why it matters:** If you want to run something from your PowerShell terminal (like testing your webcam or launching the demo), this is where the entry points live.
* **Top Script:** `scripts/demo_isl_mvp_webcam.py` runs the live translator.

### `tests/`
* **What it is:** The automated safety net.
* **Why it matters:** Running `pytest` runs every file in this folder. If you change a piece of code and accidentally break something, a test will immediately turn red and tell you what went wrong.

### `training/`
* **What it is:** The workshop where models are built.
* **Why it matters:** Contains scripts for batch feature extraction and training algorithms. Once a model is trained, its saved weights are saved to `models/`, and its inference code is placed in `setu/models/`.

---

## 4. Where to Look When Something Breaks

If you run into an error, here is where to look first:

1. **Camera won't open or crashes:**
   * Look at: [`setu/vision/camera.py`](file:///c:/MINI%20PROJ-SETU/setu/vision/camera.py).
   * Check if your camera index is wrong (try `--camera 1` or change `SETU_CAMERA_INDEX` in `.env`).
2. **MediaPipe fails to detect landmarks:**
   * Look at: [`setu/vision/landmarks.py`](file:///c:/MINI%20PROJ-SETU/setu/vision/landmarks.py).
   * Check if `models/tasks/holistic_landmarker.task` is missing.
3. **Features or numbers have the wrong shape:**
   * Look at: [`setu/features/schema.py`](file:///c:/MINI%20PROJ-SETU/setu/features/schema.py) and [`setu/features/extractor.py`](file:///c:/MINI%20PROJ-SETU/setu/features/extractor.py).
   * Check if `FEATURE_DIM` is exactly 158.
4. **Model predictions are wrong or confidence is low:**
   * Look at: [`setu/models/isl_mvp_classifier.py`](file:///c:/MINI%20PROJ-SETU/setu/models/isl_mvp_classifier.py).
   * Check if `models/isl_mvp_model.npz` is loaded and the confidence threshold (`0.50`) is appropriate.
5. **A test fails:**
   * Run: `.venv\Scripts\python -m pytest -v`
   * Read the test name to find the corresponding file in `tests/`.

---

## 5. Which Folders to Normally Leave Untouched

To keep your project healthy and stable, you should rarely or never need to manually edit:

* `data/raw/`: Original videos should never be modified by hand.
* `data/processed/`: These files are automatically generated by `scripts/process_include_subset.py`.
* `models/tasks/`: Contains Google's pre-compiled MediaPipe binary bundle (`holistic_landmarker.task`).
* `models/pretrained/`: Contains reference models kept for baseline comparisons.
* `.venv/`: Managed entirely by Python/pip. Never edit files inside `.venv` manually.

---

## 6. Where Newly Developed Features Should Be Added

When it is time to build something new, follow this simple placement rule:

* **Adding a new gesture math helper?** $\rightarrow$ Put it in `setu/utils/geometry.py`.
* **Adding a new sign recognition model?** $\rightarrow$ Put the inference code in `setu/models/` and save the trained weights in `models/`.
* **Adding a new automated test?** $\rightarrow$ Create a new test file named `test_<feature>.py` in `tests/`.
* **Adding a new runnable utility or experiment?** $\rightarrow$ Put the executable script in `scripts/`.
* **Adding a new training technique?** $\rightarrow$ Put it in `training/`.

---

## 7. Planned Future Folders (And When to Create Them)

As SETU progresses through future phases, new directories will be introduced. **Do not create these folders as empty placeholders today**; create them only when beginning the corresponding phase:

| Planned Directory | Phase | Purpose | Technology |
| :--- | :--- | :--- | :--- |
| `web/` or `frontend/` | **Phase P5** | The user-facing web translator interface featuring the movie-style subtitle bar and camera viewport. | React / Vite / TypeScript |
| `api/` or `setu/api/` | **Phase P5** | The backend REST or WebSocket API serving real-time predictions to the web browser. | FastAPI / WebSockets |
| `setu/translation/` | **Phase P6** | Sentence construction, grammar mapping, and word smoothing (turning sequences of signs into fluent English/Hindi). | Python State Machine / NLP |
| `setu/speech/` | **Phase P7** | Text-to-speech (TTS) audio synthesis that reads translated signs out loud. | Web Speech API / TTS engine |
| `storage/` | **Phase P7** | Local database storing user preferences, saved translations, and session logs. | SQLite / IndexedDB |
| `deploy/` | **Phase P7** | Docker container setups and production deployment scripts. | Docker / Nginx |

---

## 8. Summary: Why We Kept Existing Names

All existing folder names (`setu/`, `vision/`, `features/`, `models/`, `configs/`, `data/`, `training/`, `scripts/`, `tests/`) were carefully evaluated.  
**They are already industry-standard, concise, and clean.**  
Renaming `setu` to `setu_core_application` or `tests` to `automated_testing_scripts` would break Python package imports, pytest discovery, and command-line habits without adding real clarity. Instead, this navigation guide provides the plain-English bridge you need to feel completely confident exploring the code.
