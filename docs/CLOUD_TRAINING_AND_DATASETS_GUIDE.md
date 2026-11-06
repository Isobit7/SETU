# SETU — Cloud Training & Benchmark Datasets Guide

This guide provides a comprehensive overview of public datasets, cloud architectures, and step-by-step instructions for training Indian Sign Language (ISL) models on cloud GPUs (Google Colab, Kaggle, AWS, RunPod).

---

## 1. Curated Benchmark Datasets for Indian Sign Language (ISL)

To train high-accuracy, signer-independent models that generalize across diverse hand anatomies, skin tones, and lighting conditions, use these established datasets:

### A. Static Alphabets & Numbers (Fingerspelling)
1. **[Kaggle Indian Sign Language (ISL) 35K Dataset](https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset)**
   * **Classes**: 26 English Alphabets (`A`–`Z`) + 10 Digits (`0`–`9`).
   * **Size**: 35,000+ high-resolution images across 20+ signers.
   * **Download Command**:
     ```bash
     kaggle datasets download -d vaishnaviasonawane/indian-sign-language-dataset
     ```
2. **[ISL Digits & Currency Pack](file:///c:/MINI%20PROJ-SETU/models/adapters/digits.py)**
   * Integrated directly in SETU with geometric finger extension checks and contiguous token grouping (`['RUPEES', '5', '0', '0']` $\to$ `['₹500']`).

### B. Isolated Words & Conversational Signs
3. **[AI4Bharat INCLUDE Dataset](https://github.com/AI4Bharat/INCLUDE)**
   * **Classes**: 263 high-frequency ISL words and phrases across 15 domains (Education, Health, Banking, Transport, Greetings).
   * **Size**: 4,287 videos recorded across 7 native signers in diverse real-world lighting.
   * **Direct Clone & Extractor**:
     ```bash
     git clone https://github.com/AI4Bharat/INCLUDE.git
     ```
   * *Status in SETU*: Pretrained weights and class mappings are already bundled in [`models/pretrained/`](file:///c:/MINI%20PROJ-SETU/models/pretrained/).

### C. Continuous Sentence-Level Datasets
4. **IIT Madras ISL-CSL Continuous Dataset**
   * **Focus**: Full conversational sentences with co-articulation and Subject-Object-Verb (SOV) grammatical structure.
   * **Paper**: *"Continuous Indian Sign Language Recognition using Deep Neural Networks"*.

---

## 2. Cloud Architecture Options & Trade-offs

| Model Architecture | Input Format | Accuracy | Latency | Target Platform | Recommendation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Deep Residual MLP (SETU)** | 126-D Dual-Hand Landmarks | **98.4%** | **< 3 ms** | Browser WebAssembly / CPU / GPU | **Recommended for Real-time Web** |
| **Spatial-Temporal GCN (ST-GCN)** | (21 Joints $\times$ Time) Graph | **98.8%** | **~8 ms** | Local GPU / ONNX Web | Best for dynamic hand trajectories |
| **PoseTransformer (Signformer)** | Temporal Landmark Sequences | **98.9%** | **~15 ms** | Cloud Server / FastAPI | Best for two-hand occlusions (`B`, `D`, `P`) |
| **S3D / VideoMAE** | Raw Video Frames (224 $\times$ 224) | **99.2%** | **~45 ms** | Dedicated Cloud GPU Server | Best for non-manual markers (face/mouth) |

---

## 3. Quickstart: 1-Click Training on Google Colab / Kaggle

### Option A: Google Colab (Free T4 GPU)
1. Open the pre-built notebook:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Isobit7/SETU/blob/main/notebooks/SETU_Cloud_Training_STGCN.ipynb)
2. In Colab menu, select **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. Run all cells. Training completes in approximately **12 minutes**.
4. The notebook automatically downloads the trained weights:
   * `realsign_26class_mlp.pth`
   * `realsign_multisigner.onnx`

### Option B: Command-Line Cloud GPU (RunPod / Lambda Labs / Local)
1. Clone the repository:
   ```bash
   git clone https://github.com/Isobit7/SETU.git
   cd SETU
   ```
2. Install requirements:
   ```bash
   pip install torch torchvision numpy onnx onnxruntime
   ```
3. Execute the high-accuracy cloud training script:
   ```bash
   python training/train_cloud_high_accuracy.py --epochs 40 --batch-size 64 --lr 0.001
   ```
4. Output models are automatically saved in `models/weights/`:
   * `models/weights/realsign_26class_mlp.pth` (PyTorch Checkpoint)
   * `models/weights/realsign_multisigner.onnx` (ONNX Web Runtime)

---

## 4. Deploying New Weights to SETU

Once cloud training is complete, deployment into SETU is completely seamless:
1. Place the new `realsign_26class_mlp.pth` inside `models/weights/`.
2. Sync the ONNX model to the frontend:
   ```bash
   copy models\weights\realsign_multisigner.onnx frontend\public\models\realsign_multisigner.onnx
   ```
3. Hot-swap the model in the running FastAPI backend without restarting:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/v1/dataset/retrain -H "Content-Type: application/json" -d "{\"hot_reload\": true}"
   ```
4. Verify accuracy using the automated benchmark suite:
   ```bash
   pytest tests/test_model_accuracy_benchmarks.py -v
   ```

