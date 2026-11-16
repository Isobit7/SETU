# Curated Technical References & Documentation

> **Audience:** 3rd Year B.Tech AI & ML Student  
> **Verification Guarantee:** Every link listed below is an official, established reference. No fabricated URLs or obsolete tutorials.

---

## 1. Official Framework & Tool Documentation

### PyTorch Core
* **Resource:** PyTorch Official Documentation
* **Concept:** Tensor manipulation, autograd, `nn.Module`, and CUDA device allocation.
* **URL:** https://pytorch.org/docs/stable/index.html
* **Relevance:** Essential for understanding `models/adapters/realsign.py` and PyTorch training loops.
* **Status:** Highly Recommended.

### Google MediaPipe
* **Resource:** MediaPipe Hand Landmark Detection Guide
* **Concept:** 21 anatomical 3D landmark definitions and task graph pipelines.
* **URL:** https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
* **Relevance:** Direct reference for joint indices (`0` to `20`) in `backend/app/services/vision/landmarks.py`.
* **Status:** Required.

### FastAPI & Uvicorn
* **Resource:** FastAPI Official Documentation & Tutorial
* **Concept:** Asynchronous endpoints, WebSockets, request validation with Pydantic, and `lifespan`.
* **URL:** https://fastapi.tiangolo.com/
* **Relevance:** Explains the architecture of `backend/app/main.py`.
* **Status:** Highly Recommended.

### React 19 & TypeScript
* **Resource:** React Official Documentation
* **Concept:** Component state, `useRef`, `useState`, `useEffect`, and custom hooks.
* **URL:** https://react.dev/
* **Relevance:** Understand how `frontend/src/App.tsx` and custom hooks manage live video streams.
* **Status:** Recommended.

### Hugging Face Transformers & Qwen2.5
* **Resource:** Hugging Face Transformers Documentation
* **Concept:** AutoModelForCausalLM, AutoTokenizer, and auto-regressive text generation.
* **URL:** https://huggingface.co/docs/transformers/index
* **Relevance:** Explains how local Qwen SLM generates English translations from sign glosses in `neural_translator.py`.
* **Status:** Optional / Advanced.

---

## 2. Foundational Research Papers

1. **Long Short-Term Memory (LSTM)**
   * *Authors:* Sepp Hochreiter & Jürgen Schmidhuber (1997)
   * *Why it matters:* Introduces the constant error carousel, the cell state, and gating mechanisms that solve vanishing gradients in sequential gesture modeling.
2. **Connectionist Temporal Classification (CTC)**
   * *Authors:* Alex Graves et al. (ICML 2006)
   * *Paper Title:* *"Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"*
   * *Why it matters:* Introduces the blank token ($\epsilon$) that SETU adapts in `TemporalAccumulator` to eliminate character repetition and gesture stuttering.
3. **YOLO-World: Real-Time Open-Vocabulary Object Detection**
   * *Authors:* Tiancheng Cheng et al. (2024)
   * *Why it matters:* Explains vision-language pre-training and how prompt embeddings enable real-time zero-shot detection of physical objects for Show-and-Sign.
4. **Attention Is All You Need**
   * *Authors:* Vaswani et al. (NeurIPS 2017)
   * *Why it matters:* The foundational paper introducing the Transformer architecture and self-attention mechanisms powering modern SLMs like Qwen2.5.

