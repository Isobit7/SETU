# SETU — Model Evaluation Matrices & Parameter Catalog

> **Canonical Path:** `reports/evaluation_matrices/`  
> **Total Models Listed:** 13  
> **Last Synchronized:** 2026-11-07T19:33:50.212198+00:00  
> **Auto-Update Guarantee:** Automatically scans `models/weights/` whenever models are trained, updated, or removed.

---

## 1. Executive Model Comparison Matrix

| Model Name | Task / Domain | Architecture | Total Params | Weights / Biases | Learning Rate | Accuracy / F1 | Latency | Detail Link |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **amrita_isl_phrase_bilstm.pth** | General Sign / Multimodal Processing | .PTH Model | `908,585` | `903,804` / `4,781` | `N/A` | **78.57%** | `N/A` | [View Matrix](amrita_isl_phrase_bilstm_matrix.md) |
| **HuggingFace ISL 26-Class Benchmark MLP** | Benchmark Alphabet Recognition (A–Z) | 3-Layer BatchNorm MLP | `26,972` | `26,176` / `410` | `0.001` | **94.39%** | `0.16 ms` | [View Matrix](hf_isl_26class_mlp_matrix.md) |
| **isl_hindi_vowels.onnx** | General Sign / Multimodal Processing | .ONNX Model | `26,972` | `26,900` / `72` | `N/A` | **N/A** | `N/A` | [View Matrix](isl_hindi_vowels_matrix.md) |
| **isl_hindi_vowels_mlp.pth** | General Sign / Multimodal Processing | .PTH Model | `25,802` | `25,024` / `392` | `N/A` | **94.71%** | `N/A` | [View Matrix](isl_hindi_vowels_mlp_matrix.md) |
| **SETU ISL Conversational MVP Matrix** | Core Phrase Recognition (Sub-Millisecond Baseline) | L2-Regularized Ridge Classifier Matrix | `10,119` | `10,113` / `6` | `N/A (Analytic Matrix Solver)` | **80.00%** | `0.2 ms` | [View Matrix](isl_mvp_model_matrix.md) |
| **isl_numerals.onnx** | General Sign / Multimodal Processing | .ONNX Model | `26,972` | `26,900` / `72` | `N/A` | **N/A** | `N/A` | [View Matrix](isl_numerals_matrix.md) |
| **isl_numerals_mlp.pth** | General Sign / Multimodal Processing | .PTH Model | `25,932` | `25,152` / `394` | `N/A` | **94.08%** | `N/A` | [View Matrix](isl_numerals_mlp_matrix.md) |
| **Dynamic ISL Sequence BiLSTM + Attention** | Continuous Phrase & Greeting Recognition | 2-Layer Bidirectional LSTM + Multi-Head Attention | `964,631` | `959,560` / `5,071` | `0.001` | **88.00%** | `4.2 ms` | [View Matrix](isl_sequence_lstm_matrix.md) |
| **Qwen2.5-0.5B Small Language Model (SLM)** | Neural ISL Gloss-to-English Translation & Contextual Autocomplete | Qwen2ForCausalLM (24 Layers, 14 Attention Heads, RoPE) | `494,032,768` | `494,005,120` / `27,648` | `1e-05` | **Fluent Grammatical English (SOV -> SVO)** | `0.0008 ms` | [View Matrix](qwen2.5_0.5b_matrix.md) |
| **RealSign 26-Class Alphabet PyTorch MLP** | Two-Handed ISL Alphabet Recognition (A–Z) | 3-Layer BatchNorm MLP | `26,972` | `26,176` / `410` | `0.001` | **92.00%** | `0.152 ms` | [View Matrix](realsign_26class_mlp_matrix.md) |
| **realsign_multisigner.onnx** | General Sign / Multimodal Processing | .ONNX Model | `26,972` | `26,900` / `72` | `N/A` | **N/A** | `N/A` | [View Matrix](realsign_multisigner_matrix.md) |
| **realsign_multisigner_mlp.pth** | General Sign / Multimodal Processing | .PTH Model | `26,972` | `26,176` / `410` | `N/A` | **92.00%** | `N/A` | [View Matrix](realsign_multisigner_mlp_matrix.md) |
| **YOLO-World v2 Small (Open-Vocabulary Multimodal Vision)** | Show-and-Sign Real-World Object Grounding | RepVL-PAN Vision-Language Transformer Backbone + Head | `12,784,200` | `12,745,424` / `14,453` | `0.01` | **Zero False-Watch Misclassifications** | `24.8 ms` | [View Matrix](yolov8s_worldv2_matrix.md) |

---

## 2. Dynamic Auto-Sync & Retraining Protocol

Whenever a new model is trained or a checkpoint is updated:
1. **Automatic Inspection:** The engine reads layer tensors, parameter shapes, mean/std, min/max, and L2 norms.
2. **Hyperparameter Extraction:** Learning rate, optimizer, epochs, and loss functions are parsed directly from checkpoints and training manifests.
3. **Automatic Listing:** If a `.pth`, `.pt`, `.npz`, or `.safetensors` file is placed into `models/weights/`, it appears immediately in the table above.
4. **Deletion Pruning:** If an old checkpoint is deleted, its matrix reports are instantly purged, preserving zero dead documentation.

### Manual CLI Generation Command
```powershell
# Regenerate all model matrices on demand
.\.venv\Scripts\python.exe scripts/generate_model_evaluation_matrices.py
```

### Automated API Endpoint
```http
GET /api/v1/models/matrices
POST /api/v1/models/matrices/refresh
```

---

## 3. Individual Model Matrix Links

- 📄 [**amrita_isl_phrase_bilstm.pth Matrix**](amrita_isl_phrase_bilstm_matrix.md) (`amrita_isl_phrase_bilstm_matrix.json`)
- 📄 [**HuggingFace ISL 26-Class Benchmark MLP Matrix**](hf_isl_26class_mlp_matrix.md) (`hf_isl_26class_mlp_matrix.json`)
- 📄 [**isl_hindi_vowels.onnx Matrix**](isl_hindi_vowels_matrix.md) (`isl_hindi_vowels_matrix.json`)
- 📄 [**isl_hindi_vowels_mlp.pth Matrix**](isl_hindi_vowels_mlp_matrix.md) (`isl_hindi_vowels_mlp_matrix.json`)
- 📄 [**SETU ISL Conversational MVP Matrix Matrix**](isl_mvp_model_matrix.md) (`isl_mvp_model_matrix.json`)
- 📄 [**isl_numerals.onnx Matrix**](isl_numerals_matrix.md) (`isl_numerals_matrix.json`)
- 📄 [**isl_numerals_mlp.pth Matrix**](isl_numerals_mlp_matrix.md) (`isl_numerals_mlp_matrix.json`)
- 📄 [**Dynamic ISL Sequence BiLSTM + Attention Matrix**](isl_sequence_lstm_matrix.md) (`isl_sequence_lstm_matrix.json`)
- 📄 [**Qwen2.5-0.5B Small Language Model (SLM) Matrix**](qwen2.5_0.5b_matrix.md) (`qwen2.5_0.5b_matrix.json`)
- 📄 [**RealSign 26-Class Alphabet PyTorch MLP Matrix**](realsign_26class_mlp_matrix.md) (`realsign_26class_mlp_matrix.json`)
- 📄 [**realsign_multisigner.onnx Matrix**](realsign_multisigner_matrix.md) (`realsign_multisigner_matrix.json`)
- 📄 [**realsign_multisigner_mlp.pth Matrix**](realsign_multisigner_mlp_matrix.md) (`realsign_multisigner_mlp_matrix.json`)
- 📄 [**YOLO-World v2 Small (Open-Vocabulary Multimodal Vision) Matrix**](yolov8s_worldv2_matrix.md) (`yolov8s_worldv2_matrix.json`)