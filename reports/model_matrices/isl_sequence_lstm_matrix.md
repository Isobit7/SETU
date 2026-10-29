# Model Evaluation Matrix: Dynamic ISL Sequence BiLSTM + Attention

> **Model ID:** `isl_sequence_lstm`  
> **Task:** Continuous Phrase & Greeting Recognition  
> **Architecture:** 2-Layer Bidirectional LSTM + Multi-Head Attention  
> **File:** `models\weights\isl_sequence_lstm.pth` (3.69 MB)  
> **Last Modified:** 2026-10-28T13:30:00.401086+00:00  
> **Total Parameters:** `964,631` (959,560 weights, 5,071 biases)  

---

## 1. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `0.001` | Training specification |
| **Optimizer** | `AdamW` | Training specification |
| **Loss Function** | `CrossEntropyLoss` | Training specification |
| **Epochs** | `60` | Training specification |
| **Batch Size** | `16` | Training specification |
| **Weight Decay** | `0.001` | Training specification |
| **Dropout** | `0.3` | Training specification |
| **Scheduler** | `CosineAnnealingLR` | Training specification |
| **Device** | `cuda (NVIDIA GeForce RTX 3050 6GB Laptop GPU)` | Training specification |

---

## 2. Evaluation & Performance Metrics

| Metric | Empirical Result | Target Benchmark |
| :--- | :--- | :--- |
| **Test Accuracy** | **0.8800** | Production SLA Passed |
| **Fine Tuned Accuracy** | **1.0000** | Production SLA Passed |
| **Macro F1** | **0.8750** | Production SLA Passed |
| **Latency Cuda Ms** | **4.2000** | Production SLA Passed |

---

## 3. Layer-by-Layer Weights & Biases Breakdown

| Layer / Tensor Name | Type | Shape | Dtype | Parameters | Mean | Std | Min | Max | L2 Norm | Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `input_norm.weight` | `weight` | `[328]` | `float32` | `328` | 1.0079 | 0.0191 | 0.9599 | 1.0811 | 18.2577 | 0.0% |
| `input_norm.bias` | `bias` | `[328]` | `float32` | `328` | -0.0002 | 0.0119 | -0.0319 | 0.0380 | 0.2148 | 0.0% |
| `lstm.weight_ih_l0` | `weight` | `[512, 328]` | `float32` | `167,936` | -0.0002 | 0.0534 | -0.1497 | 0.1649 | 21.8851 | 0.0% |
| `lstm.weight_hh_l0` | `weight` | `[512, 128]` | `float32` | `65,536` | 0.0003 | 0.0536 | -0.1450 | 0.1441 | 13.7252 | 0.0% |
| `lstm.bias_ih_l0` | `bias` | `[512]` | `float32` | `512` | 0.0033 | 0.0530 | -0.1123 | 0.1169 | 1.1998 | 0.0% |
| `lstm.bias_hh_l0` | `bias` | `[512]` | `float32` | `512` | 0.0019 | 0.0524 | -0.1078 | 0.1180 | 1.1856 | 0.0% |
| `lstm.weight_ih_l0_reverse` | `weight` | `[512, 328]` | `float32` | `167,936` | 0.0001 | 0.0533 | -0.1544 | 0.1457 | 21.8436 | 0.0% |
| `lstm.weight_hh_l0_reverse` | `weight` | `[512, 128]` | `float32` | `65,536` | 0.0002 | 0.0533 | -0.1448 | 0.1352 | 13.6536 | 0.0% |
| `lstm.bias_ih_l0_reverse` | `bias` | `[512]` | `float32` | `512` | 0.0029 | 0.0531 | -0.1109 | 0.1458 | 1.2018 | 0.0% |
| `lstm.bias_hh_l0_reverse` | `bias` | `[512]` | `float32` | `512` | 0.0048 | 0.0495 | -0.1216 | 0.1150 | 1.1239 | 0.0% |
| `lstm.weight_ih_l1` | `weight` | `[512, 256]` | `float32` | `131,072` | 0.0003 | 0.0525 | -0.1297 | 0.1326 | 18.9957 | 0.0% |
| `lstm.weight_hh_l1` | `weight` | `[512, 128]` | `float32` | `65,536` | -0.0003 | 0.0529 | -0.1330 | 0.1350 | 13.5494 | 0.0% |
| `lstm.bias_ih_l1` | `bias` | `[512]` | `float32` | `512` | -0.0007 | 0.0526 | -0.1062 | 0.1047 | 1.1892 | 0.0% |
| `lstm.bias_hh_l1` | `bias` | `[512]` | `float32` | `512` | -0.0011 | 0.0506 | -0.1018 | 0.1001 | 1.1436 | 0.0% |
| `lstm.weight_ih_l1_reverse` | `weight` | `[512, 256]` | `float32` | `131,072` | 0.0003 | 0.0528 | -0.1502 | 0.1374 | 19.1318 | 0.0% |
| `lstm.weight_hh_l1_reverse` | `weight` | `[512, 128]` | `float32` | `65,536` | -0.0001 | 0.0530 | -0.1491 | 0.1444 | 13.5799 | 0.0% |
| `lstm.bias_ih_l1_reverse` | `bias` | `[512]` | `float32` | `512` | -0.0033 | 0.0521 | -0.1079 | 0.1054 | 1.1800 | 0.0% |
| `lstm.bias_hh_l1_reverse` | `bias` | `[512]` | `float32` | `512` | -0.0017 | 0.0540 | -0.1106 | 0.1091 | 1.2220 | 0.0% |
| `attention.0.weight` | `weight` | `[64, 256]` | `float32` | `16,384` | -0.0001 | 0.0391 | -0.1032 | 0.1006 | 5.0015 | 0.0% |
| `attention.0.bias` | `bias` | `[64]` | `float32` | `64` | -0.0012 | 0.0394 | -0.0944 | 0.0819 | 0.3132 | 0.0% |
| `attention.2.weight` | `weight` | `[1, 64]` | `float32` | `64` | -0.0106 | 0.0742 | -0.1363 | 0.1221 | 0.5951 | 0.0% |
| `fc.0.weight` | `weight` | `[256, 256]` | `float32` | `65,536` | -0.0000 | 0.0383 | -0.1263 | 0.1591 | 9.7938 | 0.0% |
| `fc.0.bias` | `bias` | `[256]` | `float32` | `256` | 0.0017 | 0.0340 | -0.0693 | 0.0809 | 0.5442 | 0.0% |
| `fc.1.weight` | `weight` | `[256]` | `float32` | `256` | 1.0447 | 0.0394 | 0.9737 | 1.2702 | 16.7269 | 0.0% |
| `fc.1.bias` | `bias` | `[256]` | `float32` | `256` | 0.0285 | 0.0186 | -0.0227 | 0.1162 | 0.5444 | 0.0% |
| `fc.4.weight` | `weight` | `[64, 256]` | `float32` | `16,384` | 0.0106 | 0.0474 | -0.3036 | 0.3692 | 6.2179 | 0.0% |
| `fc.4.bias` | `bias` | `[64]` | `float32` | `64` | 0.0149 | 0.0399 | -0.0463 | 0.0900 | 0.3383 | 0.0% |
| `fc.6.weight` | `weight` | `[7, 64]` | `float32` | `448` | -0.0197 | 0.0978 | -0.2451 | 0.1854 | 2.1084 | 0.0% |
| `fc.6.bias` | `bias` | `[7]` | `float32` | `7` | -0.0138 | 0.0825 | -0.1275 | 0.0756 | 0.2054 | 0.0% |

---

<sub>Report auto-generated by SETU Model Evaluation Engine at 2026-10-29T16:50:26.551302+00:00.</sub>