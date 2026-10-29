# Model Evaluation Matrix: YOLO-World v2 Small (Open-Vocabulary Multimodal Vision)

> **Model ID:** `yolov8s_worldv2`  
> **Task:** Show-and-Sign Real-World Object Grounding  
> **Architecture:** RepVL-PAN Vision-Language Transformer Backbone + Head  
> **File:** `models\weights\yolov8s-worldv2.pt` (24.72 MB)  
> **Last Modified:** 2026-10-29T01:15:28.014289+00:00  
> **Total Parameters:** `12,784,200` (12,745,424 weights, 14,453 biases)  

---

## 1. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `0.01` | Training specification |
| **Optimizer** | `None` | Training specification |
| **Loss Function** | `CIoU + DFL + Contrastive Text-Visual Loss` | Training specification |
| **Epochs** | `100` | Training specification |
| **Batch Size** | `16` | Training specification |
| **Imgsz** | `640` | Training specification |
| **Device** | `cuda (RTX 3050)` | Training specification |
| **Batch** | `16` | Training specification |
| **Lr0** | `0.01` | Training specification |

---

## 2. Evaluation & Performance Metrics

| Metric | Empirical Result | Target Benchmark |
| :--- | :--- | :--- |
| **Test Accuracy** | **Zero False-Watch Misclassifications** | Production SLA Passed |
| **Precision** | **0.9320** | Production SLA Passed |
| **Recall** | **0.9150** | Production SLA Passed |
| **Map50** | **0.8210** | Production SLA Passed |
| **Latency Cuda Ms** | **24.8000** | Production SLA Passed |

---

## 3. Layer-by-Layer Weights & Biases Breakdown

| Layer / Tensor Name | Type | Shape | Dtype | Parameters | Mean | Std | Min | Max | L2 Norm | Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `model.0.conv.weight` | `weight` | `[32, 3, 3, 3]` | `float16` | `864` | -0.0009 | 0.0265 | -0.1254 | 0.1823 | 0.7794 | 0.0% |
| `model.0.bn.weight` | `weight` | `[32]` | `float16` | `32` | 3.4381 | 1.5426 | 0.1993 | 6.0039 | 21.2610 | 0.0% |
| `model.0.bn.bias` | `bias` | `[32]` | `float16` | `32` | -0.2893 | 3.5957 | -8.0625 | 10.3125 | 20.0866 | 0.0% |
| `model.0.bn.running_mean` | `running_mean` | `[32]` | `float16` | `32` | -0.0107 | 0.0745 | -0.1787 | 0.2258 | 0.4194 | 0.0% |
| `model.0.bn.running_var` | `running_var` | `[32]` | `float16` | `32` | 0.0022 | 0.0045 | 0.0000 | 0.0180 | 0.0278 | 0.0% |
| `model.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.1.conv.weight` | `weight` | `[64, 32, 3, 3]` | `float16` | `18,432` | 0.0005 | 0.0258 | -0.3582 | 0.3196 | 3.4976 | 0.0% |
| `model.1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 3.8717 | 1.1284 | 1.3633 | 5.8047 | 32.2424 | 0.0% |
| `model.1.bn.bias` | `bias` | `[64]` | `float16` | `64` | 0.4129 | 1.2987 | -3.8711 | 3.5645 | 10.8241 | 0.0% |
| `model.1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.0692 | 1.1121 | -3.9902 | 5.1719 | 8.8446 | 0.0% |
| `model.1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.1626 | 0.1687 | 0.0182 | 0.8872 | 1.8668 | 0.0% |
| `model.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.2.cv1.conv.weight` | `weight` | `[64, 64, 1, 1]` | `float16` | `4,096` | -0.0049 | 0.0298 | -0.2974 | 0.2465 | 1.9333 | 0.0% |
| `model.2.cv1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.8578 | 1.1610 | 0.3560 | 4.3711 | 17.4873 | 0.0% |
| `model.2.cv1.bn.bias` | `bias` | `[64]` | `float16` | `64` | 0.0562 | 1.4064 | -2.0527 | 3.3301 | 11.1720 | 0.0% |
| `model.2.cv1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.4520 | 0.4527 | -1.2510 | 1.0664 | 5.0973 | 0.0% |
| `model.2.cv1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.6787 | 0.5078 | 0.0715 | 2.3516 | 6.7623 | 0.0% |
| `model.2.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.2.cv2.conv.weight` | `weight` | `[64, 96, 1, 1]` | `float16` | `6,144` | -0.0016 | 0.0254 | -0.2067 | 0.1630 | 1.9964 | 0.0% |
| `model.2.cv2.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.1299 | 0.2557 | 0.5195 | 1.9824 | 9.2642 | 0.0% |
| `model.2.cv2.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.0602 | 0.6895 | -1.2617 | 1.6133 | 5.4941 | 0.0% |
| `model.2.cv2.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.1422 | 0.3324 | -0.9360 | 1.1416 | 2.8730 | 0.0% |
| `model.2.cv2.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0557 | 0.0202 | 0.0174 | 0.1250 | 0.4735 | 0.0% |
| `model.2.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.2.m.0.cv1.conv.weight` | `weight` | `[32, 32, 3, 3]` | `float16` | `9,216` | -0.0010 | 0.0221 | -0.1925 | 0.1875 | 2.1192 | 0.0% |
| `model.2.m.0.cv1.bn.weight` | `weight` | `[32]` | `float16` | `32` | 1.4170 | 0.4651 | 0.9004 | 3.4219 | 8.4236 | 0.0% |
| `model.2.m.0.cv1.bn.bias` | `bias` | `[32]` | `float16` | `32` | 0.8239 | 1.8931 | -1.8496 | 8.8281 | 11.5249 | 0.0% |
| `model.2.m.0.cv1.bn.running_mean` | `running_mean` | `[32]` | `float16` | `32` | -0.1175 | 0.3261 | -1.0781 | 0.5972 | 1.9335 | 0.0% |
| `model.2.m.0.cv1.bn.running_var` | `running_var` | `[32]` | `float16` | `32` | 0.0431 | 0.0292 | 0.0126 | 0.1503 | 0.2931 | 0.0% |
| `model.2.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.2.m.0.cv2.conv.weight` | `weight` | `[32, 32, 3, 3]` | `float16` | `9,216` | -0.0016 | 0.0232 | -0.2115 | 0.1555 | 2.2318 | 0.0% |
| `model.2.m.0.cv2.bn.weight` | `weight` | `[32]` | `float16` | `32` | 1.6695 | 0.4020 | 1.0605 | 2.7148 | 9.7059 | 0.0% |
| `model.2.m.0.cv2.bn.bias` | `bias` | `[32]` | `float16` | `32` | 1.4603 | 1.3157 | -0.7432 | 3.6914 | 11.0408 | 0.0% |
| `model.2.m.0.cv2.bn.running_mean` | `running_mean` | `[32]` | `float16` | `32` | -0.3196 | 0.6687 | -2.9297 | 0.6060 | 4.1387 | 0.0% |
| `model.2.m.0.cv2.bn.running_var` | `running_var` | `[32]` | `float16` | `32` | 0.1830 | 0.0754 | 0.0812 | 0.3762 | 1.1168 | 0.0% |
| `model.2.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.3.conv.weight` | `weight` | `[128, 64, 3, 3]` | `float16` | `73,728` | -0.0012 | 0.0237 | -0.2498 | 0.1521 | 6.4374 | 0.0% |
| `model.3.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.6593 | 0.1917 | 0.2756 | 1.4346 | 7.7654 | 0.0% |
| `model.3.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.2287 | 0.8850 | -1.4844 | 2.4355 | 10.3040 | 0.0% |
| `model.3.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1875 | 0.3327 | -1.6240 | 0.7144 | 4.3081 | 0.0% |
| `model.3.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.3299 | 0.2758 | 0.1158 | 1.6387 | 4.8570 | 0.0% |
| `model.3.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.cv1.conv.weight` | `weight` | `[128, 128, 1, 1]` | `float16` | `16,384` | -0.0021 | 0.0236 | -0.2771 | 0.1552 | 3.0313 | 0.0% |
| `model.4.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.6780 | 0.3450 | 0.1835 | 1.8555 | 8.6002 | 0.0% |
| `model.4.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.1381 | 0.8294 | -2.8438 | 2.3867 | 9.4768 | 0.0% |
| `model.4.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.0255 | 0.1439 | -0.4385 | 0.3386 | 1.6465 | 0.0% |
| `model.4.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0088 | 0.0048 | 0.0030 | 0.0266 | 0.1134 | 0.0% |
| `model.4.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.cv2.conv.weight` | `weight` | `[128, 256, 1, 1]` | `float16` | `32,768` | -0.0019 | 0.0244 | -0.2303 | 0.2561 | 4.4245 | 0.0% |
| `model.4.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.8137 | 0.2046 | 0.3242 | 1.2715 | 9.4906 | 0.0% |
| `model.4.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.6393 | 0.7660 | -2.9121 | 1.4375 | 11.2621 | 0.0% |
| `model.4.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.0668 | 0.2409 | -0.9033 | 0.5415 | 2.8183 | 0.0% |
| `model.4.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0288 | 0.0116 | 0.0126 | 0.0798 | 0.3514 | 0.0% |
| `model.4.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.m.0.cv1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0012 | 0.0230 | -0.3667 | 0.1802 | 4.4224 | 0.0% |
| `model.4.m.0.cv1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.8573 | 0.1920 | 0.3608 | 1.2852 | 7.0258 | 0.0% |
| `model.4.m.0.cv1.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.5507 | 0.7453 | -1.4434 | 2.1992 | 7.3759 | 0.0% |
| `model.4.m.0.cv1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.0618 | 0.1412 | -0.3672 | 0.2917 | 1.2247 | 0.0% |
| `model.4.m.0.cv1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0237 | 0.0127 | 0.0066 | 0.0648 | 0.2149 | 0.0% |
| `model.4.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.m.0.cv2.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0017 | 0.0234 | -0.4624 | 0.1758 | 4.5003 | 0.0% |
| `model.4.m.0.cv2.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.6706 | 0.2256 | 0.2771 | 1.1924 | 5.6559 | 0.0% |
| `model.4.m.0.cv2.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.2074 | 0.7035 | -1.2363 | 1.5107 | 5.8249 | 0.0% |
| `model.4.m.0.cv2.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.0080 | 0.2345 | -0.7754 | 0.4778 | 1.8623 | 0.0% |
| `model.4.m.0.cv2.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0702 | 0.0301 | 0.0220 | 0.1825 | 0.6102 | 0.0% |
| `model.4.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.m.1.cv1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0015 | 0.0233 | -0.2228 | 0.2710 | 4.4767 | 0.0% |
| `model.4.m.1.cv1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.7488 | 0.1459 | 0.2373 | 0.9751 | 6.1014 | 0.0% |
| `model.4.m.1.cv1.bn.bias` | `bias` | `[64]` | `float16` | `64` | -1.0387 | 0.7849 | -2.7324 | 1.0312 | 10.3860 | 0.0% |
| `model.4.m.1.cv1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.1048 | 0.2600 | -1.7061 | 0.2435 | 2.2279 | 0.0% |
| `model.4.m.1.cv1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0552 | 0.0245 | 0.0240 | 0.1129 | 0.4821 | 0.0% |
| `model.4.m.1.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.4.m.1.cv2.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0015 | 0.0228 | -0.1631 | 0.1714 | 4.3888 | 0.0% |
| `model.4.m.1.cv2.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.9459 | 0.2832 | 0.4932 | 1.4766 | 7.8940 | 0.0% |
| `model.4.m.1.cv2.bn.bias` | `bias` | `[64]` | `float16` | `64` | 0.2075 | 0.9890 | -1.3682 | 2.6211 | 8.0236 | 0.0% |
| `model.4.m.1.cv2.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.1273 | 0.1539 | -0.3069 | 0.5664 | 1.5900 | 0.0% |
| `model.4.m.1.cv2.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0209 | 0.0066 | 0.0119 | 0.0387 | 0.1751 | 0.0% |
| `model.4.m.1.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.5.conv.weight` | `weight` | `[256, 128, 3, 3]` | `float16` | `294,912` | -0.0009 | 0.0232 | -0.3125 | 0.3650 | 12.6081 | 0.0% |
| `model.5.bn.weight` | `weight` | `[256]` | `float16` | `256` | 0.6480 | 0.2096 | 0.2864 | 2.8164 | 10.8950 | 0.0% |
| `model.5.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.6518 | 0.7506 | -3.2305 | 1.3857 | 15.8872 | 0.0% |
| `model.5.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | 0.0529 | 0.3118 | -1.0576 | 1.1572 | 5.0499 | 0.0% |
| `model.5.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.1816 | 0.1040 | 0.0710 | 0.8755 | 3.3473 | 0.0% |
| `model.5.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.cv1.conv.weight` | `weight` | `[256, 256, 1, 1]` | `float16` | `65,536` | -0.0024 | 0.0232 | -0.2603 | 0.2168 | 5.9605 | 0.0% |
| `model.6.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 0.9760 | 0.4469 | 0.2502 | 2.0879 | 17.1691 | 0.0% |
| `model.6.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.6180 | 0.9631 | -3.4062 | 2.7422 | 18.2841 | 0.0% |
| `model.6.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | 0.0383 | 0.1296 | -0.3313 | 0.4734 | 2.1587 | 0.0% |
| `model.6.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.0119 | 0.0046 | 0.0038 | 0.0278 | 0.2041 | 0.0% |
| `model.6.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.cv2.conv.weight` | `weight` | `[256, 512, 1, 1]` | `float16` | `131,072` | -0.0019 | 0.0245 | -0.2324 | 0.3318 | 8.9014 | 0.0% |
| `model.6.cv2.bn.weight` | `weight` | `[256]` | `float16` | `256` | 0.9385 | 0.2381 | 0.4211 | 1.7998 | 15.4907 | 0.0% |
| `model.6.cv2.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.9194 | 0.9244 | -3.6992 | 1.3213 | 20.8401 | 0.0% |
| `model.6.cv2.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.2121 | 0.2640 | -1.2158 | 0.7925 | 5.4120 | 0.0% |
| `model.6.cv2.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.1286 | 0.0413 | 0.0623 | 0.3489 | 2.1607 | 0.0% |
| `model.6.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.m.0.cv1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0019 | 0.0231 | -0.3040 | 0.2076 | 8.9013 | 0.0% |
| `model.6.m.0.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.9787 | 0.2592 | 0.3086 | 3.1816 | 11.4513 | 0.0% |
| `model.6.m.0.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -1.1436 | 0.7994 | -3.4863 | 3.1035 | 15.7655 | 0.0% |
| `model.6.m.0.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1354 | 0.1646 | -0.7642 | 0.2825 | 2.4057 | 0.0% |
| `model.6.m.0.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0903 | 0.0346 | 0.0346 | 0.3076 | 1.0933 | 0.0% |
| `model.6.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.m.0.cv2.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0019 | 0.0232 | -0.3574 | 0.1763 | 8.9391 | 0.0% |
| `model.6.m.0.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.8905 | 0.2172 | 0.3784 | 1.4121 | 10.3676 | 0.0% |
| `model.6.m.0.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.5472 | 0.6122 | -2.1055 | 1.0186 | 9.2695 | 0.0% |
| `model.6.m.0.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.0832 | 0.2726 | -0.5513 | 1.2383 | 3.2135 | 0.0% |
| `model.6.m.0.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1539 | 0.0432 | 0.0705 | 0.2764 | 1.8081 | 0.0% |
| `model.6.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.m.1.cv1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0015 | 0.0229 | -0.2822 | 0.2061 | 8.8108 | 0.0% |
| `model.6.m.1.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.9914 | 0.1769 | 0.4014 | 1.5283 | 11.3924 | 0.0% |
| `model.6.m.1.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -1.3683 | 0.8628 | -3.2930 | 1.4580 | 18.2814 | 0.0% |
| `model.6.m.1.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1806 | 0.2346 | -0.6963 | 0.4109 | 3.3415 | 0.0% |
| `model.6.m.1.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1668 | 0.0490 | 0.0901 | 0.3267 | 1.9659 | 0.0% |
| `model.6.m.1.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.6.m.1.cv2.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0016 | 0.0226 | -0.2537 | 0.1864 | 8.7119 | 0.0% |
| `model.6.m.1.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.3767 | 0.3308 | 0.7393 | 2.6816 | 16.0149 | 0.0% |
| `model.6.m.1.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.1033 | 1.0312 | -3.0059 | 2.7168 | 11.6799 | 0.0% |
| `model.6.m.1.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.2089 | 0.2291 | -1.3496 | 0.7192 | 3.5002 | 0.0% |
| `model.6.m.1.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1043 | 0.0267 | 0.0563 | 0.2155 | 1.2174 | 0.0% |
| `model.6.m.1.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.7.conv.weight` | `weight` | `[512, 256, 3, 3]` | `float16` | `1,179,648` | -0.0016 | 0.0228 | -0.2866 | 0.4883 | 24.8140 | 0.0% |
| `model.7.bn.weight` | `weight` | `[512]` | `float16` | `512` | 0.8853 | 0.1878 | 0.4341 | 1.7354 | 20.4773 | 0.0% |
| `model.7.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.0960 | 0.7028 | -3.0254 | 1.4385 | 29.4526 | 0.0% |
| `model.7.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | 0.1689 | 0.4385 | -1.2168 | 1.7324 | 10.6241 | 0.0% |
| `model.7.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.4681 | 0.1474 | 0.2201 | 1.3936 | 11.1029 | 0.0% |
| `model.7.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.8.cv1.conv.weight` | `weight` | `[512, 512, 1, 1]` | `float16` | `262,144` | -0.0033 | 0.0228 | -0.3516 | 0.6880 | 11.7854 | 0.0% |
| `model.8.cv1.bn.weight` | `weight` | `[512]` | `float16` | `512` | 1.3249 | 0.4222 | 0.3530 | 2.7344 | 31.4612 | 0.0% |
| `model.8.cv1.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.3578 | 0.9748 | -5.8281 | 3.8242 | 37.8085 | 0.0% |
| `model.8.cv1.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | 0.1893 | 0.1351 | -0.6660 | 0.8101 | 5.2610 | 0.0% |
| `model.8.cv1.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.0276 | 0.0082 | 0.0126 | 0.1179 | 0.6515 | 0.0% |
| `model.8.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.8.cv2.conv.weight` | `weight` | `[512, 768, 1, 1]` | `float16` | `393,216` | -0.0022 | 0.0241 | -0.5503 | 0.2383 | 15.1760 | 0.0% |
| `model.8.cv2.bn.weight` | `weight` | `[512]` | `float16` | `512` | 1.2483 | 0.3451 | 0.4915 | 3.3887 | 29.3021 | 0.0% |
| `model.8.cv2.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.1727 | 1.0272 | -4.0859 | 4.0391 | 35.2609 | 0.0% |
| `model.8.cv2.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | -0.3382 | 0.3281 | -1.5605 | 0.5869 | 10.6564 | 0.0% |
| `model.8.cv2.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.4262 | 0.1474 | 0.2235 | 1.4814 | 10.2038 | 0.0% |
| `model.8.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.8.m.0.cv1.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | -0.0017 | 0.0225 | -0.2998 | 0.1621 | 17.3014 | 0.0% |
| `model.8.m.0.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.3660 | 0.2722 | 0.4241 | 2.2715 | 22.2845 | 0.0% |
| `model.8.m.0.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | -1.5228 | 1.1220 | -4.8906 | 3.2227 | 30.2440 | 0.0% |
| `model.8.m.0.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.1170 | 0.3461 | -2.8047 | 0.9453 | 5.8352 | 0.0% |
| `model.8.m.0.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.3202 | 0.1516 | 0.1407 | 1.2070 | 5.6662 | 0.0% |
| `model.8.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.8.m.0.cv2.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | -0.0021 | 0.0226 | -0.2708 | 0.2106 | 17.4517 | 0.0% |
| `model.8.m.0.cv2.bn.weight` | `weight` | `[256]` | `float16` | `256` | 2.1300 | 0.5848 | 1.1924 | 4.4453 | 35.3365 | 0.0% |
| `model.8.m.0.cv2.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.8485 | 1.2837 | -4.1758 | 2.9355 | 24.5866 | 0.0% |
| `model.8.m.0.cv2.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.1284 | 0.4760 | -3.3926 | 1.1445 | 7.8739 | 0.0% |
| `model.8.m.0.cv2.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.6856 | 0.2796 | 0.3359 | 2.5137 | 11.8437 | 0.0% |
| `model.8.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.9.cv1.conv.weight` | `weight` | `[256, 512, 1, 1]` | `float16` | `131,072` | -0.0042 | 0.0238 | -0.4805 | 0.1846 | 8.7556 | 0.0% |
| `model.9.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.3314 | 0.4005 | 0.8223 | 3.7168 | 22.2424 | 0.0% |
| `model.9.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | 2.4734 | 1.9220 | -3.2383 | 9.4062 | 50.0809 | 0.0% |
| `model.9.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.0263 | 0.2428 | -1.8467 | 0.6782 | 3.8997 | 0.0% |
| `model.9.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.0955 | 0.0696 | 0.0420 | 0.4229 | 1.8892 | 0.0% |
| `model.9.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.9.cv2.conv.weight` | `weight` | `[512, 1024, 1, 1]` | `float16` | `524,288` | -0.0001 | 0.0225 | -0.4089 | 0.3704 | 16.2741 | 0.0% |
| `model.9.cv2.bn.weight` | `weight` | `[512]` | `float16` | `512` | 0.9224 | 0.2753 | 0.2964 | 3.3223 | 21.7800 | 0.0% |
| `model.9.cv2.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.6062 | 1.0987 | -5.4844 | 3.7676 | 44.0194 | 0.0% |
| `model.9.cv2.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | 0.4505 | 4.1708 | -23.9531 | 18.4531 | 94.8313 | 0.0% |
| `model.9.cv2.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 1.4783 | 0.4659 | 0.6533 | 4.3867 | 35.0693 | 0.0% |
| `model.9.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.12.cv1.conv.weight` | `weight` | `[256, 768, 1, 1]` | `float16` | `196,608` | -0.0026 | 0.0232 | -0.3291 | 0.2856 | 10.3396 | 0.0% |
| `model.12.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.1184 | 0.3169 | 0.4121 | 1.9648 | 18.5960 | 0.0% |
| `model.12.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.6036 | 1.1089 | -3.3262 | 2.6445 | 20.1698 | 0.0% |
| `model.12.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | 0.2399 | 0.1816 | -0.5244 | 0.7720 | 4.8102 | 0.0% |
| `model.12.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.0543 | 0.0159 | 0.0284 | 0.1256 | 0.9059 | 0.0% |
| `model.12.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.12.cv2.conv.weight` | `weight` | `[256, 512, 1, 1]` | `float16` | `131,072` | -0.0020 | 0.0236 | -0.4585 | 0.4265 | 8.5856 | 0.0% |
| `model.12.cv2.bn.weight` | `weight` | `[256]` | `float16` | `256` | 0.7953 | 0.2638 | 0.2842 | 1.9150 | 13.4040 | 0.0% |
| `model.12.cv2.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.7397 | 0.9397 | -2.6660 | 3.2930 | 19.1104 | 0.0% |
| `model.12.cv2.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.5135 | 8.0001 | -60.9375 | 33.7188 | 128.0157 | 0.0% |
| `model.12.cv2.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.1229 | 0.0598 | 0.0442 | 0.4287 | 2.1855 | 0.0% |
| `model.12.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.12.m.0.cv1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0019 | 0.0229 | -0.2119 | 0.2125 | 8.8113 | 0.0% |
| `model.12.m.0.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.0544 | 0.2269 | 0.5156 | 1.5918 | 12.2007 | 0.0% |
| `model.12.m.0.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -1.0621 | 0.9582 | -3.3242 | 1.1992 | 16.1555 | 0.0% |
| `model.12.m.0.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1128 | 0.3724 | -1.7812 | 0.8726 | 4.3859 | 0.0% |
| `model.12.m.0.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.2035 | 0.0715 | 0.1000 | 0.5034 | 2.4389 | 0.0% |
| `model.12.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.12.m.0.cv2.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0016 | 0.0227 | -0.2952 | 0.3008 | 8.7432 | 0.0% |
| `model.12.m.0.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.2167 | 0.3043 | 0.5640 | 2.1602 | 14.1861 | 0.0% |
| `model.12.m.0.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.6082 | 1.8037 | -3.0117 | 8.2812 | 21.4598 | 0.0% |
| `model.12.m.0.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.0538 | 0.1963 | -0.4536 | 0.6055 | 2.2938 | 0.0% |
| `model.12.m.0.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1941 | 0.1127 | 0.0831 | 0.9448 | 2.5365 | 0.0% |
| `model.12.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.12.attn.bias` | `bias` | `[4]` | `float16` | `4` | -0.7303 | 0.4938 | -1.1611 | -0.0190 | 1.6927 | 0.0% |
| `model.12.attn.gl.weight` | `weight` | `[128, 512]` | `float16` | `65,536` | 0.0004 | 0.0666 | -0.6396 | 0.4824 | 17.0461 | 0.0% |
| `model.12.attn.gl.bias` | `bias` | `[128]` | `float16` | `128` | -0.3024 | 0.8558 | -9.2266 | 0.6392 | 10.2330 | 0.0% |
| `model.12.attn.proj_conv.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0000 | 0.0236 | -0.3818 | 0.2441 | 9.0558 | 0.0% |
| `model.12.attn.proj_conv.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.6406 | 0.6810 | -0.0106 | 2.7266 | 20.0856 | 0.0% |
| `model.12.attn.proj_conv.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.0667 | 9.9711 | -37.2812 | 37.1875 | 112.3710 | 0.0% |
| `model.12.attn.proj_conv.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.0289 | 1.0246 | -4.3711 | 3.6367 | 11.5509 | 0.0% |
| `model.12.attn.proj_conv.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.3313 | 0.2881 | 0.1348 | 2.1895 | 4.9587 | 0.0% |
| `model.12.attn.proj_conv.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.15.cv1.conv.weight` | `weight` | `[128, 384, 1, 1]` | `float16` | `49,152` | -0.0014 | 0.0227 | -0.3508 | 0.2345 | 5.0487 | 0.0% |
| `model.15.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.6537 | 0.2818 | 0.2440 | 2.9160 | 8.0487 | 0.0% |
| `model.15.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | 0.1087 | 1.0585 | -2.6777 | 3.5371 | 11.9916 | 0.0% |
| `model.15.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.1073 | 0.1624 | -0.3369 | 0.5356 | 2.1963 | 0.0% |
| `model.15.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0241 | 0.0131 | 0.0125 | 0.1449 | 0.3104 | 0.0% |
| `model.15.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.15.cv2.conv.weight` | `weight` | `[128, 256, 1, 1]` | `float16` | `32,768` | -0.0014 | 0.0230 | -0.3245 | 0.3259 | 4.1754 | 0.0% |
| `model.15.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.6432 | 0.2915 | 0.2164 | 1.5605 | 7.9837 | 0.0% |
| `model.15.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.5276 | 0.9313 | -2.9668 | 1.4629 | 12.0740 | 0.0% |
| `model.15.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.2800 | 1.5494 | -10.0703 | 7.1211 | 17.7461 | 0.0% |
| `model.15.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0323 | 0.0157 | 0.0049 | 0.1119 | 0.4055 | 0.0% |
| `model.15.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.15.m.0.cv1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0015 | 0.0223 | -0.2529 | 0.1846 | 4.2966 | 0.0% |
| `model.15.m.0.cv1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.7985 | 0.1703 | 0.2959 | 1.0488 | 6.5293 | 0.0% |
| `model.15.m.0.cv1.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.7457 | 0.7637 | -2.2656 | 1.4990 | 8.5050 | 0.0% |
| `model.15.m.0.cv1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.1389 | 0.2063 | -0.7476 | 0.5532 | 1.9789 | 0.0% |
| `model.15.m.0.cv1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0279 | 0.0107 | 0.0112 | 0.0601 | 0.2390 | 0.0% |
| `model.15.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.15.m.0.cv2.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0017 | 0.0219 | -0.2456 | 0.1650 | 4.2147 | 0.0% |
| `model.15.m.0.cv2.bn.weight` | `weight` | `[64]` | `float16` | `64` | 0.8567 | 0.2121 | 0.3259 | 1.2754 | 7.0572 | 0.0% |
| `model.15.m.0.cv2.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.4821 | 0.9516 | -1.7168 | 3.0449 | 8.4806 | 0.0% |
| `model.15.m.0.cv2.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.0575 | 0.1802 | -0.4395 | 0.6128 | 1.5026 | 0.0% |
| `model.15.m.0.cv2.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0386 | 0.0196 | 0.0168 | 0.1115 | 0.3454 | 0.0% |
| `model.15.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.15.attn.bias` | `bias` | `[2]` | `float16` | `2` | -0.8670 | 0.9905 | -1.5674 | -0.1666 | 1.5762 | 0.0% |
| `model.15.attn.gl.weight` | `weight` | `[64, 512]` | `float16` | `32,768` | 0.0011 | 0.0493 | -0.3005 | 0.2874 | 8.9322 | 0.0% |
| `model.15.attn.gl.bias` | `bias` | `[64]` | `float16` | `64` | -0.3486 | 0.7870 | -3.8711 | 1.0557 | 6.8407 | 0.0% |
| `model.15.attn.proj_conv.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | 0.0001 | 0.0225 | -0.1487 | 0.2499 | 4.3269 | 0.0% |
| `model.15.attn.proj_conv.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.3221 | 0.6528 | -0.7485 | 2.3535 | 11.7779 | 0.0% |
| `model.15.attn.proj_conv.bn.bias` | `bias` | `[64]` | `float16` | `64` | -1.6090 | 5.5515 | -18.9219 | 16.5469 | 45.9053 | 0.0% |
| `model.15.attn.proj_conv.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.0936 | 0.3155 | -0.8457 | 0.8174 | 2.6139 | 0.0% |
| `model.15.attn.proj_conv.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0333 | 0.0226 | 0.0168 | 0.1561 | 0.3210 | 0.0% |
| `model.15.attn.proj_conv.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.16.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0011 | 0.0226 | -0.3130 | 0.2130 | 8.7079 | 0.0% |
| `model.16.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.8528 | 0.2423 | 0.3076 | 1.6514 | 10.0278 | 0.0% |
| `model.16.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.7594 | 0.9093 | -3.2148 | 2.1523 | 13.3724 | 0.0% |
| `model.16.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.0264 | 0.3019 | -0.9277 | 0.7251 | 3.4148 | 0.0% |
| `model.16.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1063 | 0.0444 | 0.0499 | 0.3853 | 1.3024 | 0.0% |
| `model.16.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.18.cv1.conv.weight` | `weight` | `[256, 384, 1, 1]` | `float16` | `98,304` | -0.0021 | 0.0231 | -0.3718 | 0.1870 | 7.2853 | 0.0% |
| `model.18.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.0495 | 0.2892 | 0.4438 | 2.7129 | 17.4156 | 0.0% |
| `model.18.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.4714 | 1.0180 | -3.1523 | 4.7266 | 17.9202 | 0.0% |
| `model.18.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | 0.1281 | 0.1818 | -0.5347 | 0.6514 | 3.5538 | 0.0% |
| `model.18.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.0245 | 0.0109 | 0.0111 | 0.0697 | 0.4287 | 0.0% |
| `model.18.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.18.cv2.conv.weight` | `weight` | `[256, 512, 1, 1]` | `float16` | `131,072` | -0.0015 | 0.0230 | -0.3369 | 0.3147 | 8.3544 | 0.0% |
| `model.18.cv2.bn.weight` | `weight` | `[256]` | `float16` | `256` | 0.9159 | 0.3275 | 0.3206 | 2.7988 | 15.5596 | 0.0% |
| `model.18.cv2.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.9916 | 1.1648 | -9.5156 | 2.1309 | 24.4484 | 0.0% |
| `model.18.cv2.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.2226 | 1.2581 | -9.4531 | 7.3984 | 20.4035 | 0.0% |
| `model.18.cv2.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.1299 | 0.0495 | 0.0309 | 0.4604 | 2.2244 | 0.0% |
| `model.18.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.18.m.0.cv1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0016 | 0.0228 | -0.3052 | 0.3313 | 8.7717 | 0.0% |
| `model.18.m.0.cv1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.0027 | 0.2283 | 0.3577 | 1.5625 | 11.6320 | 0.0% |
| `model.18.m.0.cv1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.9905 | 0.7863 | -2.1855 | 1.7666 | 14.2864 | 0.0% |
| `model.18.m.0.cv1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1661 | 0.3004 | -1.1729 | 0.3540 | 3.8713 | 0.0% |
| `model.18.m.0.cv1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1815 | 0.0820 | 0.0735 | 0.5049 | 2.2520 | 0.0% |
| `model.18.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.18.m.0.cv2.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0016 | 0.0226 | -0.3247 | 0.2498 | 8.7135 | 0.0% |
| `model.18.m.0.cv2.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.0806 | 0.2692 | 0.4380 | 1.9258 | 12.5958 | 0.0% |
| `model.18.m.0.cv2.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.8319 | 0.9107 | -3.7695 | 3.3691 | 13.9253 | 0.0% |
| `model.18.m.0.cv2.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | 0.0618 | 0.2544 | -1.3018 | 1.0078 | 2.9507 | 0.0% |
| `model.18.m.0.cv2.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1491 | 0.0768 | 0.0541 | 0.4761 | 1.8960 | 0.0% |
| `model.18.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.18.attn.bias` | `bias` | `[4]` | `float16` | `4` | -1.0500 | 0.9559 | -1.8867 | -0.0103 | 2.6742 | 0.0% |
| `model.18.attn.gl.weight` | `weight` | `[128, 512]` | `float16` | `65,536` | 0.0004 | 0.0413 | -0.2625 | 0.2925 | 10.5721 | 0.0% |
| `model.18.attn.gl.bias` | `bias` | `[128]` | `float16` | `128` | -0.5468 | 2.5505 | -15.3828 | 9.3906 | 29.4010 | 0.0% |
| `model.18.attn.proj_conv.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0000 | 0.0231 | -0.2345 | 0.2014 | 8.8519 | 0.0% |
| `model.18.attn.proj_conv.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.6238 | 0.3310 | 0.0018 | 3.0527 | 18.7462 | 0.0% |
| `model.18.attn.proj_conv.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.4972 | 4.5873 | -30.1250 | 14.2656 | 52.0016 | 0.0% |
| `model.18.attn.proj_conv.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.0146 | 0.3950 | -1.8887 | 0.9136 | 4.4550 | 0.0% |
| `model.18.attn.proj_conv.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.1322 | 0.0521 | 0.0804 | 0.5425 | 1.6068 | 0.0% |
| `model.18.attn.proj_conv.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.19.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | -0.0014 | 0.0229 | -0.5181 | 0.3240 | 17.5930 | 0.0% |
| `model.19.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.0434 | 0.2809 | 0.4099 | 2.6582 | 17.2862 | 0.0% |
| `model.19.bn.bias` | `bias` | `[256]` | `float16` | `256` | -0.8759 | 0.8453 | -3.4570 | 1.7930 | 19.4581 | 0.0% |
| `model.19.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.0048 | 0.4578 | -1.7051 | 1.5928 | 7.3102 | 0.0% |
| `model.19.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.5131 | 0.2481 | 0.2106 | 3.2207 | 9.1156 | 0.0% |
| `model.19.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.21.cv1.conv.weight` | `weight` | `[512, 768, 1, 1]` | `float16` | `393,216` | -0.0026 | 0.0237 | -0.8633 | 0.3999 | 14.9656 | 0.0% |
| `model.21.cv1.bn.weight` | `weight` | `[512]` | `float16` | `512` | 1.3785 | 0.4490 | 0.4775 | 2.9336 | 32.8024 | 0.0% |
| `model.21.cv1.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.1618 | 1.1537 | -4.1172 | 3.9668 | 37.0305 | 0.0% |
| `model.21.cv1.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | 0.1841 | 0.2478 | -1.5625 | 1.0244 | 6.9812 | 0.0% |
| `model.21.cv1.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.0636 | 0.0259 | 0.0352 | 0.2856 | 1.5545 | 0.0% |
| `model.21.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.21.cv2.conv.weight` | `weight` | `[512, 1024, 1, 1]` | `float16` | `524,288` | -0.0014 | 0.0232 | -0.4155 | 0.4048 | 16.8550 | 0.0% |
| `model.21.cv2.bn.weight` | `weight` | `[512]` | `float16` | `512` | 1.1556 | 0.3385 | 0.3669 | 2.8945 | 27.2448 | 0.0% |
| `model.21.cv2.bn.bias` | `bias` | `[512]` | `float16` | `512` | -1.5445 | 1.0441 | -9.6172 | 4.0781 | 42.1706 | 0.0% |
| `model.21.cv2.bn.running_mean` | `running_mean` | `[512]` | `float16` | `512` | -0.6802 | 7.4753 | -73.0625 | 57.1875 | 169.6820 | 0.0% |
| `model.21.cv2.bn.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.3652 | 0.1238 | 0.1334 | 0.8018 | 8.7246 | 0.0% |
| `model.21.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.21.m.0.cv1.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | -0.0018 | 0.0229 | -0.3760 | 0.3157 | 17.6716 | 0.0% |
| `model.21.m.0.cv1.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.2565 | 0.3108 | 0.4426 | 2.1719 | 20.7077 | 0.0% |
| `model.21.m.0.cv1.bn.bias` | `bias` | `[256]` | `float16` | `256` | -1.3743 | 1.0701 | -3.7012 | 1.6885 | 27.8476 | 0.0% |
| `model.21.m.0.cv1.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.0222 | 0.4630 | -4.4844 | 0.8960 | 7.4022 | 0.0% |
| `model.21.m.0.cv1.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.4973 | 0.2303 | 0.1785 | 1.5898 | 8.7656 | 0.0% |
| `model.21.m.0.cv1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.21.m.0.cv2.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | -0.0020 | 0.0231 | -0.5479 | 0.2620 | 17.7866 | 0.0% |
| `model.21.m.0.cv2.bn.weight` | `weight` | `[256]` | `float16` | `256` | 1.3203 | 0.3413 | 0.4097 | 2.4805 | 21.8164 | 0.0% |
| `model.21.m.0.cv2.bn.bias` | `bias` | `[256]` | `float16` | `256` | -1.2843 | 1.0919 | -3.0449 | 4.9766 | 26.9496 | 0.0% |
| `model.21.m.0.cv2.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | -0.0779 | 0.4399 | -2.6387 | 0.7998 | 7.1339 | 0.0% |
| `model.21.m.0.cv2.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.6323 | 0.3217 | 0.2803 | 2.0059 | 11.3459 | 0.0% |
| `model.21.m.0.cv2.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.21.attn.bias` | `bias` | `[8]` | `float16` | `8` | -0.8549 | 1.2976 | -2.3340 | 1.2158 | 4.1993 | 0.0% |
| `model.21.attn.gl.weight` | `weight` | `[256, 512]` | `float16` | `131,072` | -0.0001 | 0.0532 | -0.4973 | 0.4775 | 19.2641 | 0.0% |
| `model.21.attn.gl.bias` | `bias` | `[256]` | `float16` | `256` | -0.9690 | 2.8127 | -17.0000 | 7.2266 | 47.5162 | 0.0% |
| `model.21.attn.proj_conv.conv.weight` | `weight` | `[256, 256, 3, 3]` | `float16` | `589,824` | 0.0001 | 0.0237 | -0.4011 | 0.3542 | 18.1880 | 0.0% |
| `model.21.attn.proj_conv.bn.weight` | `weight` | `[256]` | `float16` | `256` | 2.1435 | 0.8467 | -0.0000 | 4.8867 | 36.8648 | 0.0% |
| `model.21.attn.proj_conv.bn.bias` | `bias` | `[256]` | `float16` | `256` | 0.7190 | 6.8975 | -32.0938 | 33.6250 | 110.7436 | 0.0% |
| `model.21.attn.proj_conv.bn.running_mean` | `running_mean` | `[256]` | `float16` | `256` | 0.0167 | 0.6105 | -3.0078 | 2.8867 | 9.7524 | 0.0% |
| `model.21.attn.proj_conv.bn.running_var` | `running_var` | `[256]` | `float16` | `256` | 0.6727 | 0.6143 | 0.2681 | 3.3809 | 14.5630 | 0.0% |
| `model.21.attn.proj_conv.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.0.0.conv.weight` | `weight` | `[64, 128, 3, 3]` | `float16` | `73,728` | -0.0013 | 0.0227 | -0.2869 | 0.2018 | 6.1612 | 0.0% |
| `model.22.cv2.0.0.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.3395 | 0.8689 | 0.4971 | 4.5039 | 12.7437 | 0.0% |
| `model.22.cv2.0.0.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.4749 | 0.9624 | -2.7988 | 3.2773 | 8.5318 | 0.0% |
| `model.22.cv2.0.0.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | 0.0115 | 0.2849 | -0.7363 | 0.6582 | 2.2632 | 0.0% |
| `model.22.cv2.0.0.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.0875 | 0.0414 | 0.0299 | 0.1814 | 0.7729 | 0.0% |
| `model.22.cv2.0.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.0.1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0010 | 0.0254 | -0.4070 | 0.2700 | 4.8866 | 0.0% |
| `model.22.cv2.0.1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 4.1274 | 2.0417 | 1.4102 | 10.2188 | 36.7815 | 0.0% |
| `model.22.cv2.0.1.bn.bias` | `bias` | `[64]` | `float16` | `64` | 0.9621 | 1.5142 | -3.4414 | 3.8242 | 14.2716 | 0.0% |
| `model.22.cv2.0.1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.0891 | 0.3334 | -1.1074 | 0.5083 | 2.7405 | 0.0% |
| `model.22.cv2.0.1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.2793 | 0.1355 | 0.1215 | 0.6704 | 2.4799 | 0.0% |
| `model.22.cv2.0.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.0.2.weight` | `weight` | `[64, 64, 1, 1]` | `float16` | `4,096` | -0.0358 | 0.1202 | -0.8076 | 0.2478 | 8.0231 | 0.0% |
| `model.22.cv2.0.2.bias` | `bias` | `[64]` | `float16` | `64` | 0.8249 | 0.5645 | -0.5459 | 1.8027 | 7.9766 | 0.0% |
| `model.22.cv2.1.0.conv.weight` | `weight` | `[64, 256, 3, 3]` | `float16` | `147,456` | -0.0010 | 0.0230 | -0.4512 | 0.3345 | 8.8319 | 0.0% |
| `model.22.cv2.1.0.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.3176 | 0.7287 | 0.4670 | 3.0938 | 12.0237 | 0.0% |
| `model.22.cv2.1.0.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.3775 | 1.0557 | -3.2910 | 2.2793 | 8.9074 | 0.0% |
| `model.22.cv2.1.0.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.2387 | 0.3798 | -1.1670 | 0.5620 | 3.5685 | 0.0% |
| `model.22.cv2.1.0.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.4183 | 0.1561 | 0.1689 | 0.8506 | 3.5686 | 0.0% |
| `model.22.cv2.1.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.1.1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0008 | 0.0253 | -0.4023 | 0.4207 | 4.8665 | 0.0% |
| `model.22.cv2.1.1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 4.1970 | 1.6457 | 1.5264 | 7.8945 | 36.0274 | 0.0% |
| `model.22.cv2.1.1.bn.bias` | `bias` | `[64]` | `float16` | `64` | 1.4049 | 1.0746 | -3.9902 | 3.2305 | 14.1096 | 0.0% |
| `model.22.cv2.1.1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.1030 | 0.2741 | -0.7822 | 0.7939 | 2.3268 | 0.0% |
| `model.22.cv2.1.1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 0.3815 | 0.1336 | 0.1276 | 0.6519 | 3.2309 | 0.0% |
| `model.22.cv2.1.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.1.2.weight` | `weight` | `[64, 64, 1, 1]` | `float16` | `4,096` | -0.0337 | 0.1168 | -0.9141 | 0.3154 | 7.7803 | 0.0% |
| `model.22.cv2.1.2.bias` | `bias` | `[64]` | `float16` | `64` | 0.8720 | 0.5908 | -1.0459 | 2.6211 | 8.4055 | 0.0% |
| `model.22.cv2.2.0.conv.weight` | `weight` | `[64, 512, 3, 3]` | `float16` | `294,912` | -0.0008 | 0.0226 | -0.4253 | 0.3674 | 12.2970 | 0.0% |
| `model.22.cv2.2.0.bn.weight` | `weight` | `[64]` | `float16` | `64` | 1.6626 | 0.7686 | 0.7007 | 3.8848 | 14.6333 | 0.0% |
| `model.22.cv2.2.0.bn.bias` | `bias` | `[64]` | `float16` | `64` | -0.0950 | 1.4776 | -3.3770 | 3.2578 | 11.7527 | 0.0% |
| `model.22.cv2.2.0.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.3453 | 0.4681 | -1.8145 | 0.4631 | 4.6298 | 0.0% |
| `model.22.cv2.2.0.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 1.3706 | 0.4330 | 0.7578 | 2.4883 | 11.4908 | 0.0% |
| `model.22.cv2.2.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.2.1.conv.weight` | `weight` | `[64, 64, 3, 3]` | `float16` | `36,864` | -0.0007 | 0.0254 | -0.4128 | 0.3999 | 4.8826 | 0.0% |
| `model.22.cv2.2.1.bn.weight` | `weight` | `[64]` | `float16` | `64` | 5.1505 | 1.8062 | 2.1582 | 8.7109 | 43.6270 | 0.0% |
| `model.22.cv2.2.1.bn.bias` | `bias` | `[64]` | `float16` | `64` | 1.8581 | 1.1351 | -1.7588 | 3.8770 | 17.3820 | 0.0% |
| `model.22.cv2.2.1.bn.running_mean` | `running_mean` | `[64]` | `float16` | `64` | -0.1455 | 0.3597 | -0.8711 | 0.7266 | 3.0828 | 0.0% |
| `model.22.cv2.2.1.bn.running_var` | `running_var` | `[64]` | `float16` | `64` | 1.0029 | 0.3268 | 0.3176 | 2.1328 | 8.4320 | 0.0% |
| `model.22.cv2.2.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv2.2.2.weight` | `weight` | `[64, 64, 1, 1]` | `float16` | `4,096` | -0.0334 | 0.1141 | -0.7910 | 0.1985 | 7.6044 | 0.0% |
| `model.22.cv2.2.2.bias` | `bias` | `[64]` | `float16` | `64` | 1.3268 | 0.9719 | -1.4971 | 3.5215 | 13.1217 | 0.0% |
| `model.22.cv3.0.0.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0017 | 0.0231 | -0.3010 | 0.2074 | 8.9100 | 0.0% |
| `model.22.cv3.0.0.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.8396 | 0.3078 | 0.3035 | 2.0586 | 10.1121 | 0.0% |
| `model.22.cv3.0.0.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.7477 | 0.7340 | -2.5898 | 2.7695 | 11.8319 | 0.0% |
| `model.22.cv3.0.0.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.0777 | 0.3502 | -1.5049 | 0.7549 | 4.0435 | 0.0% |
| `model.22.cv3.0.0.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.0972 | 0.0633 | 0.0394 | 0.5889 | 1.3113 | 0.0% |
| `model.22.cv3.0.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.0.1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0019 | 0.0233 | -0.4780 | 0.3052 | 8.9587 | 0.0% |
| `model.22.cv3.0.1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.1887 | 0.3993 | 0.3618 | 2.3594 | 14.1811 | 0.0% |
| `model.22.cv3.0.1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.0128 | 0.7755 | -1.9062 | 1.7705 | 8.7410 | 0.0% |
| `model.22.cv3.0.1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1388 | 0.3125 | -0.8643 | 0.7588 | 3.8556 | 0.0% |
| `model.22.cv3.0.1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.2751 | 0.1244 | 0.0443 | 0.7881 | 3.4135 | 0.0% |
| `model.22.cv3.0.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.0.2.weight` | `weight` | `[512, 128, 1, 1]` | `float16` | `65,536` | -0.0001 | 0.0213 | -0.1189 | 0.1593 | 5.4639 | 0.0% |
| `model.22.cv3.0.2.bias` | `bias` | `[512]` | `float16` | `512` | -0.0189 | 0.5955 | -1.5615 | 1.7422 | 13.4690 | 0.0% |
| `model.22.cv3.1.0.conv.weight` | `weight` | `[128, 256, 3, 3]` | `float16` | `294,912` | -0.0013 | 0.0233 | -0.5693 | 0.2356 | 12.6795 | 0.0% |
| `model.22.cv3.1.0.bn.weight` | `weight` | `[128]` | `float16` | `128` | 0.9744 | 0.2996 | 0.4192 | 2.0762 | 11.5295 | 0.0% |
| `model.22.cv3.1.0.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.7868 | 1.0214 | -5.0859 | 2.7520 | 14.5515 | 0.0% |
| `model.22.cv3.1.0.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.0688 | 0.4403 | -1.8594 | 0.8828 | 5.0226 | 0.0% |
| `model.22.cv3.1.0.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.4131 | 0.2228 | 0.1866 | 2.2637 | 5.3048 | 0.0% |
| `model.22.cv3.1.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.1.1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0022 | 0.0232 | -0.5801 | 0.2223 | 8.9596 | 0.0% |
| `model.22.cv3.1.1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.3691 | 0.7009 | 0.3320 | 7.5273 | 17.3872 | 0.0% |
| `model.22.cv3.1.1.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.0383 | 0.9088 | -2.7168 | 1.8564 | 10.2506 | 0.0% |
| `model.22.cv3.1.1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.3981 | 0.4331 | -2.2598 | 0.8809 | 6.6408 | 0.0% |
| `model.22.cv3.1.1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 0.5175 | 0.2525 | 0.0948 | 2.4863 | 6.5101 | 0.0% |
| `model.22.cv3.1.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.1.2.weight` | `weight` | `[512, 128, 1, 1]` | `float16` | `65,536` | -0.0000 | 0.0216 | -0.1204 | 0.1195 | 5.5333 | 0.0% |
| `model.22.cv3.1.2.bias` | `bias` | `[512]` | `float16` | `512` | 0.0220 | 0.5526 | -1.5459 | 1.3574 | 12.5006 | 0.0% |
| `model.22.cv3.2.0.conv.weight` | `weight` | `[128, 512, 3, 3]` | `float16` | `589,824` | -0.0012 | 0.0236 | -0.4160 | 0.1974 | 18.1574 | 0.0% |
| `model.22.cv3.2.0.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.2562 | 0.4283 | 0.5249 | 2.6504 | 15.0089 | 0.0% |
| `model.22.cv3.2.0.bn.bias` | `bias` | `[128]` | `float16` | `128` | -0.7425 | 1.1572 | -6.2422 | 2.5820 | 15.5122 | 0.0% |
| `model.22.cv3.2.0.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.1994 | 0.5254 | -2.9473 | 1.2803 | 6.3365 | 0.0% |
| `model.22.cv3.2.0.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 1.1894 | 0.4831 | 0.6250 | 4.7344 | 14.5166 | 0.0% |
| `model.22.cv3.2.0.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.2.1.conv.weight` | `weight` | `[128, 128, 3, 3]` | `float16` | `147,456` | -0.0024 | 0.0234 | -0.4871 | 0.1899 | 9.0286 | 0.0% |
| `model.22.cv3.2.1.bn.weight` | `weight` | `[128]` | `float16` | `128` | 1.5370 | 0.9113 | 0.7896 | 10.1172 | 20.1956 | 0.0% |
| `model.22.cv3.2.1.bn.bias` | `bias` | `[128]` | `float16` | `128` | 0.1067 | 0.8684 | -2.5898 | 1.9785 | 9.8608 | 0.0% |
| `model.22.cv3.2.1.bn.running_mean` | `running_mean` | `[128]` | `float16` | `128` | -0.6954 | 0.5409 | -3.2520 | 0.8013 | 9.9524 | 0.0% |
| `model.22.cv3.2.1.bn.running_var` | `running_var` | `[128]` | `float16` | `128` | 1.0428 | 0.3537 | 0.2280 | 2.6055 | 12.4534 | 0.0% |
| `model.22.cv3.2.1.bn.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv3.2.2.weight` | `weight` | `[512, 128, 1, 1]` | `float16` | `65,536` | -0.0001 | 0.0219 | -0.1087 | 0.1405 | 5.6189 | 0.0% |
| `model.22.cv3.2.2.bias` | `bias` | `[512]` | `float16` | `512` | -0.0313 | 0.5612 | -1.5703 | 1.6533 | 12.7061 | 0.0% |
| `model.22.dfl.conv.weight` | `weight` | `[1, 16, 1, 1]` | `float16` | `16` | 7.5000 | 4.7610 | 0.0000 | 15.0000 | 35.2136 | 6.2% |
| `model.22.cv4.0.bias` | `bias` | `[]` | `float16` | `1` | -11.9531 | 0.0000 | -11.9531 | -11.9531 | 11.9531 | 0.0% |
| `model.22.cv4.0.logit_scale` | `parameter` | `[]` | `float16` | `1` | 0.5801 | 0.0000 | 0.5801 | 0.5801 | 0.5801 | 0.0% |
| `model.22.cv4.0.norm.weight` | `weight` | `[512]` | `float16` | `512` | 1.7123 | 0.5482 | 0.0001 | 4.3242 | 40.6784 | 0.0% |
| `model.22.cv4.0.norm.bias` | `bias` | `[512]` | `float16` | `512` | -0.0583 | 1.3062 | -4.6055 | 3.4590 | 29.5570 | 0.0% |
| `model.22.cv4.0.norm.running_mean` | `running_mean` | `[512]` | `float16` | `512` | -0.0217 | 0.6104 | -1.4756 | 1.8389 | 13.8060 | 0.0% |
| `model.22.cv4.0.norm.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.0306 | 0.0130 | 0.0126 | 0.1082 | 0.7522 | 0.0% |
| `model.22.cv4.0.norm.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv4.1.bias` | `bias` | `[]` | `float16` | `1` | -10.4062 | 0.0000 | -10.4062 | -10.4062 | 10.4062 | 0.0% |
| `model.22.cv4.1.logit_scale` | `parameter` | `[]` | `float16` | `1` | 0.5596 | 0.0000 | 0.5596 | 0.5596 | 0.5596 | 0.0% |
| `model.22.cv4.1.norm.weight` | `weight` | `[512]` | `float16` | `512` | 1.5954 | 0.4654 | 0.1554 | 3.9180 | 37.6012 | 0.0% |
| `model.22.cv4.1.norm.bias` | `bias` | `[512]` | `float16` | `512` | -0.0640 | 1.3063 | -4.6680 | 3.7363 | 29.5641 | 0.0% |
| `model.22.cv4.1.norm.running_mean` | `running_mean` | `[512]` | `float16` | `512` | 0.0209 | 0.5812 | -1.4561 | 1.4834 | 13.1466 | 0.0% |
| `model.22.cv4.1.norm.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.0420 | 0.0166 | 0.0173 | 0.1743 | 1.0215 | 0.0% |
| `model.22.cv4.1.norm.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |
| `model.22.cv4.2.bias` | `bias` | `[]` | `float16` | `1` | -8.8672 | 0.0000 | -8.8672 | -8.8672 | 8.8672 | 0.0% |
| `model.22.cv4.2.logit_scale` | `parameter` | `[]` | `float16` | `1` | 0.6396 | 0.0000 | 0.6396 | 0.6396 | 0.6396 | 0.0% |
| `model.22.cv4.2.norm.weight` | `weight` | `[512]` | `float16` | `512` | 1.4609 | 0.4161 | 0.0214 | 3.5254 | 34.3686 | 0.0% |
| `model.22.cv4.2.norm.bias` | `bias` | `[512]` | `float16` | `512` | -0.0471 | 1.3412 | -4.1055 | 3.7012 | 30.3379 | 0.0% |
| `model.22.cv4.2.norm.running_mean` | `running_mean` | `[512]` | `float16` | `512` | -0.0314 | 0.6039 | -1.7900 | 1.8418 | 13.6692 | 0.0% |
| `model.22.cv4.2.norm.running_var` | `running_var` | `[512]` | `float16` | `512` | 0.0688 | 0.0349 | 0.0302 | 0.4604 | 1.7453 | 0.0% |
| `model.22.cv4.2.norm.num_batches_tracked` | `parameter` | `[]` | `int64` | `1` | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0% |

---

<sub>Report auto-generated by SETU Model Evaluation Engine at 2026-10-29T14:31:34.033360+00:00.</sub>