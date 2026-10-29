# Model Evaluation Matrix: Qwen2.5-0.5B Small Language Model (SLM)

> **Model ID:** `qwen2.5_0.5b`  
> **Task:** Neural ISL Gloss-to-English Translation & Contextual Autocomplete  
> **Architecture:** Qwen2ForCausalLM (24 Layers, 14 Attention Heads, RoPE)  
> **File:** `models\weights\qwen_0.5b` (953.28 MB)  
> **Last Modified:** 2026-10-28T18:22:28.829934+00:00  
> **Total Parameters:** `494,032,768` (494,005,120 weights, 27,648 biases)  

---

## 1. Training Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-05` | Training specification |
| **Optimizer** | `AdamW (Beta1=0.9, Beta2=0.95)` | Training specification |
| **Loss Function** | `Causal Next-Token CrossEntropy` | Training specification |
| **Hidden Size** | `896` | Training specification |
| **Intermediate Size** | `4864` | Training specification |
| **Num Hidden Layers** | `24` | Training specification |
| **Num Attention Heads** | `14` | Training specification |
| **Num Key Value Heads** | `2` | Training specification |
| **Max Position Embeddings** | `32768` | Training specification |
| **Torch Dtype** | `bfloat16 / float16` | Training specification |
| **Device** | `cuda (RTX 3050)` | Training specification |

---

## 2. Evaluation & Performance Metrics

| Metric | Empirical Result | Target Benchmark |
| :--- | :--- | :--- |
| **Test Accuracy** | **Fluent Grammatical English (SOV -> SVO)** | Production SLA Passed |
| **Cached Latency Ms** | **0.0008** | Production SLA Passed |
| **Generation Latency Ms** | **250.0000** | Production SLA Passed |

---

## 3. Layer-by-Layer Weights & Biases Breakdown

| Layer / Tensor Name | Type | Shape | Dtype | Parameters | Mean | Std | Min | Max | L2 Norm | Sparsity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `model.embed_tokens.weight` | `weight` | `[151936, 896]` | `bfloat16` | `136,134,656` | 0.0001 | 0.0152 | -0.2012 | 0.1660 | 175.0522 | 0.0% |
| `model.layers.0.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 0.0322 | 0.0925 | -0.4473 | 0.7266 | 2.9288 | 0.0% |
| `model.layers.0.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0182 | -0.3047 | 0.3027 | 37.9063 | 0.0% |
| `model.layers.0.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0230 | -0.4922 | 0.3848 | 48.1132 | 0.0% |
| `model.layers.0.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0183 | -0.2080 | 0.2793 | 38.1111 | 0.0% |
| `model.layers.0.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 0.6782 | 0.1510 | 0.4883 | 1.9922 | 20.7965 | 0.0% |
| `model.layers.0.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 3.8704 | 32.3443 | -130.0000 | 121.0000 | 367.1221 | 0.0% |
| `model.layers.0.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0806 | -1.0234 | 0.9180 | 27.2946 | 0.0% |
| `model.layers.0.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0131 | -0.3379 | 0.5625 | 11.7639 | 0.0% |
| `model.layers.0.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.1520 | 7.8570 | -79.0000 | 47.7500 | 235.0992 | 0.0% |
| `model.layers.0.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0667 | -1.2266 | 1.1719 | 59.8005 | 0.0% |
| `model.layers.0.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0020 | 0.0149 | -0.0277 | 0.0981 | 0.1691 | 0.0% |
| `model.layers.0.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0115 | -0.0947 | 0.1211 | 3.8851 | 0.0% |
| `model.layers.1.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 0.6843 | 0.3229 | -0.0608 | 3.2500 | 22.6470 | 0.0% |
| `model.layers.1.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0170 | -0.3594 | 0.3066 | 35.4849 | 0.0% |
| `model.layers.1.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0233 | -0.4121 | 0.3594 | 48.5878 | 0.0% |
| `model.layers.1.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0170 | -0.2383 | 0.2734 | 35.5917 | 0.0% |
| `model.layers.1.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.0063 | 0.2520 | 0.0001 | 1.6797 | 31.0494 | 0.0% |
| `model.layers.1.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 1.6682 | 18.4923 | -90.5000 | 147.0000 | 209.2510 | 0.0% |
| `model.layers.1.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0406 | -0.2383 | 0.2852 | 13.7371 | 0.0% |
| `model.layers.1.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0149 | -0.2031 | 0.2061 | 13.3211 | 0.0% |
| `model.layers.1.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.1425 | 1.1680 | -10.5625 | 8.3750 | 35.2024 | 0.0% |
| `model.layers.1.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0001 | 0.0287 | -0.2793 | 0.3457 | 25.7365 | 0.0% |
| `model.layers.1.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0213 | 0.2417 | -2.7031 | 0.1094 | 2.7346 | 0.0% |
| `model.layers.1.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0115 | -0.0825 | 0.0762 | 3.9040 | 0.0% |
| `model.layers.10.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.4935 | 0.9276 | -0.1367 | 4.5000 | 52.6180 | 0.0% |
| `model.layers.10.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0176 | -0.4551 | 0.3945 | 36.6885 | 0.0% |
| `model.layers.10.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0213 | -0.4336 | 0.5859 | 44.3674 | 0.0% |
| `model.layers.10.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0185 | -0.4258 | 0.4297 | 38.6707 | 0.0% |
| `model.layers.10.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3031 | 0.6394 | 0.1729 | 3.3438 | 43.4432 | 0.0% |
| `model.layers.10.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0140 | 0.9762 | -5.2812 | 2.2500 | 11.0029 | 0.0% |
| `model.layers.10.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0244 | -0.1924 | 0.1787 | 8.2719 | 0.0% |
| `model.layers.10.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0182 | -0.2217 | 0.3340 | 16.2924 | 0.0% |
| `model.layers.10.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.1632 | 2.1423 | -22.7500 | 6.7500 | 64.2761 | 0.0% |
| `model.layers.10.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0214 | -0.2461 | 0.3125 | 19.2065 | 0.0% |
| `model.layers.10.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0188 | 0.1376 | -0.4766 | 0.6523 | 1.5650 | 0.0% |
| `model.layers.10.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0202 | -0.1328 | 0.1050 | 6.8381 | 0.0% |
| `model.layers.11.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.7830 | 1.1332 | -0.0684 | 10.5625 | 63.2285 | 0.0% |
| `model.layers.11.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0179 | -0.5977 | 0.4590 | 37.3949 | 0.0% |
| `model.layers.11.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0194 | -0.4941 | 0.3711 | 40.5738 | 0.0% |
| `model.layers.11.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0185 | -0.6797 | 0.4414 | 38.6830 | 0.0% |
| `model.layers.11.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3614 | 0.6439 | 0.2109 | 2.9844 | 45.0749 | 0.0% |
| `model.layers.11.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0574 | 0.7298 | -2.5781 | 4.3125 | 8.2506 | 0.0% |
| `model.layers.11.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0234 | -0.2148 | 0.2344 | 7.9284 | 0.0% |
| `model.layers.11.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0174 | -0.5781 | 0.2949 | 15.6306 | 0.0% |
| `model.layers.11.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.3967 | 2.4302 | -25.6250 | 3.5625 | 73.6655 | 0.0% |
| `model.layers.11.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0206 | -0.2256 | 0.2314 | 18.4899 | 0.0% |
| `model.layers.11.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0239 | 0.1759 | -0.4043 | 1.3125 | 2.0010 | 0.0% |
| `model.layers.11.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0169 | -0.0908 | 0.0884 | 5.7194 | 0.0% |
| `model.layers.12.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5877 | 0.9992 | -0.1206 | 6.7188 | 56.1436 | 0.0% |
| `model.layers.12.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0182 | -0.4395 | 0.3652 | 38.0403 | 0.0% |
| `model.layers.12.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0200 | -0.2832 | 0.3359 | 41.8241 | 0.0% |
| `model.layers.12.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0191 | -0.3848 | 0.2539 | 39.9335 | 0.0% |
| `model.layers.12.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2976 | 0.6166 | 0.2139 | 4.5000 | 43.0005 | 0.0% |
| `model.layers.12.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0489 | 1.0635 | -3.4219 | 4.4688 | 11.9975 | 0.0% |
| `model.layers.12.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0236 | -0.1885 | 0.1816 | 8.0005 | 0.0% |
| `model.layers.12.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0179 | -0.3242 | 0.1963 | 16.0116 | 0.0% |
| `model.layers.12.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.2241 | 2.0307 | -9.0625 | 20.3750 | 61.1221 | 0.0% |
| `model.layers.12.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0215 | -0.2393 | 0.2598 | 19.2770 | 0.0% |
| `model.layers.12.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0014 | 0.0708 | -0.2520 | 0.2852 | 0.7981 | 0.0% |
| `model.layers.12.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0187 | -0.1348 | 0.1167 | 6.3311 | 0.0% |
| `model.layers.13.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.7091 | 0.9903 | -0.1187 | 9.0625 | 59.1192 | 0.0% |
| `model.layers.13.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0185 | -0.4648 | 0.5156 | 38.6903 | 0.0% |
| `model.layers.13.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0194 | -0.3887 | 0.4395 | 40.4088 | 0.0% |
| `model.layers.13.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0195 | -0.3770 | 0.4766 | 40.6512 | 0.0% |
| `model.layers.13.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3177 | 0.5485 | 0.2129 | 2.6875 | 42.7212 | 0.0% |
| `model.layers.13.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0905 | 1.0157 | -3.4219 | 4.2188 | 11.4920 | 0.0% |
| `model.layers.13.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0242 | -0.2754 | 0.2852 | 8.1902 | 0.0% |
| `model.layers.13.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0174 | -0.3652 | 0.2402 | 15.5946 | 0.0% |
| `model.layers.13.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.2658 | 1.9197 | -4.4062 | 16.0000 | 57.9791 | 0.0% |
| `model.layers.13.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0220 | -0.4375 | 0.4492 | 19.6936 | 0.0% |
| `model.layers.13.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0004 | 0.0984 | -0.3848 | 0.3418 | 1.1089 | 0.0% |
| `model.layers.13.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0184 | -0.1147 | 0.0977 | 6.2171 | 0.0% |
| `model.layers.14.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.6901 | 0.7390 | 0.0688 | 3.5312 | 55.2087 | 0.0% |
| `model.layers.14.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0184 | -0.4238 | 0.3672 | 38.4571 | 0.0% |
| `model.layers.14.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0201 | -0.4746 | 0.4434 | 41.9599 | 0.0% |
| `model.layers.14.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0194 | -0.3828 | 0.4746 | 40.5537 | 0.0% |
| `model.layers.14.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2739 | 0.4447 | 0.3008 | 2.5469 | 40.3871 | 0.0% |
| `model.layers.14.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0316 | 1.1077 | -5.2188 | 3.5625 | 12.4881 | 0.0% |
| `model.layers.14.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0221 | -0.1660 | 0.1865 | 7.4942 | 0.0% |
| `model.layers.14.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0183 | -0.2461 | 0.3789 | 16.4174 | 0.0% |
| `model.layers.14.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.1433 | 2.0777 | -15.6250 | 15.8125 | 62.3060 | 0.0% |
| `model.layers.14.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0211 | -0.2090 | 0.2500 | 18.9049 | 0.0% |
| `model.layers.14.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0113 | 0.1301 | -0.4043 | 0.4902 | 1.4717 | 0.0% |
| `model.layers.14.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0205 | -0.1348 | 0.1191 | 6.9548 | 0.0% |
| `model.layers.15.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5964 | 0.7992 | -0.0630 | 5.0938 | 53.4335 | 0.0% |
| `model.layers.15.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0190 | -0.4141 | 0.4355 | 39.5678 | 0.0% |
| `model.layers.15.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0200 | -0.4805 | 0.4219 | 41.8401 | 0.0% |
| `model.layers.15.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0199 | -0.7578 | 0.4727 | 41.6480 | 0.0% |
| `model.layers.15.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3249 | 0.4341 | 0.3340 | 3.0938 | 41.7304 | 0.0% |
| `model.layers.15.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0266 | 0.9881 | -4.7188 | 4.4688 | 11.1391 | 0.0% |
| `model.layers.15.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0244 | -0.1895 | 0.1592 | 8.2674 | 0.0% |
| `model.layers.15.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0180 | -0.3027 | 0.3809 | 16.1555 | 0.0% |
| `model.layers.15.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0665 | 2.1994 | -13.8750 | 21.0000 | 65.8296 | 0.0% |
| `model.layers.15.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0208 | -0.2109 | 0.1865 | 18.6281 | 0.0% |
| `model.layers.15.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0121 | 0.1237 | -0.5508 | 0.5859 | 1.4003 | 0.0% |
| `model.layers.15.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0200 | -0.1299 | 0.1133 | 6.7636 | 0.0% |
| `model.layers.16.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.6876 | 0.8475 | 0.1001 | 9.1250 | 56.5219 | 0.0% |
| `model.layers.16.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0182 | -0.4434 | 0.4316 | 37.9858 | 0.0% |
| `model.layers.16.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0213 | -0.4180 | 0.3848 | 44.4695 | 0.0% |
| `model.layers.16.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0191 | -0.3555 | 0.3828 | 39.9648 | 0.0% |
| `model.layers.16.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.4060 | 0.3485 | 0.4082 | 3.3125 | 43.3590 | 0.0% |
| `model.layers.16.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.1794 | 1.1629 | -5.0312 | 3.6094 | 13.2619 | 0.0% |
| `model.layers.16.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0226 | -0.1953 | 0.2158 | 7.6607 | 0.0% |
| `model.layers.16.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0194 | -0.3027 | 0.3145 | 17.3911 | 0.0% |
| `model.layers.16.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.2525 | 1.8062 | -15.8750 | 6.4062 | 54.5627 | 0.0% |
| `model.layers.16.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0001 | 0.0249 | -0.2451 | 0.3965 | 22.2983 | 0.0% |
| `model.layers.16.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0057 | 0.0731 | -0.4961 | 0.2832 | 0.8263 | 0.0% |
| `model.layers.16.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0183 | -0.1914 | 0.1650 | 6.1905 | 0.0% |
| `model.layers.17.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5396 | 0.3788 | 0.0903 | 3.1719 | 47.4588 | 0.0% |
| `model.layers.17.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0179 | -0.2793 | 0.3594 | 37.3637 | 0.0% |
| `model.layers.17.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0222 | -0.4805 | 0.3867 | 46.3573 | 0.0% |
| `model.layers.17.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0186 | -0.2021 | 0.2285 | 38.8485 | 0.0% |
| `model.layers.17.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.6684 | 0.4010 | 0.3184 | 4.3750 | 51.3625 | 0.0% |
| `model.layers.17.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.1031 | 0.8855 | -4.6875 | 2.1875 | 10.0473 | 0.0% |
| `model.layers.17.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0219 | -0.2051 | 0.1748 | 7.4330 | 0.0% |
| `model.layers.17.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0190 | -0.2520 | 0.2754 | 17.0172 | 0.0% |
| `model.layers.17.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0062 | 1.9724 | -16.8750 | 14.2500 | 59.0076 | 0.0% |
| `model.layers.17.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0219 | -0.2578 | 0.2734 | 19.6127 | 0.0% |
| `model.layers.17.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0382 | 0.4143 | -4.5312 | 0.4902 | 4.6889 | 0.0% |
| `model.layers.17.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0206 | -0.1582 | 0.1367 | 6.9819 | 0.0% |
| `model.layers.18.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5483 | 0.3905 | -0.0476 | 3.6562 | 47.7944 | 0.0% |
| `model.layers.18.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0187 | -0.3535 | 0.4512 | 38.9517 | 0.0% |
| `model.layers.18.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0212 | -0.4004 | 0.3281 | 44.1788 | 0.0% |
| `model.layers.18.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0194 | -0.2031 | 0.3340 | 40.4158 | 0.0% |
| `model.layers.18.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.6616 | 0.4222 | 0.3535 | 3.5938 | 51.3158 | 0.0% |
| `model.layers.18.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0404 | 0.8689 | -3.7812 | 2.0781 | 9.8029 | 0.0% |
| `model.layers.18.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0229 | -0.2148 | 0.2412 | 7.7531 | 0.0% |
| `model.layers.18.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0195 | -0.2490 | 0.2363 | 17.4982 | 0.0% |
| `model.layers.18.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0709 | 2.3132 | -22.0000 | 14.2500 | 69.2342 | 0.0% |
| `model.layers.18.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0211 | -0.3809 | 0.3867 | 18.8873 | 0.0% |
| `model.layers.18.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0085 | 0.1822 | -0.9883 | 1.4766 | 2.0551 | 0.0% |
| `model.layers.18.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0245 | -0.1201 | 0.1206 | 8.2872 | 0.0% |
| `model.layers.19.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.4084 | 0.5866 | -0.0659 | 4.1562 | 45.6632 | 0.0% |
| `model.layers.19.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0192 | -0.3145 | 0.3164 | 39.9847 | 0.0% |
| `model.layers.19.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0211 | -0.3984 | 0.5820 | 44.0992 | 0.0% |
| `model.layers.19.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0199 | -0.4238 | 0.3555 | 41.5475 | 0.0% |
| `model.layers.19.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.7984 | 0.3508 | 0.6484 | 3.7656 | 54.8450 | 0.0% |
| `model.layers.19.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.1148 | 0.9733 | -2.4531 | 4.4062 | 11.0450 | 0.0% |
| `model.layers.19.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0219 | -0.1836 | 0.1484 | 7.4264 | 0.0% |
| `model.layers.19.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0181 | -0.2207 | 0.2812 | 16.2379 | 0.0% |
| `model.layers.19.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.0456 | 2.3111 | -19.8750 | 17.3750 | 69.1542 | 0.0% |
| `model.layers.19.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0001 | 0.0216 | -0.2148 | 0.2295 | 19.3987 | 0.0% |
| `model.layers.19.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0186 | 0.1574 | -0.7461 | 0.5000 | 1.7865 | 0.0% |
| `model.layers.19.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0227 | -0.1069 | 0.1416 | 7.6755 | 0.0% |
| `model.layers.2.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 0.9054 | 0.3984 | -0.9609 | 3.5469 | 29.6072 | 0.0% |
| `model.layers.2.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0170 | -0.4414 | 0.3633 | 35.4466 | 0.0% |
| `model.layers.2.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0237 | -0.3809 | 0.4531 | 49.5635 | 0.0% |
| `model.layers.2.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0173 | -0.6016 | 0.6367 | 36.1095 | 0.0% |
| `model.layers.2.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.1278 | 0.2783 | -0.0000 | 3.8906 | 34.7690 | 0.0% |
| `model.layers.2.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.4080 | 11.9886 | -39.5000 | 58.7500 | 135.1838 | 0.0% |
| `model.layers.2.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0369 | -0.2598 | 0.2451 | 12.5065 | 0.0% |
| `model.layers.2.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0151 | -0.4570 | 0.5391 | 13.5388 | 0.0% |
| `model.layers.2.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.0121 | 1.0146 | -5.2812 | 5.0000 | 30.3553 | 0.0% |
| `model.layers.2.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0256 | -0.2910 | 0.3477 | 22.9620 | 0.0% |
| `model.layers.2.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0094 | 0.1813 | -0.4941 | 1.3125 | 2.0460 | 0.0% |
| `model.layers.2.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0112 | -0.0864 | 0.0957 | 3.7819 | 0.0% |
| `model.layers.20.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5952 | 0.6838 | -0.0952 | 4.5000 | 51.9469 | 0.0% |
| `model.layers.20.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0195 | -0.3418 | 0.3613 | 40.7082 | 0.0% |
| `model.layers.20.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0208 | -0.4824 | 0.5391 | 43.4103 | 0.0% |
| `model.layers.20.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0204 | -0.3555 | 0.4238 | 42.5906 | 0.0% |
| `model.layers.20.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.8849 | 0.3032 | 0.6289 | 4.8750 | 57.1469 | 0.0% |
| `model.layers.20.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.1661 | 1.0437 | -3.7500 | 5.3750 | 11.9114 | 0.0% |
| `model.layers.20.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0209 | -0.1895 | 0.1514 | 7.0865 | 0.0% |
| `model.layers.20.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0211 | -0.3828 | 0.2891 | 18.9197 | 0.0% |
| `model.layers.20.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0585 | 1.4950 | -11.3125 | 9.9375 | 44.7605 | 0.0% |
| `model.layers.20.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0216 | -0.2207 | 0.1904 | 19.3336 | 0.0% |
| `model.layers.20.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0608 | 0.6975 | -0.6367 | 7.7812 | 7.8901 | 0.0% |
| `model.layers.20.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0245 | -0.2295 | 0.2314 | 8.3107 | 0.0% |
| `model.layers.21.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5861 | 1.4750 | 0.0479 | 6.7812 | 64.8167 | 0.0% |
| `model.layers.21.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0198 | -0.4824 | 0.6250 | 41.3147 | 0.0% |
| `model.layers.21.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0207 | -0.4688 | 0.5820 | 43.1609 | 0.0% |
| `model.layers.21.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0208 | -0.7578 | 0.7227 | 43.4260 | 0.0% |
| `model.layers.21.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 2.0731 | 0.4757 | 0.0123 | 8.6875 | 63.6655 | 0.0% |
| `model.layers.21.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0428 | 0.9136 | -3.4531 | 3.6875 | 10.3075 | 0.0% |
| `model.layers.21.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0190 | -0.1611 | 0.1855 | 6.4345 | 0.0% |
| `model.layers.21.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0220 | -0.3359 | 0.3125 | 19.6937 | 0.0% |
| `model.layers.21.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.0857 | 1.6280 | -10.0000 | 11.5625 | 48.7702 | 0.0% |
| `model.layers.21.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0213 | -0.2295 | 0.2285 | 19.0603 | 0.0% |
| `model.layers.21.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0850 | 1.0150 | -11.3750 | 0.4473 | 11.4785 | 0.0% |
| `model.layers.21.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0299 | -0.2988 | 0.2441 | 10.1150 | 0.0% |
| `model.layers.22.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5865 | 1.4905 | 0.0845 | 7.1562 | 65.1413 | 0.0% |
| `model.layers.22.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0200 | -0.3164 | 0.3340 | 41.7671 | 0.0% |
| `model.layers.22.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0201 | -0.7617 | 0.4512 | 41.8588 | 0.0% |
| `model.layers.22.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0208 | -0.3711 | 0.2812 | 43.5219 | 0.0% |
| `model.layers.22.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.9922 | 0.2999 | 0.0005 | 3.9531 | 60.3029 | 0.0% |
| `model.layers.22.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.1840 | 1.3164 | -4.8438 | 5.6562 | 14.9809 | 0.0% |
| `model.layers.22.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0184 | -0.1582 | 0.1641 | 6.2347 | 0.0% |
| `model.layers.22.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0216 | -0.2852 | 0.2080 | 19.3372 | 0.0% |
| `model.layers.22.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.0737 | 1.1186 | -5.4375 | 6.3438 | 33.5359 | 0.0% |
| `model.layers.22.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0212 | -0.2539 | 0.2578 | 18.9799 | 0.0% |
| `model.layers.22.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0022 | 0.1527 | -0.6523 | 0.3945 | 1.7205 | 0.0% |
| `model.layers.22.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0275 | -0.2119 | 0.2109 | 9.3189 | 0.0% |
| `model.layers.23.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.7206 | 1.8950 | 0.4980 | 12.3750 | 76.5939 | 0.0% |
| `model.layers.23.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0179 | -0.5273 | 0.3828 | 37.3283 | 0.0% |
| `model.layers.23.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0206 | -0.3574 | 0.3203 | 42.9398 | 0.0% |
| `model.layers.23.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0204 | -0.2002 | 0.2969 | 42.5870 | 0.0% |
| `model.layers.23.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 2.3630 | 0.4193 | 0.0001 | 6.3125 | 71.8355 | 0.0% |
| `model.layers.23.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0543 | 1.6252 | -6.8125 | 5.0312 | 18.3259 | 0.0% |
| `model.layers.23.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0188 | -0.2070 | 0.1924 | 6.3626 | 0.0% |
| `model.layers.23.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0204 | -0.9219 | 0.8281 | 18.2689 | 0.0% |
| `model.layers.23.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0358 | 1.4627 | -14.3750 | 9.4375 | 43.7733 | 0.0% |
| `model.layers.23.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0220 | -0.2422 | 0.2354 | 19.7113 | 0.0% |
| `model.layers.23.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0335 | 1.4855 | -2.6250 | 2.7500 | 16.7448 | 0.0% |
| `model.layers.23.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0254 | -0.2441 | 0.1982 | 8.6148 | 0.0% |
| `model.layers.3.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 0.9802 | 0.4276 | -1.6172 | 3.9688 | 32.0075 | 0.0% |
| `model.layers.3.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0174 | -0.5117 | 0.4199 | 36.2667 | 0.0% |
| `model.layers.3.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0226 | -0.3125 | 0.3008 | 47.1953 | 0.0% |
| `model.layers.3.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0174 | -0.2891 | 0.3223 | 36.4001 | 0.0% |
| `model.layers.3.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2570 | 0.4664 | -0.0000 | 11.4375 | 40.1300 | 0.0% |
| `model.layers.3.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.1204 | 2.9979 | -18.0000 | 19.6250 | 33.8120 | 0.0% |
| `model.layers.3.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0298 | -0.1807 | 0.1934 | 10.0888 | 0.0% |
| `model.layers.3.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0162 | -0.3027 | 0.2734 | 14.5097 | 0.0% |
| `model.layers.3.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.1281 | 1.5539 | -9.3125 | 13.1875 | 46.6449 | 0.0% |
| `model.layers.3.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0230 | -0.2451 | 0.2578 | 20.5697 | 0.0% |
| `model.layers.3.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0463 | 0.2635 | -0.5391 | 2.0000 | 3.0154 | 0.0% |
| `model.layers.3.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0160 | -0.0815 | 0.0864 | 5.4350 | 0.0% |
| `model.layers.4.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.0213 | 0.5281 | -1.9297 | 4.2500 | 34.4119 | 0.0% |
| `model.layers.4.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0176 | -0.4395 | 0.4707 | 36.7746 | 0.0% |
| `model.layers.4.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0225 | -0.3457 | 0.4004 | 47.0704 | 0.0% |
| `model.layers.4.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0179 | -0.4316 | 0.5312 | 37.4315 | 0.0% |
| `model.layers.4.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2231 | 0.3419 | -0.0000 | 3.3906 | 38.0127 | 0.0% |
| `model.layers.4.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0262 | 2.3507 | -9.6875 | 16.7500 | 26.4929 | 0.0% |
| `model.layers.4.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0270 | -0.1816 | 0.1807 | 9.1459 | 0.0% |
| `model.layers.4.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0177 | -0.2793 | 0.3203 | 15.9003 | 0.0% |
| `model.layers.4.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.1578 | 1.9699 | -12.7500 | 17.0000 | 59.1226 | 0.0% |
| `model.layers.4.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0231 | -0.2090 | 0.2129 | 20.6635 | 0.0% |
| `model.layers.4.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0296 | 0.4412 | -2.1875 | 2.4375 | 4.9828 | 0.0% |
| `model.layers.4.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0180 | -0.1035 | 0.1104 | 6.1023 | 0.0% |
| `model.layers.5.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2032 | 0.5109 | 0.0427 | 4.4062 | 39.1255 | 0.0% |
| `model.layers.5.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0167 | -0.5078 | 0.5508 | 34.8010 | 0.0% |
| `model.layers.5.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0243 | -0.2734 | 0.3008 | 50.8205 | 0.0% |
| `model.layers.5.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0171 | -0.3789 | 0.7852 | 35.8031 | 0.0% |
| `model.layers.5.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.4186 | 0.4329 | -0.0000 | 2.7969 | 44.3962 | 0.0% |
| `model.layers.5.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0526 | 0.9217 | -4.9688 | 3.7969 | 10.4042 | 0.0% |
| `model.layers.5.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0256 | -0.2100 | 0.1992 | 8.6670 | 0.0% |
| `model.layers.5.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0180 | -0.3008 | 0.1895 | 16.1008 | 0.0% |
| `model.layers.5.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.0207 | 1.9205 | -17.0000 | 13.1875 | 57.4567 | 0.0% |
| `model.layers.5.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0221 | -0.2393 | 0.2539 | 19.8320 | 0.0% |
| `model.layers.5.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0415 | 0.3590 | -1.3984 | 1.4375 | 4.0727 | 0.0% |
| `model.layers.5.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0185 | -0.0903 | 0.0942 | 6.2763 | 0.0% |
| `model.layers.6.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.0277 | 0.3912 | -1.5078 | 4.0625 | 32.9139 | 0.0% |
| `model.layers.6.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0179 | -0.3398 | 0.4238 | 37.4661 | 0.0% |
| `model.layers.6.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0220 | -0.3516 | 0.3945 | 45.8733 | 0.0% |
| `model.layers.6.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0184 | -0.3770 | 0.3633 | 38.4654 | 0.0% |
| `model.layers.6.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2966 | 0.3593 | 0.0001 | 2.0938 | 40.2724 | 0.0% |
| `model.layers.6.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0754 | 1.2063 | -3.2656 | 4.6250 | 13.6208 | 0.0% |
| `model.layers.6.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0283 | -0.1914 | 0.1924 | 9.5796 | 0.0% |
| `model.layers.6.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0175 | -0.4277 | 0.3535 | 15.6524 | 0.0% |
| `model.layers.6.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | -0.1321 | 1.9829 | -13.9375 | 12.0000 | 59.4539 | 0.0% |
| `model.layers.6.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0001 | 0.0229 | -0.2217 | 0.3828 | 20.5545 | 0.0% |
| `model.layers.6.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0667 | 0.5264 | -1.3203 | 1.4844 | 5.9796 | 0.0% |
| `model.layers.6.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0161 | -0.0796 | 0.0869 | 5.4360 | 0.0% |
| `model.layers.7.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.1480 | 0.3614 | -2.1719 | 3.4375 | 36.0227 | 0.0% |
| `model.layers.7.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0182 | -0.4160 | 0.3809 | 37.9218 | 0.0% |
| `model.layers.7.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0212 | -0.3281 | 0.3184 | 44.3218 | 0.0% |
| `model.layers.7.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0188 | -0.4375 | 0.5781 | 39.2498 | 0.0% |
| `model.layers.7.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3060 | 0.4031 | -0.0000 | 2.0781 | 40.9112 | 0.0% |
| `model.layers.7.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | 0.0875 | 0.8038 | -2.1719 | 4.0625 | 9.1125 | 0.0% |
| `model.layers.7.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0000 | 0.0279 | -0.2148 | 0.2471 | 9.4334 | 0.0% |
| `model.layers.7.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0178 | -0.4766 | 0.3340 | 15.9916 | 0.0% |
| `model.layers.7.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.3388 | 2.6583 | -4.4375 | 24.3750 | 80.1701 | 0.0% |
| `model.layers.7.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0229 | -0.2344 | 0.2559 | 20.4866 | 0.0% |
| `model.layers.7.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0044 | 0.1734 | -0.9570 | 0.7266 | 1.9549 | 0.0% |
| `model.layers.7.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0002 | 0.0184 | -0.1030 | 0.0938 | 6.2441 | 0.0% |
| `model.layers.8.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.0842 | 0.5513 | 0.0486 | 7.6875 | 36.4031 | 0.0% |
| `model.layers.8.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0181 | -0.3438 | 0.3438 | 37.7759 | 0.0% |
| `model.layers.8.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0217 | -0.2891 | 0.3223 | 45.3704 | 0.0% |
| `model.layers.8.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0000 | 0.0188 | -0.3203 | 0.3848 | 39.3492 | 0.0% |
| `model.layers.8.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.2376 | 0.4594 | 0.0000 | 2.3438 | 39.5120 | 0.0% |
| `model.layers.8.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -3.7607 | 24.9079 | -214.0000 | 12.7500 | 283.9046 | 0.0% |
| `model.layers.8.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0322 | -0.2275 | 0.2217 | 10.8989 | 0.0% |
| `model.layers.8.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0172 | -0.3340 | 0.3340 | 15.4127 | 0.0% |
| `model.layers.8.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.2327 | 2.3745 | -16.7500 | 20.0000 | 71.3782 | 0.0% |
| `model.layers.8.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0258 | -1.6875 | 0.4102 | 23.0772 | 0.0% |
| `model.layers.8.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0462 | 0.5515 | -1.3984 | 1.4609 | 6.2368 | 0.0% |
| `model.layers.8.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0000 | 0.0163 | -0.1465 | 0.1328 | 5.5270 | 0.0% |
| `model.layers.9.input_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.5364 | 0.9681 | -0.0537 | 9.3125 | 54.3499 | 0.0% |
| `model.layers.9.mlp.down_proj.weight` | `weight` | `[896, 4864]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0180 | -0.3965 | 0.5938 | 37.5220 | 0.0% |
| `model.layers.9.mlp.gate_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | 0.0001 | 0.0205 | -0.4473 | 0.3984 | 42.8222 | 0.0% |
| `model.layers.9.mlp.up_proj.weight` | `weight` | `[4864, 896]` | `bfloat16` | `4,358,144` | -0.0000 | 0.0187 | -0.4219 | 0.3945 | 39.0760 | 0.0% |
| `model.layers.9.post_attention_layernorm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 1.3111 | 0.5723 | -0.0000 | 2.4375 | 42.8166 | 0.0% |
| `model.layers.9.self_attn.k_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0558 | 0.6571 | -3.0938 | 3.6719 | 7.4316 | 0.0% |
| `model.layers.9.self_attn.k_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | 0.0001 | 0.0244 | -0.2197 | 0.2598 | 8.2774 | 0.0% |
| `model.layers.9.self_attn.o_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | 0.0000 | 0.0168 | -0.1621 | 0.4609 | 15.0567 | 0.0% |
| `model.layers.9.self_attn.q_proj.bias` | `bias` | `[896]` | `bfloat16` | `896` | 0.0981 | 2.6872 | -19.6250 | 18.7500 | 80.4461 | 0.0% |
| `model.layers.9.self_attn.q_proj.weight` | `weight` | `[896, 896]` | `bfloat16` | `802,816` | -0.0000 | 0.0226 | -0.2324 | 0.2227 | 20.2655 | 0.0% |
| `model.layers.9.self_attn.v_proj.bias` | `bias` | `[128]` | `bfloat16` | `128` | -0.0097 | 0.1219 | -0.7695 | 0.9062 | 1.3776 | 0.0% |
| `model.layers.9.self_attn.v_proj.weight` | `weight` | `[128, 896]` | `bfloat16` | `114,688` | -0.0001 | 0.0178 | -0.0938 | 0.1094 | 6.0141 | 0.0% |
| `model.norm.weight` | `weight` | `[896]` | `bfloat16` | `896` | 7.1393 | 1.1043 | -0.0114 | 16.7500 | 216.2416 | 0.0% |

---

<sub>Report auto-generated by SETU Model Evaluation Engine at 2026-10-29T13:35:43.480149+00:00.</sub>