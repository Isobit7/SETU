---
id: PROB-10
title: "Lack of Diverse Anthropometric Signer Morphology in Synthetic Data"
stage: solved
status: solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Synthetic Augmentation & Dataset Generation
---

# PROB-10: Lack of Diverse Anthropometric Signer Morphology in Synthetic Data

## 1. Problem Description
The multi-signer dataset previously varied global rotation and isotropic scale, but failed to account for diverse anatomical variations between real human hands (such as different ratios of finger length to palm size, thumb reach, or joint flexibility).

## 2. Root Cause Analysis
In `scripts/generate_multisigner_dataset.py`:
Augmentation only scaled all 21 points by a single uniform scalar $s \sim \mathcal{U}(0.88, 1.12)$. Multiplying all points by a single scalar preserved identical geometric aspect ratios, causing the classifier to overfit to the single recording subject's hand proportions.

## 3. Implementation Plan & Proposed Fix
1. In `scripts/generate_multisigner_dataset.py`:
   - Implemented **Segmental Anthropometric Scaling**:
     - Perturbed each individual digit chain (Thumb, Index, Middle, Ring, Pinky) independently relative to its MCP joint by $s_{\text{digit}} \sim \mathcal{U}(0.88, 1.12)$.
     - Perturbed palm aspect ratio (width vs height) by $w_{\text{palm}} \sim \mathcal{U}(0.90, 1.10)$.
   - Followed by strict canonical projection to preserve $(0, -1, 0)$ orientation.

## 4. Acceptance Criteria & Verification
- [x] Synthetic dataset exhibits diverse finger-to-palm ratio variance ($std > 0.03$).
- [x] PyTest test `test_prob10_anthropometric_variation_diversity` in `tests/test_live_bugfixes.py` passes 100%.
