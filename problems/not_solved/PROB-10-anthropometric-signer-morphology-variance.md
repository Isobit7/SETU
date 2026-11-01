---
id: PROB-10
title: "Lack of Diverse Anthropometric Signer Morphology in Synthetic Data"
stage: not_solved
status: not_solved
created: 2026-11-01
updated: 2026-11-01
severity: Medium
category: Synthetic Augmentation & Dataset Generation
---

# PROB-10: Lack of Diverse Anthropometric Signer Morphology in Synthetic Data

## 1. Problem Description
The multi-signer dataset generated in `scripts/generate_multisigner_dataset.py` varies global rotation and isotropic scale, but fails to account for diverse anatomical variations between real human hands (such as different ratios of finger length to palm size, thumb reach, or joint flexibility). Consequently, the model struggles when tested against users whose hand shapes differ from the single recording subject.

## 2. Root Cause Analysis
In `scripts/generate_multisigner_dataset.py`:
- Augmented samples apply a global scaling factor $s \sim \mathcal{U}(0.85, 1.15)$ uniformly to all 21 landmarks of a hand.
- Multiplying all landmarks by a single scalar $s$ preserves identical geometric aspect ratios.
- In real populations:
  - Finger-to-palm ratio varies by up to $\pm 18\%$.
  - Thumb opposability and span varies by up to $\pm 22\%$.
  - Inter-finger joint laxity varies across signers.
Because synthetic signers are mere scaled clones of the original signer, the classifier overfits to specific proportional distances.

## 3. Implementation Plan & Proposed Fix
1. In `scripts/generate_multisigner_dataset.py`:
   - Introduce **Segmental Anthropometric Scaling**:
     - Perturb distal, intermediate, and proximal phalanges independently:
       $$s_{\text{digit}} \sim \mathcal{N}(1.0, 0.06)$$
     - Perturb palm aspect ratio (width vs height) by $w \sim \mathcal{N}(1.0, 0.05)$.
     - Simulate natural joint stiffness and hypermobility variance.
2. Regenerate multi-signer dataset with 20 distinct morphological profiles.

## 4. Acceptance Criteria & Verification
- [ ] Synthetic dataset exhibits diverse finger-to-palm ratio variance ($std > 0.04$).
- [ ] Cross-signer test evaluation accuracy improves across distinct test signers.
