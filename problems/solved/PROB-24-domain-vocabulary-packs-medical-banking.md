---
id: PROB-24
title: "Domain Vocabulary Packs for Healthcare and Banking"
stage: plan
status: solved
created: 2026-11-01
updated: 2026-11-01 13:26
severity: Medium
category: Vocabulary & Domain Adaptation
---

# PROB-24: Domain Vocabulary Packs for Healthcare and Banking

## 1. Problem Description
General-purpose sign recognition models often fail when applied to specialized high-stakes contexts like hospitals, medical clinics, bank counters, and government offices. Signers in these settings use specific terminology ("Allergy", "Prescription", "Debit Card", "Aadhaar Card", "Account Number") that does not exist in standard generic vocabulary datasets like AI4Bharat INCLUDE.

## 2. Root Cause Analysis
- Models were trained on generic conversational words (e.g. "hello", "apple", "cat").
- When a user tries to communicate a specific medical condition (e.g. "Chest Pain" or "Blood Pressure"), they are forced to fingerspell letter-by-letter, which is slow, stressful, and error-prone.

## 3. Implementation Plan & Proposed Architecture
1. **Curate Specialized ISL Domain Dictionaries**:
   - **Healthcare Pack (40 signs):** Pain, Fever, Cough, Medicine, Injection, Doctor, Nurse, Allergy, Blood Pressure, Emergency, Where does it hurt?, Since when?
   - **Banking & Official Pack (30 signs):** Bank, Money, Account, ATM, Cheque, Signature, Card, Deposit, Withdraw, Aadhaar, Receipt.
2. **Context-Aware Quick-Pick HUD**:
   - Provide a Domain Selector dropdown in the UI: `General | Medical | Banking`.
   - In `Medical` mode, NLP auto-correct and predictive word completion prioritizes clinical vocabulary.
3. **Synthetic Domain Feature Injection**:
   - Synthesize canonical dual-hand feature sequences for domain signs using standard ISL anatomical dictionaries.

## 4. Acceptance Criteria & Verification
- [x] Medical mode prioritizes clinical phrases in next-word predictive suggestions.
- [x] Quick-access visual card grid for non-signing emergencies ("Where hurts?", "Pain scale 1-10").
- [x] Unit tests verify domain dictionary loading and lookup speed $< 1$ ms.
