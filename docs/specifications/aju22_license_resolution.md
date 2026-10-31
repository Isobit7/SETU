# SETU — Resolution of aju22 Repository Licensing & Provenance

**Document Version:** 1.0.0  
**Status:** Resolved & Quarantined  
**Target Candidate:** `https://github.com/aju22/Real-Time-ISL-Translation`

---

## 1. Background & Discrepancy

In preliminary exploratory audits, `aju22/Real-Time-ISL-Translation` was referenced as an open-source reference for real-time ISL using LSTM. Certain informal reviews suggested an "MIT License". However, in adherence to strict project governance and IP safety for SETU, a primary evidence audit was required.

---

## 2. Primary License Evidence

On October 16, 2026, direct automated inspections of the GitHub API and repository contents were performed:

1. **GitHub API License Endpoint:**
   - Query: `GET https://api.github.com/repos/aju22/Real-Time-ISL-Translation`
   - Response: `"license": null`
2. **Repository File Tree:**
   - Files present: `['README.md', 'evaluate.py', 'keypoint_data', 'keypoint_extraction.py', 'main.py', 'misc', 'models.py', 'models', 'train.py', 'utils.py']`
   - Result: **No `LICENSE`, `COPYING`, or `UNLICENSE` file exists** in the repository.
3. **Repository README Inspection:**
   - The README describes the project architecture, features, and acknowledges the *INCLUDE* dataset from Zenodo.
   - It contains **no license statement, copyright waiver, or redistribution grant**.

---

## 3. Legal Analysis & Determination

Under international copyright law and GitHub's Terms of Service:
- A public repository without an explicit open-source license grant retains default copyright: **All Rights Reserved**.
- The author has not granted legal permission to copy, adapt, redistribute, or incorporate their source code or trained binary weights into downstream projects.
- The previous assumption of an "MIT License" was factually incorrect.

---

## 4. Protective Action Taken in SETU

1. **Strict Quarantine of Reference Artifacts:**
   - The Phase 0 reference model `isl_model.h5` is quarantined in `models/reference/` with an explanatory manifest. It is excluded from Git tracking via `.gitignore`.
   - It is utilized strictly for offline historical reference and comparison; it is **never** packaged into the SETU deliverable.
2. **Clean-Room Engineering:**
   - Zero lines of code from `aju22` are included in `setu/`.
   - All modules in `setu/` (`vision/`, `features/`, `utils/`, `exceptions.py`, `config.py`) are designed and written from scratch.
3. **Direct Primary Source Attribution:**
   - aju22 trained their model on a subset of the **INCLUDE dataset** (Zenodo Record 4010759).
   - The INCLUDE dataset is officially licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** by AI4Bharat.
   - SETU relies directly on primary, legally sound sources (INCLUDE and Mendeley Data) rather than unverified third-party repositories.
