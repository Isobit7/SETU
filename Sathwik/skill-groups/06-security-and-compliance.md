<!-- markdownlint-disable MD013 -->

# Skill Group 06: Security, Hardening & Compliance Engineering

## A. Group Overview

- **Group Name**: Security, Hardening & Compliance Engineering
- **Main Purpose**: Fortifying applications against malicious exploits, auditing against OWASP Top 10 vulnerabilities, and achieving regulatory compliance (PCI-DSS, GDPR, MAS TRM).
- **Problems It Helps Solve**:
  - SQL injection, Cross-Site Scripting (XSS), and insecure direct object references (IDOR).
  - Unsanitized inputs at trust boundaries.
  - Hardcoded API keys, tokens, and secret leakage in source control.
  - Data privacy non-compliance under GDPR and financial regulations.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-privacy-compliance-engineering`, `agentic-bundle-aas-secure-app-builder`, `agentic-bundle-aas-security-engineer`, `agentic-bundle-security-developer`, `agentic-bundle-security-engineer`.

---

## B. Skills Within the Group

### 1. `security-audit` & `security-auditor`

- **What It Does**: Executes comprehensive security audits covering web application penetration testing, API vulnerability scanning, and DevSecOps hardening.
- **When to Use**: Before major product launches, security reviews, or audits.
- **When to Avoid**: During early rapid exploratory prototyping.
- **Example**:

  ```text
  Run a security-audit across our FastAPI endpoints to verify rate limiting and header security policies.
  ```

### 2. `backend-security-coder` & `frontend-security-coder`

- **What It Does**: Specializes in secure coding patterns: strict input validation, parameter binding, cryptographic hashing, and Content Security Policy (CSP).
- **When to Use**: When writing authentication handlers, form inputs, or WebSocket payload deserialization.

### 3. `top-web-vulnerabilities` & `vulnerability-scanner`

- **What It Does**: Evaluates code against the 100 most critical web application vulnerabilities (OWASP 2025).
- **When to Use**: When assessing code that handles untrusted external user input.

### 4. `privacy-by-design` & `gdpr-data-handling`

- **What It Does**: Enforces data minimization, explicit user consent workflows, and secure retention policies for sensitive user biometrics and recordings.
- **When to Use**: In applications that capture webcam video or biometric landmark data.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
