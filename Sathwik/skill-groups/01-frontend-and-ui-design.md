<!-- markdownlint-disable MD013 -->

# Skill Group 01: Frontend Engineering & UI/UX Design

## A. Group Overview

- **Group Name**: Frontend Engineering & UI/UX Design
- **Main Purpose**: Creating distinctive, accessible, production-grade frontend interfaces with high aesthetic fidelity, fluid micro-interactions, and zero generic "AI slop".
- **Problems It Helps Solve**:
  - Eliminating generic, boring, or unstyled UI templates.
  - Ensuring consistent design tokens and fluid typography.
  - Complying with WCAG 2.2 accessibility standards.
  - Avoiding browser compatibility pitfalls (e.g. Safari vendor prefixes like `-webkit-backdrop-filter`).
- **Origin & Location**:
  - Workspace skill: [c:/.agent/skills/frontend-design.md](file:///c:/MINI%20PROJ-SETU/.agent/skills/frontend-design.md)
  - Global Plugins: `agentic-bundle-aas-product-design-studio`, `agentic-bundle-aas-web-app-builder`, `agentic-bundle-creative-director`, `agentic-bundle-web-designer`, `agentic-bundle-web-wizard`.

---

## B. Skills Within the Group

### 1. `frontend-design`

- **What It Does**: Guides the creation of distinctive, production-grade frontend interfaces with custom design directions (minimalist, luxury utilitarian, brutalist, etc.).
- **Purpose**: Prevents "AI slop" (cyan-on-dark, purple gradients, generic cards).
- **When to Use**: When creating new React components, designing landing pages, or establishing visual styles.
- **When to Avoid**: When writing backend database migrations or Python scripts.
- **Required Inputs**: Component requirements and desired visual direction.
- **Expected Outputs**: Polished JSX/TSX components and CSS token styling.
- **Example**:

  ```text
  Use frontend-design to build a glassmorphic LiveCameraIndicator with status pulse and optimal framing chips.
  ```

### 2. `react-best-practices`

- **What It Does**: Enforces performance optimizations from Vercel for React and Next.js applications (memoization, render trees, bundle splitting).
- **When to Use**: When diagnosing slow component re-renders or building complex interactive views.
- **When to Avoid**: When writing vanilla CSS.
- **Example**:

  ```text
  Review WebcamAlphabetRecognition.tsx against react-best-practices to eliminate unnecessary re-renders.
  ```

### 3. `tailwind-patterns`

- **What It Does**: Directs modern Tailwind CSS v4 setup, container queries, and CSS-first configuration.
- **When to Use**: When using Tailwind for utility-based responsive design.
- **When to Avoid**: In projects explicitly using vanilla CSS or CSS modules.

### 4. `animejs-animation` & `threejs-skills`

- **What It Does**: Provides advanced JavaScript animation patterns and 3D WebGL scene rendering.
- **When to Use**: When building immersive product showcases, 3D hand visualizers, or custom micro-interactions.

---

## C. Companion Workflows

This skill group connects directly to the 17 UI workflows in [c:/.agent/workflows/](file:///c:/MINI%20PROJ-SETU/.agent/workflows/):

- `/polish`: Refines spacing, borders, and alignment.
- `/adapt`: Adapts layouts for mobile and touch devices.
- `/animate`: Enhances motion and easing.
- `/audit`: Scans accessibility and performance.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
