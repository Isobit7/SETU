# SETU Course Exports & Reading Guide

> **Audience:** 3rd Year B.Tech AI & ML Student  
> **Mission:** Guidance on reading, sharing, and exporting the SETU project-based learning course.

---

## Primary Format: Enhanced Markdown

The entire SETU course is authored in standard **GitHub Flavored Markdown (GFM)** with embedded KaTeX mathematical notation and Mermaid architectural diagrams.
This provides:
1. **Interactive Navigation:** Relative file links allow you to click between lessons, exercises, solutions, and code files.
2. **Version Control:** All documents are tracked in your Git repository.
3. **Instant Searchability:** You can search terms across all lessons directly inside VS Code or GitHub (`Ctrl+Shift+F`).

---

## How to View or Export to PDF

Because external headless rendering tools (like Pandoc or WeasyPrint) are not pre-installed in your environment, we preserve the clean Markdown source rather than generating a broken, low-fidelity PDF.

### Method 1: In VS Code (Recommended)
1. Open any lesson (e.g. `learning/README.md` or `lessons/01_foundations/lesson_01_how_software_and_python_execute.md`).
2. Press `Ctrl+K V` to open the rich interactive Markdown preview with rendered diagrams and equations.
3. If you wish to save a PDF, install the popular **"Markdown PDF"** or **"Markdown Preview Enhanced"** extension in VS Code $\to$ Right-click $\to$ **Export to PDF**.

### Method 2: Via Web Browser
1. In VS Code or your editor, open the rendered Markdown preview.
2. Right-click $\to$ **Print** (or `Ctrl+P`) $\to$ Select **Save as PDF**.
3. Enable **Background Graphics** in print settings for color syntax highlighting.

