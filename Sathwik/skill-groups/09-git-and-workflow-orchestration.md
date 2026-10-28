<!-- markdownlint-disable MD013 -->

# Skill Group 09: Git Operations, Code Review & Orchestration

## A. Group Overview

- **Group Name**: Git Operations, Code Review & Orchestration
- **Main Purpose**: Enforcing disciplined version control, conventional commit standards, peer review checklists, and subagent work allocation.
- **Problems It Helps Solve**:
  - Messy git histories with uninformative messages like "update" or "fix bug".
  - Accidental commits containing temporary logs, scratch files, or credentials.
  - Missed regressions during pull request review.
  - Coordinating parallel subagent workflows without branch collisions.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-oss-maintainer`, `agentic-bundle-essentials`, `agentic-bundle-skill-author`.

---

## B. Skills Within the Group

### 1. `commit` & `git-pushing`

- **What It Does**: Stages intended files, validates working trees, formats conventional commits (`feat:`, `fix:`, `refactor:`, `test:`), and verifies tests before pushing.
- **When to Use**: Whenever finalizing work on a feature, bugfix, or refactor.
- **When to Avoid**: When midway through an unverified, broken experiment.
- **Example**:

  ```text
  Use commit to stage changes in recognize.py and realsign.py with an accurate conventional commit message.
  ```

### 2. `create-branch` & `pr-writer`

- **What It Does**: Names git branches cleanly (`fix/realsign-streak-lockout`), and generates structured pull request summaries with test evidence, diff breakdowns, and linked issues.
- **When to Use**: When opening a pull request for team review.

### 3. `code-reviewer` & `code-review-checklist`

- **What It Does**: Performs an elite code review examining edge-case handling, performance impacts, architectural adherence, and security.
- **When to Use**: Before merging any branch into `main`.

### 4. `subagent-orchestrator` & `executing-plans`

- **What It Does**: Manages quota-aware parallel child agents to execute independent tasks across multi-file repositories simultaneously.
- **When to Use**: When executing a large 10-step implementation plan with non-overlapping tasks.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
