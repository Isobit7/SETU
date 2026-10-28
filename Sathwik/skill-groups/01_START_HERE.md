<!-- markdownlint-disable MD013 -->

# Antigravity Quick Start — 5-Minute Beginner Guide

Welcome to the **Antigravity Commands & Skills Handbook**. If you are new to Antigravity, this document is designed to give you complete clarity in five minutes on what tools exist in your environment, how they work, and how to use them with confidence.

---

## 1. What is Antigravity?

Antigravity is an agentic AI pair programming environment developed by Google DeepMind. Unlike simple code completion tools, Antigravity acts as an autonomous pair programmer that can:

- Read and navigate your workspace code.
- Run terminal commands on your system (builds, tests, linters).
- Edit existing files with minimal diffs and create new files.
- Launch browser subagents to inspect web interfaces.
- Utilize specialized domain **skills**, **workflows**, and **Model Context Protocol (MCP)** tools.

---

## 2. The Five Core Concepts Explained Simply

Beginners often get confused between slash commands, skills, agents, workflows, and plugins. Here is the simple distinction:

```text
+-------------------------------------------------------------------------------+
|                             USER CHAT INTERFACE                               |
+-------------------------------------------------------------------------------+
         |                                                 |
         v                                                 v
  [Slash Command]                                   [Natural Language Prompt]
  Typed with a leading "/"                          e.g., "Find and fix the bug
  e.g., /polish, /goal, /plan                        in our alphabet classifier"
         |                                                 |
         +------------------------+------------------------+
                                  |
                                  v
                       [ANTIGRAVITY AGENT]
                      Orchestrator & Reasoning
                                  |
         +------------------------+------------------------+
         |                        |                        |
         v                        v                        v
    [Workflows]                [Skills]               [MCP Tools]
  Step-by-step playbooks    Domain knowledge,        Live connections to
  in .agent/workflows/      guidelines & patterns    Neon, Context7, 21st,
  (e.g., adapt, harden)     in plugins & global      Firebase, and DBs
```

### Definitions at a Glance

1. **Slash Command (`/command`)**: A shortcut you type directly into the chat prompt (e.g., `/plan` or `/polish`). It signals to Antigravity a specific mode or structured procedure.
2. **Workflow (`.agent/workflows/*.md`)**: A step-by-step guide stored as Markdown in your project. When you type `/adapt` or `/harden`, Antigravity loads that specific guide and follows its structured checklist.
3. **Skill (`SKILL.md`)**: A self-contained package of domain expertise, code examples, best practices, and scripts (e.g., `fastapi-pro`, `react-best-practices`, `systematic-debugging`). Antigravity consults skills automatically when handling related tasks.
4. **Plugin**: A bundle of related skills and subagents grouped together under a single topic (e.g., `agentic-bundle-aas-qa-test-automation`).
5. **MCP Server (Model Context Protocol)**: A bridge that connects Antigravity to live external systems, such as looking up official library documentation (`context7`), querying PostgreSQL databases (`Neon`), or managing cloud infrastructure.

---

## 3. How to Use Antigravity in Your Daily Work

### Scenario A: You Want to Plan Before Touching Code

Type:

```text
/plan I want to add audio feedback whenever a sign is recognized
```

**What happens:** Antigravity analyzes your project, lists requirements, breaks the implementation into atomic checklist items, and asks for your confirmation before writing any code.

---

### Scenario B: You Encounter a Bug or Failing Test

Simply say:

```text
The test test_realsign_classifier.py is failing on letter C. Use systematic-debugging to find the root cause and fix it.
```

**What happens:** Antigravity follows the `systematic-debugging` skill—reproducing the issue, isolating the root cause, verifying edge cases, and testing the minimal working diff.

---

### Scenario C: You Want to Refine and Polish Your Web UI

Type:

```text
/polish Check the CameraControls component and make sure spacing, transitions, and hover states look agency-grade.
```

**What happens:** Antigravity loads [c:/.agent/workflows/polish.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/polish.md) and methodically audits alignments, border radii, color contrast, and micro-interactions.

---

## 4. The Golden Rules for Beginners

1. **Be Specific About the Goal**: Describe what you want the end result to look like. "Make it better" gives vague results; "Reduce the hold duration from 600ms to 380ms and unclamp confidence" gives exact results.
2. **Never Guess Command Names**: Use the [Commands Index](02_COMMANDS_INDEX.md) to see the exact 22 verified commands available in your workspace.
3. **Let the Agent Test Its Own Changes**: Always ask Antigravity to run your tests or linter after modifying code (`pytest`, `npm run build`, `npm run lint`).
4. **Refer to the Decision Guide**: If you don't know what skill applies to your problem, open [04_WHICH_SKILL_SHOULD_I_USE.md](04_WHICH_SKILL_SHOULD_I_USE.md).

---

## 5. Handbook Document Roadmap

| Document | Purpose |
| --- | --- |
| [README.md](README.md) | Central portal & table of contents |
| [01_START_HERE.md](01_START_HERE.md) | This 5-minute onboarding guide |
| [02_COMMANDS_INDEX.md](02_COMMANDS_INDEX.md) | Complete list of all 22 `/` commands |
| [03_SKILL_GROUPS.md](03_SKILL_GROUPS.md) | 10 major skill domains categorized |
| [04_WHICH_SKILL_SHOULD_I_USE.md](04_WHICH_SKILL_SHOULD_I_USE.md) | "I want to..." task-to-skill decision tree |
| [05_COMMANDS_REFERENCE.md](05_COMMANDS_REFERENCE.md) | Detailed command syntax & parameters |
| [06_SKILL_ORIGIN_MAP.md](06_SKILL_ORIGIN_MAP.md) | Verification and provenance table |
| [07_EXAMPLES_AND_WORKFLOWS.md](07_EXAMPLES_AND_WORKFLOWS.md) | Real-world multi-step pipelines |
| [08_GLOSSARY.md](08_GLOSSARY.md) | Antigravity vocabulary defined |
| [09_COVERAGE_AND_LIMITATIONS.md](09_COVERAGE_AND_LIMITATIONS.md) | Scope boundaries and environment notes |
