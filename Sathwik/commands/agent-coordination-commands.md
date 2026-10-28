<!-- markdownlint-disable MD013 -->

# Agent Coordination Slash Commands

These 5 commands govern **how** Antigravity reasons, plans, executes, and retains memory. They are built into the Antigravity core platform.

---

## 1. `/plan`

- **Invocation**: Type `/plan <your feature or refactoring description>`
- **Mechanism**: The agent stops execution before touching code. It inspects current files, identifies dependencies, and writes an atomic checklist with verification checkpoints.
- **Why It Matters**: Prevents high-cost mistakes and ensures developer approval before files are altered.
- **Best Practice**: Use when a task touches 2 or more files or alters an API contract.

---

## 2. `/goal`

- **Invocation**: Type `/goal <high-level objective>`
- **Mechanism**: Activates persistent, iterative mode. The agent autonomously runs commands, reads terminal outputs, diagnoses errors, and self-corrects until all tests pass.
- **Why It Matters**: Perfect for overnight or unsupervised tasks where you want complete results rather than constant pauses asking for permission.

---

## 3. `/grill-me`

- **Invocation**: Type `/grill-me <proposal or feature idea>`
- **Mechanism**: The agent acts as a skeptical senior software architect, asking you 3-5 hard questions regarding edge cases, failure states, and trade-offs.
- **Why It Matters**: Uncovers blind spots in design thinking before you write a single line of implementation.

---

## 4. `/schedule`

- **Invocation**: Type `/schedule <duration or cron> <action>`
- **Mechanism**: Utilizes Antigravity's background scheduling engine to fire notifications or execute health monitoring checks at scheduled intervals.
- **Why It Matters**: Enables monitoring long-running builds or polling service uptime without keeping the user active.

---

## 5. `/learn`

- **Invocation**: Type `/learn <correction or environment discovery>`
- **Mechanism**: Writes the rule into project-level configuration (`AGENTS.md` or `.agents/rules/`).
- **Why It Matters**: Guarantees that the agent adapts permanently to your specific OS, environment quirks, and repository rules.

---

Back to [Commands Index](README.md) | [Main Handbook README](../README.md)
