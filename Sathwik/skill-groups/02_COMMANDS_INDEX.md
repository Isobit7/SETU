<!-- markdownlint-disable MD013 -->

# Antigravity Slash Commands Index

This index contains every verified `/` command available in this workspace environment. In Antigravity, commands fall into two distinct operational classes:

1. **Agent Coordination Commands (5 commands)**: Built-in system slash commands that guide the agent's reasoning loop, planning depth, schedule, or memory persistence.
2. **Interface & Design Workflows (17 commands)**: Workspace-level structured workflows located in `.agent/workflows/` that execute systematic visual, functional, and accessibility passes over frontend code.

---

## Master Commands Table

| Command | Category | Purpose | Source File / Origin | Primary Input |
| --- | --- | --- | --- | --- |
| [`/plan`](05_COMMANDS_REFERENCE.md#1-plan) | Coordination | Generates a structured, phased implementation checklist before touching code | Antigravity Core | Feature goal or specification |
| [`/goal`](05_COMMANDS_REFERENCE.md#2-goal) | Coordination | Runs an autonomous, persistent execution loop until full completion | Antigravity Core | High-level objective |
| [`/grill-me`](05_COMMANDS_REFERENCE.md#3-grill-me) | Coordination | Conducts an interactive Socratic interview to resolve ambiguous decisions | Antigravity Core | Architectural proposal / idea |
| [`/schedule`](05_COMMANDS_REFERENCE.md#4-schedule) | Coordination | Configures one-shot timers or recurring cron triggers | Antigravity Core | Duration or cron expression |
| [`/learn`](05_COMMANDS_REFERENCE.md#5-learn) | Coordination | Persists corrections and setup discoveries into project instructions | Antigravity Core | Learned rule or pattern |
| [`/adapt`](05_COMMANDS_REFERENCE.md#6-adapt) | UI Workflow | Adapts layouts across devices, screen sizes, touch inputs, and orientations | `.agent/workflows/adapt.md` | Target component or page |
| [`/animate`](05_COMMANDS_REFERENCE.md#7-animate) | UI Workflow | Enhances interfaces with purposeful micro-interactions and transitions | `.agent/workflows/animate.md` | Target element / interaction |
| [`/audit`](05_COMMANDS_REFERENCE.md#8-audit) | UI Workflow | Runs diagnostic scan of a11y, performance, theming, and responsive design | `.agent/workflows/audit.md` | Page URL or component file |
| [`/bolder`](05_COMMANDS_REFERENCE.md#9-bolder) | UI Workflow | Amplifies timid or bland designs with striking contrast and distinct style | `.agent/workflows/bolder.md` | Component or section |
| [`/clarify`](05_COMMANDS_REFERENCE.md#10-clarify) | UI Workflow | Improves confusing UX copy, error messages, labels, and microcopy | `.agent/workflows/clarify.md` | Form, dialog, or text area |
| [`/colorize`](05_COMMANDS_REFERENCE.md#11-colorize) | UI Workflow | Adds cohesive OKLCH/HSL palettes to monochromatic or washed-out views | `.agent/workflows/colorize.md` | Target UI section |
| [`/critique`](05_COMMANDS_REFERENCE.md#12-critique) | UI Workflow | Conducts a holistic design review evaluating usability and visual hierarchy | `.agent/workflows/critique.md` | Component or screen |
| [`/delight`](05_COMMANDS_REFERENCE.md#13-delight) | UI Workflow | Introduces subtle personality, celebratory feedback, and memorable touches | `.agent/workflows/delight.md` | Feature or user milestone |
| [`/extract`](05_COMMANDS_REFERENCE.md#14-extract) | UI Workflow | Consolidates repeated styling, tokens, and markup into reusable components | `.agent/workflows/extract.md` | Duplicate code blocks |
| [`/harden`](05_COMMANDS_REFERENCE.md#15-harden) | UI Workflow | Hardens interfaces against long strings, network failures, and edge cases | `.agent/workflows/harden.md` | Form or data display component |
| [`/normalize`](05_COMMANDS_REFERENCE.md#16-normalize) | UI Workflow | Standardizes arbitrary styles into design tokens and established patterns | `.agent/workflows/normalize.md` | Non-standard styles |
| [`/onboard`](05_COMMANDS_REFERENCE.md#17-onboard) | UI Workflow | Designs first-run user experiences, empty states, and progressive guides | `.agent/workflows/onboard.md` | New feature or empty view |
| [`/optimize`](05_COMMANDS_REFERENCE.md#18-optimize) | UI Workflow | Optimizes render performance, frame rates, bundle size, and reflows | `.agent/workflows/optimize.md` | Component or page |
| [`/polish`](05_COMMANDS_REFERENCE.md#19-polish) | UI Workflow | Final craftsmanship pass fixing alignment, optical spacing, and borders | `.agent/workflows/polish.md` | Screen or component |
| [`/quieter`](05_COMMANDS_REFERENCE.md#20-quieter) | UI Workflow | Tones down visual noise, high-frequency animations, and over-saturation | `.agent/workflows/quieter.md` | Overly loud interface |
| [`/simplify`](05_COMMANDS_REFERENCE.md#21-simplify) | UI Workflow | Strips unnecessary decoration, redundant containers, and visual clutter | `.agent/workflows/simplify.md` | Complex or messy screen |
| [`/teach-impeccable`](05_COMMANDS_REFERENCE.md#22-teach-impeccable) | UI Workflow | Extracts project aesthetic rules and writes persistent design guidelines | `.agent/workflows/teach-impeccable.md` | Codebase design scan |

---

## Section Summaries

### 1. Agent Coordination Commands

These commands alter **how** Antigravity operates. Use `/plan` when embarking on multi-file work to align on an atomic checklist. Use `/grill-me` when you aren't sure about architectural tradeoffs. Use `/goal` for large tasks that should run through validation without stopping mid-way.

### 2. Interface Quality Workflows (`.agent/workflows/`)

These 17 workflows originate from the local project directory [c:/.agent/workflows/](file:///c:/MINI%20PROJ-SETU/.agent/workflows/). Each file contains a structured markdown procedure designed to improve a specific dimension of user experience. They do not execute arbitrary shell scripts; instead, they instruct Antigravity to audit, refactor, and verify against professional frontend guidelines.

---

## Detailed Command Documentation

For complete technical details, input parameters, failure modes, and examples for each command, see [05_COMMANDS_REFERENCE.md](05_COMMANDS_REFERENCE.md).
