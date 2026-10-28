<!-- markdownlint-disable MD013 -->

# Antigravity Slash Commands Reference Guide

This document provides a complete technical specification for all **22 verified slash commands** available in this workspace environment.

---

## Part 1: Agent Coordination Commands (5 Commands)

---

### 1. `/plan`

- **Command Name**: `/plan`
- **What It Does**: Analyzes project requirements and produces a step-by-step, atomic implementation plan before any code is modified.
- **Purpose**: Eliminates wasted effort, identifies architectural dependencies early, and ensures the user approves the technical direction.
- **When to Use**: Whenever starting a task that touches more than one file, introduces a new feature, or modifies a core data pipeline.
- **When Not to Use**: For trivial one-line fixes, simple typo corrections, or quick queries.
- **Required Input**: Description of the feature, refactoring, or architectural change desired.
- **Expected Result**: A markdown checklist with categorized phases (Analysis, Core Changes, Testing, Verification).
- **Example**:

  ```text
  /plan Add an audio chime toggle to CameraControls that plays a short sound when an alphabet letter commits.
  ```

- **Skill Group**: Coordination & Planning
- **Related Skills**: `concise-planning`, `writing-plans`, `executing-plans`
- **Source**: Antigravity Core Orchestrator

---

### 2. `/goal`

- **Command Name**: `/goal`
- **What It Does**: Instructs Antigravity to run an autonomous, highly persistent execution loop that does not terminate until the stated goal is 100% completed and verified.
- **Purpose**: Enables unsupervised, long-running agent tasks (e.g. overnight batch test fixing, full refactors).
- **When to Use**: For extensive tasks where you want the agent to iterate through test failures and self-correct without asking for step-by-step approval.
- **When Not to Use**: When you want interactive collaboration or need to review every small diff.
- **Required Input**: A comprehensive goal with measurable success criteria (e.g., "all 125 tests passing").
- **Example**:

  ```text
  /goal Fix all remaining linter warnings in the frontend and make sure npm run build succeeds with zero warnings.
  ```

- **Skill Group**: Autonomous Execution
- **Related Skills**: `goal-loop`, `subagent-driven-development`
- **Source**: Antigravity Core Orchestrator

---

### 3. `/grill-me`

- **Command Name**: `/grill-me`
- **What It Does**: Conducts an interactive Socratic interview, asking you probing technical questions to uncover hidden edge cases and resolve ambiguous design decisions.
- **Purpose**: Prevents building the wrong thing by forcing clarity on tradeoffs before coding begins.
- **When to Use**: When you have an abstract idea (e.g. "let's add a database cache") but haven't settled on the schema, invalidation policy, or failure modes.
- **When Not to Use**: When requirements are already fully settled and documented.
- **Required Input**: An architectural proposal or concept.
- **Example**:

  ```text
  /grill-me I want to migrate our sequence recognition model from rolling window LSTM to a Transformer.
  ```

- **Skill Group**: Design Alignment
- **Related Skills**: `ask-questions-if-underspecified`, `brain-to-docs`
- **Source**: Antigravity Core Orchestrator

---

### 4. `/schedule`

- **Command Name**: `/schedule`
- **What It Does**: Sets a one-shot timer or recurring cron schedule that alerts the agent to perform checks or remind the user.
- **Purpose**: Long-running background polling (e.g., polling build status or checking service health).
- **When to Use**: When waiting on a long external job or wanting periodic status checks.
- **When Not to Use**: Normal synchronous tasks that finish in a few seconds.
- **Required Input**: Duration in seconds or standard 5-field cron expression, plus a prompt.
- **Example**:

  ```text
  /schedule Check server health at http://127.0.0.1:8000/health every 5 minutes for the next hour.
  ```

- **Skill Group**: Background Automation
- **Related Skills**: `schedule` tool in Antigravity Core
- **Source**: Antigravity Core Orchestrator

---

### 5. `/learn`

- **Command Name**: `/learn`
- **What It Does**: Captures a mistake, correction, or project-specific setup detail and persists it into project instructions (`AGENTS.md` or `.agents/rules/`).
- **Purpose**: Teaches the agent permanent project conventions so you never have to correct the same issue twice.
- **When to Use**: Immediately after correcting the agent on a specific project idiosyncrasy (e.g., "always use .venv\Scripts\pytest on Windows").
- **When Not to Use**: For one-off temporary hacks that shouldn't be remembered permanently.
- **Required Input**: Explanation of the rule or pattern to persist.
- **Example**:

  ```text
  /learn Remember that on this Windows system, running global pytest fails because mediapipe is only in .venv. Always use .venv\Scripts\pytest.
  ```

- **Skill Group**: Memory & Learning
- **Related Skills**: `cc-skill-continuous-learning`, `writing-for-agents`
- **Source**: Antigravity Core Orchestrator

---

## Part 2: Interface Quality Workflows (17 Commands)

All 17 workflows originate from [c:/.agent/workflows/](file:///c:/MINI%20PROJ-SETU/.agent/workflows/).

---

### 6. `/adapt`

- **Source File**: `.agent/workflows/adapt.md`
- **What It Does**: Adapts existing UI components and layouts to work across varying screen sizes, devices, input modalities (touch vs mouse), and orientations.
- **When to Use**: When desktop views look broken on mobile devices, touch targets are too small, or layouts overflow on narrow viewports.
- **When Not to Use**: When designing a backend API or writing unit tests.
- **Example**:

  ```text
  /adapt Make CameraWorkspace.tsx responsive down to 360px mobile portrait screens with thumb-reachable controls.
  ```

---

### 7. `/animate`

- **Source File**: `.agent/workflows/animate.md`
- **What It Does**: Enhances user interfaces with intentional, 60fps micro-interactions, layout transitions, and fluid motion effects.
- **When to Use**: When interactive states feel abrupt, buttons pop instantaneously without hover transitions, or badges appear jarringly.
- **When Not to Use**: In text-heavy static documentation or where user preferences request `prefers-reduced-motion`.
- **Example**:

  ```text
  /animate Add smooth 180ms ease-out transitions and subtle scale feedback when sign recognition chips update.
  ```

---

### 8. `/audit`

- **Source File**: `.agent/workflows/audit.md`
- **What It Does**: Conducts a diagnostic quality scan across accessibility (WCAG 2.2), performance, responsive design, and CSS token usage.
- **When to Use**: Before releasing a major UI update or when reviewing a new screen.
- **When Not to Use**: When you want immediate code edits rather than a diagnostic inspection report.
- **Example**:

  ```text
  /audit Run a diagnostic scan on the CameraWorkspace and SubtitlePanel components.
  ```

---

### 9. `/bolder`

- **Source File**: `.agent/workflows/bolder.md`
- **What It Does**: Amplifies safe, timid, or generic designs with bold typography, sharp visual contrast, and distinctive aesthetic identity.
- **When to Use**: When a UI looks like generic grey-on-white bootstrap or lacks brand punch.
- **When Not to Use**: When the interface is already visually intense or needs quiet utilitarian restraint.
- **Example**:

  ```text
  /bolder Give the standby state card in index.css a distinctive optical viewfinder aesthetic.
  ```

---

### 10. `/clarify`

- **Source File**: `.agent/workflows/clarify.md`
- **What It Does**: Rewrites confusing microcopy, error toasts, button labels, and onboarding prompts to be clear, polite, and scannable.
- **When to Use**: When users misinterpret status badges or error messages are technical and unhelpful.
- **When Not to Use**: When rewriting internal code comments.
- **Example**:

  ```text
  /clarify Improve the guidance banner text in CameraWorkspace when the hand is positioned too far from the lens.
  ```

---

### 11. `/colorize`

- **Source File**: `.agent/workflows/colorize.md`
- **What It Does**: Injects strategic, accessible color palettes (using OKLCH or semantic CSS tokens) into monochrome interfaces.
- **When to Use**: When a dashboard is completely greyscale and lacks semantic color coding for success, warnings, or active states.
- **When Not to Use**: When an intentionally brutalist black-and-white theme is requested.
- **Example**:

  ```text
  /colorize Add emerald, amber, and slate-blue accent tokens to the live framing evaluator chips.
  ```

---

### 12. `/critique`

- **Source File**: `.agent/workflows/critique.md`
- **What It Does**: Provides a holistic design review evaluating visual hierarchy, emotional resonance, spacing rhythm, and usability.
- **When to Use**: When you want expert design feedback like a Creative Director before writing code.
- **When Not to Use**: When searching for compiler syntax errors.
- **Example**:

  ```text
  /critique Evaluate the overall layout of the live translation screen.
  ```

---

### 13. `/delight`

- **Source File**: `.agent/workflows/delight.md`
- **What It Does**: Adds playful or memorable touches—such as subtle completion pulses, celebratory micro-copy, or rewarding tactile feedback.
- **When to Use**: On key user milestone moments (e.g., successfully completing an alphabet fingerspelling drill).
- **When Not to Use**: On high-stress critical error dialogs or legal disclaimers.
- **Example**:

  ```text
  /delight Add a subtle celebratory particle pulse when the user holds a sign for 2 continuous seconds.
  ```

---

### 14. `/extract`

- **Source File**: `.agent/workflows/extract.md`
- **What It Does**: Scans components for duplicated inline styles or markup and extracts them into reusable CSS utility classes or shared UI primitives.
- **When to Use**: When multiple components duplicate identical border-radius, backdrop-filter, or button markup.
- **When Not to Use**: When variations between components are truly unique.
- **Example**:

  ```text
  /extract Consolidate all glassmorphic badge styles in index.css into a shared .glass-chip token class.
  ```

---

### 15. `/harden`

- **Source File**: `.agent/workflows/harden.md`
- **What It Does**: Fortifies UI components against edge cases: truncated text, network dropouts, empty states, and rapid clicks.
- **When to Use**: Before production deployment to prevent layout breaks on unexpected user inputs.
- **When Not to Use**: In early disposable wireframe prototypes.
- **Example**:

  ```text
  /harden Ensure the SubtitlePanel gracefully wraps long sentences without overflowing the camera viewfinder.
  ```

---

### 16. `/normalize`

- **Source File**: `.agent/workflows/normalize.md`
- **What It Does**: Harmonizes inconsistent margins, arbitrary pixel values, and rogue hex colors into your project's design token system.
- **When to Use**: When multiple developers have added ad-hoc CSS rules like `margin: 13px` or `#293144`.
- **When Not to Use**: When an element intentionally breaks the grid for a special visual effect.
- **Example**:

  ```text
  /normalize Replace all hardcoded hex values in CameraControls.tsx with var(--setu-*) theme variables.
  ```

---

### 17. `/onboard`

- **Source File**: `.agent/workflows/onboard.md`
- **What It Does**: Designs first-time user guidance, empty state cards, and progressive disclosure for complex features.
- **When to Use**: When a new user lands on a blank camera screen and doesn't know where to hold their hand.
- **When Not to Use**: On power-user utility screens where users already know the workflow.
- **Example**:

  ```text
  /onboard Create a friendly optical positioning guide for first-time signers when camera permissions are granted.
  ```

---

### 18. `/optimize`

- **Source File**: `.agent/workflows/optimize.md`
- **What It Does**: Eliminates redundant React re-renders, optimizes CSS layout thrashing, and reduces DOM tree depth.
- **When to Use**: When the webcam UI drops frames or feels sluggish during live recognition.
- **When Not to Use**: Prematurely on non-critical screens that already render in <5ms.
- **Example**:

  ```text
  /optimize Review useRecognitionStream.ts to prevent unnecessary re-renders of CameraWorkspace.
  ```

---

### 19. `/polish`

- **Source File**: `.agent/workflows/polish.md`
- **What It Does**: The definitive craftsmanship pass: fixes 1px misalignments, uneven padding, jittery borders, and focus rings.
- **When to Use**: As the very last step before presenting a feature or committing UI changes.
- **When Not to Use**: When the underlying logic is still broken and requires heavy refactoring.
- **Example**:

  ```text
  /polish Inspect the top navigation bar, status pill, and camera header for pixel-perfect alignment.
  ```

---

### 20. `/quieter`

- **Source File**: `.agent/workflows/quieter.md`
- **What It Does**: Tones down visual clutter, harsh neon colors, excessive drop shadows, and high-frequency animations.
- **When to Use**: When a user or designer complains that a screen feels chaotic, over-stimulating, or exhausting.
- **When Not to Use**: When the client specifically requested high-energy maximalism.
- **Example**:

  ```text
  /quieter Soften the neon glow and reduce border saturation on the diagnostic stats overlay.
  ```

---

### 21. `/simplify`

- **Source File**: `.agent/workflows/simplify.md`
- **What It Does**: Strips interfaces down to their core essence by removing unnecessary borders, redundant cards, and non-essential controls.
- **When to Use**: When a screen has too many competing buttons, nested boxes, or duplicate labels.
- **When Not to Use**: When all visible elements are strictly required for mission-critical operations.
- **Example**:

  ```text
  /simplify Remove unnecessary borders and nested card wrappers from the LiveCameraIndicator.
  ```

---

### 22. `/teach-impeccable`

- **Source File**: `.agent/workflows/teach-impeccable.md`
- **What It Does**: Analyzes the aesthetic patterns in your codebase and writes persistent design rules into your AI configuration.
- **When to Use**: Once at the beginning of a project to establish consistent visual DNA for all future agent tasks.
- **When Not to Use**: On every regular feature request.
- **Example**:

  ```text
  /teach-impeccable Scan frontend/src/index.css and write our dark-mode glassmorphic rules into project memory.
  ```
