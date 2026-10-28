<!-- markdownlint-disable MD013 -->

# UI & UX Design Workflows

These 17 workflows originate from the local project directory [c:/.agent/workflows/](file:///c:/MINI%20PROJ-SETU/.agent/workflows/). Typing any of these with a leading `/` instructs Antigravity to load and execute that specific markdown playbook.

---

## The 17 Design Workflows Catalog

### 1. `/adapt`

- **Playbook**: [adapt.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/adapt.md)
- **Focus**: Responsive cross-device layouts (Mobile, Tablet, Desktop, Touch vs Mouse).
- **Checklist Summary**:
  - Convert multi-column desktop grids into vertical stacks for mobile viewports.
  - Enforce minimum touch target sizes of 44x44px.
  - Replace desktop hover-only patterns with tap/focus patterns.

### 2. `/animate`

- **Playbook**: [animate.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/animate.md)
- **Focus**: Fluid 60fps micro-interactions, cubic-bezier easing, and reduced-motion fallbacks.
- **Checklist Summary**:
  - Replace jarring 0ms state switches with 150ms-250ms transitions.
  - Animate only `transform` and `opacity` to maintain 60fps GPU acceleration.
  - Respect `@media (prefers-reduced-motion: reduce)`.

### 3. `/audit`

- **Playbook**: [audit.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/audit.md)
- **Focus**: Full diagnostic scan of WCAG accessibility, performance, and CSS tokens.
- **Checklist Summary**:
  - Generates a prioritized issue report without immediately modifying code.

### 4. `/bolder`

- **Playbook**: [bolder.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/bolder.md)
- **Focus**: Amplifying safe or boring designs into distinctive, memorable aesthetics.

### 5. `/clarify`

- **Playbook**: [clarify.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/clarify.md)
- **Focus**: Polishing UX copy, error messages, form field placeholders, and labels.

### 6. `/colorize`

- **Playbook**: [colorize.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/colorize.md)
- **Focus**: Introducing semantic color systems using OKLCH and CSS custom properties.

### 7. `/critique`

- **Playbook**: [critique.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/critique.md)
- **Focus**: Holistic UX design critique evaluating hierarchy, visual rhythm, and cognitive load.

### 8. `/delight`

- **Playbook**: [delight.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/delight.md)
- **Focus**: Subtle personality, celebratory feedback, and satisfying interactions.

### 9. `/extract`

- **Playbook**: [extract.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/extract.md)
- **Focus**: Consolidating duplicated styles and markup into reusable components and tokens.

### 10. `/harden`

- **Playbook**: [harden.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/harden.md)
- **Focus**: Fortifying UI resilience against long text overflows, offline states, and edge cases.

### 11. `/normalize`

- **Playbook**: [normalize.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/normalize.md)
- **Focus**: Standardizing arbitrary hex codes and magic pixel numbers into design tokens.

### 12. `/onboard`

- **Playbook**: [onboard.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/onboard.md)
- **Focus**: Crafting welcoming empty states, first-run checklists, and feature introductions.

### 13. `/optimize`

- **Playbook**: [optimize.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/optimize.md)
- **Focus**: Frontend performance tuning (render frequency, reflows, bundle size).

### 14. `/polish`

- **Playbook**: [polish.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/polish.md)
- **Focus**: Pixel-perfect craftsmanship pass on borders, alignment, padding, and subtle shadows.

### 15. `/quieter`

- **Playbook**: [quieter.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/quieter.md)
- **Focus**: Calming overly aggressive, hyper-saturated, or visually noisy interfaces.

### 16. `/simplify`

- **Playbook**: [simplify.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/simplify.md)
- **Focus**: Stripping unnecessary borders, card nesting, and visual noise.

### 17. `/teach-impeccable`

- **Playbook**: [teach-impeccable.md](file:///c:/MINI%20PROJ-SETU/.agent/workflows/teach-impeccable.md)
- **Focus**: Scanning codebase styles and saving design guidelines into persistent memory.

---

Back to [Commands Index](README.md) | [Main Handbook README](../README.md)
