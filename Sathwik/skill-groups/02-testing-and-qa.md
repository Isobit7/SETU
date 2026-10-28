<!-- markdownlint-disable MD013 -->

# Skill Group 02: Testing, Quality Assurance & Verification

## A. Group Overview

- **Group Name**: Testing, Quality Assurance & Verification
- **Main Purpose**: Guaranteeing software correctness, preventing regressions, automating end-to-end user journeys, and load-testing APIs.
- **Problems It Helps Solve**:
  - Catching breaking changes before shipping code.
  - Automating the Red-Green-Refactor testing cycle.
  - Diagnosing and systematically resolving failing test suites.
  - Ensuring browser UI flows work correctly under realistic user interactions.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-qa-test-automation`, `agentic-bundle-qa-testing`, `agentic-bundle-essentials`.

---

## B. Skills Within the Group

### 1. `test-driven-development` (TDD)

- **What It Does**: Guides the strict Red-Green-Refactor cycle: writing failing unit tests first, implementing minimal code to make them pass, then refactoring.
- **When to Use**: When implementing new features, algorithms, or bugfixes.
- **When to Avoid**: When creating rapid throwaway prototypes or exploratory research scripts.
- **Example**:

  ```text
  Use test-driven-development to write tests for the thumb-to-index aperture formula before updating geometry.py.
  ```

### 2. `test-fixing`

- **What It Does**: Systematically groups, isolates, and repairs failing test suites without weakening assertion logic or introducing mock cheats.
- **When to Use**: When running `pytest` or `npm test` reveals multiple broken tests after a refactor.
- **When to Avoid**: When no tests are failing.
- **Example**:

  ```text
  Run test-fixing on tests/test_realsign_classifier.py to address the assertion failure on letter C.
  ```

### 3. `e2e-testing-patterns` & `playwright-skill`

- **What It Does**: Designs robust end-to-end browser tests using Playwright, handling async waiting, selectors, and screenshot diffs.
- **When to Use**: When verifying full browser journeys (e.g. webcam permission grant -> canvas stream -> WebSocket prediction -> subtitle display).
- **When to Avoid**: For pure mathematical unit logic that can be tested in <10ms with pytest.

### 4. `k6-load-testing`

- **What It Does**: Simulates realistic concurrent traffic to test backend latency, throughput, and WebSocket connection saturation.
- **When to Use**: Before production deployment to test how many concurrent WebSocket video streams the FastAPI backend can support.

---

## C. Sub-Skills & Workflows

- `tdd-workflows-tdd-red`: Generates clean failing test assertions.
- `tdd-workflows-tdd-green`: Implements minimal code to make tests pass.
- `tdd-workflows-tdd-refactor`: Cleans up implementation while keeping tests green.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
