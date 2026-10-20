# SETU --- AI Agent and Engineering Rules

**Status:** Binding working rules for Antigravity and contributors.\
**Last updated:** 2026-10-17

## 1. Product and scope

1.  SETU is **ISL-first**. Never silently substitute ASL data, labels or
    assumptions.
2.  The immediate target is a working, evidence-backed five-sign MVP. Do
    not add advanced features before it passes.
3.  Reuse-first is mandatory: look for suitable licensed repositories,
    models, datasets and training code before rebuilding equivalents.
4.  Check source code, model weights, training data, labels,
    preprocessing, dependencies and licenses separately.
5.  A repository title or README claim does not prove authentic ISL
    vocabulary or accuracy.
6.  Do not claim recognition works because code imports, synthetic tests
    pass or the model emits plausible labels. Test against known real
    signs.
7.  Do not claim full ISL translation; state the exact vocabulary and
    tested conditions.
8.  Product decisions belong to the user. Recommend, but do not silently
    expand scope.

## 2. Phase discipline

Use: **Plan briefly → Implement approved scope → Test → Demonstrate →
Report evidence/limitations → Git checkpoint → Request approval.** -
Work on one milestone at a time; do not skip failed prerequisites. - Do
not train before positive real ISL samples and labels are verified. - Do
not build React/FastAPI/database features before the local recognition
MVP works unless approved. - Avoid repeated long reports when the next
action is clear. - Stop and ask before destructive operations or scope
changes. - Update `docs/memory.md` after a meaningful decision,
milestone, blocker or changed assumption.

## 3. Reuse and licensing

For every third-party component, record repository URL, exact revision,
code license, separate model-weight/data permissions, attribution,
dependencies, modifications and tests. A code license does not
automatically license weights or data. If permissions are unclear, do
not copy, modify, redistribute or bundle the asset. Keep experimental
components isolated and preserve notices. Never rename an unverified
model in a way that implies a verified ISL identity.

## 4. Technology rules

**Use where already present and useful:** Python, OpenCV, MediaPipe,
NumPy and pytest.

**Add only for a demonstrated need:** TensorFlow/Keras, PyTorch,
scikit-learn, FastAPI, WebSockets, React/Vite, SQLite or other
dependencies. Explain why the current stack cannot do the job, license,
compatibility and dependency impact before adding one.

**Avoid:** competing ML frameworks without need, large upgrades during
unrelated work, rebuilding functioning components, generic plugin
systems/event buses/abstraction layers without a requirement, and
"future-proofing" that does not support the current milestone.

## 5. Code style and boundaries

-   Python modules/functions/variables use `snake_case`; classes use
    `PascalCase`; constants use uppercase.
-   Add type hints to public functions and docstrings for non-obvious
    behavior.
-   Keep functions focused and modules cohesive.
-   Keep feature extraction deterministic; avoid hidden
    camera/filesystem/global-state dependencies in pure feature
    functions.
-   Separate UI/demo code from recognition/model logic.
-   Keep preprocessing inside a model-specific adapter.
-   Centralize configuration; avoid scattered magic numbers, paths,
    thresholds and label maps.
-   Document units and coordinate conventions.
-   Version feature schemas and label maps; changes require migration
    notes and tests.

## 6. Errors and logging

-   Use project exceptions where available; do not silently swallow
    errors.
-   Catch exceptions where recovery is possible and provide actionable
    user messages.
-   Validate file existence, tensor shapes, feature dimensions, finite
    values and label counts.
-   Handle camera failure/disconnect, missing model, invalid landmarks,
    corrupt data and inference failure.
-   Release camera/GUI resources on normal exit and exceptions.
-   Never convert failed inference into a valid-looking prediction.
-   Do not expose secrets, raw biometric data or internal stack traces
    to end users.

## 7. Configuration and secrets

-   Never hardcode API keys, passwords, tokens, personal data or
    environment-specific absolute paths.
-   Use environment variables or typed settings when needed.
-   `.env.example` contains placeholders only; real `.env` files stay
    out of Git.
-   Prefer configurable/project-relative paths.
-   No cloud credentials or external service in the MVP without
    approval.

## 8. Data and ML integrity

-   Record data provenance and license/attribution.
-   Never fabricate labels, signer IDs, sessions, URLs or counts.
-   Mark samples as real, synthetic or augmented; synthetic samples are
    not human recordings.
-   Never treat ASL and ISL labels as equivalent.
-   Prevent leakage across splits; group by signer when verified IDs and
    enough signers exist.
-   Keep test data out of training and tuning.
-   Report per-class metrics, macro-F1, confusion matrix and
    false-positive behavior where counts support them.
-   Record random seeds, preprocessing, split strategy, model
    configuration and dependency versions.
-   Never claim accuracy/FPS/latency without logs and a defined
    measurement method.
-   Do not assume model scores are calibrated probabilities.
-   Do not force predictions for absent, invalid or unsupported inputs.

## 9. Files and destructive operations

Without explicit user approval, do not delete/overwrite/bulk-rename
files; remove models/data/license notices; rewrite Git history or
discard work; replace the feature schema/label map; change the core
framework/base repository; move folders in bulk; or download/bundle a
large dataset without a bounded licensed plan.

Before refactoring, inspect `git status`, inventory affected files and
create a safe checkpoint. Preserve existing package, tests, reference
artifacts and user-authored documentation.

## 10. Testing

-   Add a regression test for each fixed bug.
-   Run narrow tests first, then the full suite.
-   Report exact commands and actual outcomes.
-   Distinguish unit tests, synthetic pipeline tests, offline model
    tests and real-camera human-sign tests.
-   Passing tests do not prove semantic correctness.
-   Never weaken/delete tests just to make them pass.
-   If a test cannot run, report the blocker.

## 11. Documentation and reporting

Keep docs aligned with code; label plans as plans. Every milestone
report states changed files, commands/tests and results, limitations,
Git checkpoint and next action/approval. Separate verified facts from
assumptions. Never invent completion evidence.

## 12. Privacy, security and accessibility

Process camera data locally by default. Do not record/upload/retain
video without explicit purpose and consent. Avoid storing identifiable
recordings unless approved and protected. Review security impact before
dependency upgrades. Do not rely on color alone for state. Future web
controls must be readable and keyboard accessible.

## 13. Required milestone report

1.  **Status:** Not started / In progress / Blocked / Completed with
    evidence.
2.  **Changes:** exact files and behavior.
3.  **Verification:** commands, tests, real-data/camera checks and
    results.
4.  **Limitations:** what is not proven.
5.  **Checkpoint:** commit hash or why none was created.
6.  **Next step:** one concrete action and whether approval is needed.

Do not report "complete" if an acceptance criterion is unmet.
