<!-- markdownlint-disable MD013 -->

# Skill Group 05: Debugging, Observability & Incident Response

## A. Group Overview

- **Group Name**: Debugging, Observability & Incident Response
- **Main Purpose**: Rapidly identifying root causes of bugs, instrumenting production systems with distributed tracing and metrics, and managing operational reliability.
- **Problems It Helps Solve**:
  - Guesswork and superficial trial-and-error bug fixes.
  - Silent production failures where logs provide zero actionable context.
  - Flaky asynchronous race conditions and memory leaks.
  - Writing blameless postmortems to prevent repeat incidents.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-observability-ir`, `agentic-bundle-observability-monitoring`.

---

## B. Skills Within the Group

### 1. `systematic-debugging`

- **What It Does**: Enforces a strict 4-step diagnostic protocol:
  1. Reproduce the bug reliably.
  2. Inspect inputs and trace data flow end-to-end to identify the root cause.
  3. Grep all callers and formulate a single-point fix.
  4. Verify with runnable tests before declaring completion.
- **When to Use**: Proactively on ANY test failure, unexpected behavior, or reported bug.
- **When to Avoid**: When no issue exists and you are writing new code from scratch.
- **Example**:

  ```text
  Use systematic-debugging to find out why the confidence margin gate rejects valid letters in realsign.py.
  ```

### 2. `observability-engineer` & `observability-and-instrumentation`

- **What It Does**: Instruments application code with OpenTelemetry spans, structured logs, and Prometheus metrics.
- **When to Use**: When shipping production microservices and needing visibility into request latency and error rates.
- **Example**:

  ```text
  Add OpenTelemetry trace spans around each WebSocket frame inference step in recognize.py.
  ```

### 3. `distributed-tracing` & `grafana-dashboards`

- **What It Does**: Configures Jaeger, Tempo, and Grafana to visualize multi-service request flows and design operational dashboards.
- **When to Use**: In distributed systems to pinpoint which downstream service is slowing down requests.

### 4. `incident-responder` & `postmortem-writing`

- **What It Does**: Guides SRE incident response during outages and structures comprehensive blameless postmortems.
- **When to Use**: Immediately following a production service degradation.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
