<!-- markdownlint-disable MD013 -->

# Skill Group 04: Software Architecture, DDD & System Design

## A. Group Overview

- **Group Name**: Software Architecture, DDD & System Design
- **Main Purpose**: Structuring large, maintainable, evolvable software systems using Domain-Driven Design (DDD), Clean Architecture, CQRS, and high-performance systems languages.
- **Problems It Helps Solve**:
  - Spaghetti code and unmaintainable monolithic files.
  - Undocumented technical decisions that leave future teams confused.
  - Tight coupling between business logic and database/framework code.
  - Designing event streams and distributed state transitions.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-architecture-design`, `agentic-bundle-ddd-evented-architecture`, `agentic-bundle-systems-programming`.

---

## B. Skills Within the Group

### 1. `senior-architect` & `architecture-patterns`

- **What It Does**: Provides architectural guidance across Hexagonal Architecture, Clean Architecture, and microservice boundaries.
- **When to Use**: When structuring a new repository or refactoring a subsystem with high coupling.
- **Example**:

  ```text
  Use senior-architect to decouple feature extraction from the PyTorch model adapters in our models directory.
  ```

### 2. `architecture-decision-records` (ADR)

- **What It Does**: Creates standardized Architecture Decision Records documenting context, options considered, decisions made, and consequences.
- **When to Use**: Whenever making significant technical choices (e.g., choosing PyTorch MLP over Random Forest, or selecting WebSocket protocol over HTTP polling).
- **Example**:

  ```text
  Write an ADR documenting why SETU uses 63-D wrist-normalized features for alphabet recognition rather than raw pixels.
  ```

### 3. `domain-driven-design` (DDD), `cqrs-implementation` & `event-sourcing-architect`

- **What It Does**: Maps business subdomains into Bounded Contexts and separates write models (Commands) from read models (Queries/Projections).
- **When to Use**: In complex domains with intricate business invariants or temporal audit trail requirements.

### 4. `rust-pro`, `golang-pro` & `cpp-pro`

- **What It Does**: Writes idiomatic, memory-safe, high-performance systems code with modern patterns (RAII, concurrency channels, move semantics).
- **When to Use**: When writing native computer vision bindings or low-latency audio/video processing engines.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
