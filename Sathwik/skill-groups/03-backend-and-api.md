<!-- markdownlint-disable MD013 -->

# Skill Group 03: Backend, API Architecture & Frameworks

## A. Group Overview

- **Group Name**: Backend, API Architecture & Frameworks
- **Main Purpose**: Building high-performance, asynchronous, type-safe backend services and REST/WebSocket APIs.
- **Problems It Helps Solve**:
  - Designing clean, production-grade endpoints with Pydantic V2 validation.
  - Handling asynchronous WebSocket streams without connection drops.
  - Generating OpenAPI 3.1 specifications and typed client SDKs.
  - Preventing memory leaks in deep learning inference loops.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-api-platform-builder`, `agentic-bundle-aas-python-api-builder`, `agentic-bundle-full-stack-developer`, `agentic-bundle-python-pro`, `agentic-bundle-typescript-javascript`.

---

## B. Skills Within the Group

### 1. `fastapi-pro` & `fastapi-templates`

- **What It Does**: Implements production-ready FastAPI applications with dependency injection, router splitting, async endpoints, and lifespan event management.
- **When to Use**: When adding new API routes, WebSocket handlers, or backend services.
- **When to Avoid**: In frontend React components.
- **Example**:

  ```text
  Use fastapi-pro to add a new /api/v1/calibration endpoint for recording user baseline landmark scales.
  ```

### 2. `pydantic-models-py`

- **What It Does**: Crafts multi-model Pydantic V2 schemas (Base, Create, Read, Update) ensuring strict validation and clean serialization.
- **When to Use**: When defining request payloads or database models.
- **Example**:

  ```text
  Use pydantic-models-py to define the schema for CalibrationRequest and CalibrationResponse.
  ```

### 3. `api-design-principles` & `openapi-spec-generation`

- **What It Does**: Enforces RESTful conventions, proper HTTP status codes, structured error envelopes, and automated OpenAPI 3.1 spec generation.
- **When to Use**: When designing public APIs or documenting internal microservice contracts.

### 4. `python-pro` & `async-python-patterns`

- **What It Does**: Guides idiomatic Python 3.12+ async programming, non-blocking I/O, concurrency with asyncio, and resource management.
- **When to Use**: When optimizing high-frequency frame processing in WebSocket loops.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
