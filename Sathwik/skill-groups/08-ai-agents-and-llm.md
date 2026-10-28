<!-- markdownlint-disable MD013 -->

# Skill Group 08: AI Agents, LLM Orchestration & RAG

## A. Group Overview

- **Group Name**: AI Agents, LLM Orchestration & RAG
- **Main Purpose**: Architecting autonomous multi-agent systems, Retrieval-Augmented Generation (RAG) pipelines, Model Context Protocol (MCP) servers, and LLM observability.
- **Problems It Helps Solve**:
  - LLM hallucinations when answering domain-specific technical questions.
  - Context window overflow and context degradation in long agent conversations.
  - Designing stateful cyclic multi-actor workflows using LangGraph.
  - Building standards-compliant MCP servers so LLMs can securely access tools.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-agent-mcp-builder`, `agentic-bundle-agent-architect`, `agentic-bundle-llm-application-developer`, `agentic-bundle-aas-ai-product-evaluation-ops`.

---

## B. Skills Within the Group

### 1. `ai-agents-architect` & `langgraph`

- **What It Does**: Designs stateful multi-agent architectures with memory checkpointers, human-in-the-loop gates, subagent orchestration, and cyclic graphs.
- **When to Use**: When building custom AI agents that collaborate to solve multi-stage programming tasks.
- **When to Avoid**: When building a simple deterministic Python script.
- **Example**:

  ```text
  Use langgraph to design a stateful review graph that passes code through a linter agent, test runner agent, and architect reviewer.
  ```

### 2. `rag-engineer` & `rag-implementation`

- **What It Does**: Implements Retrieval-Augmented Generation pipelines: optimal chunking strategies, embedding model selection, vector indexing, and hybrid semantic search.
- **When to Use**: When creating an AI question-answering assistant over your private project documentation.

### 3. `mcp-builder` & `mcp-tool-developer`

- **What It Does**: Builds Model Context Protocol (MCP) servers from scratch in TypeScript or Python, exposing custom tools and resources with verified JSON schemas.
- **When to Use**: When you want to give Antigravity access to an internal corporate API or proprietary database.
- **Example**:

  ```text
  Use mcp-builder to create a custom Model Context Protocol server that lets Antigravity inspect our sign landmark dataset.
  ```

### 4. `langfuse` & `agent-evaluation`

- **What It Does**: Integrates open-source LLM tracing, prompt versioning, benchmark evaluation, and latency/cost monitoring.
- **When to Use**: When evaluating model output quality and measuring prompt costs in production.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
