<!-- markdownlint-disable MD013 -->

# Skill Group 07: Cloud Infrastructure, DevOps & Containerization

## A. Group Overview

- **Group Name**: Cloud Infrastructure, DevOps & Containerization
- **Main Purpose**: Automating cloud infrastructure deployment, container packaging, CI/CD pipeline automation, and multi-cloud operations across AWS, Azure, and GCP.
- **Problems It Helps Solve**:
  - Inconsistent local vs production development environments ("works on my machine").
  - Bloated container images with slow build times and high cold starts.
  - Manual, error-prone production deployments without automated rollbacks.
  - Unstructured cloud infrastructure lacking Infrastructure-as-Code (IaC) governance.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-devops-cloud`, `agentic-bundle-devops-cloud`, `agentic-bundle-azure-ai-cloud`.

---

## B. Skills Within the Group

### 1. `docker-expert`

- **What It Does**: Crafts production-grade multi-stage Dockerfiles, optimizes layer caching, minimizes final image footprints, and applies rootless security hardening.
- **When to Use**: When containerizing the Python FastAPI backend or Vite frontend.
- **When to Avoid**: When running local test suites in a virtualenv.
- **Example**:

  ```text
  Use docker-expert to create an optimized multi-stage Dockerfile for SETU with GPU/CPU PyTorch inference support.
  ```

### 2. `kubernetes-architect`

- **What It Does**: Designs Kubernetes manifests, Helm charts, ingress controllers, HPA (Horizontal Pod Autoscaling), and GitOps deployment workflows (ArgoCD/Flux).
- **When to Use**: When orchestrating multi-pod microservice deployments in cloud clusters.

### 3. `terraform-specialist`

- **What It Does**: Writes modular, reusable Terraform/OpenTofu configurations with remote state locking, least-privilege IAM, and multi-environment workspaces.
- **When to Use**: When provisioning cloud resources (S3/GCS buckets, databases, VPCs, compute instances).

### 4. `github-actions-templates` & `ci-cd-and-automation`

- **What It Does**: Scaffolds reliable CI/CD pipelines that automate linting, testing, security vulnerability scanning, and container registry publishing on every push.
- **When to Use**: When setting up automated pull request verification.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
