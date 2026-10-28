<!-- markdownlint-disable MD013 -->

# Skill Group 10: Business Analysis, SaaS Growth & Product Launch

## A. Group Overview

- **Group Name**: Business Analysis, SaaS Growth & Product Launch
- **Main Purpose**: Accelerating digital product monetization, integrating payment gateways (Stripe/PayPal), optimizing organic search visibility (SEO), and tracking product KPI metrics.
- **Problems It Helps Solve**:
  - Shipping a technical feature without a clear business monetization or pricing strategy.
  - Insecure, non-PCI-compliant payment flows and unhandled webhook web events.
  - Zero search engine indexability due to missing Schema.org metadata and poor Core Web Vitals.
  - Designing dashboards that track Vanity metrics instead of North Star KPIs.
- **Origin & Location**:
  - Global Plugins: `agentic-bundle-aas-saas-launch-revenue`, `agentic-bundle-startup-founder`, `agentic-bundle-commerce-payments`, `agentic-bundle-business-analyst`, `agentic-bundle-aas-marketing-seo-growth`, `agentic-bundle-seo-specialist`.

---

## B. Skills Within the Group

### 1. `stripe-integration`

- **What It Does**: Implements PCI-compliant Stripe checkout sessions, customer billing portals, subscription tier webhooks, and refund workflows.
- **When to Use**: When adding paid subscriptions, usage-based billing, or one-time purchases to an app.
- **When to Avoid**: When building internal non-commercial utilities.
- **Example**:

  ```text
  Use stripe-integration to set up a subscription webhook listener in our FastAPI backend verifying stripe-signature headers.
  ```

### 2. `saas-mvp-launcher` & `micro-saas-launcher`

- **What It Does**: Provides structured launch checklists covering tech stack validation, authentication, billing, analytics events, and launch day checklists.
- **When to Use**: When bootstrapping a new SaaS product from an idea to paying customers in weeks.

### 3. `seo-fundamentals`, `seo-audit` & `schema-markup`

- **What It Does**: Audits technical search engine optimization: crawler indexing, Core Web Vitals, JSON-LD Schema.org structured data, and meta tags.
- **When to Use**: When optimizing landing pages or public web applications for Google search rankings.

### 4. `business-analyst` & `kpi-dashboard-design`

- **What It Does**: Constructs business analytics models: Customer Acquisition Cost (CAC), Lifetime Value (LTV), monthly churn rates, and executive KPI dashboards.
- **When to Use**: When presenting business performance to investors or stakeholders.

---

Back to [Skill Groups Index](README.md) | [Main Handbook README](../README.md)
