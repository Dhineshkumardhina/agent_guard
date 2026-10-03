# AgentGuard: Planned System Architecture

## Overview

AgentGuard is structured as a modular research platform. Each layer is independently testable and replaceable. The architecture strictly separates data collection, ML modeling, evaluation, and serving concerns.

---

## 1. High-Level Component Map

```
┌─────────────────────────────────────────────────────┐
│              MULTI-AGENT ENVIRONMENT                 │
│   Configurable topologies: Pipeline / Star / Mesh    │
│   Tasks: Research / Coding / Analysis / Planning     │
└──────────────────────┬──────────────────────────────┘
                       │ interaction events (telemetry)
                       ▼
┌─────────────────────────────────────────────────────┐
│              FAULT INJECTION ENGINE                  │
│   Controlled injection of L1/L2/L3 failure modes    │
│   Fault types: hallucination, tool error, loop ...  │
└──────────────────────┬──────────────────────────────┘
                       │ annotated event stream
                       ▼
┌─────────────────────────────────────────────────────┐
│           TELEMETRY COLLECTOR (ml/telemetry/)        │
│   Typed AgentTelemetryEvent schemas                 │
│   Trajectory-level metadata capture                 │
│   Credential sanitization at ingestion              │
└──────────────────────┬──────────────────────────────┘
                       │ structured trajectories
                       ▼
┌─────────────────────────────────────────────────────┐
│        TEMPORAL GRAPH BUILDER (ml/graph/)            │
│   Continuous-time dynamic graph snapshots           │
│   Node features: rolling latency, error rate        │
│   Edge features: contradiction score, token count   │
│   Graph motif detection: loops, bottlenecks         │
└──────────────────────┬──────────────────────────────┘
                       │ graph snapshots + features
             ┌─────────┴──────────┐
             ▼                    ▼
┌────────────────────┐  ┌──────────────────────────────┐
│  BASELINE MODELS   │  │  TEMPORAL GRAPH MODEL (GNN)  │
│  (ml/baselines/)   │  │  (ml/gnn/)                   │
│  - Rule-based      │  │  - Static GNN                │
│  - Logistic Reg.   │  │  - Temporal GNN (CTDG)       │
│  - Random Forest   │  │  - Attention attribution     │
│  - XGBoost         │  │                              │
│  - LSTM            │  │                              │
└─────────┬──────────┘  └──────────────┬───────────────┘
          └─────────────┬──────────────┘
                        ▼
┌─────────────────────────────────────────────────────┐
│              EVALUATION ENGINE (ml/evaluation/)      │
│   Metrics: AUROC, AUPRC, F1, lead time              │
│   Horizons K ∈ {1, 3, 5, 10, 20} interaction steps │
│   Temporal cross-validation (trajectory-level)      │
└──────────────────────┬──────────────────────────────┘
                       │ evaluation metrics
                       ▼
┌─────────────────────────────────────────────────────┐
│           EXPLAINABILITY ENGINE (ml/explainability/) │
│   Attention weight visualization                    │
│   Feature attribution (SHAP / GNNExplainer)        │
│   Causal graph motif attribution                    │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           FASTAPI REST BACKEND (backend/)            │
│   /health, /experiments, /runs, /predictions        │
│   SQLAlchemy ORM + SQLite / PostgreSQL              │
│   Pydantic schema validation                        │
└──────────────────────┬──────────────────────────────┘
                       │ REST API
                       ▼
┌─────────────────────────────────────────────────────┐
│           RESEARCH DASHBOARD (frontend/)             │
│   Real-time run monitoring                          │
│   Prediction horizon visualizations                 │
│   Dynamic graph viewer                              │
│   Comparative model leaderboard                     │
└─────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
agentguard/
├── backend/
│   └── app/
│       ├── api/            # Route handlers (future)
│       ├── core/           # Settings, logging
│       ├── database/       # SQLAlchemy session, ORM models
│       ├── models/         # Domain logic models
│       ├── schemas/        # Pydantic request/response schemas
│       └── services/       # Business services (future)
│
├── ml/
│   ├── config/             # Experiment, simulation configurations
│   ├── data/
│   │   ├── raw/            # Raw telemetry (excluded from git)
│   │   ├── processed/      # Feature matrices and graph snapshots
│   │   └── synthetic/      # Procedurally generated trajectories
│   ├── simulation/         # Agent, topology, task, fault injection
│   ├── graph/              # Temporal graph construction
│   ├── baselines/          # Rule-based, tabular ML, LSTM models
│   ├── gnn/                # Static GNN and Temporal GNN
│   ├── training/           # Training loop and curriculum
│   ├── evaluation/         # Metric computation and horizon analysis
│   ├── explainability/     # Attention and attribution engines
│   ├── experiments/        # Experiment orchestration
│   ├── telemetry/          # Event schema and sanitization
│   └── utils/              # Seeding, hashing, reproducibility
│
├── frontend/               # React + TypeScript dashboard (future)
├── datasets/               # Trajectory datasets and split manifests
├── configs/                # Reproducible YAML experiment specs
├── tests/                  # Unit, integration, ML integrity tests
├── docs/                   # Research documentation
├── results/                # Evaluation outputs and figures
├── scripts/                # Data generation and utility scripts
└── notebooks/              # Exploratory analysis notebooks
```

---

## 3. Data Flow and Temporal Integrity

All feature computation strictly obeys the **no-future-leakage** constraint:

- At prediction step $t$, only events $\{e_0, e_1, \dots, e_t\}$ are visible.
- Rolling statistics use a causal sliding window.
- Graph snapshots are built incrementally, never retroactively.
- A programmatic leakage checker (`verify_no_future_leakage`) is enforced during feature extraction.

---

## 4. Database Schema Relationships

```
Experiment ──< Run ──< Agent
                   ──< Event
                   ──< AgentInteraction
                   ──< FaultInjection
                   ──< Failure
                   ──< Prediction
           ──< ModelResult
```

---

## 5. Phase Delivery Plan

| Phase | Description                          | Status   |
|-------|--------------------------------------|----------|
| 1     | Foundation (config, DB, schemas)     | ✅ Done  |
| 2     | Multi-Agent Simulator                | 🔜 Next  |
| 3     | Telemetry Collector                  | ⬜       |
| 4     | Fault Injection Engine               | ⬜       |
| 5     | Temporal Graph Builder               | ⬜       |
| 6     | Dataset Generator                    | ⬜       |
| 7     | ML Baselines                         | ⬜       |
| 8     | Static GNN                           | ⬜       |
| 9     | Temporal GNN                         | ⬜       |
| 10    | Evaluation Engine                    | ⬜       |
| 11    | Explainability Engine                | ⬜       |
| 12    | FastAPI Business APIs                | ⬜       |
| 13    | Research Dashboard (Frontend)        | ⬜       |
| 14    | End-to-End Integration               | ⬜       |
