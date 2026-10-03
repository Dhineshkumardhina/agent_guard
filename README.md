# AgentGuard: Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AgentGuard is an experimental research platform designed to investigate:

> **Whether the temporal evolution of interactions between agents provides predictive information about impending cascading failures beyond conventional agent-level trajectory features.**

---

## 1. Research Motivation & Core Questions

Autonomous multi-agent systems often experience complex cascading failure dynamics. A failure in an upstream agent (e.g. planner or researcher) can propagate silently across multiple hops before manifesting as an unrecoverable system crash or corrupted output.

AgentGuard addresses the following core question:
* *Can temporal interaction patterns among agents improve prediction of impending cascading failures compared with agent-level behavioral features and static graph representations?*

### Formal Hypotheses
* **H0 (Null):** Agent-level behavioral features are sufficient for predicting impending failures.
* **H1:** Temporal interaction-graph information provides additional predictive value beyond agent-level behavioral features.
* **H2:** Temporal graph models provide better early-warning performance (lead time) than static graph models.
* **H3:** Certain evolving graph motifs (loops, delegation bottlenecks, reciprocity shifts) are associated with cascading failures.

---

## 2. High-Level Architecture

```
                MULTI-AGENT ENVIRONMENT
                          |
                          v
                 EVENT / TRACE COLLECTOR
                          |
                          v
                TEMPORAL GRAPH BUILDER
                          |
                          v
                 FEATURE ENGINEERING
                          |
         +----------------+----------------+
         |                                 |
         v                                 v
   BASELINE MODELS                 TEMPORAL GRAPH MODEL
         |                                 |
         +----------------+----------------+
                          |
                          v
                FAILURE PREDICTION
                          |
                          v
                EARLY WARNING ENGINE
                          |
                          v
                EXPLANATION ENGINE
                          |
                          v
                   WEB DASHBOARD
```

---

## 3. Project Structure

```
agentguard/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── database/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   └── requirements.txt
│
├── ml/
│   ├── config/             # Experiment and simulation configurations
│   ├── data/               # Raw, processed, and synthetic datasets
│   ├── simulation/         # Multi-agent simulation environments and agents
│   ├── telemetry/          # Structured event telemetry schemas and logging
│   ├── graph/              # Dynamic graph snapshots and feature builders
│   ├── baselines/          # Rule-based, tabular ML, and LSTM models
│   ├── gnn/                # Static GNN and Continuous-Time Temporal GNN
│   ├── training/           # Training pipelines
│   ├── evaluation/         # Early warning lead time & discrimination metrics
│   ├── explainability/     # Attention and feature attribution
│   └── utils/              # Deterministic seeding and reproducibility tools
│
├── frontend/               # Interactive React / TypeScript dashboard
├── datasets/               # Trajectory datasets and split manifests
├── configs/                # Reproducible YAML experiment specifications
├── tests/                  # Unit, ML integrity, leakage, and integration tests
├── docs/                   # Scientific methodology and protocol documentation
├── results/                # Experiment evaluation outputs and figures
└── pyproject.toml
```

---

## 4. Quickstart & Installation

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
```

### Run Test Suite
```bash
pytest tests/ -v
```

### Launch Backend Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```

---

## 5. Development Status & Phases

- [x] **Phase 1: Research-Grade Project Foundation** (Config, DB schemas, telemetry contracts, reproducibility, initial test suite)
- [ ] **Phase 2: Multi-Agent Simulator** (Configurable agents, topologies, deterministic tasks)
- [ ] **Phase 3: Telemetry / Event Collector** (Structured interaction recording)
- [ ] **Phase 4: Fault Injection Engine** (Controlled multi-level fault injection)
- [ ] **Phase 5: Temporal Graph Builder** (Continuous dynamic graph snapshots & features)
- [ ] **Phases 6–20:** Dataset generator, baselines, GNNs, evaluation, explainability, API, UI, documentation.
