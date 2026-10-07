# AgentGuard: Temporal Graph-Based Detection and Prediction of Failures in Multi-Agent AI Systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Test Suite: 339 Passed](https://img.shields.io/badge/tests-339%20passed-brightgreen.svg)](docs/testing.md)
[![Security: Bandit Clean](https://img.shields.io/badge/security-bandit%20clean-success.svg)](docs/security_audit.md)

---

## 1. Project Overview

**AgentGuard** is an open-source, research-grade software and experimentation platform investigating early failure detection and prediction in collaborative multi-agent AI systems.

### The Problem Investigated
As LLM-based autonomous agents are deployed in complex multi-agent workflows (collaborative teams of Planners, Researchers, Analysts, Software Engineers, and Verifiers), they become susceptible to **cascading failures**. An isolated error in an upstream agent—such as a subtle factual hallucination, unhandled tool timeout, or conflicting premise—frequently propagates across intermediate message-passing handoffs and delegation chains, culminating in a catastrophic, unrecoverable system breakdown.

### Why Multi-Agent Failures Are Difficult
1. **Multi-Hop Propagation:** The triggering fault often occurs several steps and multiple agent interactions away from where the failure ultimately manifests.
2. **Latent Semantic Divergence:** Intermediate agents may accept subtly corrupt inputs and generate plausible but erroneous artifacts without raising immediate tool exceptions.
3. **Temporal Dynamics:** Interactions exhibit non-stationary burstiness, communication loops, and shifting reciprocity that static snapshots fail to capture.

### What Temporal Interaction Graphs Contribute
AgentGuard models the evolving communication history as a **continuous-time dynamic interaction graph** $\mathcal{G}(t) = (\mathcal{V}(t), \mathcal{E}(t), \mathbf{X}_V(t), \mathbf{X}_E(t))$. By combining continuous sinusoidal Fourier time encoding $\phi(\Delta t)$, dynamic per-agent recurrent memory banks $\mathbf{m}_v(t)$, and relational graph attention, the system captures pre-failure communication bottlenecks, delegation cycles, and latency shifts before physical system collapse occurs.

### What the System Evaluates
AgentGuard does **not** assert universal superiority of any single architecture. Instead, it provides a controlled, reproducible benchmark comparing:
* **9 Model Families:** Rule-Based, Logistic Regression, Random Forest, XGBoost, LSTM, GRU, GCN, GAT, and Continuous-Time Temporal GNN.
* **5 Prediction Horizons:** Forecasting failures $K \in \{1, 3, 5, 10, 20\}$ interaction steps ahead.
* **9 Systematic Ablations:** Isolating the marginal contribution of time encodings, graph topology, node telemetry, edge attributes, and recurrent memory.
* **4 Generalization Shift Dimensions:** Evaluating transfer across agent population scale ($N \in \{3, 5\} \to \{8, 12\}$), topology shifts, task domains, and unseen failure modes.
* **5 Explainability Attribution Levels:** Providing actionable diagnostics through feature rankings, agent attributions, directed interaction channels, and counterfactual sensitivity deltas.

---

## 2. High-Level Architecture

```
  MULTI-AGENT SIMULATOR (ml/simulation/)
  ├── Topologies: Pipeline, Star, Mesh, Custom (3 to 12 agents)
  └── Tasks: Research, Coding, Data Analysis, Planning
             │
             ▼
  CONTROLLED FAULT INJECTION (ml/simulation/fault_injection/)
  └── 12 Canonical Fault Modes across 3 Severity Levels (L1/L2/L3)
             │
             ▼
  STRUCTURED TELEMETRY COLLECTOR (ml/telemetry/)
  └── Credential sanitization, continuous timestamps, latency, tokens
             │
             ▼
  TEMPORAL GRAPH BUILDER (ml/graph/)
  └── Dynamic interaction stream G(t) with strict zero-leakage invariant
             │
             ▼
  DATASET GENERATION & PARTITIONING (ml/data/)
  └── Run-level stratified splits (train/val/test) across horizons K
             │
       ┌─────┴─────────────────────────────────────┐
       ▼                                           ▼
  BENCHMARK BASELINES (ml/baselines/)         CONTINUOUS-TIME TEMPORAL GNN
  - Rule-Based Thresholds                     - Fourier Time Encoding φ(Δt)
  - Classical ML (LogReg, RF, XGBoost)        - Persistent Node Memory Bank m_v
  - Sequence Models (LSTM, GRU)               - Temporal Graph Attention
  - Static GNNs (GCN, GAT)                    - Multi-Pooling Readout & Hazard Head
       │                                           │
       └─────┬─────────────────────────────────────┘
             │
             ▼
  COMPREHENSIVE EVALUATION ENGINE (ml/evaluation/)
  ├── Metrics: AUROC, AUPRC, F1, Precision, Recall, Brier, ECE, Lead Time
  ├── Trajectory Block Bootstrap (B=500, 95% Confidence Intervals, paired p-values)
  └── Incident-level early warning verification (t_warn <= t_fail)
             │
       ┌─────┼─────────────────────────────────────┐
       ▼     ▼                                     ▼
  ABLATION   GENERALIZATION BENCHMARKS             EXPLAINABILITY FRAMEWORK
  9 studies  4 OOD dimensions                      5 attribution levels, case studies
       │     │                                     │
       └─────┴─────────────────┬───────────────────┘
                               │
                               ▼
  FASTAPI REST BACKEND (backend/app/) ──> REACT DASHBOARD (frontend/src/)
  Sub-20ms latencies, SQLAlchemy ORM     10 interactive views, graph visualizer
```

---

## 3. Key Empirical Findings (Phase 12 Benchmark Summary)

Evaluated on the standardized $N=35$ test population of `agentguard_dataset_v1` at horizon $K=1$:

| Model Family | Model | Precision | Recall | F1 Score | AUROC | AUPRC | Brier Score | ECE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical ML | Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | 0.017 |
| Classical ML | Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 0.016 |
| Classical ML | XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 0.025 |
| Temporal Sequence | GRU | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.029 | 0.105 |
| Static GNN | GCN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.092 |
| Static GNN | GAT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.035 | 0.174 |
| Temporal Sequence | LSTM | 1.000 | 0.923 | 0.960 | 1.000 | 1.000 | 0.049 | 0.158 |
| Temporal GNN | Temporal GNN | 1.000 | 0.538 | 0.700 | 0.923 | 0.995 | 0.258 | 0.440 |
| Rule-Based | Rule-Based | 1.000 | 0.154 | 0.267 | 1.000 | 1.000 | 0.531 | 0.699 |

### Statistical Rigor & Epistemological Stance:
* **No Premature Significance Claims:** Paired trajectory block bootstrap testing indicates that the point performance difference between Temporal GNN and Classical ML (XGBoost) does not reach conventional statistical significance on the current sample size ($p = 0.080 > 0.05$).
* **Role-Specific Strengths:** Classical ML excels on local single-agent failures where aggregated telemetry signals are intense. Graph-aware architectures provide superior structural attribution and cascade tracing along complex multi-hop communication corridors.

---

## 4. Quickstart and Installation

### 4.1 Clone and Environment Setup
```powershell
# Clone the repository
git clone https://github.com/Dhineshkumardhina/agent_guard.git
cd agent_guard

# Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Linux/macOS: source .venv/bin/activate

# Install Python scientific and backend dependencies
pip install -r requirements.txt

# Verify environment imports and system provenance
python scripts/verify_env.py
```

### 4.2 Run Test Pyramid & Quality Gates
```powershell
# Run the complete test suite (327 tests)
pytest tests/ -v

# Run rapid quality gate verification (SAST scan, Vitest, 69 invariant tests, API perf)
python scripts/run_full_validation.py --quick
```

### 4.3 Launch Backend Server
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
Interactive API documentation is accessible at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4.4 Launch Frontend Research Dashboard
In a separate terminal:
```powershell
cd frontend
npm.cmd install    # On Linux/macOS: npm install
npm.cmd run dev    # On Linux/macOS: npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 5. Documentation Architecture

Comprehensive documentation is structured into the following specialized guides:

| Category | Document | Description |
|---|---|---|
| **Research Foundations** | [`docs/research_problem.md`](docs/research_problem.md) | Problem formulation, multi-hop failure cascading, and failure hierarchy. |
| | [`docs/research_questions.md`](docs/research_questions.md) | Formal Research Questions RQ1 through RQ6. |
| | [`docs/hypotheses.md`](docs/hypotheses.md) | Testable hypotheses H0 through H3 and evaluation criteria. |
| **System & Pipeline** | [`docs/architecture.md`](docs/architecture.md) | 13-stage end-to-end system architecture, data flows, and storage models. |
| | [`docs/methodology.md`](docs/methodology.md) | Mathematical formulation: $\mathcal{G}(t)$ and $P(F(t+K) \mid \mathcal{G}_{\le t})$. |
| | [`docs/simulator.md`](docs/simulator.md) | Multi-agent discrete-event simulation, agent roles, topologies, and tasks. |
| | [`docs/telemetry.md`](docs/telemetry.md) | Structured `AgentTelemetryEvent` schemas, collector, and credential sanitization. |
| | [`docs/fault_injection.md`](docs/fault_injection.md) | 3-level failure hierarchy and 12 canonical fault modes. |
| | [`docs/temporal_graph.md`](docs/temporal_graph.md) | Dynamic interaction graph snapshot construction and causal windowing. |
| **Data & Labeling** | [`docs/dataset.md`](docs/dataset.md) | Specifications for `agentguard_dataset_v1` and `agentguard_generalization_v1`. |
| | [`docs/labeling.md`](docs/labeling.md) | Causal labeling policy, forward horizons $K$, and post-cascade exclusion. |
| **Machine Learning** | [`docs/baselines.md`](docs/baselines.md) | Rule-Based, Classical ML, Sequence, and Static GNN baseline specifications. |
| | [`docs/temporal_gnn.md`](docs/temporal_gnn.md) | Continuous-Time Temporal GNN architecture, memory bank, and time encoding. |
| | [`docs/evaluation.md`](docs/evaluation.md) | Evaluation metrics, threshold freezing, lead time, and bootstrap testing. |
| | [`docs/ablation.md`](docs/ablation.md) | 9 systematic architectural and feature ablations vs. full reference model. |
| | [`docs/generalization.md`](docs/generalization.md) | 4 distribution shift dimensions (population scaling, topology, task, faults). |
| | [`docs/explainability.md`](docs/explainability.md) | 5 attribution levels, counterfactual perturbations, and case study archetypes. |
| **Serving & Applications**| [`docs/api.md`](docs/api.md) | REST API endpoints, Pydantic schemas, and query parameters. |
| | [`docs/dashboard.md`](docs/dashboard.md) | React 18 / TypeScript 5 interactive research dashboard and 10 views. |
| **Quality & Assurance** | [`docs/testing.md`](docs/testing.md) | Testing pyramid: 327 Python tests, 12 Vitest tests, and verified invariants. |
| | [`docs/security_audit.md`](docs/security_audit.md) | Bandit SAST scan, SQL injection immunity, and safe deserialization. |
| | [`docs/leakage_audit.md`](docs/leakage_audit.md) | Formal proofs and automated assertions for zero future leakage. |
| | [`docs/reproducibility_audit.md`](docs/reproducibility_audit.md) | Deterministic seeding, config hashing, and software provenance audit. |
| | [`docs/research_integrity_audit.md`](docs/research_integrity_audit.md) | Anti-cherry-picking assurance and epistemological classification. |
| | [`docs/reproducibility.md`](docs/reproducibility.md) | Step-by-step reproduction guide with verified repository commands. |
| | [`docs/reproducibility_checklist.md`](docs/reproducibility_checklist.md) | 13-point verified reproducibility checklist with empirical evidence. |
| | [`docs/limitations.md`](docs/limitations.md) | Empirical sample bounds ($N=35$), horizon $K=20$, and non-causal caveats. |
| | [`docs/troubleshooting.md`](docs/troubleshooting.md) | Operational diagnostic procedures for Python, PyTorch, database, and UI. |
| | [`docs/commands.md`](docs/commands.md) | Single-reference catalog of all verified operational commands. |
| | [`docs/research_artifacts.md`](docs/research_artifacts.md) | Master traceability index mapping RQs to datasets, models, plots, and reports. |

---

## 6. License and Citation

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

```bibtex
@article{agentguard2026,
  title={AgentGuard: Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems},
  author={AgentGuard Research Team},
  year={2026},
  url={https://github.com/Dhineshkumardhina/agent_guard}
}
```
