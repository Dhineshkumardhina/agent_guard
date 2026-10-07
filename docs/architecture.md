# System Architecture

## Overview

AgentGuard is an end-to-end research platform for detecting, predicting, and explaining cascading failures in multi-agent AI systems. The architecture implements a strictly modular 13-stage pipeline spanning simulation, telemetry capture, temporal graph representation learning, model evaluation, explainability, REST API serving, and an interactive research dashboard.

---

## 1. End-to-End System Pipeline

```
  +─────────────────────────────────────────────────────────────+
  | 1. Multi-Agent Simulator (ml/simulation/)                   |
  |    - Configurable agent roles: Planner, Researcher, Analyst,|
  |      Coder, Verifier, Critic, Decision                      |
  |    - Topologies: Pipeline, Star, Mesh, Custom               |
  |    - Benchmark tasks: Research, Coding, Analysis, Planning  |
  +──────────────────────────────┬──────────────────────────────+
                                 │
                                 ▼
  +─────────────────────────────────────────────────────────────+
  | 2. Fault Injection Engine (ml/simulation/fault_injection/)  |
  |    - 12 canonical fault modes (L1/L2/L3)                    |
  |    - Controlled injection step, target agent, severity      |
  +──────────────────────────────┬──────────────────────────────+
                                 │
                                 ▼
  +─────────────────────────────────────────────────────────────+
  | 3. Telemetry Event Collector (ml/telemetry/)                |
  |    - Structured AgentTelemetryEvent JSON/Pydantic schemas    |
  |    - Credential sanitization, token tracking, latency logs  |
  +──────────────────────────────┬──────────────────────────────+
                                 │
                                 ▼
  +─────────────────────────────────────────────────────────────+
  | 4. Temporal Graph Construction (ml/graph/)                  |
  |    - Dynamic interaction stream G(t) = (V(t), E(t), X_V, X_E)|
  |    - Time-varying node & directed edge feature computation  |
  |    - Temporal snapshot sliding windows & motif metrics      |
  +──────────────────────────────┬──────────────────────────────+
                                 │
                                 ▼
  +─────────────────────────────────────────────────────────────+
  | 5. Dataset Generation & Storage (ml/data/)                  |
  |    - Prediction samples across horizons K ∈ {1, 3, 5, 10, 20}|
  |    - Run-level stratified train/val/test partitioning       |
  |    - Parquet tabular partitions & JSONL graph sequences     |
  +──────────────────────────────┬──────────────────────────────+
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
  +───────────────────────────────+   +───────────────────────────────+
  | 6. Baseline Models            |   | 7. Temporal GNN Model         |
  |    (ml/baselines/)            |   |    (ml/baselines/temporal_gnn)|
  |    - Rule-Based Thresholds    |   |    - Continuous-Time Dynamic  |
  |    - Logistic Regression      |   |      Graph Neural Network     |
  |    - Random Forest, XGBoost   |   |    - Fourier Time Encoding    |
  |    - Temporal Sequence        |   |    - Persistent Node Memory   |
  |      (LSTM, GRU)              |   |    - Temporal Graph Readout   |
  |    - Static GNN (GCN, GAT)    |   |    - Binary Hazard Head       |
  +───────────────┬───────────────+   +───────────────┬───────────────+
                  └───────────────┬───────────────────┘
                                  │
                                  ▼
  +─────────────────────────────────────────────────────────────+
  | 8. Comprehensive Evaluation Engine (ml/evaluation/)         |
  |    - Metrics: AUROC, AUPRC, F1, Precision, Recall, Brier,ECE|
  |    - Lead time calculation (earliest warning t_warn <= t_fail)|
  |    - Trajectory block bootstrap (B=500, 95% CI)             |
  |    - Calibration curves, ROC/PR curves                      |
  +──────────────────────────────┬──────────────────────────────+
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
  +───────────────+       +───────────────+       +───────────────+
  | 9. Ablation   |       | 10. General-  |       | 11. Explain-  |
  |    Study      |       |     ization   |       |     ability   |
  | (ml/ablation/)|       | (ml/general-  |       | (ml/explain-  |
  | 9 systematic  |       |   ization/)   |       |   ability/)   |
  | architectural |       | 4 OOD shifts: |       | 5 attribution |
  | & feature     |       | N-scaling,    |       | levels, case  |
  | ablations     |       | topology, task|       | studies, risk |
  | vs full TGN   |       | unseen faults |       | trajectories  |
  +───────┬───────+       +───────┬───────+       +───────┬───────+
          └───────────────────────┼───────────────────────┘
                                  │
                                  ▼
  +─────────────────────────────────────────────────────────────+
  | 12. FastAPI Application Backend (backend/app/)              |
  |     - Endpoints: /health, /runs, /agents, /events, /graph,  |
  |       /predictions, /evaluations, /ablations, /explanations |
  |     - SQLAlchemy 2.0 ORM + SQLite / PostgreSQL database     |
  |     - Sub-20ms response latencies, masked error responses   |
  +──────────────────────────────┬──────────────────────────────+
                                 │ REST API (JSON)
                                 ▼
  +─────────────────────────────────────────────────────────────+
  | 13. Research Dashboard (frontend/src/)                      |
  |     - React 18, TypeScript 5, Vite                          |
  |     - 10 interactive views: Overview, Runs, Model Comparison|
  |       Early Warning, Ablations, Generalization, Explainability|
  |     - Interactive graph canvas, timeline, case study viewer |
  +─────────────────────────────────────────────────────────────+
```

---

## 2. Component Responsibilities

### 1. Multi-Agent Simulator (`ml/simulation/`)
- Manages discrete-event simulation sessions across four standard topologies (`pipeline`, `star`, `mesh`, `custom`).
- Instantiates agents possessing specific operational roles (`planner`, `researcher`, `analyst`, `coder`, `verifier`, `critic`, `decision`).
- Executes structured tasks (`research`, `coding`, `analysis`, `planning`) with deterministic pseudo-random seeds.

### 2. Fault Injection Engine (`ml/simulation/fault_injection/`)
- Introduces controlled faults from a 12-mode taxonomy across three failure severity levels (Level 1: Agent, Level 2: Interaction, Level 3: Cascading).
- Records deterministic injection metadata (`injection_step`, `target_agent`, `fault_type`, `severity`, `parameters`).
- Simulates realistic error propagation chains across message-passing channels.

### 3. Telemetry Event Collector (`ml/telemetry/`)
- Intercepts inter-agent communication, emitting structured `AgentTelemetryEvent` records.
- Automatically captures continuous timestamps, step indices, message lengths, token consumption, execution latencies, self-reported confidence, output quality, contradiction scores, tool usage, and retry counts.
- Sanitizes sensitive tokens and credential strings at ingestion.

### 4. Temporal Graph Builder (`ml/graph/`)
- Transforms linear telemetry event logs into continuous-time dynamic interaction graphs $\mathcal{G}(t) = (\mathcal{V}(t), \mathcal{E}(t), \mathbf{X}_V(t), \mathbf{X}_E(t))$.
- Dynamically updates agent state features $\mathbf{X}_V(t)$ (rolling error rates, latency moving averages, active states).
- Computes directed edge features $\mathbf{X}_E(t)$ (message tokens, contradiction scores, tool flags).
- Enforces strict causal event boundaries ($t_{\text{event}} \le t_{\text{eval}}$).

### 5. Dataset Generation & Storage (`ml/data/`)
- Extracts predictive observation windows across five prediction horizons $K \in \{1, 3, 5, 10, 20\}$ interaction steps.
- Partitions datasets at the run level (`run_id`) to prevent temporal autocorrelation leakage between splits.
- Persists tabular samples as Apache Parquet (`all_samples.parquet`, `train.parquet`, `val.parquet`, `test.parquet`) and graph histories as JSON Lines (`graph_sequences_*.jsonl`).
- Generates cryptographic `manifest.json` metadata defining class ratios and feature schemas.

### 6. Baseline Models (`ml/baselines/`)
- Implements 8 reference models across 4 paradigm families:
  - **Rule-Based**: Heuristic threshold detector on rolling retries, latency, and contradiction.
  - **Classical ML**: Logistic Regression, Random Forest, XGBoost trained on aggregated agent features.
  - **Temporal Sequence**: Recurrent neural networks (LSTM, GRU) processing temporal event series.
  - **Static GNN**: Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) operating on collapsed adjacency matrices.

### 7. Continuous-Time Temporal GNN (`ml/baselines/temporal_gnn/`)
- Implements dynamic graph neural network architecture with:
  - **Fourier Time Encoding**: Maps continuous inter-event intervals $\Delta t$ into dense harmonic embeddings.
  - **Persistent Node Memory Bank**: Updates per-agent GRU memory states upon each message exchange.
  - **Temporal Message Function & Aggregation**: Combines sender state, receiver state, edge attributes, and time encoding.
  - **Temporal Graph Readout & Hazard Head**: Pools dynamic agent embeddings into a graph-level hazard probability.

### 8. Comprehensive Evaluation Engine (`ml/evaluation/`)
- Evaluates models across classification discrimination (AUROC, AUPRC, F1, Precision, Recall, Specificity, Brier score, Expected Calibration Error).
- Computes incident-level early warning metrics (mean and median lead time, false alarm rate per trajectory).
- Executes trajectory-level block bootstrap resampling ($B=500$) to yield non-parametric 95% confidence intervals and paired permutation p-values.

### 9. Ablation Study Runner (`ml/ablation/`)
- Executes 9 controlled ablations removing continuous time encoding, graph topology, node features, edge attributes, temporal memory, interaction frequency, contradiction indicators, confidence scores, and historical failure counts.
- Verifies architectural and information source contributions against the full reference model.

### 10. Generalization Benchmark Runner (`ml/generalization/`)
- Evaluates model transferability across 4 distribution shift dimensions:
  - Population scaling ($N \in \{3, 5\} \to \{8, 12\}$).
  - Communication topology shifts (Leave-One-Topology-Out).
  - Task domain transfer (e.g., Planning $\to$ Data Analysis).
  - Unseen, held-out failure modes.

### 11. Explainability Framework (`ml/explainability/`)
- Decomposes failure predictions into 5 analytical levels:
  - Level 1: Global feature importance (Random Forest Gini, XGBoost gain).
  - Level 2: Graph feature attribution (node vs. edge impact).
  - Level 3: Agent attribution (per-agent risk score contribution).
  - Level 4: Communication channel attribution (directed pair $u \to v$).
  - Level 5: Temporal event attribution (chronological interaction events).
- Provides counterfactual sensitivity perturbations and risk trajectories over time.

### 12. REST API Backend (`backend/app/`)
- FastAPI application exposing modular endpoints under `/api/v1/`:
  - Simulation runs (`/runs`, `/runs/{id}`)
  - Agents and telemetry events (`/agents`, `/events`, `/runs/{id}/events`)
  - Dynamic graph snapshots (`/runs/{id}/graph`)
  - Inference predictions (`/predictions`)
  - Evaluation summaries (`/evaluations`, `/evaluations/comparison`)
  - Ablation metrics (`/ablations`)
  - Generalization benchmarks (`/generalization`)
  - Multi-level explanations (`/explanations`)
- Backed by SQLAlchemy 2.0 ORM with SQLite (development) and PostgreSQL (production) compatibility.
- Enforces strict input validation, pagination limits ($\le 200$), and sanitized 500 error responses.

### 13. Interactive Research Dashboard (`frontend/src/`)
- Single Page Application (SPA) built with React 18, TypeScript 5, and Vite.
- Includes 10 dedicated views: Overview, Runs List, Run Detail, Experiment Detail, Model Comparison, Early Warning, Ablations, Generalization, Explainability, and Case Study Detail.
- Renders SVG dynamic interaction graphs, interactive timelines, calibration curves, and counterfactual sensitivity comparisons.

---

## 3. Data Flow Architecture

The runtime data flow proceeds through four primary stages:

```
[Simulation / Generation]
  Simulation Environment ──(raw events)──> Telemetry Collector ──(validated events)──> SQLite/Postgres DB
                                                 │
                                                 ▼
[Graph & Dataset Pipeline]                Temporal Graph Builder
                                                 │
                                                 ▼
                                     Feature Windows & Labeling
                                                 │
                                                 ▼
                                     Dataset Storage (Parquet / JSONL)
                                                 │
                                                 ▼
[Model Training & Evaluation]        Model Trainers (Rule, ML, RNN, GNN, TGN)
                                                 │
                                                 ▼
                                     Inference Predictions (Parquet / JSON)
                                                 │
                                                 ▼
                                     Evaluation Engine / Ablation / Generalization
                                                 │
                                                 ▼
[Serving & Presentation]             Results Storage (results/*/*.json, plots, tables)
                                                 │
                                                 ▼
                                     Database Seeder ──> Relational DB Tables
                                                 │
                                                 ▼
                                     FastAPI Backend ──(REST JSON)──> React Dashboard
```

---

## 4. Storage Flow and Schema Mapping

AgentGuard separates transient research artifacts from relational application data:

1. **Relational Database (`agentguard.db` / PostgreSQL):**
   - Tables: `users`, `experiments`, `datasets`, `runs`, `agents`, `events`, `agent_interactions`, `fault_injections`, `failures`, `predictions`, `model_results`.
   - Optimized with compound indexes (`idx_events_run_step`, `idx_events_src_tgt`, `idx_predictions_run_step_model`).
2. **File-Based Research Storage (`data/` and `results/`):**
   - `data/processed/{version}/`: Immutable parquet feature tables and JSONL graph histories.
   - `results/baselines/`: Trained model weights (`.pt`, `.json`, `.pkl`), per-step predictions (`predictions.parquet`, `predictions.json`), and metrics (`metrics.json`).
   - `results/evaluation/`: Comparative metric tables, calibration curves, and markdown research reports.
   - `results/ablation/`: Ablation comparison matrices and delta-F1 tables.
   - `results/generalization/`: In-distribution vs. out-of-distribution gap analyses and plots.
   - `results/explainability/`: Case study profiles, feature importance rankings, and counterfactual sensitivity deltas.
