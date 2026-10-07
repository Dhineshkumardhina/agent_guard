# Multi-Agent Simulator Documentation

## Overview

The AgentGuard Multi-Agent Simulator (`ml/simulation/`) is a controlled, reproducible discrete-event simulation environment designed to generate realistic collaborative multi-agent interaction trajectories, inject controlled faults, and capture high-fidelity telemetry.

---

## 1. Simulator Architecture

The simulation engine is structured into five core submodules:

```
ml/simulation/
├── agents/         # Base agent abstractions and role implementations
├── environment/    # Discrete-event runtime environment and state management
├── topologies/     # Network communication topologies (Pipeline, Star, Mesh, Custom)
├── tasks/          # Domain workflow tasks (Research, Coding, Analysis, Planning)
├── fault_injection/# Injector engine and cascade propagation logic
├── interfaces.py   # Protocol interfaces and abstract base classes
├── events.py       # Simulation event definitions
├── run.py          # Trajectory execution runner and orchestrator
└── db.py           # Relational persistence into SQLite / PostgreSQL
```

---

## 2. Agent Roles and Behavioral Capabilities

Defined in `ml/simulation/agents/roles.py` and `base.py`, agents are stateful entities possessing individual memory, capability sets, and communication behaviors:

| Role Name | Identifier | Primary Responsibilities | Common Interaction Channels |
|---|---|---|---|
| **Planner** | `planner` | High-level goal decomposition, task delegation, dependency scheduling. | Emits tasks to Researcher, Coder, Analyst. |
| **Researcher** | `researcher` | Context discovery, simulated knowledge retrieval, document summarization. | Receives tasks from Planner; hands off to Analyst/Coder. |
| **Analyst** | `analyst` | Data transformation, metric aggregation, anomaly assessment. | Queries Researcher; outputs to Verifier and Critic. |
| **Coder** | `coder` | Logic synthesis, programmatic script generation, mock unit testing. | Receives specifications from Planner; passes code to Verifier. |
| **Verifier** | `verifier` | Constraint verification, fact-checking, schema validation, quality scoring. | Intercepts outputs; returns approval or rejection to sender. |
| **Critic** | `critic` | Peer review, reasoning flaw detection, qualitative critique. | Deliberates with Analyst and Coder. |
| **Decision** | `decision` | Terminal synthesis, voting resolution, final output formatting. | Receives verified outputs from all agents; terminates session. |

Each agent maintains internal telemetry accumulators:
- `total_events`: Count of processed interactions.
- `error_count`: Number of encountered exceptions.
- `retry_count`: Cumulative retry attempts.
- `rolling_latency`: Exponential moving average of response latency.
- `confidence`: Self-reported confidence score $\in [0.0, 1.0]$.
- `status`: Operational state (`active`, `degraded`, `failed`).

---

## 3. Communication Topologies

Implemented in `ml/simulation/topologies/`, topologies enforce message routing rules between agents:

### 3.1 Pipeline Topology (`pipeline.py`)
- **Structure:** Linear directed graph $v_1 \to v_2 \to \dots \to v_N$.
- **Behavior:** Each agent processes input exclusively from its immediate predecessor and delegates to its immediate successor.
- **Cascade Characteristics:** Unidirectional error propagation. If an upstream agent fails, all downstream stages are starved or corrupted.

### 3.2 Star Topology (`star.py`)
- **Structure:** Central hub agent ($v_0$, typically the Planner) connected bidirectionally to $N-1$ peripheral workers ($v_i$).
- **Behavior:** Peripheral agents communicate exclusively through the central hub.
- **Cascade Characteristics:** Central hub is a single point of failure and communication bottleneck. Hub corruption instantly propagates to all worker agents.

### 3.3 Mesh Topology (`mesh.py`)
- **Structure:** Complete or dense directed graph where $\forall u, v \in \mathcal{V}, (u, v) \in \mathcal{E}$.
- **Behavior:** Any agent can query or delegate to any other agent based on task requirements.
- **Cascade Characteristics:** Complex, multi-path propagation with high susceptibility to communication loops and conflicting cross-agent updates.

### 3.4 Custom Topology (`custom.py`)
- **Structure:** Clustered hierarchical graph with dedicated verification loops (e.g., Planner $\to$ {Researcher, Coder} $\to$ Analyst $\leftrightarrow$ Verifier $\to$ Decision).
- **Behavior:** Models realistic enterprise multi-agent workflows with division of labor and specialized quality control gates.

---

## 4. Benchmark Task Categories

Implemented in `ml/simulation/tasks/`, tasks define multi-step execution graphs:

1. **Research Task (`research.py`):** Multi-agent investigation requiring literature retrieval, claim extraction, and executive briefing synthesis.
2. **Coding Task (`coding.py`):** Specification analysis, code scaffold generation, automated test generation, and syntax verification.
3. **Data Analysis Task (`analysis.py`):** Tabular data parsing, outlier detection, statistical summary generation, and conclusion verification.
4. **Planning Task (`planning.py`):** Multi-stage project decomposition, milestone assignment, contingency mapping, and dependency checking.

---

## 5. Execution Lifecycle and Persistence

The execution orchestrator (`ml/simulation/run.py`) executes trajectories through standardized lifecycle phases:

```
Initialize Seed & Config
       │
       ▼
Instantiate Agents & Topology
       │
       ▼
Loop Step k = 1 to MaxSteps:
  ├── Check Scheduled Fault Injections
  ├── Apply Fault Perturbations (if triggered)
  ├── Route Message: Agent u ──> Agent v
  ├── Agent v Processes Input & Updates Telemetry
  ├── Emit Structured AgentTelemetryEvent
  ├── Evaluate Interaction Failure Level (L1, L2, L3)
  └── Terminate early if Level 3 Cascade or Terminal Goal achieved
       │
       ▼
Persist Trajectory, Events, and Failures to Database (SQLite / Postgres)
```

### Deterministic Seeding
Every simulation run takes an explicit `random_seed`. Pseudo-random operations (agent response variance, tool latency fluctuations, fault selection) are seeded to guarantee bit-for-bit trajectory reproducibility across repeated executions.
