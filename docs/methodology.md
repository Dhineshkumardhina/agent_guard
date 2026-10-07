# Scientific Methodology

## Overview

This document specifies the research methodology of AgentGuard, including the multi-agent simulation framework, telemetry acquisition, fault injection protocol, mathematical formulation of dynamic interaction graphs, failure prediction formulation, benchmark model families, and the rigorous statistical evaluation protocol.

---

## 1. Multi-Agent Simulation Environment

To investigate cascading failure propagation under controlled conditions, AgentGuard implements a discrete-event multi-agent execution environment.

### 1.1 Agent Roles
The environment instantiates specialized agent roles with defined operational scopes:
* **Planner (`planner`):** Decomposes high-level objectives into subtasks, establishes dependency schedules, and coordinates delegations.
* **Researcher (`researcher`):** Gathers external context, executes simulated search operations, and synthesizes reference material.
* **Analyst (`analyst`):** Performs data transformations, evaluates statistical patterns, and generates intermediate analytical summaries.
* **Coder (`coder`):** Generates structured code blocks, formats programmatic logic, and executes mock execution routines.
* **Verifier (`verifier`):** Validates outputs produced by other agents against task constraints, schema expectations, and consistency criteria.
* **Critic (`critic`):** Evaluates reasoning quality and proposes revisions.
* **Decision (`decision`):** Synthesizes multi-agent deliberations and emits the terminal task output.

### 1.2 Communication Topologies
Agents communicate according to fixed or evolving network topologies:
1. **Pipeline (`pipeline`):** Sequential execution chain ($v_1 \to v_2 \to \dots \to v_N$) where error propagation is strictly unidirectional.
2. **Star (`star`):** Central coordinator (e.g., Planner) mediating all communication with leaf specialist agents ($v_{\text{hub}} \leftrightarrow v_i$).
3. **Mesh (`mesh`):** Fully connected or dense peer-to-peer communication where any agent can query any other agent.
4. **Custom (`custom`):** Irregular, clustered, or hierarchical topology with multiple delegation clusters and verifier feedback loops.

### 1.3 Benchmark Task Categories
Trajectories simulate four distinct collaborative workflow tasks:
1. **Research (`research`):** Multi-hop knowledge gathering, source synthesis, and executive briefing.
2. **Coding (`coding`):** Architecture design, code generation, mock unit testing, and verification.
3. **Data Analysis (`analysis` / `data_analysis`):** Metric calculation, trend identification, anomaly flagging, and visualization planning.
4. **Planning (`planning`):** Multi-stage project scheduling, resource allocation, and risk contingency formulation.

---

## 2. Telemetry and Fault Injection Protocol

### 2.1 Structured Telemetry Ingestion
At each discrete step $k$ in trajectory $r$, the environment emits an `AgentTelemetryEvent` record:
$$e_k = \left( u_k, v_k, t_k, \mathbf{x}_{e_k}, \mathbf{x}_{u_k}, \mathbf{x}_{v_k} \right)$$
capturing interaction metadata, continuous simulated timestamp $t_k$, message length, token consumption, response latency, self-reported confidence, output quality, contradiction score, tool invocation flags, error codes, and retry counts.

### 2.2 Controlled Fault Injection
Synthetic faults are injected according to a 12-mode failure taxonomy spanning three severity levels:
* **Level 1 (Agent-Level):** Local tool timeouts, unhandled tool exceptions, malformed output, local hallucinations.
* **Level 2 (Interaction-Level):** Pairwise contradiction spikes, rejected handoffs, delegation ping-pong, stale context reuse.
* **Level 3 (Cascading System-Level):** Multi-agent cascade where propagated upstream inconsistencies cause system deadlock, unrecoverable exception cycles, or corrupted terminal output.

---

## 3. Mathematical Formulation

### 3.1 Continuous-Time Dynamic Interaction Graph
A multi-agent execution session is formalized as a continuous-time dynamic interaction graph:
$$\mathcal{G}(t) = \left( \mathcal{V}(t), \mathcal{E}(t), \mathbf{X}_V(t), \mathbf{X}_E(t) \right)$$

where:
* $\mathcal{V}(t) = \{v_1, v_2, \dots, v_N\}$ is the set of agents active at or before timestamp $t$.
* $\mathcal{E}(t) = \{ e_k = (u_k, v_k, t_k) \mid t_k \le t \}$ is the ordered set of directed communication events from agent $u_k$ to agent $v_k$ occurring at continuous timestamp $t_k \le t$.
* $\mathbf{X}_V(t) \in \mathbb{R}^{|\mathcal{V}(t)| \times d_v}$ is the time-varying agent state feature matrix at timestamp $t$. For agent $v$, $\mathbf{x}_v(t)$ encodes rolling error rate, latency moving average, self-reported confidence, retry velocity, and active status.
* $\mathbf{X}_E(t) = \{ \mathbf{x}_{e_k} \in \mathbb{R}^{d_e} \mid t_k \le t \}$ is the set of edge attribute vectors for interaction events up to timestamp $t$.

### 3.2 Dynamic Graph History
The observable history up to timestamp $t$ is denoted:
$$\mathcal{G}(\le t) = \left\{ \left( u_k, v_k, t_k, \mathbf{x}_{e_k}, \mathbf{X}_V(t_k) \right) \mid t_k \le t \right\}$$

### 3.3 Failure Prediction Formulation
Let $F(t + k) \in \{0, 1\}$ denote the binary occurrence of a Level 3 cascading failure within the forward horizon of $k$ interaction steps into the future, i.e., at any step $\tau$ such that $t < \tau \le t + k$.

The failure prediction objective is to estimate the conditional probability of impending cascading failure given the observed temporal interaction graph history:
$$P\left( F(t + k) = 1 \mid \mathcal{G}(\le t) \right)$$

where $k \in \{1, 3, 5, 10, 20\}$ denotes the prediction horizon in interaction steps.

### 3.4 Strict Temporal Isolation Invariant
To prevent future data leakage:
1. Feature extraction at evaluation timestamp $t$ strictly conditions on events occurring at or before $t$:
   $$\forall e_i \in \mathcal{G}(\le t), \quad t_i \le t$$
2. Target label $Y_{t, k}$ is computed over the strictly forward interval $(t, t+k]$:
   $$Y_{t, k} = \mathbb{I}\left( \exists \tau \in (t, t + k] : \text{FailureLevel}(\tau) = 3 \right)$$
3. Samples where a Level 3 cascade has already occurred at or prior to $t$ ($\tau \le t$) are excluded from early warning training datasets, ensuring models learn *pre-failure early warning signals* rather than post-mortem crash detection.

---

## 4. Evaluated Model Families

AgentGuard evaluates 9 models spanning 5 architectural paradigm families:

1. **Rule-Based Baseline:**
   - Heuristic thresholding on rolling retry count, latency spikes, and contradiction scores.
   - Serves as the operational baseline used in conventional heuristic alerting.
2. **Classical Machine Learning:**
   - **Logistic Regression (L2-regularized):** Linear baseline on aggregated agent features.
   - **Random Forest:** Ensemble of decision trees capturing non-linear feature interactions.
   - **XGBoost:** Gradient-boosted decision trees optimizing binary cross-entropy.
3. **Temporal Sequence Models:**
   - **LSTM (Long Short-Term Memory):** Recurrent neural network processing sequential event vectors.
   - **GRU (Gated Recurrent Unit):** Lightweight gated recurrence capturing temporal event transitions.
4. **Static Graph Neural Networks:**
   - **GCN (Graph Convolutional Network):** Multi-hop neighborhood aggregation over static communication adjacency.
   - **GAT (Graph Attention Network):** Anisotropic attention weights over static inter-agent communication channels.
5. **Continuous-Time Temporal GNN (Core Model):**
   - **Temporal GNN:** Dynamic graph architecture combining continuous Fourier time encoding $\phi(\Delta t)$, persistent per-agent recurrent memory bank ($\mathbf{m}_v$), temporal message-passing aggregation, graph readout, and an MLP hazard prediction head.

---

## 5. Statistical Evaluation Protocol

### 5.1 Standardized Population Alignment
All models are evaluated on strictly identical test sample partitions for each horizon $k$. No model is permitted different test instances.

### 5.2 Frozen Validation Thresholding
To prevent test-set optimization bias:
- Operating decision thresholds $\theta^* \in [0.10, 0.90]$ are chosen exclusively on the validation split by maximizing the validation F1 score.
- Thresholds are frozen prior to test inference. The test set is evaluated once with the frozen threshold $\theta^*$.

### 5.3 Trajectory Block Bootstrap
Because interaction events within the same simulation run are autocorrelated, standard independent-and-identically-distributed (i.i.d.) sample bootstrapping yields pseudo-replication and underestimated confidence intervals.

AgentGuard executes **trajectory-level block bootstrap resampling**:
- The resampling unit is the entire simulation trajectory (`run_id`).
- For $B = 500$ iterations, $N_{\text{runs}}$ trajectories are sampled with replacement.
- Metrics are recomputed on the aggregated samples of the bootstrapped runs.
- 95% empirical bootstrap confidence intervals and paired permutation p-values are derived from this distribution.

### 5.4 Lead Time Evaluation
For each trajectory with a cascading failure at step $t_{\text{fail}}$:
- A candidate warning emitted at step $t_{\text{warn}}$ is valid if and only if $t_{\text{warn}} \le t_{\text{fail}}$.
- Lead time is defined as:
  $$\Delta t_{\text{lead}} = t_{\text{fail}} - t_{\text{warn}} \quad (\ge 0)$$
- Post-failure predictions ($t_{\text{warn}} > t_{\text{fail}}$) are explicitly disqualified from lead time credit.
