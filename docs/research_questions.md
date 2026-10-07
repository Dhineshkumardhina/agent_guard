# Research Questions (RQ1 – RQ6)

## Overview

AgentGuard investigates whether temporal graph representations of inter-agent communication provide predictive information about impending cascading failures in multi-agent AI systems, compared with isolated agent behavioral features and static graph representations.

The investigation is governed by six formal Research Questions (RQs).

---

### RQ1: Predictive Utility of Agent-Level Behavioral Features
> **Can isolated agent-level behavioral telemetry features (e.g., error rates, latency variance, token counts, retry counts, self-reported confidence) predict impending failures in multi-agent systems?**

* **Motivation:** Before invoking complex relational or temporal graph architectures, we must establish what baseline predictive capacity is provided by standard single-agent operational telemetry alone.
* **Evaluated Paradigms:** Rule-Based Thresholds, Tabular Ensembles (Logistic Regression, Random Forest, XGBoost).
* **Target Outcome:** Quantify the extent to which local agent symptoms signal impending failure without graph topological context.

---

### RQ2: Value of Relational Graph Topology
> **Does inter-agent communication graph topology provide additional predictive information for impending failures beyond isolated agent-level features?**

* **Motivation:** Multi-agent workflows depend on directed communication pathways. An agent experiencing latency may only trigger failure if downstream dependent agents rely on its output.
* **Evaluated Paradigms:** Static Graph Neural Networks (GCN, GAT) vs. Tabular Ensembles (XGBoost, Random Forest).
* **Target Outcome:** Measure whether incorporating static interaction topology (who talks to whom) improves discrimination (AUROC, AUPRC, F1) across localized vs. distributed failure modes.

---

### RQ3: Contribution of Continuous-Time Temporal Dynamics
> **Does continuous-time temporal interaction information (event arrival order, message inter-arrival times, burstiness) improve failure prediction over static graph and sequence representations?**

* **Motivation:** Aggregating communication into a static adjacency matrix discards the sequence, velocity, and timing of interactions. Standard sequence models (LSTM, GRU) capture order but lack relational inductive bias.
* **Evaluated Paradigms:** Continuous-Time Temporal Graph Neural Network (Temporal GNN with Fourier time encoding and persistent per-agent memory) vs. Static GNNs (GCN, GAT) and Sequence Models (LSTM, GRU).
* **Target Outcome:** Determine whether fine-grained temporal message timestamps $\phi(\Delta t)$ and evolving dynamic node memories yield superior predictive calibration and classification performance.

---

### RQ4: Early Warning Lead Time for Cascading Failures
> **Can temporal interaction patterns provide earlier warnings (higher lead time in interaction steps or simulated seconds) before cascading system-level failures manifest?**

* **Motivation:** A warning emitted at the exact moment of failure is operationally useless for mitigation. An effective guard system must alert while the cascade is still propagating across intermediate agent hops.
* **Evaluated Paradigms:** Earliest valid warning time ($t_{\text{warn}} \le t_{\text{fail}}$) evaluated across horizons $K \in \{1, 3, 5, 10, 20\}$ interaction steps ahead.
* **Target Outcome:** Quantify mean and median early warning lead time, false alarm rates, and detection rates at varying prediction horizons across model families.

---

### RQ5: Generalization Under Distribution Shifts
> **How robust are the learned failure prediction models under out-of-distribution (OOD) shifts in agent population size, communication topology, task domain, and unseen failure types?**

* **Motivation:** Production multi-agent environments dynamically add agents, modify collaboration topologies, and encounter unexpected failure modes. A research model that only functions on its training configuration is practically brittle.
* **Evaluated Dimensions:**
  1. **Population Scaling:** Train on $N \in \{3, 5\}$ agents $\to$ Evaluate on $N \in \{8, 12\}$.
  2. **Topology Shift:** Train on Star/Mesh $\to$ Evaluate on Custom/Pipeline (Leave-One-Topology-Out).
  3. **Task Domain Shift:** Train on Research/Coding/Planning $\to$ Evaluate on Data Analysis.
  4. **Unseen Failure Modes:** Train on seen faults $\to$ Evaluate on held-out unseen fault types.
* **Target Outcome:** Quantify the generalization gap ($\Delta \text{F1} = \text{F1}_{\text{ID}} - \text{F1}_{\text{OOD}}$) and identify failure boundaries of inductive biases.

---

### RQ6: Component Attribution and Information Decomposition
> **Which behavioral signals, relational components, and architectural mechanisms contribute most to failure prediction accuracy?**

* **Motivation:** To prevent ungrounded architectural complexity, each component of the proposed framework must be empirically justified through controlled ablations and attribution analysis.
* **Evaluated Dimensions:**
  1. **Architectural Ablations:** Systematic removal of continuous time encoding, graph topology, node features, edge attributes, and recurrent temporal memory.
  2. **Feature Masking:** Removal of interaction frequency, contradiction indicators, confidence scores, and historical error counts.
  3. **Multi-Level Explainability:** Global feature importance, agent attribution, directed edge channel attribution, and counterfactual sensitivity perturbations.
* **Target Outcome:** Isolate the precise marginal contribution of each architectural and behavioral component.
