# Hypotheses (H0 – H3)

## Overview

The empirical investigation in AgentGuard is guided by formal hypotheses comparing behavioral, relational, and temporal information sources for impending failure prediction in multi-agent systems.

These hypotheses are evaluated empirically using controlled benchmarks, standardized test partitions, frozen validation thresholds, and trajectory-level block bootstrap statistical testing. They are formulated below as scientific propositions undergoing rigorous experimental test, rather than preordained conclusions.

---

### Hypothesis 0: Null Hypothesis (Sufficiency of Agent-Level Telemetry)
> **$H_0$:** Agent-level behavioral features (e.g., local execution latency, token counts, retry velocity, and confidence indicators) are sufficient for predicting impending multi-agent failures; adding relational graph topology or continuous temporal interaction dynamics yields no statistically significant improvement in failure discrimination or lead time.

* **Rationale for Testing:** Standard production monitoring typically relies on agent-level APM metrics (e.g., individual tool timeouts, memory consumption). If $H_0$ holds, the operational complexity and computational overhead of dynamic graph models cannot be justified.
* **Test Criteria:** Comparison of Tabular Ensembles (Logistic Regression, Random Forest, XGBoost) against Graph-based models (GCN, GAT, Temporal GNN) across standardized AUROC, AUPRC, F1, and lead time metrics on identical test instances.

---

### Hypothesis 1: Additional Predictive Value of Temporal Interaction Information
> **$H_1$:** Temporal interaction information (the chronological sequence, inter-arrival intervals, and directional propagation of inter-agent messages) provides additional predictive value for impending failures beyond isolated agent-level behavioral features.

* **Rationale for Testing:** Multi-agent systems execute interdependent reasoning chains. Failures frequently originate as subtle semantic divergences or latency shifts in upstream agents that cascade across several communication steps before manifesting as a system breakdown.
* **Test Criteria:** Evaluated via performance comparison between temporal sequence/graph models and tabular single-agent models, as well as controlled feature ablations removing temporal and communication dynamics.

---

### Hypothesis 2: Early-Warning Lead Time Advantage of Temporal Graph Models
> **$H_2$:** Continuous-time temporal graph models can improve early-warning performance (higher warning lead time before failure onset and fewer premature false alarms) relative to static graph representations that aggregate interaction histories into static adjacency matrices.

* **Rationale for Testing:** Static graph aggregations collapse the temporal dimension, treating old communication edges equally with recent high-frequency bursts. A continuous-time formulation preserves event ordering, message inter-arrival times, and dynamic memory states, potentially capturing cascade signatures earlier in the execution trajectory.
* **Test Criteria:** Evaluation of incident-level warning lead time ($t_{\text{warn}} \le t_{\text{fail}}$) across prediction horizons $K \in \{1, 3, 5, 10, 20\}$ interaction steps ahead, alongside false alarm rates per trajectory.

---

### Hypothesis 3: Association of Evolving Graph Motifs with Failure Risk
> **$H_3$:** Specific evolving interaction patterns—such as communication ping-pong loops, sudden drops in inter-agent reciprocity, abnormal delegation bottlenecks, and spikes in semantic contradiction scores—are statistically associated with elevated risk of system-level cascading failure.

* **Rationale for Testing:** Human distributed systems (e.g., microservices, organizations) exhibit characteristic communication bottlenecks and infinite retry cycles preceding systemic collapse. Identifying whether multi-agent AI systems exhibit similar structural failure signatures provides actionable diagnostic primitives.
* **Test Criteria:** Measured through multi-level explainability analysis (Level 3 agent attribution, Level 4 communication channel attribution, Level 5 event attribution), graph motif feature ablations, and counterfactual sensitivity perturbations.

---

### Methodological Protocol for Hypothesis Evaluation

1. **Strict Causal Event Boundaries:** Telemetry is bounded by the observation horizon $t \le t_{\text{eval}}$. Under no circumstances are future events leaked to models.
2. **Standardized Test Population:** All model families are evaluated on identical test sample instances.
3. **Frozen Validation Thresholding:** Decision thresholds $\theta^* \in [0.10, 0.90]$ are calibrated solely on validation trajectories and frozen prior to test inference.
4. **Trajectory-Level Resampling:** Statistical confidence intervals (95% CI) and p-values are computed via trajectory block bootstrap ($B=500$) resampling at the simulation run level to respect autocorrelation and prevent pseudo-replication.
5. **Scientific Neutrality:** Results are reported with measured confidence intervals and statistical significance tests without claiming universal superiority where evidence is inconclusive.
