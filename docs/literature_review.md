# Literature Review and Research Positioning

## 1. Research Context and Grounding
AgentGuard positions its inquiry at the intersection of several established fields. We do **not** claim to invent multi-agent systems, temporal graph neural networks, or failure auditing. Rather, we empirically evaluate whether continuous-time temporal graph representations offer statistically measurable predictive advantages over agent-isolated baselines for anticipating cascading failures.

---

## 2. Related Research Areas

### 2.1 Multi-Agent Failure Taxonomies & Attribution
Recent works have characterized failure modes in LLM-based autonomous agent societies:
* **Failure Modes in Multi-Agent Interaction:** Cascading misunderstandings, goal drift, communication overhead, and deadlocks in collaborative problem-solving.
* **Causal Attribution:** Tracing system-level failure back to upstream prompts, tool errors, or reasoning deficits.
* **AgentGuard Differentiation:** Rather than performing post-hoc post-mortem attribution after an execution finishes, AgentGuard focuses on **online, ahead-of-time prediction** of cascading failures while the interaction unfolds.

### 2.2 Agent Observability & Auditing
* **Telemetry Tracing:** Standard distributed tracing frameworks (OpenTelemetry) adapted for agentic loops (recording LLM calls, tool latencies, token consumption).
* **Trajectory-level Auditing:** Evaluating step-by-step reasoning steps for validity.
* **AgentGuard Differentiation:** Treats the communication fabric itself as a first-class dynamic graph, extracting structural and topological dynamics rather than isolated trace spans.

### 2.3 Temporal Graph Neural Networks (TGNNs / TGNs)
* **Foundations:** Rossi et al. (Temporal Graph Networks - TGN), Kumar et al. (JODIE), Xu et al. (TGAT).
* **Applications:** Dynamic link prediction, continuous-time node classification, social network anomaly detection.
* **AgentGuard Differentiation:** Applies temporal graph formulations to the domain of multi-agent communication networks where nodes represent heterogeneous AI agents and edges represent asynchronous, semantic message flows.

### 2.4 Graph Anomaly Detection & Early Warning Systems
* **Static Graph Anomaly Detection:** Outlier node and edge identification using GCNs/GATs.
* **Dynamic Early Warning Systems:** Anticipating tipping points, systemic risk, and contagion in financial and ecological networks.
* **AgentGuard Differentiation:** Evaluates early warning lead times in discrete interaction steps, assessing whether network topology informs cascading propagation velocity.

---

## 3. Position and Hypotheses

| ID | Hypothesis Statement | Rejection Criteria |
|---|---|---|
| **H0 (Null)** | Agent-level behavioral features (latency, retries, confidence) are sufficient to predict cascading failures. | Temporal graph models achieve statistically significant gains in AUROC/AUPRC and Lead Time ($p < 0.05$). |
| **H1** | Inter-agent interaction graph structure provides predictive information beyond agent-level behavioral features. | GNN models outperform tabular agent-level models (XGBoost, Random Forest). |
| **H2** | Continuous temporal graph models provide earlier warnings (higher lead time) than static graph models. | Temporal GNN achieves higher mean lead time prior to failure than static GAT/GCN. |
| **H3** | Distinct evolving interaction motifs (e.g., reciprocity collapse, communication bursts) precede cascading failures. | Ablation of dynamic edge features significantly degrades prediction performance. |
