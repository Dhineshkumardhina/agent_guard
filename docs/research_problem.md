# Research Problem Formulation

## Title
**AgentGuard: Temporal Graph-Based Detection and Prediction of Cascading Failures in Multi-Agent AI Systems**

---

## 1. Executive Summary
Multi-agent AI systems (e.g., collaborative ensembles of LLMs operating as planners, coders, researchers, and verifiers) are susceptible to cascading failures. An isolated error in an early agent (such as a subtle hallucination, an unhandled tool exception, or a contradictory assertion) often propagates through intermediate reasoning loops and delegation chains, culminating in a catastrophic system-level failure.

Conventional failure auditing examines either:
1. **Agent-level behavioral telemetry** (token counts, isolated confidence, individual retry rates), or
2. **Static communication graphs** (aggregated interaction counts over an entire session).

Neither paradigm adequately captures the **evolving temporal dynamics** of inter-agent interactions, including burstiness, reciprocity shifts, and delayed cascading phenomena.

---

## 2. Core Research Question
> **Can temporal interaction patterns among agents improve the prediction of impending cascading failures in multi-agent AI systems compared with agent-level behavioral features and static graph representations?**

### Sub-Questions
1. **Early Warning Lead Time:** Can temporal graph models predict system-level failures earlier (higher lead time in interaction steps) than non-graph and static-graph baselines?
2. **Information Decomposition:** How much predictive information is encoded purely within inter-agent interaction dynamics versus isolated node behavior?
3. **Graph Signatures:** Which evolving graph motifs (e.g., communication loops, delegation ping-pong, sudden drop in reciprocity) correlate most strongly with cascading failures?
4. **Generalization:** Does a model trained on smaller agent counts (e.g., $N=5$) and specific topologies (e.g., Pipeline, Star) generalize to larger topologies (e.g., $N=12$, Mesh)?
5. **Interpretability:** Can temporal attention weights or message attribution provide actionable explanations for automated early warnings?

---

## 3. Mathematical Problem Formulation

### Dynamic Interaction Graph
A multi-agent system execution is formalized as a continuous-time dynamic graph:
$$G(t) = (V(t), E(t), X_V(t), X_E(t))$$
where:
* $V(t) = \{v_1, \dots, v_N\}$ is the set of agents active at or before time $t$.
* $E(t) = \{(u, v, t_k)\}$ represents directed interaction events from agent $u$ to agent $v$ at discrete interaction timestamp $t_k \le t$.
* $X_V(t) \in \mathbb{R}^{|V| \times d_v}$ denotes time-varying agent state features (error rate, rolling latency, confidence, retry counts).
* $X_E(t_k) \in \mathbb{R}^{d_e}$ denotes interaction edge features (message length, semantic contradiction score, tool error flags).

### Prediction Target
Given observed telemetry up to current event index $t$, the objective is to predict the probability of a system-level cascading failure ($Y \in \{0, 1\}$) occurring within the forward horizon $K \in \{1, 3, 5, 10, 20\}$:
$$P(Y_{t, K} = 1 \mid \mathcal{H}_t)$$
where $\mathcal{H}_t = \{ (u_i, v_i, t_i, X_V(t_i), X_E(t_i)) \}_{i=1}^t$.

---

## 4. Failure Hierarchy
* **Level 1 (Agent Failure):** Isolated failure (e.g., tool call timeout, JSON malformation, local hallucination).
* **Level 2 (Interaction Failure):** Pairwise inconsistency (e.g., contradicting upstream outputs, rejected delegations).
* **Level 3 (Cascading Failure):** System-level breakdown resulting from multi-hop error propagation across dependencies. **AgentGuard primarily targets Level 3 prediction.**
