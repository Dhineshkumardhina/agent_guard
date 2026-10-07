# System and Research Limitations

## Overview

In accordance with scientific integrity and transparent research standards, this document details the empirical, architectural, and methodological limitations of AgentGuard. These boundaries define the scope of validity for reported findings and outline open challenges for future research.

---

## 1. Test Population Sample Size ($N=35$)

* **Observed Limitation:** In the standardized benchmark dataset (`agentguard_dataset_v1`), the held-out test split comprises 3 runs and 35 total prediction samples distributed across horizons ($K \in \{1, 3, 5, 10\}$).
* **Statistical Impact:** While the point F1 score of Classical ML (XGBoost: 1.000) differs from Temporal GNN (0.700) on this split, trajectory block bootstrap testing demonstrates that the difference does not reach conventional statistical significance ($p = 0.080 > 0.05$).
* **Scientific Caveat:** Claims that one model family is definitively superior to another across all operating conditions cannot be substantiated on this sample size alone.

---

## 2. Finite Simulation Trajectory Lengths and Horizon $K=20$

* **Observed Limitation:** Trajectories in `agentguard_dataset_v1` execute for a maximum of 20 interaction steps.
* **Impact on Long-Horizon Evaluation:**
  - Positive cascading failure labels require an observation cutoff $t$ followed by a forward interval $(t, t+K]$.
  - For $K=20$, any cutoff step $t \ge 1$ exceeds the 20-step trajectory boundary before the forward window completes.
  - Consequently, **horizon $K=20$ contains 0 test instances in `v1`**.
* **Remediation:** Long-horizon trajectory hazard forecasting requires multi-hour or continuous multi-agent sessions ($T \ge 100$ steps).

---

## 3. Synthetic Multi-Agent Simulation vs. Live Production Systems

* **Controlled Environment vs. Production Reality:**
  - AgentGuard evaluates agents within a controlled discrete-event simulator.
  - In production, multi-agent workflows interface with proprietary LLM APIs (e.g., GPT-4, Claude 3.5, Gemini 1.5 Pro) across live network connections.
* **Discrepancy Factors:**
  1. **Token & Latency Jitter:** Real API calls experience network congestion, rate limiting (HTTP 429), and dynamic model latency variance not perfectly modeled by synthetic distributions.
  2. **Prompt Non-Determinism:** Temperature $> 0.0$ introduces conversational drift that may yield novel conversational breakdown modes.
  3. **Complex External Tool APIs:** Live tools (e.g., SQL databases, GitHub APIs, web scrapers) fail with idiosyncratic network and permission errors.
* **Interpretation Guideline:** AgentGuard findings demonstrate structural graph properties and temporal dynamics under controlled experimental conditions; they must be validated on production telemetry before mission-critical deployment.

---

## 4. Association vs. Causality in Explainability

* **Limitation:** All feature importances, agent attributions, and interaction channel rankings describe **model-internal sensitivity and statistical association**.
* **Non-Causal Interpretation:**
  - If agent `analyst_1` has the highest attribution score for an impending failure, this indicates that the neural network's risk output is highly sensitive to `analyst_1`'s features.
  - It does **not** prove that `analyst_1` was the root cause of the breakdown. In multi-agent cascades, a verifier may exhibit high attribution simply because it was the first to detect an upstream corruption caused by the planner.
  - Intervening on the highest-attribution agent in production will not necessarily avert the system failure.

---

## 5. Subgroup Sample Imbalance on Rare Failure Modes

* **Observed Distribution:** Across the 72 trajectories of `agentguard_generalization_v1`, certain fault modes occur frequently (`delayed_response`: 23 runs, `incorrect_delegation`: 16 runs), while others occur rarely (`malformed_output`: 3 runs, `communication_loop`: 4 runs).
* **Impact:** Performance estimates on rarely triggered failure modes have wider confidence intervals and higher sensitivity to random seed initialization.

---

## 6. Hardware and Computational Bounds

* **Environment Provenance:** Evaluated on Windows 11 x86_64 CPU (`torch_version: 2.14.1+cpu`, no CUDA acceleration active during standard CI testing).
* **Scalability Boundary:** The continuous-time graph builder and in-memory node memory bank scale effectively to networks of $N \le 50$ agents. Scaling to massive multi-agent ecosystems ($N > 1,000$ agents) requires distributed graph databases (e.g., Neo4j, Memgraph) and GPU-accelerated sparse temporal message passing.
