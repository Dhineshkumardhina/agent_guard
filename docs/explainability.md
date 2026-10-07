# Explainability and Attribution Framework

## Overview

AgentGuard provides a multi-level explainability framework (`ml/explainability/`) to diagnose why the predictive model forecasted an impending cascading failure at a given cutoff step $t$.

> ### SCIENTIFIC NON-CAUSALITY MANDATE
> **IMPORTANT: All feature importances, agent attributions, edge channel scores, and counterfactual sensitivity deltas reported by this framework describe model-internal associations, sensitivity patterns, and algorithmic representations. They do NOT establish physical or empirical causal relationships.**  
> High attribution indicates that the neural network's risk output is statistically sensitive to that component; it does not prove that modifying the agent's behavior in production will prevent a physical system failure.

---

## 1. Multi-Level Attribution Hierarchy

The framework decomposes failure forecasts across five distinct analytical levels:

```
[Level 1: Global Feature Importance]
   └── Which behavioral and telemetry metrics generally drive risk across the population?
[Level 2: Graph Attribute Attribution]
   └── How do node telemetry attributes vs. edge communication attributes impact embedding formation?
[Level 3: Agent Attribution]
   └── Which specific agent in the network contributes most to the elevated risk score?
[Level 4: Communication Channel Attribution]
   └── Which directed communication edge (u -> v) exhibits anomalous message flow?
[Level 5: Temporal Event Attribution]
   └── Which recent interaction events (e <= t) triggered the early warning alert?
```

---

## 2. Global Feature Importance (Level 1)

Across tree ensembles and Temporal GNN feature masking, the most salient predictors of cascading failures are:

1. **`contradiction_rate` / `mean_contradiction_score`:** Strongest positive predictor of impending multi-agent coordination breakdown. Spikes indicate that agents are generating logically diverging intermediate assertions.
2. **`total_retries` / `retry_velocity`:** Consecutive retry attempts indicate that tool invocations or inter-agent task handoffs are encountering friction.
3. **`mean_latency` / `average_latency`:** Latency spikes frequently precede agent timeout cascades and communication loops.
4. **`mean_confidence` / `average_confidence`:** Downward degradation in agent self-reported confidence serves as a reliable early indicator before explicit crashes manifest.

---

## 3. Agent and Edge Attribution (Levels 3 & 4)

Attribution analysis decomposes risk across agent roles and communication channels:

### Agent Role Vulnerability Profiles:
* **Analyst & Coder Agents:** Frequently emerge with highest attribution during tool failure cascades and response timeouts due to high execution intensity.
* **Planner Agent:** Exhibits elevated attribution during communication loop failures and incorrect delegation breakdowns.
* **Verifier Agent:** Exhibits high attribution when contradiction rates surge, reflecting repeated verification rejections.

### Directed Communication Channels:
* **`Planner -> Researcher` & `Researcher -> Analyst`:** Primary task delegation corridors; latency spikes here create downstream starvation bottlenecks.
* **`Analyst -> Verifier`:** Frequent source of contradiction events prior to task abandonment.

---

## 4. Counterfactual Perturbation Sensitivity Analysis

To examine model sensitivity without retraining, the framework applies controlled perturbations to individual test samples:

* **`reduce_contradiction_50pct`:** Reduces edge contradiction scores by $50\%$. Measures model response to conflict reduction.
* **`reduce_retry_frequency_50pct`:** Halves retry counters. Measures sensitivity to retry velocity.
* **`modify_confidence_plus20`:** Increases agent confidence scores by $0.20$. Tests whether confidence boosts suppress false alarms.
* **`remove_top_agent`:** Masks the highest-attribution agent from the graph embedding. Demonstrates agent-level risk reliance.

### Representative Sensitivity Delta (True Positive Instance `run_0031`):
* Base Predicted Failure Probability: $P(F) = 0.8423$
* `reduce_contradiction_50pct`: $\Delta P = -0.0806$ ($-9.6\%$) $\to$ Risk decreased
* `modify_confidence_plus20`: $\Delta P = -0.0122$ ($-1.4\%$) $\to$ Risk decreased
* `remove_top_agent`: $\Delta P = -0.2563$ ($-30.4\%$) $\to$ Risk decreased

---

## 5. Temporal Risk Trajectories

For every evaluated trajectory, AgentGuard traces the predicted failure hazard $P(F_{t, K} \mid \mathcal{G}_{\le t})$ across all execution steps $t = 1, 2, \dots, T$:
- **Nominal Trajectories:** Risk remains bounded within the `NORMAL` operational envelope ($\le 0.10$).
- **Cascading Trajectories:** Risk transitions predictably from `NORMAL` $\to$ `WATCH` ($\ge 0.30$) $\to$ `HIGH_RISK` ($\ge 0.60$) $\to$ `PREDICTED_CASCADE` ($\ge 0.80$) several steps in advance of physical failure.

---

## 6. Representative Case Study Archetypes

The explainability engine profiles six representative case archetypes to avoid selective reporting:

| Case Archetype | Description | Ground Truth | Model Prediction | Primary Diagnostic Finding |
|---|---|:---:|:---:|---|
| **True Positive Early** | Early warning emitted $\ge 3$ steps prior to cascade. | Class 1 | Alert Class 1 | Early contradiction surge on Planner-Analyst channel correctly detected. |
| **True Positive Late** | Warning emitted only 1 step prior to cascade. | Class 1 | Alert Class 1 | Sudden tool failure with rapid cascade propagation. |
| **False Positive** | Model alerts on contained Level 1 glitch. | Class 0 | Alert Class 1 | High retry velocity on non-critical tool call triggered premature alert. |
| **False Negative** | Model misses subtle silent failure. | Class 1 | Alert Class 0 | Factual hallucination with normal latency and high agent confidence. |
| **High Risk No Cascade** | Elevated risk where resilience prevented crash. | Class 0 | Alert Class 1 | Verifier successfully recovered the invalid state, halting propagation. |
| **Low Risk Success** | Nominal workflow with steady progression. | Class 0 | Alert Class 0 | Stable latencies, zero contradictions, high confidence throughout. |

---

## 7. Attribution Stability & Consistency

Attribution stability was empirically validated across random seeds:
* Feature Ranking Spearman Correlation: $\rho = 0.916$
* Feature Top-5 Jaccard Similarity: $J = 0.508$
* Agent Attribution Spearman Correlation: $\rho = 0.820$
* Cross-Horizon Alignment ($K \in \{1, 3, 5\}$): $\rho = 0.750$

These metrics confirm that the attributions capture consistent structural signals rather than stochastic gradient noise.
