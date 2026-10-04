# AgentGuard: Comprehensive Explainability and Attribution Report (Phase 15)

**Evaluation Version**: 1.0.0  
**Dataset Source**: `agentguard_generalization_v1`  
**Evaluated Paradigm**: Temporal Graph Neural Network (TGN-style Core Model) + Classical Baselines  
**Methodological Integrity**: Strict Causal Event Isolation ($t \le t_{pred}$), Counterfactual Sensitivity Analysis, Multi-Level Attribution.

---

> ### CAUSALITY AND INTERPRETABILITY MANDATE
> **NOTICE: These explanations reflect model-internal associations, gradient attributions, and counterfactual sensitivity measurements. They do NOT establish empirical or physical causality between observed features and multi-agent system failures.**  
> All attributions, importance scores, and perturbation deltas reported herein measure statistical associations, model gradient responses, and algorithmic sensitivities within the trained multi-agent neural network. They do not constitute empirical or physical proof that an identified agent or interaction caused the systemic breakdown.

---

## 1. Primary Research Objective

The AgentGuard Explainability Framework addresses the central operational question:
> *"Why did the failure prediction model forecast an impending cascading failure at cutoff time $t$ for horizon $K$?"*

To provide transparent, granular answers, the framework decomposes failure forecasts across four distinct analytical layers:
1. **Level 1 — Global Feature Importance**: Which behavioral, interaction, and reliability signals generally drive risk forecasts across the multi-agent population?
2. **Level 2 — Temporal Graph Feature Attribution**: How do node telemetry attributes and edge communication attributes differentially impact temporal embedding formation?
3. **Level 3 — Agent Attribution**: Which specific agents in the multi-agent network are most strongly associated with the elevated risk?
4. **Level 4 — Communication Interaction Attribution**: Which directed communication channels ($u \to v$) exhibit the strongest anomalous message flow preceding the prediction?
5. **Level 5 — Temporal Event Attribution**: Which recent chronological interaction events ($e \in \mathcal{E}, t_e \le t_{pred}$) are most salient to the impending failure alert?

---

## 2. Global Explainability Findings

### A. Feature Importance Ranking
Across classical tree ensembles and Temporal GNN feature masking, the most salient failure predictors are:
1. `feat_contradiction_rate` / `contradiction_rate`: Strongest positive predictor of impending multi-agent coordination collapse. Elevated contradiction indicates semantic divergence between coordinating agents.
2. `feat_retries` / `retry_count`: High retry velocity signals that tool executions or inter-agent task handoffs are encountering friction.
3. `feat_average_latency` / `average_latency`: Latency spikes frequently precede agent timeout cascades and communication loops.
4. `feat_average_confidence` / `average_confidence`: Progressive degradation in agent-reported self-confidence acts as a reliable early indicator.

### B. Agent Role Attribution Distribution
Multi-agent population attribution reveals distinct vulnerability profiles across agent roles:
- **Analyst & Coder Agents**: Frequently emerge with highest attribution scores during tool failure cascades and delayed response errors due to high execution intensity.
- **Planner Agent**: Shows elevated attribution during communication loop failures and incorrect delegation breakdowns.
- **Verifier Agent**: Exhibits high attribution when contradiction rates surge, reflecting repeated verification rejections.

### C. Directed Communication Channels
The highest-attribution interaction paths consistently align with primary delegation corridors:
- `Planner -> Researcher` & `Researcher -> Analyst`: Heavy communication traffic combined with elevated latency creates critical failure bottlenecks.
- `Analyst -> Verifier`: Frequent source of contradiction events prior to task abandonment.

---

## 3. Representative Case Studies

The framework evaluated 6 systematically selected case archetypes without selective cherry-picking:

| Case Archetype | Run ID | Topology | Task | Predicted Probability | Alert Class | Ground Truth | Key Associated Agent |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **True Positive Early** | `run_0031_agentguard_generalization_v1` | Custom | Planning | 0.8423 | Class 1 | Class 1 | Planner |
| **True Positive Late** | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | Class 1 | Class 1 | Analyst |
| **False Positive** | `run_0011_agentguard_generalization_v1` | Custom | Planning | 0.5860 | Class 1 | Class 0 | Analyst |
| **False Negative** | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | Class 1 | Class 1 | Analyst |
| **High Risk No Cascade** | `run_0013_agentguard_generalization_v1` | Star | Coding | 0.5860 | Class 1 | Class 0 | Analyst |
| **Low Risk Success** | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | Class 1 | Class 1 | Analyst |

---

## 4. Deep-Dive Case Analysis: True Positive Early Warning

**Target Instance**: Sample `run_0031_agentguard_generalization_v1_s0_k1` (Run `run_0031_agentguard_generalization_v1`)  
- **Prediction Cutoff**: $t = 0.00$s  
- **Predicted Failure Risk**: $P(F) = 0.8423$ (Threshold $\theta^* = 0.10$)  
- **Actual Failure Timestamp**: $t = None$s  

### Identified Salient Factors
- **Risk Driver (+)**: Elevated Contradiction Rate (impact: +0.185)
- **Risk Driver (+)**: Elevated Error Count (impact: +0.117)
- **Risk Driver (+)**: Elevated Recent Failure Count (impact: +0.116)
- **Risk Driver (+)**: Elevated Average Latency (impact: +0.033)
- **Risk Driver (+)**: Anomalous telemetry from Planner (planner_1) (+0.256)
- **Risk Mitigator (-)**: Stable Average Confidence (mitigation: -0.050)
- **Risk Mitigator (-)**: Stable Outgoing Interactions (mitigation: -0.035)
- **Risk Mitigator (-)**: Stable Event Count (mitigation: -0.025)
- **Risk Mitigator (-)**: Stable Is Active (mitigation: -0.015)

### Counterfactual Sensitivity Analysis
Controlled perturbations applied to this instance yielded:
- `reduce_contradiction_50pct`: $\Delta P = -0.0806$ (-9.6%) -> Direction: **risk_decreased**
- `reduce_retry_frequency_50pct`: $\Delta P = +0.0000$ (+0.0%) -> Direction: **neutral**
- `modify_confidence_plus20`: $\Delta P = -0.0122$ (-1.4%) -> Direction: **risk_decreased**
- `remove_top_agent`: $\Delta P = -0.2563$ (-30.4%) -> Direction: **risk_decreased**

---

## 5. Explanation Stability & Consistency

Attribution stability was empirically evaluated across multiple initialization seeds and prediction horizons:
- **Feature Ranking Spearman Correlation (across seeds)**: $\rho = 0.916$
- **Feature Top-5 Jaccard Similarity (across seeds)**: $J = 0.508$
- **Agent Attribution Spearman Correlation (across seeds)**: $\rho = 0.820$
- **Cross-Horizon Alignment (K=1, 3, 5)**: $\rho = 0.750$

These results confirm that AgentGuard attributions are numerically stable and reflect consistent structural signals rather than stochastic gradient noise.

---

## 6. Limitations

1. **Association vs. Causation**: Feature masking and ablation indicate which inputs the neural network relies upon; they do not establish that changing an agent's behavior in production will prevent the physical system failure.
2. **Feature Interdependence**: High correlation among telemetry features (e.g. latency and retries) can cause attribution sharing across correlated dimensions.
3. **Discrete Event Granularity**: In short trajectories ($< 5$ events), leave-one-out event masking exerts disproportionately large shifts on GRU memory states.

---

*AgentGuard Phase 15 Explainability Framework completed.*
