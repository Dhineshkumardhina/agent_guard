# Supplementary Materials: AgentGuard

**Paper Title:** AgentGuard: Temporal Interaction-Graph-Based Prediction of Failures in Multi-Agent AI Systems  
**Authors:** AgentGuard Research Team  
**Artifact Archive:** `paper/supplementary/`

---

## S1. Detailed Model Hyperparameters and Optimization Configurations

All neural models were implemented in PyTorch 2.14.1 and PyTorch Geometric 2.8.0.post1 and trained on CPU using AdamW with weight decay $\lambda = 1 \times 10^{-4}$.

### S1.1 Rule-Based Baseline
* **Retry Threshold ($\theta_r$):** 3 consecutive retry events.
* **Latency Threshold ($\theta_l$):** 2.5 seconds (or $\mu + 3\sigma$ of historical baseline).
* **Contradiction Threshold ($\theta_c$):** 0.70 semantic contradiction score.
* **Calibration:** Decision threshold $\theta^* \in [0.10, 0.90]$ chosen via grid search on validation F1.

### S1.2 Classical Machine Learning Baselines
* **Logistic Regression:** L2 regularization ($C = 1.0$), solver: `lbfgs`, max iterations: 1,000, tolerance: $1 \times 10^{-4}$.
* **Random Forest:** $N_{\text{trees}} = 100$, maximum tree depth: 10, minimum split samples: 2, criterion: Gini impurity, class weighting: `"balanced"`.
* **XGBoost:** Number of boosting rounds: 100, learning rate: 0.05, maximum tree depth: 5, subsample ratio: 0.80, column subsample ratio: 0.80, objective: `binary:logistic`, evaluation metric: `logloss`.

### S1.3 Recurrent Sequence Baselines (LSTM and GRU)
* **Architecture:** 2 recurrent layers, hidden dimension $h = 64$, dropout: 0.20, bidirectional: False.
* **Input Feature Dimension:** 12 event features (continuous timestamp delta, message length, tokens, latency, confidence, contradiction, tool error flag, retry count, etc.).
* **Sequence Length ($L$):** Evaluated at $L=10$ and $L=20$ steps (shorter sequences left-padded with zeros).
* **Optimization:** AdamW ($\eta = 1 \times 10^{-3}$), batch size: 16, loss: `BCEWithLogitsLoss`.

### S1.4 Static Graph Neural Networks (GCN and GAT)
* **GCN:** 2 layers of `GCNConv`, hidden dimension: 64, activation: ReLU, dropout: 0.20, readout: global mean pooling.
* **GAT:** 2 layers of `GATConv`, 4 attention heads ($d_{\text{head}} = 16$, total hidden dimension 64), negative slope: 0.2, dropout: 0.20, readout: global mean pooling.
* **Hazard Head:** 2-layer MLP ($64 \to 32 \to 1$) with ReLU activation and dropout 0.10.
* **Optimization:** AdamW ($\eta = 5 \times 10^{-4}$), batch size: 8 graph snapshots.

### S1.5 Continuous-Time Temporal GNN (Core Research Model)
* **Fourier Time Encoding:** Dimension $D_{\text{time}} = 16$, sinusoidal formulation with base frequency $10000.0$.
* **Node Memory Bank:** Dimension $d_{\text{mem}} = 64$ per agent, initialized to zero tensors; strict reset on each new trajectory run.
* **Message Function:** 2-layer MLP mapping $[ \mathbf{m}_u, \mathbf{m}_v, \mathbf{x}_e, \phi(\Delta t) ] \in \mathbb{R}^{64 + 64 + 10 + 16 = 154} \to \mathbb{R}^{64}$.
* **Memory Updater:** Recurrent `nn.GRUCell(input_size=64, hidden_size=64)`.
* **Temporal Attention:** 1-hop dynamic attention over recent interaction edges with scaled dot-product attention ($K=4$ heads).
* **Graph Readout:** Concatenation of Global Mean Pooling and Global Max Pooling: $\mathbf{h}_G = [\text{MeanPool}(\{\mathbf{z}_v\}) \,\|\, \text{MaxPool}(\{\mathbf{z}_v\})] \in \mathbb{R}^{128}$.
* **Hazard Prediction Head:** 2-layer MLP ($128 \to 64 \to 1$) with dropout 0.20.
* **Optimization:** AdamW ($\eta = 5 \times 10^{-4}$), gradient clipping max norm $\| \mathbf{g} \|_2 = 1.0$, early stopping patience: 10 epochs.

---

## S2. Tabular Feature Schema and Extraction Definitions

The 17-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{17}$ computed at evaluation cutoff step $t$ conditions strictly on causal history $\mathcal{H}_t$:

| Index | Feature Identifier | Mathematical Definition | Operational Meaning |
|---|---|---|---|
| 1 | `total_events_observed` | $|\mathcal{E}_{\le t}|$ | Cumulative message exchanges up to step $t$. |
| 2 | `mean_output_quality` | $\frac{1}{|\mathcal{E}_{\le t}|} \sum_{e \le t} q_e$ | Average output quality across participating agents. |
| 3 | `min_output_quality` | $\min_{e \le t} q_e$ | Lowest quality score recorded in the session. |
| 4 | `mean_confidence` | $\frac{1}{|\mathcal{E}_{\le t}|} \sum_{e \le t} c_e$ | Average self-reported agent confidence. |
| 5 | `min_confidence` | $\min_{e \le t} c_e$ | Lowest recorded confidence indicator. |
| 6 | `total_token_count` | $\sum_{e \le t} \text{tokens}_e$ | Cumulative tokens expended by the ensemble. |
| 7 | `total_latency` | $\sum_{e \le t} \text{latency}_e$ | Cumulative execution latency in simulated seconds. |
| 8 | `mean_latency` | $\frac{1}{|\mathcal{E}_{\le t}|} \sum_{e \le t} \text{latency}_e$ | Average single-step response latency. |
| 9 | `max_latency` | $\max_{e \le t} \text{latency}_e$ | Maximum recorded latency spike. |
| 10 | `total_retries` | $\sum_{e \le t} \text{retry}_e$ | Cumulative retry queries across all agents. |
| 11 | `mean_contradiction_score` | $\frac{1}{|\mathcal{E}_{\le t}|} \sum_{e \le t} \text{contra}_e$ | Mean semantic contradiction metric. |
| 12 | `max_contradiction_score` | $\max_{e \le t} \text{contra}_e$ | Highest recorded pairwise contradiction spike. |
| 13 | `tool_call_count` | $\sum_{e \le t} \mathbb{I}(\text{tool\_used} \ne \emptyset)$ | Total external tools invoked. |
| 14 | `tool_failure_count` | $\sum_{e \le t} \mathbb{I}(\text{tool\_error} = \text{True})$ | Total tool exceptions or timeouts. |
| 15 | `agent_count_active` | $\sum_{v \in \mathcal{V}} \mathbb{I}(\text{status}_v = \text{active})$ | Number of surviving non-crashed agents. |
| 16 | `error_count` | $\sum_{e \le t} \mathbb{I}(\text{error\_type} \ne \emptyset)$ | Total caught runtime errors. |
| 17 | `interaction_density` | $\frac{|\text{UniqueDirectedEdges}_{\le t}|}{N(N-1)}$ | Ratio of active communication channels to total possible directed pairs. |

---

## S3. Subgroup Evaluation Breakdowns (Phase 12 Benchmark)

### S3.1 Breakdown by Communication Topology (Horizon K=1)
* **Custom Topology (12 test samples, all positive):**
  - Classical ML (LogReg, RF, XGBoost): Precision = 1.000, Recall = 1.000, F1 = 1.000, AUROC = 1.000
  - Static GNN (GCN, GAT): Precision = 1.000, Recall = 1.000, F1 = 1.000, AUROC = 1.000
  - Temporal GNN: Precision = 1.000, Recall = 0.583, F1 = 0.737, AUROC = 1.000
  - Rule-Based: Precision = 1.000, Recall = 0.083, F1 = 0.154, AUROC = 1.000
* **Pipeline Topology (2 test samples, 1 positive, 1 negative):**
  - Classical ML: Precision = 1.000, Recall = 1.000, F1 = 1.000, AUROC = 1.000
  - Static GNN: Precision = 1.000, Recall = 1.000, F1 = 1.000, AUROC = 1.000
  - Temporal GNN: Precision = 0.000, Recall = 0.000, F1 = 0.000, AUROC = 0.000 (missed single positive point)

### S3.2 Breakdown by Benchmark Task (Horizon K=1)
* **Planning Task (12 test samples):**
  - Classical ML: F1 = 1.000
  - Temporal GNN: F1 = 0.737 (Recall = 0.583)
  - Rule-Based: F1 = 0.154
* **Research Task (2 test samples):**
  - Classical ML: F1 = 1.000
  - Temporal GNN: F1 = 0.000 (due to conservative threshold $\theta^*$ missing the single positive point)

---

## S4. Qualitative Error Case Studies (Phase 15 Diagnostics)

### Case 1: True Positive Early Warning (`run_0031_agentguard_generalization_v1`)
* **Task:** Planning, Topology: Custom Hierarchy.
* **Event:** Early contradiction surge on `Planner -> Analyst` delegation channel at step $t=3$.
* **Prediction:** $P(F) = 0.8423$ (Alert emitted 4 steps before terminal cascade).
* **Attributed Drivers:** Contradiction rate ($+0.185$), Error count ($+0.117$), Planner node anomaly ($+0.256$).
* **Outcome:** Valid early warning; successful pre-cascade detection.

### Case 2: False Positive Contained Glitch (`run_0011_agentguard_generalization_v1`)
* **Task:** Planning, Topology: Custom Hierarchy.
* **Event:** Isolated tool timeout occurred during intermediate data parsing. Agent retried 3 times and succeeded; task completed nominal execution.
* **Prediction:** $P(F) = 0.5860 > \theta^* = 0.10$ $\to$ False alarm.
* **Root Cause:** Tree ensemble and GNN were over-sensitive to local retry burst, assuming it would cascade across the verifier boundary.

### Case 3: False Negative Silent Hallucination (`run_0003_agentguard_generalization_v1`)
* **Task:** Planning, Topology: Custom Hierarchy.
* **Event:** Factual hallucination in upstream agent with high self-reported confidence ($0.95$) and nominal latency ($1.1$s).
* **Prediction:** $P(F) = 0.082 < \theta^* = 0.10$ at step $t=2$ $\to$ False Negative.
* **Root Cause:** Telemetry signals remained within normal operational envelopes until downstream verifier rejected output at step $t=6$.
