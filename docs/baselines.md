# Baseline Models Documentation

## Overview

AgentGuard implements eight reference baseline models across four paradigm families to benchmark against the proposed Continuous-Time Temporal Graph Neural Network. All baselines are evaluated on standardized test partitions across horizons $K \in \{1, 3, 5, 10, 20\}$.

---

## 1. Summary of Evaluated Model Families

```
                                 ALL EVALUATED MODELS
                                          │
       ┌──────────────────┬───────────────┴───────────────┬──────────────────┐
       ▼                  ▼                               ▼                  ▼
  [Rule-Based]     [Classical ML]                 [Sequence Models]     [Static GNN]
  - Heuristic      - Logistic Regression          - LSTM                - GCN
    Detector       - Random Forest                - GRU                 - GAT
                   - XGBoost
```

---

## 2. Rule-Based Baseline (`ml/baselines/rule_based/`)

### Role in Experiment
Establishes the performance ceiling of naive threshold-based observability systems commonly deployed in DevOps and LLM application telemetry.

### Input
Rolling telemetry counters at step $t$:
* `retry_count`: Consecutive retry attempts by any agent.
* `mean_latency`: Moving average of interaction latencies.
* `contradiction_rate`: Average semantic contradiction metric.

### Architecture & Rules
* **Rule 1 (Retry Spike):** Trigger alert if `retry_count >= 3`.
* **Rule 2 (Latency Spike):** Trigger alert if `mean_latency >= 2.5s` (or $> 3\sigma$ above historical baseline).
* **Rule 3 (Contradiction Surge):** Trigger alert if `contradiction_rate >= 0.70`.
* **Composite Score:** Weighted sum normalized $\in [0.0, 1.0]$. An alert triggers if composite score $\ge \theta^*$.

### Training & Calibration
Non-parametric; decision threshold $\theta^*$ is calibrated on validation trajectories to maximize validation F1 score.

### Output
Predicted failure probability $\hat{p} \in [0.0, 1.0]$ and binary alert flag.

---

## 3. Classical Machine Learning Baselines (`ml/baselines/classical_ml/`)

### Role in Experiment
Evaluates whether isolated agent-level behavioral aggregates without relational connectivity or continuous time order are sufficient for impending failure prediction (tests Null Hypothesis $H_0$).

### Input
A 17-dimensional tabular feature vector $\mathbf{x} \in \mathbb{R}^{17}$ computed over the execution history up to cutoff step $t$:
* Telemetry aggregates: `total_events_observed`, `mean_output_quality`, `min_output_quality`, `mean_confidence`, `min_confidence`, `total_token_count`, `total_latency`, `mean_latency`, `max_latency`, `total_retries`, `mean_contradiction_score`, `max_contradiction_score`, `tool_call_count`, `tool_failure_count`, `agent_count_active`, `error_count`, `interaction_density`.
* Features are standardized via `StandardScaler` fit strictly on the training partition.

### Models Implemented
1. **Logistic Regression:**
   - Architecture: L2-regularized generalized linear model with logistic sigmoid link function.
   - Solver: `lbfgs`, max iterations: 1,000, $C = 1.0$.
2. **Random Forest:**
   - Architecture: Ensemble of 100 de-correlated classification trees (`n_estimators=100`).
   - Hyperparameters: `max_depth=10`, `min_samples_split=2`, `class_weight="balanced"`.
3. **XGBoost:**
   - Architecture: Gradient-boosted decision trees optimizing binary logistic loss.
   - Hyperparameters: `n_estimators=100`, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`.

### Training & Calibration
Trained on `train.parquet`. Validation split (`val.parquet`) is used to calibrate decision threshold $\theta^*$ and early stopping.

### Output
Probability $\hat{p} \in [0.0, 1.0]$ of Level 3 cascading failure within horizon $K$.

---

## 4. Temporal Sequence Baselines (`ml/baselines/sequence/`)

### Role in Experiment
Evaluates whether chronological event sequence ordering alone (without explicit graph topology) is sufficient for early warning detection.

### Input
Ordered sequence of interaction event feature vectors:
$$\mathbf{S}_t = [\mathbf{x}_{e_1}, \mathbf{x}_{e_2}, \dots, \mathbf{x}_{e_L}] \in \mathbb{R}^{L \times d_e}$$
where $L$ is sequence length (default $L=10$ or $20$) and $d_e = 12$ represents event attributes (timestamp delta, message length, tokens, latency, confidence, contradiction, tool error flag, retry count). Shorter sequences are zero-padded on the left.

### Models Implemented
1. **LSTM (Long Short-Term Memory):**
   - Architecture: 2-layer bidirectional LSTM, hidden dimension $h = 64$, dropout $p = 0.20$.
   - Output Head: Linear layer mapping final hidden state $\mathbf{h}_L \in \mathbb{R}^{64} \to \mathbb{R}^1 \to \sigma(\cdot)$.
2. **GRU (Gated Recurrent Unit):**
   - Architecture: 2-layer GRU, hidden dimension $h = 64$, dropout $p = 0.20$.
   - Output Head: Linear projection on final hidden state followed by sigmoid activation.

### Training Protocol
- Optimizer: AdamW with learning rate $\eta = 1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$.
- Loss Function: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- Batch size: 16; early stopping with patience of 10 epochs on validation loss.

---

## 5. Static Graph Neural Network Baselines (`ml/baselines/static_gnn/`)

### Role in Experiment
Evaluates whether relational communication graph structure (who talks to whom) provides predictive value when temporal order is collapsed into a static graph (tests Research Question RQ2).

### Input
A static graph snapshot $G_{\text{static}} = (\mathcal{V}, \mathcal{E}, \mathbf{X}_V, \mathbf{A})$:
* Adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ where $A_{ij} = 1$ if agent $i$ communicated with agent $j$ at least once in $[0, t]$.
* Node feature matrix $\mathbf{X}_V \in \mathbb{R}^{N \times d_v}$ summarizing cumulative agent telemetry up to step $t$.

### Models Implemented
1. **GCN (Graph Convolutional Network):**
   - Architecture: 2-layer GCN (`GCNConv` from PyTorch Geometric).
   - Message passing: $\mathbf{H}^{(l+1)} = \sigma\left( \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{\tilde{A}} \mathbf{\tilde{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$.
   - Hidden dimension: 64, ReLU activation, dropout $p = 0.20$.
   - Graph Readout: Global mean pooling across all agent nodes $\mathbf{h}_G = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \mathbf{h}_v$.
   - Prediction Head: 2-layer MLP mapping $\mathbf{h}_G \to 1$.
2. **GAT (Graph Attention Network):**
   - Architecture: 2-layer GAT (`GATConv` from PyTorch Geometric).
   - Attention mechanism: Multi-head attention ($K=4$ heads) weighting directed communication edges.
   - Hidden dimension: 64, LeakyReLU ($\alpha = 0.2$), global mean pooling readout, 2-layer MLP hazard head.

### Training Protocol
- Optimizer: AdamW, learning rate $\eta = 5 \times 10^{-4}$.
- Loss: Binary Cross-Entropy with Logits (`pos_weight` calibrated to inverse class ratio).
- Evaluated on identical test graph snapshots.
