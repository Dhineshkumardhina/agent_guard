# AgentGuard Phase 14: Generalization and Robustness Evaluation Plan

## 1. Primary Research Question

The core objective of Phase 14 is to assess whether learned failure-prediction models generalize beyond the exact multi-agent configurations encountered during training:

> **Primary Research Question**: Does a model trained on particular agent counts, interaction topologies, task categories, or fault configurations retain predictive performance when evaluated under previously unseen out-of-distribution (OOD) configurations?

In accordance with strict scientific methodology:
1. Generalization capability is **not assumed**; it is empirically quantified.
2. We evaluate whether graph-based architectures (Static GNN, Temporal GNN) provide superior structural invariance under distribution shifts compared to agent-level classical ML and sequence models.
3. Every performance degradation or transfer gap is reported transparently without cherry-picking.

---

## 2. Generalization Dimensions

We systematically investigate four orthogonal distribution shift dimensions:

### A. Number of Agents (Population Scaling)
- **Supported Counts**: $N \in \{3, 5, 8, 12\}$.
- **Shift Mechanics**: Multi-agent communication graphs exhibit varying density, path lengths, and node degrees as population scales. Models must predict cascading failures without overfitting to fixed graph diameters or fixed input sizes.
- **Experimental Splits**:
  - **G1**: Train on $\{3, 5\}$ agents $\rightarrow$ Test on $\{8\}$ agents.
  - **G2**: Train on $\{3, 5, 8\}$ agents $\rightarrow$ Test on $\{12\}$ agents.

### B. Communication Topology (Structural Shift)
- **Supported Topologies**: Pipeline, Star, Mesh, Custom (hierarchical/clustered).
- **Shift Mechanics**: Communication pathways govern how errors propagate. In a Pipeline, errors propagate linearly; in a Star, errors funnel through the hub; in a Mesh, errors diffuse across redundant edges.
- **Experimental Splits**:
  - **G3**: Train on $\{\text{Pipeline}, \text{Star}, \text{Mesh}\} \rightarrow$ Test on $\{\text{Custom}\}$ (Leave-One-Topology-Out).
  - **G3b**: Train on $\{\text{Star}, \text{Mesh}, \text{Custom}\} \rightarrow$ Test on $\{\text{Pipeline}\}$.

### C. Multi-Agent Task Workflow (Semantic Shift)
- **Supported Tasks**: Research, Coding, Data Analysis, Planning.
- **Shift Mechanics**: Different task domains exhibit distinct conversational vocabularies, message length distributions, verification stages, and tool execution patterns.
- **Experimental Splits**:
  - **G4**: Train on $\{\text{Research}, \text{Coding}, \text{Planning}\} \rightarrow$ Test on $\{\text{Analysis}\}$ (Leave-One-Task-Out).
  - **G4b**: Train on $\{\text{Coding}, \text{Analysis}, \text{Planning}\} \rightarrow$ Test on $\{\text{Research}\}$.

### D. Failure-Type Generalization (Fault Mode Transfer)
- **Failure Taxonomy (12 Modes)**:
  1. `hallucinated_output`
  2. `incorrect_information`
  3. `tool_failure`
  4. `tool_timeout`
  5. `delayed_response`
  6. `malformed_output`
  7. `low_confidence_output`
  8. `contradictory_output`
  9. `communication_loop`
  10. `incorrect_delegation`
  11. `stale_context`
  12. `agent_dropout`
- **Shift Mechanics**: Models may learn to detect specific syntactic or telemetry artifacts of seen failures. We evaluate whether models learn general interaction pre-failure precursors that transfer to unobserved failure mechanisms.
- **Experimental Split**:
  - **G5**: Train on **Seen Failure Modes** (e.g. `hallucinated_output`, `tool_failure`, `delayed_response`, `contradictory_output`, `malformed_output`, `communication_loop`) $\rightarrow$ Test on **Held-Out / Unseen Failure Modes** (e.g. `incorrect_information`, `tool_timeout`, `low_confidence_output`, `incorrect_delegation`, `agent_dropout`).
  - Nominal clean runs (fault-free) are included in both training and evaluation splits to ensure realistic class balance and false alarm measurement.

---

## 3. Strict Train / Test Separation & Integrity Controls

Scientific validity mandates zero information leakage:

1. **Run-Level Isolation**: Splitting is strictly performed at the whole simulation trajectory level ($\text{run\_id}$). No step, event, or snapshot from a test run is ever accessible during training or validation.
2. **Temporal Causality**: Observations up to time $t$ predict failures in $(t, t+k]$. No future timestamps or future events leak into features or graph states.
3. **Threshold Calibration Integrity**:
   - The decision threshold $\theta^*$ is calibrated **strictly on the In-Distribution (ID) validation split** ($\theta^* = \arg\max_{\theta \in [0.10, 0.90]} F1_{\text{val}}$).
   - $\theta^*$ is subsequently **frozen** and applied unchanged to both the In-Distribution test set and the Out-of-Distribution test set.
   - Out-of-distribution test labels are never inspected for threshold optimization.
4. **Independent Preprocessing**: Scalers, normalizers, and feature encoders are fitted exclusively on training trajectories.

---

## 4. Generalization Experiment Matrix

| Exp ID | Dimension | In-Distribution (Train / Val / Test-ID) | Out-of-Distribution (Test-OOD) | Research Hypothesis |
|:---|:---|:---|:---|:---|
| **G1** | Agent Count | Agents $\in \{3, 5\}$ | Agents $= 8$ | Failure dynamics scale to larger multi-agent networks without node count retuning. |
| **G2** | Agent Count | Agents $\in \{3, 5, 8\}$ | Agents $= 12$ | Structural predictors generalize to dense 12-agent coordination environments. |
| **G3** | Topology | Topology $\in \{\text{Pipeline}, \text{Star}, \text{Mesh}\}$ | Topology $= \text{Custom}$ | Graph convolutions generalize to irregular, unobserved communication graphs. |
| **G3b** | Topology | Topology $\in \{\text{Star}, \text{Mesh}, \text{Custom}\}$ | Topology $= \text{Pipeline}$ | Network trained on rich topologies transfers to constrained linear sequences. |
| **G4** | Task | Task $\in \{\text{Research}, \text{Coding}, \text{Planning}\}$ | Task $= \text{Analysis}$ | Cascading warning signals transfer across multi-agent prompt/task domains. |
| **G4b** | Task | Task $\in \{\text{Coding}, \text{Analysis}, \text{Planning}\}$ | Task $= \text{Research}$ | Anomaly signals learned from structured tasks apply to open-ended research. |
| **G5** | Failure Type | Seen Faults + Nominal | Held-Out Unseen Faults | Interaction graph models detect emergent failure precursors without memorizing fault signatures. |

---

## 5. Model Families Evaluated

To characterize how different inductive biases handle distribution shift, we compare:

1. **Family A: Classical Machine Learning** (Agent-Level Tabular Features):
   - Logistic Regression ($L_2$ regularized)
   - Random Forest (Bagging ensemble)
   - XGBoost (Gradient boosted decision trees)
2. **Family B: Temporal Sequence Baselines** (Historical Node Telemetry):
   - Recurrent Neural Networks: LSTM, GRU
3. **Family C: Static Graph Neural Networks** (Structural Graph Topology):
   - Graph Convolutional Network (GCN)
   - Graph Attention Network (GAT)
4. **Family D: Temporal Graph Neural Network** (TGN-style Core Research Model):
   - Continuous time encodings $\phi(\Delta t)$, continuous GRU memory ($m_v$), temporal graph convolution.

*Computational Note*: Full matrix benchmark focuses on the core Temporal GNN across all horizons and seeds, with comparative evaluation across all 6 model families on representative transfer tasks.

---

## 6. Prediction Horizons and Seeds

- **Prediction Horizons**: $K \in \{1, 3, 5, 10, 20\}$ steps advance warning.
- **Random Seeds**: Reproducibility framework evaluated across $\{42, 123, 456\}$.

---

## 7. Metrics and Generalization Gap Formulation

For both In-Distribution ($\text{ID}$) and Out-of-Distribution ($\text{OOD}$) test sets, we measure:
- **Precision, Recall, F1 Score**
- **AUROC, AUPRC**
- **False Positive Rate (FPR), False Alarm Rate (FAR)**
- **Mean & Median Early Warning Lead Time** (seconds advance notice)
- **Successful Early Warnings & Warnings Per Trajectory**

### Generalization Gap Formulation
For any metric $M$:
$$\text{Generalization Gap}(M) = M_{\text{In-Distribution}} - M_{\text{Out-of-Distribution}}$$
- For utility metrics ($F1, \text{AUPRC}, \text{AUROC}, \text{Recall}$), a **positive gap** indicates performance degradation under distribution shift.
- For lead time, a positive gap indicates earlier warnings in-distribution than out-of-distribution.
- For false positive rate, the gap is inverted: $\Delta \text{FPR} = \text{FPR}_{\text{OOD}} - \text{FPR}_{\text{ID}}$.

### Statistical Rigor
- Confidence intervals (95%) and transfer significance are derived via **trajectory-level block bootstrap** ($B=500$) resampling runs with replacement.
- Trajectory clustering prevents pseudoreplication.
