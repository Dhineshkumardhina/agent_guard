# Formal Data Leakage and Temporal Invariant Audit

**Project**: AgentGuard — Early Cascading Failure Detection in Multi-Agent Systems  
**Phase**: Phase 18 — Comprehensive Testing, Security, and Research Validation  
**Date**: October 2026  
**Auditor**: Senior Research-Software & ML Assurance Engineer  
**Status**: APPROVED — ZERO LEAKAGE DETECTED

---

## Executive Summary

Data leakage is the foremost threat to scientific validity in predictive temporal modeling, especially for early warning systems where anticipating future failure modes must strictly rely on past behavioral trajectories. This audit formally verifies and documents the mathematical and architectural mechanisms preventing temporal leakage, identity leakage, preprocessing leakage, and state pollution in AgentGuard.

Every invariant defined below is covered by automated regression tests in `tests/test_fault_and_label_validation.py`, `tests/test_temporal_gnn_invariants.py`, and `tests/test_end_to_end_pipeline.py`.

---

## 1. Temporal Cutoff Invariant ($t \le t_{\text{eval}}$)

### Invariant Definition
At any designated prediction step or evaluation timestamp $t_{\text{eval}}$, the model and feature extractors MUST NOT inspect, query, or aggregate any event $e_i$ whose timestamp $t(e_i) > t_{\text{eval}}$ or discrete step index $s(e_i) > s_{\text{eval}}$.

### Implementation Verification
* **Temporal Graph Builder (`ml/graph/graph_builder.py`)**:
  Graph construction takes a chronological slice of events:
  ```python
  history_events = [e for e in run_events if e.timestamp <= current_timestamp]
  ```
  Edge weights, communication frequencies, sentiment trajectories, and anomaly scores are computed strictly over `history_events`.
* **Dynamic Neighborhood Aggregation (`ml/baselines/temporal_gnn/models.py`)**:
  The `DynamicNeighborhoodTracker` updates edge memories and interactions using chronological sorting. Temporal edge queries strictly mask out edges where $t_{\text{edge}} > t_{\text{query}}$.
* **Automated Test Proof**:
  `tests/test_temporal_gnn_invariants.py::test_temporal_neighborhood_no_future_edges` verifies that future interactions injected at $t = 25.0$ are invisible to neighbor queries executed at $t = 15.0$.

---

## 2. Horizon Boundary Invariant ($y_{t}^{(k)} \in (t, t + k]$)

### Invariant Definition
For a prediction horizon $k \in \{1, 3, 5\}$, the target ground truth binary label $y_t^{(k)}$ indicates whether a cascading failure occurs within the forward window $(t, t + k]$. Events occurring at or before $t$ are part of the past feature history and MUST NOT contaminate the target definition. Failures occurring beyond $t + k$ MUST NOT be labeled as positive for horizon $k$.

### Implementation Verification
* **Prediction Sample Generator (`ml/data/prediction_samples.py`)**:
  ```python
  # Cascading failure step is tau_cascade
  # Sample is generated at step s
  is_positive = (s < tau_cascade <= s + k)
  ```
  Samples evaluated after the cascade has already started ($s \ge \tau_{\text{cascade}}$) are flagged as post-failure and excluded from early warning training datasets to prevent trivial retrospective classification.
* **Automated Test Proof**:
  `tests/test_fault_and_label_validation.py::test_horizon_boundary_exclusion` tests ground truth assignment for cascading steps at $s=4$. Samples evaluated at $s=0$ for $k=3$ evaluate to negative ($4 \notin (0, 3]$), while sample at $s=1$ for $k=3$ evaluates to positive ($4 \in (1, 4]$).

---

## 3. Scaler and Preprocessing Invariant

### Invariant Definition
Any normalization, standardization (e.g., `StandardScaler`), categorical encoding, or imputation parameter MUST be computed solely on the training partition:
$$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} x_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum_{i \in \text{train}} (x_i - \mu_{\text{train}})^2}$$
Under no circumstances may validation or test split feature values influence $\mu$ or $\sigma$.

### Implementation Verification
* **Tabular and Sequence Preprocessing (`ml/baselines/classical_ml/features.py`, `ml/baselines/sequence_models/dataset.py`)**:
  Scalers are fit in `fit(X_train)` and saved to artifacts. During inference or evaluation, `scaler.transform(X_val)` and `scaler.transform(X_test)` apply frozen parameters.
* **Automated Test Proof**:
  `tests/test_end_to_end_pipeline.py` enforces separate fitting on `X_train` before applying to `X_test`.

---

## 4. Trajectory-Level Splitting Invariant ($\text{Runs}_{\text{train}} \cap \text{Runs}_{\text{test}} = \emptyset$)

### Invariant Definition
Multi-agent simulations produce multiple sequential prediction samples per run. If samples from run $R$ are split randomly into both training and test sets, the model can memorize run-specific agent IDs, task topics, or topological quirks. Therefore, dataset partitioning MUST be executed strictly at the **trajectory/run** level:
$$\mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{val}} = \emptyset, \quad \mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{test}} = \emptyset, \quad \mathcal{R}_{\text{val}} \cap \mathcal{R}_{\text{test}} = \emptyset$$

### Implementation Verification
* **Trajectory Partitioning (`ml/data/split.py`)**:
  `split_trajectories()` partitions unique `run_id`s with stratification on failure labels. `split_samples()` then filters samples by membership in partition run sets:
  ```python
  train_samples = [s for s in all_samples if s.run_id in train_run_ids]
  ```
* **Automated Test Proof**:
  `tests/test_end_to_end_pipeline.py` verifies zero `run_id` intersection between train and test splits.

---

## 5. State and Memory Isolation Invariant

### Invariant Definition
Recurrent neural networks (LSTM, GRU), temporal GNN memory banks (TGN node memories), simulation environments, and fault trackers MUST NOT carry persistent hidden states across distinct simulation runs.

### Implementation Verification
* **TGN Node Memory Reset (`ml/baselines/temporal_gnn/models.py`)**:
  `TemporalGraphFailurePredictor.reset_memory()` initializes all node memory embeddings to zero tensors.
* **Simulation Environment Reset (`ml/simulation/run.py`)**:
  Each `SimulationRun.execute()` instantiates a fresh `SimulationEnvironment(seed=...)` and resets all agent states and message mailboxes.
* **Automated Test Proof**:
  `tests/test_temporal_gnn_invariants.py::test_memory_isolation_between_runs` proves that processing Run A changes memory states, but invoking `reset_memory()` before Run B produces identical outputs to a freshly instantiated model.

---

## Audit Matrix Summary

| Leakage Dimension | Potential Risk | Prevention Mechanism | Test Verification | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Temporal Horizon** | Peeking into future steps | Strict inequality $t \le t_{\text{eval}}$ | `test_temporal_neighborhood_no_future_edges` | PASS |
| **Cascade Boundary** | Labeling post-cascade steps as early warning | Horizon $(t, t+k]$ boundary filtering | `test_horizon_boundary_exclusion` | PASS |
| **Feature Scaling** | Test distribution informing scalers | Fit exclusively on train split | `test_complete_end_to_end_research_pipeline` | PASS |
| **Run Leakage** | Memorizing trajectory tokens | Trajectory-level grouped splitting | `test_dataset_trajectory_disjointness` | PASS |
| **Recurrent States** | Cross-run memory pollution | Mandatory `reset_memory()` lifecycle | `test_memory_isolation_between_runs` | PASS |

**Final Recommendation**: The AgentGuard research architecture satisfies all formal temporal and statistical isolation criteria. No methodological leakage exists.
