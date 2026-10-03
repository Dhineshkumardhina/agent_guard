# Experiment Protocol and Scientific Evaluation

## 1. Data Splitting Strategy
To prevent information leakage across steps of the same collaborative task:
* **Trajectory-Level Split:** All events from a specific execution trajectory $T_i$ are assigned exclusively to one partition:
  * **Train Set:** 70% of trajectories
  * **Validation Set:** 15% of trajectories (used for hyperparameter tuning & early stopping)
  * **Test Set:** 15% of trajectories (held-out evaluation)
* **Topology Generalization Split:** To test generalization across communication topologies, models are trained on Pipeline and Star configurations and tested on unseen Mesh and Custom configurations.

---

## 2. Evaluation Metrics

### 2.1 Standard Classification Metrics
* **AUROC (Area Under Receiver Operating Characteristic):** Discrimination ability across all probability thresholds.
* **AUPRC (Area Under Precision-Recall Curve):** Critical for imbalanced datasets where cascading failures are rarer than normal interactions.
* **F1-Score, Precision, and Recall at Operating Threshold:** Calibrated using the validation split.

### 2.2 Temporal Early Warning Metrics
* **Early Warning Lead Time ($\Delta t_{\text{lead}}$):** Number of interaction steps between the first alert exceeding threshold $\tau$ and the actual system failure event $t_F$:
  $$\Delta t_{\text{lead}} = t_F - t_{\text{alert}}$$
* **Mean and Median Lead Time** across all true positive trajectories.
* **False Alarms per Trajectory:** Count of warning triggers in trajectories that successfully conclude without failure.

---

## 3. Ablation Experiments
1. **Ablation 1 (- Temporal):** Replace dynamic timestamps with static aggregated interaction frequencies.
2. **Ablation 2 (- Edge Features):** Remove contradiction score, message length, and latency from edges.
3. **Ablation 3 (- Node Memory):** Remove persistent agent memory vectors from the temporal graph network.
4. **Ablation 4 (- Graph Structure):** Treat interaction events as flat sequential time-series (LSTM).
