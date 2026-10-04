# AgentGuard: Research Ablation Study Plan (Phase 13)

## 1. Primary Research Question & Objective

> **Central Research Question**: *Which information sources and architectural components contribute to early prediction of cascading failures in multi-agent AI systems?*

In Phases 7–12, five paradigm families were evaluated:
- Rule-Based Baseline
- Family A: Classical Machine Learning (Logistic Regression, Random Forest, XGBoost)
- Family B: Temporal Sequence Models (LSTM, GRU)
- Family C: Static Graph Neural Networks (GCN, GAT)
- Family D: Temporal Graph Neural Network (TGN-style Core Model)

Phase 12 established baseline comparative metrics across families, noting that both agent-level behavioral aggregates (XGBoost) and temporal dynamic graphs (Temporal GNN) achieve strong predictive capability.

Phase 13 isolates and quantifies the exact contribution of each component within the Temporal GNN architecture.

---

## 2. Baseline Configuration (Full Temporal GNN)

The reference benchmark model is the complete **TemporalGraphFailurePredictor**:
- **Continuous Fourier Time Encoding**: $\phi(\Delta t) \in \mathbb{R}^{16}$ mapping elapsed inter-interaction intervals to sinusoidal embeddings.
- **Dynamic Node Memory**: $m_v(t) \in \mathbb{R}^{64}$ with GRU-based state transitions updated per chronological interaction event.
- **Temporal Neighborhood Attention**: Multi-head attention over recent neighbor interactions $k \le 10$.
- **Graph Topology & Edge Features**: 10-dimensional interaction vectors $e_{uv}(t)$.
- **Node Behavioral Features**: 14-dimensional per-agent telemetry vectors $x_v(t)$.
- **Multi-Pooling Readout**: Concat(Mean, Max) $\in \mathbb{R}^{128}$ pooling node states into a system-level representation.
- **Classification Head**: MLP mapping graph embedding to impending failure probability.
- **Threshold Policy**: Frozen threshold $\theta^*$ calibrated strictly on the validation set ($\theta^* \in [0.10, 0.90]$). No test-set tuning.

---

## 3. Systematic Ablation Suite

| ID | Ablation Code | Component Manipulated | Specific Intervention | Targeted Research Question |
|:---|:---|:---|:---|:---|
| **Full** | `full_temporal_gnn` | None (Reference) | All architectural mechanisms active | Reference ceiling performance |
| **A** | `no_temporal_info` | Continuous Time Encoding | Time encodings $\phi(\Delta t) = \mathbf{0}$, $\Delta t = 0$ | Does elapsed continuous time contribute beyond graph sequence order? |
| **B** | `no_graph_structure` | Interaction Graph Connectivity | Graph edge passing disabled; isolated per-agent temporal processing | Does graph topology provide predictive value beyond agent telemetry? |
| **C** | `no_node_features` | Node Behavioral Features | Node feature vectors $x_v(t) = \mathbf{0}$ | Can interaction topology alone detect cascading failures without local node metrics? |
| **D** | `no_edge_features` | Edge Behavioral Features | Edge feature vectors $e_{uv}(t) = \mathbf{0}$ | Do communication edge attributes contribute beyond network connectivity? |
| **E** | `no_temporal_memory`| Persistent Node Memory | Memory updates disabled; $m_v(t) = \mathbf{0}$ | What is the contribution of recurrent state persistence across event sequences? |
| **F** | `no_interaction_freq`| Interaction Frequency | Zero out message counts, event counts, interaction rates | Does communication intensity/traffic volume predict system breakdown? |
| **G** | `no_contradiction` | Contradiction Signals | Zero out contradiction rate and semantic dissonance features | Does conflicting inter-agent communication serve as an early warning signal? |
| **H** | `no_confidence` | Agent Confidence Metrics | Zero out average confidence and confidence degradation features | Does agent self-reported confidence correlate with system failure? |
| **I** | `no_failure_history`| Recent Failure Indicators | Zero out recent error, retry, and failure counts | Is the model learning proactive predictive dynamics or reactive failure detection? |

---

## 4. Controlled Variables & Experimental Protocol

To guarantee experimental integrity:
1. **Identical Dataset Split**:
   - `data/processed/agentguard_dataset_v1/train.parquet` & `graph_sequences_train.jsonl`
   - `data/processed/agentguard_dataset_v1/val.parquet` & `graph_sequences_val.jsonl`
   - `data/processed/agentguard_dataset_v1/test.parquet` & `graph_sequences_test.jsonl`
2. **Identical Test Population**:
   - Every ablated model is evaluated on the exact same test trajectories and prediction samples used in Phase 12.
3. **No Test-Set Tuning**:
   - Decision thresholds $\theta^*$ are selected strictly on the validation partition via F1-optimization.
   - Test data is strictly held-out until frozen evaluation.
4. **Prediction Horizons**:
   - Evaluated across $K \in \{1, 3, 5, 10, 20\}$ steps ahead.
5. **Multiple Random Seeds**:
   - Seeds: $\{42, 123, 456, 789, 2026\}$.
   - Models are retrained and evaluated across seeds to measure mean, standard deviation, and stability.

---

## 5. Evaluation Metrics

For every ablation experiment, the framework computes:
- **Classification Performance**: Precision, Recall, F1 Score, AUROC, AUPRC, False Positive Rate (FPR), False Alarm Rate (FAR).
- **Incident Early Warning**: Mean Lead Time, Median Lead Time, Successful Early Warnings, Warnings Per Trajectory, Detection Coverage.
- **Probabilistic Calibration**: Brier Score, Expected Calibration Error (ECE).

---

## 6. Statistical Comparison Methodology

To compare each ablated model $M_{\text{ablated}}$ against the full model $M_{\text{full}}$:
1. **Paired Trajectory Difference**:
   $$\Delta \text{Metric} = \text{Metric}(M_{\text{ablated}}) - \text{Metric}(M_{\text{full}})$$
2. **Trajectory Block Bootstrap**:
   - Resampling unit is the simulation trajectory (`run_id`), preserving within-run temporal correlations.
   - $B = 500$ bootstrap resamples to construct 95% empirical confidence intervals $[\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}]$.
3. **Hypothesis Testing**:
   - Two-sided empirical p-value tested against $\alpha = 0.05$.
   - Differences are declared statistically significant only when the 95% CI strictly excludes zero.

---

## 7. Ablation Matrix Specification

| Experiment | Temporal Info | Graph Topology | Node Features | Edge Features | Temporal Memory | Interaction Freq | Contradiction Info | Confidence Info | Failure History |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `full_temporal_gnn` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_temporal_info`  | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_graph_structure`| ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_node_features`  | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_edge_features`  | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `no_temporal_memory`| ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| `no_interaction_freq`| ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| `no_contradiction`  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| `no_confidence`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| `no_failure_history`| ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

---

## 8. Artifact Generation Plan

Outputs will be stored in `results/ablation/`:
- `metrics/`: `ablation_metrics.json` containing unified records for all experiments.
- `comparisons/`: `pairwise_ablation_comparisons.json` documenting $\Delta \text{F1}$, $\Delta \text{AUROC}$, $\Delta \text{Lead Time}$, 95% CIs, and p-values.
- `tables/`: `ablation_matrix.md`, `ablation_summary_table.md`, `seed_variability_table.md`.
- `plots/`:
  1. `ablation_performance_bars.png` (F1 by ablation)
  2. `metric_degradation.png` ($\Delta \text{F1}$ and $\Delta \text{AUROC}$)
  3. `horizon_wise_ablation.png` (Performance vs $K$)
  4. `seed_variability.png` (Boxplot across seeds)
  5. `lead_time_comparison.png` (Lead-time changes)
  6. `pr_curves_ablation.png` (Precision-Recall comparison)
  7. `auprc_comparison.png` (AUPRC bar chart)
  8. `component_contribution_summary.png` (Radar/waterfall chart of impact)
- `ablation_report.md`: Complete research report summarizing empirical findings.
