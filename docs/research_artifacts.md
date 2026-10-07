# Research Artifact Index

## Overview

This index establishes end-to-end traceability between each Research Question (RQ), experimental evaluation, dataset version, model family, empirical result, figure, table, and written report.

Researchers can use this mapping to trace every claim in the project back to its underlying code, data, and experimental artifact.

---

## 1. Master Research Question Traceability Map

| RQ | Research Scope | Experiment Script | Dataset | Evaluated Models | Key Tables | Generated Figures | Formal Report |
|---|---|---|---|---|---|---|---|
| **RQ1** | Predictive Utility of Agent-Level Telemetry | `scripts/train_classical_baselines.py`, `scripts/compare_baselines.py` | `agentguard_dataset_v1` | Logistic Regression, Random Forest, XGBoost | `Table A (K=1)`, `results/evaluation/metrics/` | `roc_curves.png`, `pr_curves.png`, `calibration_curves.png` | `results/evaluation/research_evaluation_report.md` |
| **RQ2** | Value of Relational Graph Topology | `scripts/train_static_gnn.py`, `scripts/compare_baselines.py` | `agentguard_dataset_v1` | GCN, GAT vs. XGBoost, RF | `Table A (K=1)`, `Table I (Bootstrap Diff)` | `subgroup_topology.png`, `roc_curves.png` | `results/evaluation/research_evaluation_report.md` |
| **RQ3** | Continuous-Time Temporal Interaction Dynamics | `scripts/train_temporal_gnn.py`, `scripts/train_sequence_baselines.py` | `agentguard_dataset_v1` | Temporal GNN vs. LSTM, GRU, Static GNN | `Table A (K=1)`, `Table I (p-values)` | `f1_vs_horizon.png`, `auprc_vs_horizon.png` | `results/evaluation/research_evaluation_report.md` |
| **RQ4** | Early Warning Lead Time for Cascading Failures | `scripts/run_full_evaluation.py` | `agentguard_dataset_v1` | All 9 Model Families across $K \in \{1, 3, 5, 10\}$ | `Table C (Lead Time Summary)` | `lead_time_distributions.png`, `f1_vs_horizon.png` | `results/evaluation/research_evaluation_report.md` |
| **RQ5** | Generalization Under Distribution Shifts | `scripts/run_generalization_experiments.py` | `agentguard_generalization_v1` | Temporal GNN, Baselines across G1–G5 | `results/generalization/tables/generalization_summary_table.md` | `generalization_gap_plots.png`, `id_vs_ood_performance.png`, `agent_count_curves.png` | `results/generalization/generalization_report.md` |
| **RQ6** | Component Attribution & Information Decomposition | `scripts/run_ablation_study.py`, `scripts/run_explainability.py` | `agentguard_dataset_v1`, `agentguard_generalization_v1` | 9 Ablated GNN Variants, Multi-Level Explainers | `results/ablation/tables/ablation_summary_table.md`, `results/explainability/tables/` | `component_contribution_summary.png`, `global_feature_importance.png`, `risk_trajectory_cases.png` | `results/ablation/ablation_report.md`, `results/explainability/explainability_report.md` |

---

## 2. Artifact Locations and Directory Index

### Benchmark Evaluation Artifacts (`results/evaluation/`)
* **Primary Report:** `results/evaluation/research_evaluation_report.md`
* **Performance Tables:**
  - Overall Performance (Table A, $K=1$): Section 3 of evaluation report
  - Incident Lead Time Summary (Table C): Section 5 of evaluation report
  - Trajectory Bootstrap Hypothesis Tests (Table I): Section 6 of evaluation report
* **Figures & Plots (`results/evaluation/plots/`):**
  - `roc_curves.png`: Receiver Operating Characteristic curves across all models.
  - `pr_curves.png`: Precision-Recall curves highlighting low-prevalence behavior.
  - `calibration_curves.png`: Reliability diagrams and ECE calibration plots.
  - `f1_vs_horizon.png`: F1 performance degradation across horizons $K \in \{1, 3, 5, 10\}$.
  - `auprc_vs_horizon.png`: AUPRC stability across horizons.
  - `lead_time_distributions.png`: Lead time distributions for incident-level warnings.
  - `subgroup_topology.png`: Detection performance broken down by topology.
  - `subgroup_task.png`: Detection performance broken down by benchmark task.
  - `subgroup_failure_level.png`: Performance broken down by failure levels 1, 2, and 3.

### Ablation Study Artifacts (`results/ablation/`)
* **Primary Report:** `results/ablation/ablation_report.md`
* **Summary Table:** `results/ablation/tables/ablation_summary_table.md`
* **Component Matrix:** `results/ablation/tables/ablation_component_matrix.md`
* **Figures & Plots (`results/ablation/plots/`):**
  - `component_contribution_summary.png`: Marginal delta-F1 rankings across ablations.
  - `ablation_performance_bars.png`: Comparative bar chart of F1, AUROC, and AUPRC.
  - `lead_time_comparison.png`: Lead time impact when temporal memory is removed.
  - `pr_curves_ablation.png`: Precision-recall curves across all 9 ablated models.
  - `seed_variability.png`: Performance stability across multiple initializations.

### Generalization Study Artifacts (`results/generalization/`)
* **Primary Report:** `results/generalization/generalization_report.md`
* **Summary Table:** `results/generalization/tables/generalization_summary_table.md`
* **Shift Matrix:** `results/generalization/tables/generalization_matrix.md`
* **Figures & Plots (`results/generalization/plots/`):**
  - `generalization_gap_plots.png`: Measured $\Delta \text{F1}$ gaps across 4 dimensions.
  - `id_vs_ood_performance.png`: In-distribution vs. out-of-distribution F1 bars.
  - `agent_count_curves.png`: Performance scaling curves as agents increase to 8 and 12.
  - `topology_comparison.png`: Cross-topology transfer performance (Pipeline vs. Custom).
  - `task_comparison.png`: Cross-task transfer performance (Planning vs. Analysis).
  - `failure_type_heatmap.png`: Sensitivity matrix across unseen held-out fault types.

### Explainability Study Artifacts (`results/explainability/`)
* **Primary Report:** `results/explainability/explainability_report.md`
* **Case Study Profiles:** `results/explainability/case_studies/` (6 archetypes)
* **Figures & Plots (`results/explainability/plots/`):**
  - `global_feature_importance.png`: Top-ranked behavioral features (Contradiction, Retries, Latency).
  - `agent_importance_distribution.png`: Attribution distribution across agent roles.
  - `interaction_edge_importance.png`: Attribution matrix across directed communication channels.
  - `temporal_event_contributions.png`: Attribution weights across chronological events.
  - `perturbation_sensitivity_analysis.png`: Counterfactual $\Delta P(F)$ sensitivity responses.
  - `risk_trajectory_cases.png`: Step-by-step risk trajectory lines for case archetypes.
  - `explanation_stability.png`: Stability correlation across seeds ($\rho = 0.916$).
