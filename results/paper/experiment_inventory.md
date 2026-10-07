# Experiment Inventory

## Overview

This inventory lists all empirical experiments executed and recorded in the AgentGuard repository across Phases 7 through 15. Every experiment listed corresponds to serialized model weights, per-step predictions, evaluation metric files, or explanation records on disk. No unexecuted or fabricated experiments are included.

---

## 1. Baseline Model Experiments (Phases 7–11)

| Experiment ID | Model Family | Model Name | Dataset Version | Seed | Horizons Evaluated | Configuration Details | Result Location | Status |
|---|---|---|---|:---:|:---:|---|---|:---:|
| `exp_rule_baseline_20261003_125321_22d666` | Rule-Based | Rule Detector | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10 | Heuristic thresholds: retries $\ge 3$, latency $\ge 2.5$s, contradiction $\ge 0.70$ | `results/baselines/rule_based/exp_rule_baseline_20261003_125321_22d666/` | **Completed** |
| `exp_classical_20261003_125823_3d61fc_lr` | Classical ML | Logistic Regression | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10, 20 | L2 penalty ($C=1.0$), lbfgs solver, 17 standardized agent features | `results/baselines/classical_ml/logistic_regression/exp_classical_20261003_125823_3d61fc/` | **Completed** |
| `exp_classical_20261003_125823_3d61fc_rf` | Classical ML | Random Forest | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10, 20 | 100 trees, max depth 10, balanced class weights | `results/baselines/classical_ml/random_forest/exp_classical_20261003_125823_3d61fc/` | **Completed** |
| `exp_classical_20261003_125823_3d61fc_xgb`| Classical ML | XGBoost | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10, 20 | 100 estimators, max depth 5, lr 0.05, subsample 0.8 | `results/baselines/classical_ml/xgboost/exp_classical_20261003_125823_3d61fc/` | **Completed** |
| `exp_seq_20261003_130744_446422_lstm_10` | Temporal Sequence | LSTM | `agentguard_dataset_v1` | 42 | 1, 3, 5 | 2-layer LSTM, hidden dim 64, dropout 0.2, sequence length 10 | `results/baselines/sequence/lstm/seq_10/` | **Completed** |
| `exp_seq_20261003_130744_446422_lstm_20` | Temporal Sequence | LSTM | `agentguard_dataset_v1` | 42 | 1, 3, 5 | 2-layer LSTM, hidden dim 64, dropout 0.2, sequence length 20 | `results/baselines/sequence/lstm/seq_20/` | **Completed** |
| `exp_seq_20261003_130744_446422_gru_10`  | Temporal Sequence | GRU | `agentguard_dataset_v1` | 42 | 1, 3, 5 | 2-layer GRU, hidden dim 64, dropout 0.2, sequence length 10 | `results/baselines/sequence/gru/seq_10/` | **Completed** |
| `exp_seq_20261003_130744_446422_gru_20`  | Temporal Sequence | GRU | `agentguard_dataset_v1` | 42 | 1, 3, 5 | 2-layer GRU, hidden dim 64, dropout 0.2, sequence length 20 | `results/baselines/sequence/gru/seq_20/` | **Completed** |
| `exp_static_gnn_20261003_185027_gcn`      | Static GNN | GCN | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10 | 2-layer GCNConv, hidden dim 64, global mean pooling readout | `results/baselines/static_gnn/gcn/` | **Completed** |
| `exp_static_gnn_20261003_185027_gat`      | Static GNN | GAT | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10 | 2-layer GATConv, 4 attention heads, global mean pooling readout | `results/baselines/static_gnn/gat/` | **Completed** |
| `exp_temporal_gnn_20261003_190957`        | Temporal GNN | Temporal GNN (Core Model) | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10 | Continuous Fourier time encoding $\phi(\Delta t)$, GRU memory $\mathbf{m}_v$, temporal attention | `results/baselines/temporal_gnn/` | **Completed** |

---

## 2. Comprehensive Evaluation Benchmark (Phase 12)

| Experiment ID | Evaluation Target | Dataset Version | Seed | Prediction Horizons | Key Analysis Slices | Result Location | Status |
|---|---|---|:---:|:---:|---|---|:---:|
| `eval_full_comparison_phase12` | Comparative Benchmark of All 9 Models | `agentguard_dataset_v1` | 42 | 1, 3, 5, 10 | Overall metrics, lead time, failure levels (L0, L3), topologies (Pipeline, Custom), tasks (Planning, Research) | `results/evaluation/reports/` | **Completed** |
| `eval_pairwise_bootstrap_k1` | Paired Trajectory Block Bootstrap | `agentguard_dataset_v1` | 42 | 1 | $B=500$ resamples at run level, 95% CIs, paired permutation tests | `results/evaluation/comparisons/pairwise_comparisons_k1.json` | **Completed** |

---

## 3. Systematic Ablation Study (Phase 13)

Evaluated on `agentguard_dataset_v1` at horizon $K=1$, Seed 42:

| Experiment ID | Ablation Condition | Removed Component | Preserved Subsystems | Result Location | Status |
|---|---|---|---|---|:---:|
| `abl_full_reference` | Full Model Reference | *None* | Complete Temporal GNN architecture | `results/ablation/predictions/full_temporal_gnn.json` | **Completed** |
| `abl_no_temporal_info`| No Temporal Info | Continuous Time Encoding $\phi(\Delta t)$ | Dynamic memory, graph topology, node/edge features | `results/ablation/predictions/no_temporal_info.json` | **Completed** |
| `abl_no_graph_structure`| No Graph Structure | Inter-Agent Graph Message Passing | Node memory, time encoding, node/edge features | `results/ablation/predictions/no_graph_structure.json` | **Completed** |
| `abl_no_node_features`| No Node Features | Node Behavioral Telemetry $\mathbf{X}_V$ | Dynamic memory, time encoding, edge attributes | `results/ablation/predictions/no_node_features.json` | **Completed** |
| `abl_no_edge_features`| No Edge Features | Edge Attributes $\mathbf{X}_E$ | Dynamic memory, time encoding, node features | `results/ablation/predictions/no_edge_features.json` | **Completed** |
| `abl_no_temporal_memory`| No Temporal Memory | Persistent Memory Bank $\mathbf{m}_v(t)$ | Time encoding, graph topology, node/edge features | `results/ablation/predictions/no_temporal_memory.json` | **Completed** |
| `abl_no_interaction_freq`| No Interaction Frequency | Communication Rate Metrics | Dynamic memory, time encoding, remaining features | `results/ablation/predictions/no_interaction_freq.json` | **Completed** |
| `abl_no_contradiction`| No Contradiction Info | Semantic Contradiction Score | Dynamic memory, time encoding, remaining features | `results/ablation/predictions/no_contradiction.json` | **Completed** |
| `abl_no_confidence`| No Confidence Info | Self-Reported Confidence Telemetry | Dynamic memory, time encoding, remaining features | `results/ablation/predictions/no_confidence.json` | **Completed** |
| `abl_no_failure_history`| No Failure History | Historical Error & Retry Counts | Proactive interaction dynamics, memory, time encoding | `results/ablation/predictions/no_failure_history.json` | **Completed** |

---

## 4. Generalization & Robustness Suite (Phase 14)

Evaluated on `agentguard_generalization_v1` (72 runs, 1,098 samples) at horizon $K=1$, Seed 42:

| Experiment ID | Shift Dimension | In-Distribution (Training / Val) | Out-of-Distribution (Held-Out Test) | Evaluated Architectures | Result Location | Status |
|---|---|---|---|---|---|:---:|
| `gen_g1_agent_count_8` | Population Scaling | Agents $\in \{3, 5\}$ | Agents $= 8$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g2_agent_count_12`| Population Scaling | Agents $\in \{3, 5, 8\}$ | Agents $= 12$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g3_topology_custom`| Topology Shift | Topologies $\in \{\text{Pipeline, Star, Mesh}\}$ | Topology $= \text{Custom}$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g3b_topology_pipe` | Topology Shift | Topologies $\in \{\text{Star, Mesh, Custom}\}$ | Topology $= \text{Pipeline}$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g4_task_analysis`  | Task Domain Shift | Tasks $\in \{\text{Research, Coding, Planning}\}$ | Task $= \text{Analysis}$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g4b_task_research` | Task Domain Shift | Tasks $\in \{\text{Coding, Analysis, Planning}\}$ | Task $= \text{Research}$ | Temporal GNN | `results/generalization/predictions/` | **Completed** |
| `gen_g5_unseen_faults`  | Failure Mode Transfer | Seen training faults | Held-out unseen faults | Temporal GNN | `results/generalization/predictions/` | **Completed** |

---

## 5. Explainability & Attribution Suite (Phase 15)

Evaluated on `agentguard_generalization_v1`, Horizon $K=1$:

| Experiment ID | Attribution Level | Method / Analytical Protocol | Target Instances | Result Location | Status |
|---|---|---|---|---|:---:|
| `xai_level1_feature_imp` | Level 1: Global Feature Importance | Random Forest Gini + XGBoost Gain + Permutation Importance | Full population | `results/explainability/tables/global_feature_importance.md` | **Completed** |
| `xai_level3_agent_attr` | Level 3: Agent Attribution | Gradient $\times$ Input & Masking Attribution per Agent Node | Multi-agent runs | `results/explainability/tables/agent_role_importance.md` | **Completed** |
| `xai_level4_channel_attr`| Level 4: Directed Channel Attribution | Edge Masking Sensitivity on Directed Message Channels ($u \to v$) | Multi-agent runs | `results/explainability/plots/interaction_edge_importance.png`| **Completed** |
| `xai_level5_event_attr` | Level 5: Temporal Event Attribution | Chronological Event Masking Sensitivity ($e \le t$) | Trajectory events | `results/explainability/plots/temporal_event_contributions.png`| **Completed** |
| `xai_counterfactual_sens`| Counterfactual Sensitivity | Controlled feature perturbations ($-50\%$ contradiction, etc.) | Test samples | `results/explainability/plots/perturbation_sensitivity_analysis.png` | **Completed** |
| `xai_case_studies` | Representative Case Archetypes | In-depth diagnostic profiling across 6 distinct archetypes | 6 selected runs | `results/explainability/case_studies/` | **Completed** |
| `xai_attribution_stability`| Attribution Stability | Spearman rank correlation & Top-5 Jaccard across random seeds | Multi-seed runs | `results/explainability/plots/explanation_stability.png` | **Completed** |
