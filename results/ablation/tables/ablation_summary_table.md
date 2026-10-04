# Ablation Performance Summary (Horizon K=1, Seed 42)

| Ablation | Removed Component | Precision | Recall | F1 Score | AUROC | AUPRC | FPR | Lead Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Temporal GNN | None (Reference) | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Temporal Information | Continuous Time Encoding | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Graph Structure | Graph Topology | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Node Features | Node Behavioral Telemetry | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Edge Features | Edge Behavioral Attributes | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Temporal Memory | Persistent Node Memory (m_v) | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Interaction Frequency | Communication Frequency Metrics | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Contradiction Information | Contradiction / Conflict Signals | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Confidence Information | Confidence Telemetry | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
| No Failure History | Recent Failure / Error History | 0.000 | 0.000 | 0.000 | 0.500 | 0.827 | 0.000 | 0.00s |
