# Table 3: Overall Failure Prediction Performance (Horizon K=1, N=35 Test Population)

| Paradigm Family | Model Architecture | Precision | Recall | F1 Score | AUROC | AUPRC | False Positive Rate | Brier Score | Expected Calibration Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical ML** | Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.004 | 0.017 |
| **Classical ML** | Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.001 | 0.016 |
| **Classical ML** | XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.001 | 0.025 |
| **Sequence Model** | GRU | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.029 | 0.105 |
| **Static GNN** | GCN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.013 | 0.092 |
| **Static GNN** | GAT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.035 | 0.174 |
| **Sequence Model** | LSTM | 1.000 | 0.923 | 0.960 | 1.000 | 1.000 | 0.000 | 0.049 | 0.158 |
| **Temporal GNN** | Temporal GNN (Core Model) | 1.000 | 0.538 | 0.700 | 0.923 | 0.995 | 0.000 | 0.258 | 0.440 |
| **Rule-Based** | Rule-Based Thresholds | 1.000 | 0.154 | 0.267 | 1.000 | 1.000 | 0.000 | 0.531 | 0.699 |

*Note: Evaluated on strictly identical test sample partitions in `agentguard_dataset_v1` using frozen validation thresholds $\theta^* \in [0.10, 0.90]$.*
