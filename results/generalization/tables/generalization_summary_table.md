# Generalization Performance Summary (Temporal GNN, Horizon K=1, Seed 42)

| Experiment | Dimension | Split Type | Precision | Recall | F1 Score | AUROC | AUPRC | FPR | Lead Time (s) |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| G1 | agent_count | In Distribution | 0.056 | 1.000 | 0.105 | 0.500 | 0.056 | 1.000 | 0.00s |
| G1 | agent_count | Out Of Distribution | 0.528 | 1.000 | 0.691 | 0.500 | 0.528 | 1.000 | 0.00s |
| G2 | agent_count | In Distribution | 0.488 | 1.000 | 0.656 | 0.500 | 0.488 | 1.000 | 0.00s |
| G2 | agent_count | Out Of Distribution | 0.801 | 1.000 | 0.889 | 0.500 | 0.801 | 1.000 | 0.00s |
| G3 | topology | In Distribution | 0.488 | 1.000 | 0.656 | 0.500 | 0.488 | 1.000 | 0.00s |
| G3 | topology | Out Of Distribution | 0.801 | 1.000 | 0.889 | 0.500 | 0.801 | 1.000 | 0.00s |
| G3b | topology | In Distribution | 0.566 | 1.000 | 0.723 | 0.500 | 0.566 | 1.000 | 0.00s |
| G3b | topology | Out Of Distribution | 0.444 | 1.000 | 0.615 | 0.500 | 0.444 | 1.000 | 0.00s |
| G4 | task | In Distribution | 0.816 | 1.000 | 0.899 | 0.500 | 0.816 | 1.000 | 0.00s |
| G4 | task | Out Of Distribution | 0.528 | 1.000 | 0.691 | 0.500 | 0.528 | 1.000 | 0.00s |
| G4b | task | In Distribution | 0.566 | 1.000 | 0.723 | 0.500 | 0.566 | 1.000 | 0.00s |
| G4b | task | Out Of Distribution | 0.444 | 1.000 | 0.615 | 0.500 | 0.444 | 1.000 | 0.00s |
| G5 | failure_type | In Distribution | 0.604 | 1.000 | 0.753 | 0.500 | 0.604 | 1.000 | 0.00s |
| G5 | failure_type | Out Of Distribution | 0.867 | 1.000 | 0.929 | 0.500 | 0.867 | 1.000 | 0.00s |
