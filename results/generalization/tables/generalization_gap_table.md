# Generalization Gap Summary (Temporal GNN, Horizon K=1)

| Experiment | Dimension | Metric | In-Dist (ID) | Out-of-Dist (OOD) | Gap (ID - OOD) | % Degradation | 95% CI | p-value | Interpretation |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| G1 | agent_count | F1 | 0.105 | 0.691 | -0.586 | +556.1% | [-0.820, -0.192] | 0.013 | Negative transfer |
| G2 | agent_count | F1 | 0.656 | 0.889 | -0.234 | +35.7% | [-0.610, +0.059] | 0.120 | Negative transfer |
| G3b | topology | F1 | 0.723 | 0.615 | +0.107 | -14.8% | [-0.179, +0.400] | 0.440 | Moderate degradation |
| G3 | topology | F1 | 0.656 | 0.889 | -0.234 | +35.7% | [-0.610, +0.059] | 0.120 | Negative transfer |
| G4b | task | F1 | 0.723 | 0.615 | +0.107 | -14.8% | [-0.179, +0.400] | 0.440 | Moderate degradation |
| G4 | task | F1 | 0.899 | 0.691 | +0.208 | -23.1% | [+0.023, +0.461] | 0.027 | Moderate degradation |
| G5 | failure_type | F1 | 0.753 | 0.929 | -0.176 | +23.3% | [-0.569, +0.029] | 0.100 | Negative transfer |
