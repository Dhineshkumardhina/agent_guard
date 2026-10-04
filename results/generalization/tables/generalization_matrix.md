# Generalization Experiment Matrix

| Experiment | Dimension | In-Distribution (Train / Val) | Out-of-Distribution (Test-OOD) | Models Evaluated |
|:---|:---|:---|:---|:---|
| `G1` | agent_count | [3, 5] agents | [8] agents | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G2` | agent_count | [3, 5, 8] agents | [12] agents | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G3` | topology | pipeline, star, mesh | custom | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G3b` | topology | star, mesh, custom | pipeline | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G4` | task | research, coding, planning | analysis | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G4b` | task | coding, analysis, planning | research | Temporal GNN, Classical ML, Sequence, Static GNN |
| `G5` | failure_type | Seen Faults + Nominal | Held-Out Unseen Faults | Temporal GNN, Classical ML, Sequence, Static GNN |
