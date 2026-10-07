# AgentGuard Research Publication Package

This directory contains the complete scientific manuscript, publication figures, formatted tables, supplementary materials, reference audits, and artifact generation scripts for:

**AgentGuard: Temporal Interaction-Graph-Based Prediction of Failures in Multi-Agent AI Systems**

---

## 1. Directory Structure

```
paper/
├── README.md                              # This navigation and reproducibility guide
├── agentguard_paper.md                    # Complete 18-section research paper manuscript
├── claim_audit.md                         # Claim-level evidence audit and verification status
├── figures/                               # Publication figures (10 high-resolution charts/diagrams)
│   ├── fig1_system_architecture.png       # End-to-end framework architecture
│   ├── fig2_temporal_interaction_example.png # Multi-agent message stream & dynamic graph
│   ├── fig3_failure_propagation_example.png # Cascade timeline & fault injection trace
│   ├── fig4_model_comparison_pr.png       # Precision-Recall curves across 9 models
│   ├── fig4_model_comparison_roc.png      # ROC curves across 9 baseline & neural models
│   ├── fig5_horizon_performance.png       # F1 and ROC-AUC degradation across horizons K
│   ├── fig6_early_warning_lead_time.png   # Lead time distribution boxplot (T_lead)
│   ├── fig7_ablation_contributions.png    # Ablation relative drops (F1 and AUROC)
│   ├── fig8_generalization_gaps.png       # Topology, agent-count, and task generalization gaps
│   ├── fig9_risk_trajectories.png         # Trajectory risk score progression vs step
│   └── fig10_interaction_edge_importance.png # Explainability GNNExplainer edge attributions
├── tables/                                # Publication tables in Markdown
│   ├── table1_dataset_configuration.md    # Multi-agent dataset splits and horizon counts
│   ├── table2_failure_taxonomy.md         # 5 fault categories, triggers, and propagation
│   ├── table3_model_comparison.md         # Benchmark results on test set across 9 models
│   ├── table4_horizon_wise_performance.md # Degradation across horizons K in {1, 3, 5, 10, 20}
│   ├── table5_early_warning_performance.md # Lead time, warning rate, false alarm rate
│   ├── table6_ablation_study.md           # 9 component ablation variants and relative delta
│   ├── table7_generalization_results.md   # Out-of-distribution evaluation results
│   ├── table8_failure_type_analysis.md    # Category-specific failure prediction performance
│   ├── table9_explainability_summary.md   # GNNExplainer and Integrated Gradients top factors
│   └── table10_reproducibility_environment.md # Hardware, OS, Python, and PyTorch environment
├── references/
│   └── citation_audit.md                  # Comprehensive verification of all 14 academic citations
└── supplementary/
    └── supplementary_materials.md         # Detailed hyperparameters, formulas, case studies
```

---

## 2. Origin of Experimental Results and Data

All experimental results presented in the manuscript and tables originate strictly from verified execution logs in the `results/` directory:

| Paper Artifact | Source Result Directory / File | Experiment Run ID |
| :--- | :--- | :--- |
| **Table 3** (Model Comparison) | `results/evaluation/evaluation_metrics.json` | `EXP-EVAL-001` |
| **Table 4** (Horizon-wise Performance) | `results/evaluation/horizon_metrics.json` | `EXP-EVAL-002` |
| **Table 5** (Early-Warning Performance) | `results/evaluation/lead_time_distribution.json` | `EXP-EVAL-003` |
| **Table 6** (Ablation Study) | `results/ablation/ablation_summary.json` | `EXP-ABL-001` through `EXP-ABL-009` |
| **Table 7** (Generalization) | `results/generalization/generalization_summary.json` | `EXP-GEN-001` through `EXP-GEN-004` |
| **Table 8** (Failure Type Breakdown) | `results/evaluation/subgroup_metrics.json` | `EXP-EVAL-004` |
| **Table 9** (Explainability Summary) | `results/explainability/explainability_summary.json` | `EXP-EXP-001` through `EXP-EXP-005` |
| **Figures 4–10** | Generated from numerical metrics in above files via `scripts/prepare_paper_figures.py` | — |

---

## 3. How to Reproduce All Figures and Tables

To regenerate all figures and markdown tables directly from the experimental run outputs:

```powershell
# Activate the project virtual environment
.\.venv\Scripts\Activate.ps1

# Regenerate all publication figures (saved to paper/figures/)
python scripts/prepare_paper_figures.py

# Regenerate all publication tables (saved to paper/tables/)
python scripts/prepare_paper_tables.py
```

### Dependencies
- Python 3.11+
- `matplotlib >= 3.8.0`
- `numpy >= 1.26.0`
- `torch >= 2.0.0`
- `torch_geometric >= 2.5.0`

---

## 4. Scientific Integrity & Reporting Standards

1. **No Data Fabrication**: Every metric reported corresponds to actual executed test sets recorded during Phases 7–15.
2. **Empirical Caveats Reported Neutrally**:
   - Temporal GNN achieved F1 = 0.700 vs. XGBoost F1 = 1.000 on test set ($N=35$). The block-bootstrap significance test yielded $p = 0.080 > 0.05$. Therefore, the manuscript does not claim universal statistical superiority for the neural model.
   - Horizon $K=20$ yielded 0 evaluable test points because maximum trajectory length in `v1` was 20 steps. This is explicitly reported as a dataset horizon boundary.
3. **Non-Causal Explainability**: Feature and edge attribution scores are explicitly reported as internal model sensitivities, not ground-truth physical causal mechanisms.
4. **Verified Citations**: All 14 citations in `paper/references/citation_audit.md` and the manuscript bibliography are verified against official academic venues (NeurIPS, ICLR, ICML, ACM, IEEE).
