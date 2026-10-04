# Representative Case Study Overview

| Case Archetype | Run ID | Topology | Task | Pred Risk | Pred Class | Ground Truth | Top Agent | Top Interaction |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| True Positive Early | `run_0031_agentguard_generalization_v1` | Custom | Planning | 0.8423 | 1 | 1 | Planner | None |
| True Positive Late | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | 1 | 1 | Analyst | analyst_1->verifier_1 |
| False Positive | `run_0011_agentguard_generalization_v1` | Custom | Planning | 0.5860 | 1 | 0 | Analyst | analyst_1->verifier_1 |
| False Negative | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | 1 | 1 | Analyst | analyst_1->verifier_1 |
| High Risk No Cascade | `run_0013_agentguard_generalization_v1` | Star | Coding | 0.5860 | 1 | 0 | Analyst | planner_1->analyst_1 |
| Low Risk Success | `run_0003_agentguard_generalization_v1` | Custom | Planning | 0.5860 | 1 | 1 | Analyst | analyst_1->verifier_1 |
