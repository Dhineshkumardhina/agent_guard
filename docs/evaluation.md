# Scientific Evaluation Protocol

## Overview

AgentGuard implements a rigorous evaluation methodology (`ml/evaluation/`) combining standard statistical classification metrics, probabilistic calibration diagnostics, incident-level early-warning lead time measurements, and non-parametric trajectory block bootstrap hypothesis testing.

---

## 1. Metrics and Their Scientific Significance

| Metric | Mathematical Definition | Why It Matters for Multi-Agent Failure Prediction |
|---|---|---|
| **AUPRC** (Area Under Precision-Recall Curve) | $\int_0^1 P(R) \, dR$ | **Primary Discrimination Metric:** Highly sensitive to performance on the minority positive class (cascading failures). Unlike AUROC, AUPRC does not present an artificially optimistic score under high class imbalance. |
| **AUROC** (Area Under ROC Curve) | $\int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})$ | Measures threshold-independent ranking ability (probability that a randomly chosen failure instance receives a higher risk score than a nominal instance). |
| **F1 Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall at the operational decision threshold $\theta^*$. Balances missed cascades against false alarms. |
| **Precision** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | Proportion of emitted failure warnings that correspond to true impending cascades. Prevents alert fatigue for human operators. |
| **Recall** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | Proportion of actual cascading failures successfully detected prior to onset. High recall ensures critical failures are not missed. |
| **Brier Score** | $\frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$ | Strictly proper scoring rule quantifying mean squared probability error. Penalizes overconfident wrong predictions. |
| **ECE** (Expected Calibration Error) | $\sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$ | Measures whether predicted risk probabilities reflect actual empirical frequencies (e.g., when model outputs $0.80$, do $80\%$ of those cases fail?). |
| **Lead Time** | $t_{\text{fail}} - t_{\text{warn}} \quad (\ge 0)$ | Operational utility metric. Quantifies how far in advance (in simulated seconds or interaction steps) an operator receives warning before system collapse. |
| **False Alarm Rate** | $\frac{\text{FP}}{\text{Nominal Trajectories}}$ | Average number of false alarms generated per non-failing workflow execution. |

---

## 2. Threshold Calibration Protocol

To prevent test-set optimization bias (a major source of inflated claims in ML literature):
1. **Validation Optimization:** The operating decision threshold $\theta^* \in [0.10, 0.90]$ is selected exclusively on the validation partition (`val.parquet`) by maximizing validation F1:
   $$\theta^* = \arg\max_{\theta \in [0.10, 0.90]} \text{F1}_{\text{val}}(\theta)$$
2. **Threshold Freezing:** The selected $\theta^*$ is frozen. No test predictions or labels are examined during threshold selection.
3. **Single Test Evaluation:** Test set predictions are binarized using $\theta^*$ exactly once to produce final test metrics.

---

## 3. Incident-Level Early Warning and Lead Time Protocol

For every simulation trajectory containing a Level 3 cascading failure at timestamp $t_{\text{fail}}$:

### Warning Eligibility Policy:
1. **Causal Pre-Failure Requirement:** A candidate alert emitted at timestamp $t_{\text{warn}}$ is valid if and only if:
   $$t_{\text{warn}} \le t_{\text{fail}}$$
2. **First-Hit Matching:** The **earliest** valid alert emitted prior to $t_{\text{fail}}$ defines the incident lead time:
   $$\Delta t_{\text{lead}} = t_{\text{fail}} - t_{\text{earliest\_warn}}$$
3. **Subsequent Warnings:** Additional alerts emitted after the first warning but before $t_{\text{fail}}$ do not artificially inflate the lead time metric.
4. **Post-Failure Disqualification:** Alerts emitted after the failure has already initiated ($t > t_{\text{fail}}$) are explicitly disqualified and receive 0 lead time.

---

## 4. Class Imbalance Handling

In realistic multi-agent workflows, catastrophic failures are relatively rare compared to nominal message passing.
- In `agentguard_dataset_v1`, the positive class ratio is $38.69\%$.
- In training, positive instances are re-weighted via `pos_weight = N_neg / N_pos` in binary cross-entropy loss functions.
- Evaluation emphasizes AUPRC and precision-recall curves over accuracy and raw AUROC.

---

## 5. Trajectory-Level Block Bootstrap Hypothesis Testing

Standard i.i.d. bootstrap resampling samples individual rows independently. Because events within the same simulation trajectory share temporal autocorrelation, standard bootstrapping violates independence assumptions and produces artificially narrow confidence intervals.

AgentGuard implements **Trajectory-Level Block Bootstrapping** (`ml/evaluation/uncertainty.py`):
1. **Resampling Unit:** The simulation run ID (`run_id`).
2. **Iterations:** $B = 500$ bootstrap iterations.
3. **Procedure:**
   - Draw $M$ run IDs with replacement from the test set of $M$ runs.
   - Aggregate all prediction samples belonging to the selected runs.
   - Recompute F1, AUROC, AUPRC, and lead time.
4. **Confidence Intervals:** 95% non-parametric empirical percentile intervals ($[2.5\%, 97.5\%]$).
5. **Pairwise Hypothesis Testing:** For model comparison (e.g., Temporal GNN vs. XGBoost), the paired delta $\Delta = \text{Metric}_A - \text{Metric}_B$ is computed on each bootstrap resample. The two-tailed empirical p-value tests whether $\Delta$ significantly differs from 0.

---

## 6. Empirical Results Summary (Phase 12 Benchmark, Horizon K=1)

Below are the empirical metrics measured on the standardized $N=35$ test population in `agentguard_dataset_v1`:

### Overall Performance (Horizon K=1)
| Model Family | Model | Precision | Recall | F1 Score | AUROC | AUPRC | Brier Score | ECE |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Classical ML | Logistic Regression | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.004 | 0.017 |
| Classical ML | Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 0.016 |
| Classical ML | XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.001 | 0.025 |
| Temporal Sequence | GRU | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.029 | 0.105 |
| Static GNN | GCN | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.013 | 0.092 |
| Static GNN | GAT | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.035 | 0.174 |
| Temporal Sequence | LSTM | 1.000 | 0.923 | 0.960 | 1.000 | 1.000 | 0.049 | 0.158 |
| Temporal GNN | Temporal GNN | 1.000 | 0.538 | 0.700 | 0.923 | 0.995 | 0.258 | 0.440 |
| Rule-Based | Rule-Based | 1.000 | 0.154 | 0.267 | 1.000 | 1.000 | 0.531 | 0.699 |

### Pairwise Bootstrap Hypothesis Tests (XGBoost vs Temporal GNN, K=1)
* **Point Difference ($\Delta \text{F1}$):** $+0.300$
* **95% Trajectory Bootstrap CI:** $[+0.000, +1.000]$
* **Empirical p-value:** $p = 0.080$ (Not statistically significant at $\alpha = 0.05$)
* **Scientific Conclusion:** The point performance difference between the models does not reach conventional statistical significance on the $N=35$ test sample population. Claims that Temporal GNN is universally superior to Classical ML are **unsupported** by current sample evidence.
