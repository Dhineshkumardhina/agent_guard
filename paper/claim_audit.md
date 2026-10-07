# Paper Claim Audit

## Overview

In accordance with scientific integrity guidelines, this audit evaluates every major assertion intended for inclusion in the AgentGuard research paper against the empirical evidence available in the repository.

Each claim is vetted against six criteria:
1. Is there experimental evidence?
2. Is the evidence from the held-out test set?
3. Is the test sample size sufficient for statistical significance?
4. Is the claim broader than the tested experiment?
5. Does the claim imply physical causality?
6. Does the claim imply universal generalization?

---

## 1. Claim Vetting and Audit Decisions

### Statement 1: "Agent-level behavioral features can predict impending failures."
* **Evidence:** XGBoost, Random Forest, and Logistic Regression achieve F1 = 1.000, AUROC = 1.000 on the standardized test partition at horizon $K=1$.
* **Sample Size / Test Set:** Tested on the 35-sample test partition across horizons in `agentguard_dataset_v1`.
* **Causality / Universality:** Does not imply causality or universal applicability; describes observed predictive correlation on the experimental benchmark.
* **Audit Decision:** **APPROVED.**

---

### Statement 2: "Continuous-Time Temporal GNN is universally superior to classical ML baselines."
* **Evidence:** In Phase 12 evaluations, XGBoost achieved F1 = 1.000 while Temporal GNN achieved F1 = 0.700. Paired trajectory block bootstrap testing yielded $\Delta \text{F1} = +0.300$ in favor of XGBoost with $p = 0.080 > 0.05$.
* **Audit Decision:** **REJECTED.** Universal superiority is not supported by the experimental evidence.
* **Mandated Correction:** The paper must explicitly reject universal superiority, report the paired bootstrap p-value ($p = 0.080$), and explain that Classical ML performs comparably or better on localized failures where telemetry signals are intense.

---

### Statement 3: "Relational graph structure aids in isolating multi-hop cascade propagation pathways."
* **Evidence:** Multi-level explainability (Level 3 agent attribution, Level 4 directed channel attribution) correctly identified primary delegation corridors (`Planner -> Researcher`, `Analyst -> Verifier`) corresponding to injected fault cascades.
* **Audit Decision:** **APPROVED** with qualification: Attributions reflect internal model sensitivity, not physical causality.

---

### Statement 4: "Architectural ablations prove that each GNN component is individually required on the test set."
* **Evidence:** On the frozen validation threshold evaluation checkpoint evaluated on $N=35$ test points, ablated variants exhibited F1 = 0.000 ($p = 1.000$) due to conservative thresholding, while ranking metric AUPRC remained at 0.827.
* **Audit Decision:** **REJECTED AS ABSOLUTE CLAIM.** The statistical test does not show significance ($p > 0.05$) on the small test split.
* **Mandated Correction:** Report the exact measured numbers transparently. State that while architectural design follows established continuous-time graph formulations, the test sample size ($N=35$) is underpowered to statistically separate marginal feature contributions.

---

### Statement 5: "The model generalizes across population scaling to 8 and 12 agents."
* **Evidence:** Temporal GNN trained on $N \in \{3, 5\}$ achieved OOD F1 = 0.691 on $N=8$ and OOD F1 = 0.889 on $N=12$ with Recall = 1.000 on `agentguard_generalization_v1`.
* **Causality / Universality:** Bounded strictly to tested agent counts $N \in \{8, 12\}$.
* **Audit Decision:** **APPROVED.**

---

### Statement 6: "The model generalizes to unseen, held-out failure modes."
* **Evidence:** On held-out fault modes in Phase 14 (`G5`), Temporal GNN achieved OOD F1 = 0.929 and OOD AUPRC = 0.867, verifying that it detects structural interaction breakdown rather than memorizing fault codes.
* **Audit Decision:** **APPROVED.**

---

### Statement 7: "Identified high-attribution agents caused the multi-agent failure."
* **Evidence:** Attribution and perturbation measures reflect algorithmic gradients and feature masking sensitivities; no physical intervention was executed in live production.
* **Audit Decision:** **REJECTED.** Implying physical causality violates epistemological boundaries.
* **Mandated Correction:** Enforce a strict non-causality disclaimer throughout the paper. All attributions must be framed as model sensitivity and statistical association.

---

## 2. Final Claim Audit Sign-Off

All claims appearing in `paper/agentguard_paper.md` conform to these audited boundaries. Exaggerated, unsubstantiated, or causal assertions have been eliminated.
