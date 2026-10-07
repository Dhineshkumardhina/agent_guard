# Labeling and Prediction Horizon Protocol

## Overview

A fundamental challenge in failure prediction is defining a scientifically rigorous, causal labeling policy that avoids temporal leakage and prevents models from trivially detecting post-mortem system crashes.

This document details AgentGuard's labeling policy, prediction horizon definitions, and sample exclusion rules.

---

## 1. Mathematical Label Formulation

Let $t$ denote the current evaluation step (cutoff step) within trajectory $r$, and let $K \ge 1$ denote the forward prediction horizon in interaction steps.

### Definition of Binary Target $Y_{t, K}$
The binary label $Y_{t, K} \in \{0, 1\}$ indicates whether a Level 3 cascading failure initiates or manifests strictly within the forward interval $(t, t+K]$:

$$Y_{t, K} = \begin{cases} 
1 & \text{if } \exists \tau \in (t, t + K] \text{ such that } \text{FailureLevel}(\tau) = 3 \\
0 & \text{otherwise}
\end{cases}$$

### Key Properties:
1. **Strictly Forward Window:** The evaluation interval $(t, t+K]$ is open on the left. A failure occurring at or before step $t$ ($\tau \le t$) **never** counts toward $Y_{t, K}$.
2. **Horizon Specificity:** A sample at cutoff $t=3$ may be labeled $Y_{3, 1} = 0$ (no failure at step 4) but $Y_{3, 5} = 1$ (failure occurs at step 7).

---

## 2. Prediction Horizons ($K \in \{1, 3, 5, 10, 20\}$)

AgentGuard evaluates model performance across five discrete prediction horizons:

| Horizon | Forward Steps | Operational Interpretation |
|---|---|---|
| **$K = 1$** | Immediate next step | **Immediate Alert:** Failure is imminent on the very next interaction handoff. |
| **$K = 3$** | Next 3 interaction steps | **Short-Range Warning:** Cascade has initiated upstream; early mitigation possible. |
| **$K = 5$** | Next 5 interaction steps | **Medium-Range Warning:** Multi-hop error is propagating; opportunity for agent reset or human check. |
| **$K = 10$** | Next 10 interaction steps | **Long-Range Forecast:** Early structural divergence; ample time for task rescheduling. |
| **$K = 20$** | Next 20 interaction steps | **Trajectory-Level Hazard:** Systemic architectural mismatch for long workflows. |

### Finite Trajectory Limitation for $K = 20$:
In `agentguard_dataset_v1`, simulation trajectories run for a maximum of 20 interaction steps. Consequently, samples evaluated at cutoff steps $t \ge 1$ have fewer than 20 remaining forward steps before natural trajectory termination. Therefore, $K=20$ contains 0 test instances in `v1`. This is documented as a known empirical dataset boundary rather than a model defect.

---

## 3. Negative Sample Generation

To train discriminative models, the dataset contains diverse negative instances ($Y_{t, K} = 0$):
1. **Nominal Runs:** Trajectories where no failure of any level is injected ($100\%$ negative samples).
2. **Contained Glitches (Level 1 / Level 2):** Trajectories where a tool timeout or local format error occurred but was recovered or halted, never escalating to a Level 3 cascade.
3. **Pre-Failure Calm Steps:** In failure-bound runs, steps that occur well before the cascade window ($t + K < t_{\text{fail}}$).

---

## 4. Post-Cascade Sample Exclusion Policy

A pervasive error in naive early-warning evaluations is including samples collected *after* a failure has already broken the system ($t \ge t_{\text{fail}}$). 

Post-failure states exhibit obvious trivial signals (crashed processes, infinite latency, 100% error rates). Classifying post-failure states as positive early warnings is invalid because the system has already collapsed.

### AgentGuard Exclusion Invariant:
$$\forall \text{ sample } s_t \in \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}, \quad t < t_{\text{first\_cascade}}$$

* Any step evaluated at or after the initiation of a Level 3 cascading failure is strictly discarded from the dataset.
* As a result, **100% of positive training and testing samples represent true early warnings**, compelling models to learn pre-failure structural signals rather than post-crash artifacts.
