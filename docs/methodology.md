# Methodology and Technical Architecture

## 1. System Pipeline Overview
The AgentGuard architecture processes raw agent executions into continuous temporal graph streams, computing risk probabilities and early warning alerts in real time:

```
  +-----------------------------------------------------------+
  |              Controlled Multi-Agent Environment           |
  |  (Agents: Planner, Researcher, Analyst, Coder, Verifier)  |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |             Structured Telemetry Event Collector          |
  | (Timestamps, latencies, tokens, contradictions, retries)  |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |                Temporal Graph Builder G(t)                |
  |  - Node features: Rolling error, latency, confidence      |
  |  - Edge features: Message delta, reciprocity, semantics   |
  +-----------------------------+-----------------------------+
                                |
                +---------------+---------------+
                |                               |
                v                               v
  +---------------------------+   +---------------------------+
  |    Non-Graph Baselines    |   |    Temporal Graph Model   |
  |  - Rule-based Thresholds  |   |  - Continuous-Time TGN    |
  |  - Logistic Regression    |   |  - Dynamic Edge Memory    |
  |  - XGBoost & Random Forest|   |  - Temporal Graph Attention|
  |  - Sequence LSTM / GRU    |   +-------------+-------------+
  +-------------+-------------+                 |
                |                               |
                +---------------+---------------+
                                |
                                v
  +-----------------------------------------------------------+
  |                   Early Warning Engine                    |
  |        P(Failure within K events | Observed History)      |
  |    Status: NORMAL -> WATCH -> HIGH RISK -> CASCADE        |
  +-----------------------------+-----------------------------+
                                |
                                v
  +-----------------------------------------------------------+
  |                 Explainability Engine                     |
  |  (Attribution, critical interaction edges, lead times)   |
  +-----------------------------------------------------------+
```

---

## 2. Temporal Graph Construction
At each interaction event $e_t = (u, v, t, x_e)$:
1. Update directed edge $(u, v)$ in dynamic adjacency with feature vector $x_e(t)$.
2. Update temporal memory state for sender $u$ and receiver $v$.
3. Maintain sliding-window statistics for edge reciprocity, message frequency, and error propagation.

---

## 3. Strict Temporal Isolation (Zero Future Leakage)
To prevent temporal data contamination:
* The feature representation at step $t$ depends strictly on events up to $t$: $\mathcal{H}_t = \{e_1, \dots, e_t\}$.
* Ground truth labels $Y(t, K)$ evaluate the occurrence of Level 3 failures within the window $[t+1, t+K]$.
* In no circumstance does any model access future events during feature extraction.
* Automated leakage check assertions (`verify_no_future_leakage`) are integrated into every test and training batch generator.
