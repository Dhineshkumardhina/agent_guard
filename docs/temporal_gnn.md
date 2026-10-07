# Continuous-Time Temporal Graph Neural Network (Core Model)

## Overview

The central research model in AgentGuard is the **Continuous-Time Temporal Graph Failure Predictor** (`ml/baselines/temporal_gnn/models.py`). It formalizes multi-agent communication sessions as continuous-time dynamic interaction graphs (CTDGs) and tracks evolving agent memory states to forecast impending cascading failures.

---

## 1. Architectural Blueprint

```
Interaction Event e_k = (u, v, t_k, x_e)
  ├── Compute Elapsed Interval: Δt = t_k - t_{last}(u)
  ├── Time Encoding: φ(Δt) via Fourier Harmonics
  │
  ├── Dynamic Message Construction:
  │     msg_u = MLP([ m_u(t^-), m_v(t^-), x_e, φ(Δt) ])
  │     msg_v = MLP([ m_v(t^-), m_u(t^-), x_e, φ(Δt) ])
  │
  ├── Dynamic Memory Update:
  │     m_u(t) = GRUCell(m_u(t^-), msg_u)
  │     m_v(t) = GRUCell(m_v(t^-), msg_v)
  │
  ├── Dynamic Neighborhood Aggregation:
  │     Temporal Attention over recent interaction edges
  │
  ├── Node Temporal Embedding:
  │     z_i(t) = MLP([ m_i(t), x_i(t), h_{neigh}(i) ])
  │
  ├── Multi-Pooling Graph Readout:
  │     h_G(t) = [ MeanPool({z_i}), MaxPool({z_i}) ]
  │
  └── Hazard Prediction Head:
        P(F(t+K) = 1 | G(<=t)) = σ( MLP(h_G(t)) )
```

---

## 2. Mathematical Components

### 2.1 Continuous-Time Fourier Encoding (`time_encoding.py`)
Given elapsed time interval between consecutive interactions $\Delta t = t_{\text{current}} - t_{\text{previous}} \ge 0$, continuous sinusoidal Fourier temporal encoding maps $\Delta t$ into a $D_{\text{time}}$-dimensional dense representation ($D_{\text{time}} = 16$):

$$\phi_k(\Delta t) = \cos\left( \Delta t \cdot \omega_k + \psi_k \right), \quad k = 1, \dots, D_{\text{time}} / 2$$
$$\phi_{k + D/2}(\Delta t) = \sin\left( \Delta t \cdot \omega_k + \psi_k \right)$$

where frequencies $\omega_k$ are logarithmically spaced:
$$\omega_k = \frac{1}{10000^{2k / D_{\text{time}}}}$$

This guarantees smooth mathematical representation of interaction delays, rapid bursts, and temporal idle periods.

### 2.2 Persistent Node Memory Bank (`memory.py`)
Each participating agent $v \in \mathcal{V}$ maintains an internal dynamic memory vector:
$$\mathbf{m}_v(t) \in \mathbb{R}^{d_{\text{mem}}} \quad (d_{\text{mem}} = 64)$$

* **Run-Level Isolation Invariant:** At the start of every independent simulation trajectory, all agent memory states $\mathbf{m}_v$ and timestamp records are strictly reset to zero tensors to prevent cross-run state contamination.
* Memory acts as a compressed summary of the agent's historical interactions and behavioral trajectory up to the present.

### 2.3 Interaction Message Function (`MessageFunction`)
When an interaction event $e_k = (u, v, t_k)$ occurs with edge attributes $\mathbf{x}_{e_k}$:
$$\mathbf{msg}_u(t_k) = \text{MLP}_{\text{msg}}\left( \left[ \mathbf{m}_u(t_k^-), \mathbf{m}_v(t_k^-), \mathbf{x}_{e_k}, \phi(\Delta t) \right] \right)$$
$$\mathbf{msg}_v(t_k) = \text{MLP}_{\text{msg}}\left( \left[ \mathbf{m}_v(t_k^-), \mathbf{m}_u(t_k^-), \mathbf{x}_{e_k}, \phi(\Delta t) \right] \right)$$

The message incorporates both agents' prior memories, the edge feature vector, and the continuous time delta.

### 2.4 Gated Memory Updater (`MemoryUpdater`)
Upon receiving messages, agent memory transitions via a recurrent Gated Recurrent Unit (GRU):
$$\mathbf{m}_v(t_k) = \text{GRUCell}\left( \mathbf{m}_v(t_k^-), \mathbf{msg}_v(t_k) \right)$$

### 2.5 Temporal Graph Attention Embedding
To incorporate 1-hop relational context without incurring multi-hop leakage, agent embeddings aggregate information from their recent interaction neighbors:
$$\mathbf{h}_{\mathcal{N}(v)}(t) = \sum_{j \in \mathcal{N}(v)} \alpha_{v, j} \mathbf{W}_v \left[ \mathbf{m}_j(t), \mathbf{x}_j(t), \mathbf{x}_{e_{vj}} \right]$$

where attention coefficients $\alpha_{v, j}$ are computed using scaled dot-product attention over the temporal neighbor set.

Agent node embedding is then computed as:
$$\mathbf{z}_v(t) = \text{MLP}_{\text{node}}\left( \left[ \mathbf{m}_v(t), \mathbf{x}_v(t), \mathbf{h}_{\mathcal{N}(v)}(t) \right] \right)$$

### 2.6 Multi-Pooling Graph Readout (`GraphReadout`)
To obtain a fixed-size system-level representation invariant to the number of agents participating in the trajectory:
$$\mathbf{h}_G(t) = \left[ \text{MeanPool}\left( \{ \mathbf{z}_v(t) \}_{v \in \mathcal{V}} \right) \;\Vert\; \text{MaxPool}\left( \{ \mathbf{z}_v(t) \}_{v \in \mathcal{V}} \right) \right] \in \mathbb{R}^{2 \cdot d_{\text{embed}}}$$

Concatenating mean and max pooling preserves both global average network stress and localized maximum anomaly peaks (e.g., a single failing hub agent).

### 2.7 Hazard Prediction Head (`FailurePredictionHead`)
The system representation $\mathbf{h}_G(t)$ is passed to a 2-layer MLP classifier with dropout:
$$\hat{p}_{t, K} = \sigma\left( \mathbf{W}_2 \cdot \text{ReLU}\left( \mathbf{W}_1 \mathbf{h}_G(t) + \mathbf{b}_1 \right) + b_2 \right)$$

Output $\hat{p}_{t, K} \in [0.0, 1.0]$ represents the predicted probability of a Level 3 cascading failure within horizon $K$.

---

## 3. Training Protocol and Hyperparameters

* **Loss Function:** Weighted Binary Cross-Entropy (`BCEWithLogitsLoss`) with positive class weighting:
  $$\mathcal{L} = - \sum_{i=1}^B \left[ w_{\text{pos}} y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
* **Optimizer:** AdamW with learning rate $\eta = 5 \times 10^{-4}$ and weight decay $\lambda = 1 \times 10^{-4}$.
* **Gradient Clipping:** Max norm $\| \mathbf{g} \|_2 \le 1.0$.
* **Batch Size:** 8 simulation trajectory sequences.
* **Epochs & Early Stopping:** Maximum 50 epochs with early stopping patience of 10 epochs based on validation AUPRC.
* **Hardware Detection:** Dynamically detects CUDA GPU or falls back gracefully to CPU.
