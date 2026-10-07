# Dataset Documentation

## Overview

AgentGuard provides two standardized, version-controlled research datasets: `agentguard_dataset_v1` and `agentguard_generalization_v1`. Both datasets are produced by the controlled multi-agent simulation pipeline and persisted with immutable manifests.

> **CRITICAL SCIENTIFIC INTEGRITY STATEMENT:**  
> The simulator-generated datasets documented herein constitute controlled experimental data designed to isolate structural and temporal failure dynamics. They should **NOT** automatically be interpreted as equivalent to production enterprise multi-agent system data. Real-world multi-agent deployments exhibit distinct token distributions, open-ended tool interfaces, varying network latencies, and diverse model backends that may introduce failure patterns not captured in this benchmark.

---

## 1. Dataset Versions and Summary Statistics

| Attribute | `agentguard_dataset_v1` (Standard Benchmark) | `agentguard_generalization_v1` (Robustness Suite) |
|---|---|---|
| **Primary Purpose** | Benchmark comparison across 9 model families & ablations | Out-of-distribution evaluation across $N$, topology, task, faults |
| **Total Simulation Runs** | 20 runs | 72 runs |
| **Total Prediction Samples**| 305 samples | 1,098 samples |
| **Class Distribution (Pos/Neg)**| 118 positive (38.69%) / 187 negative (61.31%) | 769 positive (70.04%) / 329 negative (29.96%) |
| **Train Split (Runs / Samples)**| 14 runs (70%) / 224 samples (73.4%) | 50 runs (69.4%) / 746 samples (67.9%) |
| **Val Split (Runs / Samples)** | 3 runs (15%) / 46 samples (15.1%) | 11 runs (15.3%) / 138 samples (12.6%) |
| **Test Split (Runs / Samples)**| 3 runs (15%) / 35 samples (11.5%) | 11 runs (15.3%) / 214 samples (19.5%) |
| **Prediction Horizons $K$** | $\{1, 3, 5, 10, 20\}$ | $\{1, 3, 5, 10, 20\}$ |
| **Agent Counts Tested** | $\{3, 5, 8, 12\}$ | $\{3, 5, 8, 12\}$ |
| **Topologies** | Pipeline, Star, Mesh, Custom | Pipeline, Star, Mesh, Custom |
| **Tasks** | Research, Coding, Analysis, Planning | Research, Coding, Analysis, Planning |
| **Fault Modes Present** | 3 modes (`none`, `hallucinated_output`, `delayed_response`) | All 12 canonical fault modes |
| **Storage Formats** | Apache Parquet (`.parquet`) + JSON Lines (`.jsonl`) | Apache Parquet (`.parquet`) + JSON Lines (`.jsonl`) |

---

## 2. Simulation and Generation Parameters

The dataset generation script (`scripts/generate_dataset.py`) parameterizes each simulation run with:
- **`random_seed = 42`**: Fixed master seed guaranteeing deterministic replication.
- **`fault_probability`**: 0.40 in `v1` (lower fault frequency, higher nominal baseline) and 0.70 in `generalization_v1` (stress-testing across diverse fault configurations).
- **`max_steps`**: 20 interaction steps per trajectory.
- **Topologies**:
  - `pipeline`: 3 to 5 agents.
  - `star`: 3 to 12 agents.
  - `mesh`: 3 to 8 agents.
  - `custom`: 5 to 12 agents.

---

## 3. Run-Level Partitioning (Zero Pseudo-Replication)

A critical flaw in time-series and graph ML benchmarks is pseudo-replication: randomly splitting samples from the same trajectory across train and test sets. When consecutive steps from one run appear in both splits, temporal autocorrelation artificially inflates validation metrics.

AgentGuard enforces **strict run-level stratified partitioning**:
$$\mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{val}} = \emptyset, \quad \mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{test}} = \emptyset, \quad \mathcal{R}_{\text{val}} \cap \mathcal{R}_{\text{test}} = \emptyset$$

* In `agentguard_dataset_v1`, exactly 14 runs comprise training, 3 runs comprise validation, and 3 runs comprise testing (yielding the standardized $N=35$ test sample population across horizons).
* No simulation run appears in more than one partition.

---

## 4. Feature Schema Specification

Tabular feature vectors generated for each prediction sample include 17 agent-level and network-level telemetry features:

```
feature_schema:
  agent_level_features:
    1. total_events_observed      (Cumulative communication events up to cutoff t)
    2. mean_output_quality        (Average quality score across all agents)
    3. min_output_quality         (Lowest quality score among participating agents)
    4. mean_confidence            (Average self-reported confidence)
    5. min_confidence             (Lowest self-reported confidence)
    6. total_token_count          (Cumulative tokens expended)
    7. total_latency              (Cumulative execution latency)
    8. mean_latency               (Mean step response latency)
    9. max_latency                (Maximum single-step response latency)
   10. total_retries              (Cumulative retry count)
   11. mean_contradiction_score   (Average semantic contradiction score)
   12. max_contradiction_score    (Highest recorded contradiction score)
   13. tool_call_count            (Total external tools invoked)
   14. tool_failure_count         (Total tool runtime exceptions)
   15. agent_count_active         (Number of non-crashed active agents)
   16. error_count                (Total caught errors across session)
   17. interaction_density        (Ratio of active edges to possible edges)
```

---

## 5. Storage Formats and Directory Layout

Datasets reside in `data/processed/{dataset_version}/`:

```
data/processed/agentguard_dataset_v1/
├── manifest.json              # Cryptographic manifest, parameters, and class distributions
├── all_samples.parquet        # Combined tabular dataset with all split tags
├── train.parquet              # Training partition (224 samples)
├── val.parquet                # Validation partition (46 samples)
├── test.parquet               # Test partition (35 samples)
├── graph_sequences_train.jsonl# Dynamic graph snapshots for training trajectories
├── graph_sequences_val.jsonl  # Dynamic graph snapshots for validation trajectories
└── graph_sequences_test.jsonl # Dynamic graph snapshots for testing trajectories
```

### Fallback Format Support
In environments lacking Apache Arrow/PyArrow (`pyarrow`), `ml/data/storage.py` provides fallback serialization via standard JSON Lines (`.jsonl`).
