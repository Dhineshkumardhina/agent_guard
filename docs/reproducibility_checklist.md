# Research Reproducibility Checklist

## Overview

In accordance with open science practices and empirical research guidelines, this checklist documents verified reproducibility artifacts and evidence within the AgentGuard repository.

An item is marked complete (`[x]`) only when explicit empirical or codebase evidence exists.

---

## Reproducibility Verification Matrix

- [x] **Fixed Pseudo-Random Seeds**
  - *Evidence:* Master seed `random_seed = 42` is explicitly fixed across `manifest.json`, `ml/config/experiment_config.py`, all baseline model trainers (`train_classical_baselines.py`, `train_temporal_gnn.py`), and dataset generators. Verified deterministic execution across runs.

- [x] **Dataset Version Recorded**
  - *Evidence:* Immutable version identifiers (`agentguard_dataset_v1` and `agentguard_generalization_v1`) are stored in `manifest.json` alongside ISO-8601 creation timestamps and SHA-256 parameter hashes.

- [x] **Model Version Recorded**
  - *Evidence:* Model checkpoint schemas and `metrics.json` records include model family tags, horizon parameters, and architecture version metadata (`version: 1.0.0`).

- [x] **Configuration Recorded**
  - *Evidence:* Experiment configurations are tracked via Pydantic schemas (`ExperimentConfig`), serialized into YAML (`configs/default.yaml`), and stored in the relational database (`experiments.config`).

- [x] **Software Versions Recorded**
  - *Evidence:* Automated provenance captures Python runtime (`3.14.7`), PyTorch (`2.14.1+cpu`), PyTorch Geometric (`2.8.0.post1`), Scikit-Learn (`1.9.1`), and FastAPI (`0.142.2`) via `scripts/verify_env.py`.

- [x] **Git Commit Hash Recorded**
  - *Evidence:* The Git commit SHA (`f2d56bcb09894ba1ef5585bcead77e9387b53a81`) is automatically resolved and embedded into experiment metadata via `ml/utils/reproducibility.py`.

- [x] **Train / Validation / Test Split Recorded**
  - *Evidence:* Exact trajectory run IDs and sample distributions (`train`: 14 runs/224 samples, `val`: 3 runs/46 samples, `test`: 3 runs/35 samples) are permanently persisted in `manifest.json` and parquet partition tags.

- [x] **Prediction Horizons Recorded**
  - *Evidence:* Forward horizons $K \in \{1, 3, 5, 10, 20\}$ are documented in dataset manifests, database schema (`predictions.horizon_k`), and evaluation result directories.

- [x] **Results Stored Losslessly**
  - *Evidence:* All per-step prediction probabilities (`predictions.parquet`, `predictions.json`) and summary metrics (`metrics.json`) are preserved in `results/baselines/`, `results/evaluation/`, `results/ablation/`, `results/generalization/`, and `results/explainability/`.

- [x] **Executable Evaluation Scripts Available**
  - *Evidence:* Complete evaluation and comparison logic is executable via `scripts/compare_baselines.py`, `scripts/run_full_evaluation.py`, and `scripts/check_inventory.py`.

- [x] **Zero Future Information Leakage Verified**
  - *Evidence:* Strict causal event filtering ($t_{\text{event}} \le t_{\text{eval}}$) and horizon boundary exclusion ($(t, t+k]$) are formally verified by automated tests (`tests/test_temporal_neighborhood_no_future_edges`, `tests/test_fault_and_label_validation.py`) and documented in `docs/leakage_audit.md`.

- [x] **No Test-Set Tuning (Frozen Thresholds)**
  - *Evidence:* Decision thresholds $\theta^* \in [0.10, 0.90]$ are chosen exclusively on the validation split and frozen prior to test inference. No hyperparameter tuning or threshold adjustment occurs on the test set.

- [x] **Failed / Neutral Experiments Transparently Documented**
  - *Evidence:* Non-statistically significant outcomes (e.g., paired bootstrap test showing $p = 0.080$ between XGBoost and Temporal GNN on $N=35$; lack of test-set difference in Phase 13 ablation checkpoint) are explicitly reported in `results/evaluation/research_evaluation_report.md` without selective suppression.
