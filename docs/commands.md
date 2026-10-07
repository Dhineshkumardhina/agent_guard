# Command Reference

## Overview

This document provides a single-reference catalog of all verified operational commands in the AgentGuard repository.

---

## 1. Environment & Diagnostics

```powershell
# Verify Python runtime and package imports
python scripts/verify_env.py

# Check test suite and experiment results inventory
python scripts/check_inventory.py

# Inspect active compute device (CPU or CUDA)
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## 2. Test Execution & Quality Gates

```powershell
# Run the complete test suite (327 tests)
pytest tests/ -v

# Run tests silently with summary
pytest tests/ -q

# Run specific subsystem test suites
pytest tests/test_model_sanity.py -v
pytest tests/test_fault_and_label_validation.py -v
pytest tests/test_temporal_gnn_invariants.py -v
pytest tests/test_security_and_failure_recovery.py -v
pytest tests/test_end_to_end_pipeline.py -v

# Run Bandit security SAST scan
bandit -r ml/ backend/ -ll -q

# Run unified validation gate (Quick mode, ~35 seconds)
python scripts/run_full_validation.py --quick

# Run unified validation gate (Full mode)
python scripts/run_full_validation.py
```

---

## 3. Simulation & Trajectory Generation

```powershell
# Run a single interactive simulation trajectory
python scripts/run_simulation.py --topology star --task coding --num-agents 5 --seed 42

# Run simulation with specific fault injection
python scripts/run_fault_experiment.py --fault-type tool_timeout --severity 0.9 --seed 42

# Export a completed trajectory to JSON
python scripts/export_run.py --run-id run_0001 --output run_0001_export.json
```

---

## 4. Dataset Generation & Management

```powershell
# Generate standard benchmark dataset (20 runs, 305 samples)
python scripts/generate_dataset.py --num-runs 20 --output-dir data/processed/agentguard_dataset_v1

# Generate generalization benchmark dataset (72 runs, 1098 samples)
python scripts/generate_dataset.py --num-runs 72 --output-dir data/processed/agentguard_generalization_v1

# Validate dataset integrity, schema conformity, and zero future leakage
python scripts/validate_dataset.py --dataset-dir data/processed/agentguard_dataset_v1

# Print dataset statistics and class distribution summary
python scripts/dataset_summary.py --dataset-dir data/processed/agentguard_dataset_v1
```

---

## 5. Model Training

```powershell
# Evaluate Rule-Based Baseline
python scripts/evaluate_rule_baseline.py --dataset-dir data/processed/agentguard_dataset_v1

# Train Classical ML Baselines (Logistic Regression, Random Forest, XGBoost)
python scripts/train_classical_baselines.py --dataset-dir data/processed/agentguard_dataset_v1

# Train Temporal Sequence Baselines (LSTM, GRU)
python scripts/train_sequence_baselines.py --dataset-dir data/processed/agentguard_dataset_v1

# Train Static Graph Neural Networks (GCN, GAT)
python scripts/train_static_gnn.py --dataset-dir data/processed/agentguard_dataset_v1

# Train Continuous-Time Temporal GNN (Core Research Model)
python scripts/train_temporal_gnn.py --dataset-dir data/processed/agentguard_dataset_v1
```

---

## 6. Comprehensive Evaluation & Benchmarking

```powershell
# Compare all trained baselines and print performance table
python scripts/compare_baselines.py

# Run full evaluation pipeline (generates calibration plots, lead-time distributions, report)
python scripts/run_full_evaluation.py
```

---

## 7. Research Studies (Ablation, Generalization, Explainability)

```powershell
# Execute systematic ablation study (Phase 13: 9 ablations vs full model)
python scripts/run_ablation_study.py --dataset-dir data/processed/agentguard_dataset_v1

# Execute out-of-distribution generalization suite (Phase 14: 4 shift dimensions)
python scripts/run_generalization_experiments.py --dataset-dir data/processed/agentguard_generalization_v1

# Regenerate generalization plot figures and tables
python scripts/regenerate_generalization_artifacts.py

# Execute multi-level explainability analysis (Phase 15: 5 attribution levels, case studies)
python scripts/run_explainability.py --dataset-dir data/processed/agentguard_generalization_v1
```

---

## 8. Backend API & Web Dashboard

```powershell
# Launch FastAPI REST backend
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

# Run API latency smoke benchmark
python scripts/perf_smoke_test.py

# Launch React frontend development server
cd frontend
npm.cmd run dev    # (or 'npm run dev' on Linux/macOS)

# Run frontend Vitest test suite
cd frontend
npm.cmd test -- --run

# Run frontend TypeScript type checking & build
cd frontend
npm.cmd run build

# Run frontend Oxlint linter
cd frontend
npm.cmd run lint
```
