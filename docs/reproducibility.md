# Reproducibility Guide

## Overview

This guide provides end-to-end, verified instructions for replicating all research findings, datasets, models, evaluations, backend services, and dashboard interfaces in AgentGuard.

All commands below are actual, verified scripts from the repository.

---

## 1. System Requirements and Environment

### Software Versions Verified
* **Operating System:** Windows 10/11 x86_64, Linux (Ubuntu 22.04+), or macOS
* **Python Runtime:** Python 3.11+ (Verified on Python 3.14.7)
* **Node.js Runtime:** Node.js v20+ (Verified on Node v24.21.0, npm 11.19.0)
* **Git:** Version 2.40+ (Captures commit hash in metadata)

---

## 2. Environment Setup & Dependency Installation

### Step 2.1: Clone and Create Virtual Environment
```powershell
# Clone the repository
git clone https://github.com/Dhineshkumardhina/agent_guard.git
cd agent_guard

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### Step 2.2: Install Python Dependencies
```powershell
# Install root scientific, ML, and backend dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Verify environment imports and provenance
python scripts/verify_env.py
```

### Step 2.3: Install Frontend Dependencies
```powershell
cd frontend
npm.cmd install    # On Windows (use 'npm install' on Linux/macOS)
cd ..
```

---

## 3. Configuration & Environment Variables

Copy the template environment configuration:
```powershell
copy .env.example .env   # On Linux/macOS: cp .env.example .env
```

### Key Environment Variables (`.env`):
```ini
ENV=development
LOG_LEVEL=INFO
RANDOM_SEED=42

HOST=127.0.0.1
PORT=8000
DEBUG=True

# SQLite default is preconfigured; PostgreSQL is optionally supported:
DATABASE_URL=sqlite:///./agentguard.db
# DATABASE_URL=postgresql+psycopg2://agentguard:password@localhost:5432/agentguard

DATASET_DIR=./data/processed
RESULTS_DIR=./results
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

---

## 4. Database Setup & Initialization

The database is self-initializing. Upon backend startup, SQLAlchemy executes `init_db()` and seeds baseline experiment runs, agents, and prediction records if empty:
```powershell
# Verify database connection and schema tables
python -c "from backend.app.database.session import init_db; init_db(); print('Database tables initialized successfully.')"
```

---

## 5. Dataset Generation & Validation

Generate synthetic multi-agent trajectories and compile feature partitions:

```powershell
# 1. Generate standardized benchmark dataset (20 runs, 305 samples)
python scripts/generate_dataset.py --num-runs 20 --output-dir data/processed/agentguard_dataset_v1

# 2. Validate dataset partitions, schemas, and zero future leakage
python scripts/validate_dataset.py --dataset-dir data/processed/agentguard_dataset_v1

# 3. Print dataset class balance and split statistics
python scripts/dataset_summary.py --dataset-dir data/processed/agentguard_dataset_v1
```

---

## 6. Model Training & Evaluation

Train each model family across prediction horizons $K \in \{1, 3, 5, 10\}$:

### Step 6.1: Rule-Based Heuristic Baseline
```powershell
python scripts/evaluate_rule_baseline.py --dataset-dir data/processed/agentguard_dataset_v1 --results-dir results/baselines/rule_based
```

### Step 6.2: Classical ML (Logistic Regression, Random Forest, XGBoost)
```powershell
python scripts/train_classical_baselines.py --dataset-dir data/processed/agentguard_dataset_v1 --results-dir results/baselines/classical_ml
```

### Step 6.3: Temporal Sequence Models (LSTM, GRU)
```powershell
python scripts/train_sequence_baselines.py --dataset-dir data/processed/agentguard_dataset_v1 --results-dir results/baselines/sequence
```

### Step 6.4: Static Graph Neural Networks (GCN, GAT)
```powershell
python scripts/train_static_gnn.py --dataset-dir data/processed/agentguard_dataset_v1 --results-dir results/baselines/static_gnn
```

### Step 6.5: Continuous-Time Temporal GNN (Core Research Model)
```powershell
python scripts/train_temporal_gnn.py --dataset-dir data/processed/agentguard_dataset_v1 --results-dir results/baselines/temporal_gnn
```

### Step 6.6: Comparative Benchmark Analysis
```powershell
# Generate comparative tables, calibration curves, and research report
python scripts/compare_baselines.py
python scripts/run_full_evaluation.py
```

---

## 7. Research Studies: Ablation, Generalization, Explainability

### Step 7.1: Systematic Ablation Study (Phase 13)
```powershell
# Runs all 9 architectural and feature ablations vs full model
python scripts/run_ablation_study.py --dataset-dir data/processed/agentguard_dataset_v1 --output-dir results/ablation
```

### Step 7.2: Out-of-Distribution Generalization Study (Phase 14)
```powershell
# Evaluates 4 distribution shift dimensions (agents, topology, task, faults)
python scripts/run_generalization_experiments.py --dataset-dir data/processed/agentguard_generalization_v1 --output-dir results/generalization
```

### Step 7.3: Multi-Level Explainability and Attribution (Phase 15)
```powershell
# Generates 5 attribution levels, counterfactual deltas, and case studies
python scripts/run_explainability.py --dataset-dir data/processed/agentguard_generalization_v1 --output-dir results/explainability
```

---

## 8. Starting the Web Services

### Launch Backend API Server
```powershell
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```
* **Swagger Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Launch Frontend Research Dashboard
In a separate terminal window:
```powershell
cd frontend
npm.cmd run dev    # On Windows (or 'npm run dev' on Linux/macOS)
```
* **Dashboard URL:** [http://localhost:5173](http://localhost:5173)

---

## 9. Comprehensive System Validation Gate

Execute the unified quality gate runner to verify all system components, security posture, test suites, and API response latencies:

```powershell
# Rapid Critical Quality Gate (~35 seconds)
python scripts/run_full_validation.py --quick

# Full Test Pyramid Validation (327 Python tests + 12 Vitest tests)
python scripts/run_full_validation.py
```
