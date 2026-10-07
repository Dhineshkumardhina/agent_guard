# Testing and Quality Assurance Documentation

## Overview

AgentGuard implements an exhaustive testing pyramid spanning unit tests, ML invariant validation, data leakage prevention, security hardening, API integration, and frontend components. The test suite comprises **327 automated Python tests** and **12 React/TypeScript tests**, totaling **339 tests** with a verified 100% pass rate.

---

## 1. Testing Pyramid Breakdown

| Test Level | Scope & Target Files | Test Count | Pass Rate | Key Invariants Verified |
|---|---|:---:|:---:|---|
| **Unit Tests** | `tests/test_model_sanity.py`, `tests/test_phase2_*.py`, `tests/test_simulator_interfaces.py` | 118 | **100%** | Empty batches, extreme values, class imbalance, zero-variance handling across all 9 baseline models. |
| **Integration Tests** | `tests/test_temporal_gnn_invariants.py`, `tests/test_fault_and_label_validation.py`, `tests/test_phase*.py` | 137 | **100%** | Memory isolation between runs, strict horizon filtering $(t, t+k]$, 12 fault propagation cascades. |
| **Security & Recovery** | `tests/test_security_and_failure_recovery.py` | 11 | **100%** | Limit $> 200$ rejection (422), negative offset rejection (422), unknown ID (404), method rejection (405), SQL injection resistance, traceback suppression. |
| **API Integration** | `tests/test_phase16_api.py`, `tests/test_api_health.py` | 41 | **100%** | OpenAPI 3.1 compliance, schema response validation, nested relational queries, filter combinations. |
| **Frontend Tests** | `frontend/src/test/` (Vitest + Testing Library) | 12 | **100%** | Component rendering, graph visualization canvas, metric cards, model comparison table, alert banners. |
| **End-to-End Pipeline** | `tests/test_end_to_end_pipeline.py` | 1 | **100%** | Complete 11-step research pipeline from raw simulation through API and DB persistence. |
| **Full System Total** | **Entire Test Pyramid** | **339** | **100%** | **Zero failures, zero regressions.** |

---

## 2. Core Invariants Verified

### 1. Zero Future Information Leakage Invariant
* **Rule:** Every graph snapshot $G(t)$, node feature vector $\mathbf{x}_v(t)$, and edge feature $\mathbf{x}_e(t)$ must condition strictly on events occurring at or before timestamp $t$.
* **Verification:** `test_temporal_neighborhood_no_future_edges` and `verify_no_future_leakage()` in `ml/graph/graph_builder.py`.

### 2. Horizon Boundary Exclusion Invariant
* **Rule:** Target labels $Y_{t, K}$ must evaluate failures strictly within $(t, t+K]$. Samples collected after a Level 3 cascade has already initiated are strictly purged from early-warning evaluation sets.
* **Verification:** `test_horizon_boundary_exclusion` in `tests/test_fault_and_label_validation.py`.

### 3. Trajectory Memory Isolation Invariant
* **Rule:** Dynamic agent memory states in the Continuous-Time Temporal GNN must be completely reset between independent simulation trajectories to prevent cross-run state bleeding.
* **Verification:** `test_memory_isolation_between_runs` in `tests/test_temporal_gnn_invariants.py`.

### 4. Data Partition Isolation Invariant
* **Rule:** Data partitioning must operate strictly at the run level (`run_id`), ensuring $\mathcal{R}_{\text{train}} \cap \mathcal{R}_{\text{test}} = \emptyset$. Normalization scalers must be fit strictly on training runs.
* **Verification:** `test_run_level_split_isolation` in `tests/test_phase6_dataset.py`.

### 5. Safe Model Deserialization Invariant
* **Rule:** PyTorch model weights must be loaded using `weights_only=True` to mitigate arbitrary code execution vulnerabilities during checkpoint loading.
* **Verification:** Verified via Bandit AST SAST scan and `test_model_weight_deserialization_safety`.

### 6. Production Traceback Suppression Invariant
* **Rule:** Unhandled server exceptions (500) must return sanitized error envelopes without leaking Python internal stack traces to HTTP clients.
* **Verification:** `test_unhandled_exception_masks_traceback` in `tests/test_security_and_failure_recovery.py`.

---

## 3. Test Execution Commands

```powershell
# 1. Run all Python unit and integration tests (327 tests)
.\.venv\Scripts\pytest.exe tests/ -v

# 2. Run rapid critical quality gates (69 tests + Bandit + Vitest + API perf)
.\.venv\Scripts\python.exe scripts/run_full_validation.py --quick

# 3. Run frontend Vitest test suite (12 tests)
cd frontend
npm.cmd test -- --run

# 4. Run Bandit security SAST scan
.\.venv\Scripts\bandit.exe -r ml/ backend/ -ll -q
```
