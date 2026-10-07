# Troubleshooting and Operational Diagnostics

## Overview

This guide documents practical troubleshooting procedures for common runtime, dependency, database, and hardware issues encountered in AgentGuard.

---

## 1. Environment & Package Issues

### 1.1 PowerShell Script Execution Policy (`npm.ps1` or `Activate.ps1` Blocked)
* **Symptom:**  
  `npm : File C:\Program Files\nodejs\npm.ps1 cannot be loaded because running scripts is disabled on this system.`  
  `File .venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled...`
* **Root Cause:** Windows PowerShell restricted execution policy for `.ps1` script execution.
* **Resolution:**
  - Option A (Recommended): Use the `.cmd` binary wrapper directly:
    ```powershell
    npm.cmd run dev
    npm.cmd test -- --run
    ```
  - Option B: Enable execution for current user:
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    ```

### 1.2 `pyarrow` Missing and Parquet `UnicodeDecodeError`
* **Symptom:**  
  `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfa...` when calling `storage.load_tabular("...parquet")`.
* **Root Cause:** Running with system Python instead of the virtual environment where `pyarrow` is installed, or `pyarrow` missing from the active interpreter.
* **Resolution:**
  Verify that the virtual environment interpreter is being invoked:
  ```powershell
  .\.venv\Scripts\python.exe -c "import pyarrow; print(pyarrow.__version__)"
  ```
  If missing:
  ```powershell
  pip install pyarrow>=15.0.0
  ```

### 1.3 Python 3.14 PyTorch JIT and PyG Deprecation Warnings
* **Symptom:**  
  `FutureWarning: torch.jit.script is not supported in Python 3.14+ and may break...`  
  `DeprecationWarning: Failing to pass a value to the 'type_params' parameter of 'typing._eval_type'...`
* **Status:** These are non-fatal upstream library warnings emitted by PyTorch Geometric on Python 3.14. All 327 test invariants and model forward passes execute and pass without failure. They can be silenced during test execution with `pytest -W ignore`.

---

## 2. Hardware and Acceleration Diagnostics

### 2.1 CUDA Not Detected / Running on CPU
* **Symptom:**  
  `[Device] Selected Device: CPU` despite NVIDIA GPU presence.
* **Root Cause:** PyTorch CPU wheel was installed, or CUDA drivers are mismatched.
* **Resolution:**
  AgentGuard automatically falls back to CPU computation with full functional parity. To enable CUDA on Windows/Linux:
  ```powershell
  pip uninstall torch
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```

---

## 3. Database & Backend Startup Diagnostics

### 3.1 Port 8000 Already in Use
* **Symptom:**  
  `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000): address already in use`
* **Resolution:**
  Launch the backend on an alternate port:
  ```powershell
  uvicorn backend.app.main:app --port 8001
  ```
  Update `frontend/vite.config.ts` or `frontend/.env` to point proxy/API requests to port 8001.

### 3.2 Switching from SQLite to PostgreSQL
* **Symptom:**  
  `ModuleNotFoundError: No module named 'psycopg2'`
* **Resolution:**
  Install PostgreSQL driver and configure `.env`:
  ```powershell
  pip install psycopg2-binary>=2.9.9
  ```
  Set `.env`:
  ```ini
  DATABASE_URL=postgresql+psycopg2://agentguard:password@localhost:5432/agentguard
  ```

### 3.3 Database Seeder Warnings on Startup
* **Symptom:**  
  `Database seeder encountered an issue: UNIQUE constraint failed: runs.id`
* **Cause:** The database already contains seeded runs from a prior startup.
* **Impact:** Safe to ignore; `init_db()` is idempotent and leaves existing records intact.

---

## 4. Dataset Generation and Experiment Pipelines

### 4.1 Missing Dataset Directory
* **Symptom:**  
  `FileNotFoundError: Dataset manifest not found at data/processed/agentguard_dataset_v1/manifest.json`
* **Resolution:**
  Run the automated dataset generator:
  ```powershell
  python scripts/generate_dataset.py --num-runs 20 --output-dir data/processed/agentguard_dataset_v1
  ```

### 4.2 Corrupted or Missing Baseline Prediction Files
* **Symptom:**  
  `compare_baselines.py` reports missing predictions for a specific horizon.
* **Resolution:**
  Inspect inventory to see completed runs:
  ```powershell
  python scripts/check_inventory.py
  ```
  Re-train the missing baseline family using the appropriate script:
  ```powershell
  python scripts/train_classical_baselines.py
  python scripts/train_static_gnn.py
  python scripts/train_temporal_gnn.py
  ```

---

## 5. Dashboard Connection & CORS Diagnostics

### 5.1 Dashboard Cannot Fetch API Endpoints
* **Symptom:**  
  Browser console shows `CORS error` or `NetworkError when attempting to fetch resource`.
* **Resolution:**
  1. Confirm the backend server is running: `curl http://127.0.0.1:8000/health`.
  2. Verify that `ALLOWED_ORIGINS` in `.env` includes the Vite origin:
     ```ini
     ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
     ```
  3. Restart the backend server so new environment variables take effect.
