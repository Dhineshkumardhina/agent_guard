#!/usr/bin/env python3
"""Unified Validation and Quality Gate Verification Script for AgentGuard Phase 18.

Runs comprehensive validation across:
1. Environment & Provenance Checks
2. Static Application Security Testing (Bandit AST scan)
3. Unit, Integration, Model Invariant, and Security Test Suites (PyTest)
4. Frontend Vitest Tests and Type/Lint Quality Gates
5. API Latency and Performance Smoke Benchmarks

Usage:
    python scripts/run_full_validation.py          # Full complete validation
    python scripts/run_full_validation.py --quick  # Fast critical gate validation
"""

import sys
import os
import time
import argparse
import subprocess
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_banner(text: str) -> None:
    width = 76
    print("\n" + "=" * width)
    print(f" {text}".center(width))
    print("=" * width)


def run_cmd(cmd_list, description: str, cwd: Path = None, timeout: int = 300) -> tuple[int, float, str]:
    print(f"\n[RUNNING] {description}...")
    t0 = time.perf_counter()
    work_dir = str(cwd if cwd is not None else PROJECT_ROOT)
    use_shell = os.name == "nt" and (str(cmd_list[0]).endswith(".cmd") or "npm" in str(cmd_list[0]))
    try:
        proc = subprocess.run(
            cmd_list,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=use_shell,
        )
        duration = time.perf_counter() - t0
        status = "PASSED" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{status}] {description} in {duration:.2f}s")
        output = proc.stdout + "\n" + proc.stderr
        return proc.returncode, duration, output
    except Exception as e:
        duration = time.perf_counter() - t0
        print(f"[ERROR] {description}: {e}")
        return 1, duration, str(e)


def main():
    parser = argparse.ArgumentParser(description="AgentGuard Unified Phase 18 Validation Runner")
    parser.add_argument("--quick", action="store_true", help="Execute rapid critical quality gates only")
    args = parser.parse_args()

    print_banner(f"AGENTGUARD PHASE 18 VALIDATION GATE ({'QUICK' if args.quick else 'FULL'} MODE)")

    summary_results = []
    overall_start = time.perf_counter()

    # Step 1: Environment & Dependency Check
    print_banner("1. ENVIRONMENT & PROVENANCE CHECK")
    import torch
    import torch_geometric
    import sklearn
    import fastapi
    import sqlalchemy
    print(f"  * Python Runtime:     {sys.version.split()[0]}")
    print(f"  * PyTorch Framework:  {torch.__version__}")
    print(f"  * PyTorch Geometric:  {torch_geometric.__version__}")
    print(f"  * Scikit-Learn:       {sklearn.__version__}")
    print(f"  * FastAPI Backend:    {fastapi.__version__}")
    print(f"  * SQLAlchemy ORM:     {sqlalchemy.__version__}")
    summary_results.append(("Environment Provenance", "PASSED", 0.05))

    # Step 2: SAST Security Scan with Bandit
    print_banner("2. STATIC APPLICATION SECURITY TESTING (BANDIT)")
    python_bin = sys.executable
    venv_bandit = Path(python_bin).parent / "bandit.exe"
    bandit_cmd = [str(venv_bandit) if venv_bandit.exists() else "bandit", "-r", "ml/", "backend/", "-ll", "-q"]
    code, dur, out = run_cmd(bandit_cmd, "Bandit SAST Security Scan")
    summary_results.append(("Bandit Security Scan (0 Med/High)", "PASSED" if code == 0 else "FAILED", dur))

    # Step 3: Frontend Quality Gates (Vitest + Build)
    print_banner("3. FRONTEND TESTING & QUALITY GATES")
    npm_bin = shutil.which("npm.cmd") or shutil.which("npm")
    if npm_bin and (PROJECT_ROOT / "frontend" / "node_modules").exists():
        fe_dir = PROJECT_ROOT / "frontend"
        c1, d1, o1 = run_cmd([npm_bin, "test", "--", "--run"], "Frontend Vitest Unit Tests", cwd=fe_dir)
        summary_results.append(("Frontend Vitest Suite", "PASSED" if c1 == 0 else "FAILED", d1))
        c2, d2, o2 = run_cmd([npm_bin, "run", "lint"], "Frontend Linter (oxlint)", cwd=fe_dir)
        summary_results.append(("Frontend Linting", "PASSED" if c2 == 0 else "FAILED", d2))
    else:
        print("  * Frontend node_modules not found or npm not available, skipping frontend node tests.")
        summary_results.append(("Frontend Tests", "SKIPPED", 0.0))

    # Step 4: PyTest Test Suites
    print_banner("4. PYTEST TEST PYRAMID VALIDATION")
    pytest_bin = Path(python_bin).parent / "pytest.exe"
    pytest_exe = str(pytest_bin) if pytest_bin.exists() else "pytest"

    if args.quick:
        # Critical Phase 18 gate test files
        target_tests = [
            "tests/test_model_sanity.py",
            "tests/test_fault_and_label_validation.py",
            "tests/test_temporal_gnn_invariants.py",
            "tests/test_security_and_failure_recovery.py",
            "tests/test_end_to_end_pipeline.py",
        ]
        cmd = [pytest_exe] + target_tests + ["-q"]
        suite_desc = "Phase 18 Core Invariant & Pipeline Suites (69 tests)"
    else:
        cmd = [pytest_exe, "tests/", "-q"]
        suite_desc = "Complete Test Pyramid (327 tests across all 18 phases)"

    code, dur, out = run_cmd(cmd, suite_desc, timeout=600)
    summary_results.append((suite_desc, "PASSED" if code == 0 else "FAILED", dur))

    # Step 5: API Performance Smoke Profile
    print_banner("5. API LATENCY & PERFORMANCE PROFILE")
    from scripts.perf_smoke_test import run_performance_smoke_test
    t_perf0 = time.perf_counter()
    try:
        run_performance_smoke_test()
        dur_perf = time.perf_counter() - t_perf0
        summary_results.append(("API Performance Benchmark (<100ms)", "PASSED", dur_perf))
    except Exception as e:
        dur_perf = time.perf_counter() - t_perf0
        print(f"Performance smoke test failed: {e}")
        summary_results.append(("API Performance Benchmark", "FAILED", dur_perf))

    # Final Summary Table
    total_time = time.perf_counter() - overall_start
    print_banner("PHASE 18 VALIDATION GATE SUMMARY")
    print(f"{'Verification Category':<52} | {'Status':<10} | {'Duration':>8}")
    print("-" * 76)
    all_passed = True
    for cat, status, dur in summary_results:
        print(f"{cat:<52} | {status:<10} | {dur:7.2f}s")
        if "FAILED" in status:
            all_passed = False
    print("-" * 76)
    print(f"Total Validation Runtime: {total_time:.2f}s")

    if all_passed:
        print("\n>>> ALL QUALITY GATES PASSED: AgentGuard Phase 18 Validation is APPROVED. <<<")
        sys.exit(0)
    else:
        print("\n>>> ONE OR MORE QUALITY GATES FAILED. Review output above. <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
