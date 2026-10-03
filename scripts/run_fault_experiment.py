"""Demo Script: Controlled Fault Injection & Propagation Experiment (Phase 4).

Runs comparative experiments:
1. Normal (fault-free) multi-agent execution.
2. Fault-injected execution demonstrating Level 3 cascading failure propagation.

Prints:
- fault
- origin
- affected agents
- propagation path
- final outcome
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silence SQL engine logs for clean terminal output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from backend.app.database.session import SessionLocal, init_db
from ml.simulation.run import SimulationRun
from ml.simulation.fault_injection.injector import FaultInjector


def run_experiment():
    print("=" * 75)
    print(" AgentGuard -- Controlled Fault Injection & Cascading Failure Demo (Phase 4)")
    print("=" * 75)

    init_db()

    # -------------------------------------------------------------
    # 1. NORMAL (FAULT-FREE) EXECUTION
    # -------------------------------------------------------------
    print("\n[EXPERIMENT 1: FAULT-FREE EXECUTION]")
    normal_run = SimulationRun(
        run_id="exp_normal_001",
        task_type="research",
        topology="pipeline",
        random_seed=42,
        task_input={"topic": "Autonomous Network Topology Adaptation"},
    ).execute()

    print(f"  Run ID:              {normal_run.run_id}")
    print(f"  Injected Fault:      None")
    print(f"  Origin:              None")
    print(f"  Affected Agents:     None")
    print(f"  Propagation Path:    {normal_run.propagation_tracker.to_path_string()}")
    print(f"  Has Cascade:         {normal_run.has_cascading_failure}")
    print(f"  Final Outcome:       {normal_run.final_status}")

    # -------------------------------------------------------------
    # 2. FAULT-INJECTED EXECUTION (Cascading Failure Propagation)
    # -------------------------------------------------------------
    print("\n[EXPERIMENT 2: FAULT-INJECTED EXECUTION]")
    
    # Inject hallucinated output into the researcher early in the pipeline
    injector = FaultInjector(
        fault_type="hallucinated_output",
        target_agent="researcher_1",
        injection_step=1,
        probability=1.0,
        severity=0.85,
        random_seed=42,
    )

    fault_run = SimulationRun(
        run_id="exp_fault_injected_001",
        task_type="research",
        topology="pipeline",
        random_seed=42,
        fault_injector=injector,
        task_input={"topic": "Autonomous Network Topology Adaptation"},
    ).execute()

    tracker = fault_run.propagation_tracker

    print(f"  Run ID:              {fault_run.run_id}")
    print(f"  Injected Fault:      {tracker.originating_fault_type}")
    print(f"  Origin:              {tracker.originating_agent} (Step {tracker.originating_step})")
    print(f"  Affected Agents:     {', '.join(tracker.affected_agents)}")
    print(f"  Propagation Path:    {tracker.to_path_string()}")
    print(f"  Failure Level:       Level {tracker.failure_level} (Cascading Failure)")
    print(f"  Has Cascade:         {fault_run.has_cascading_failure}")
    print(f"  Final Outcome:       {fault_run.final_status}")

    # Persist to database
    db = SessionLocal()
    try:
        fault_run.save_to_db(db)
        print(f"\n[+] Successfully persisted fault experiment to database (run_id: {fault_run.run_id}).")
    finally:
        db.close()

    print("\n" + "=" * 75)
    print(" Fault Injection Experiment Completed Successfully")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
