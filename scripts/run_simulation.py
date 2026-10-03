"""Demo Script: Multi-Agent Simulation Run.

Executes a controlled, reproducible 5-agent pipeline simulation
performing an empirical research task with seed=42.
Prints run ID, participating agents, chronological events, and final status,
and validates database persistence.
"""

import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silence verbose SQL engine logger for clean terminal output
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

from backend.app.database.session import SessionLocal, init_db
from backend.app.database.models import Run, Agent as DBAgent, Event as DBEvent
from ml.simulation import SimulationRun
from ml.simulation.agents import (
    Planner,
    Researcher,
    Analyst,
    Verifier,
    DecisionAgent,
)


def run_demo():
    print("=" * 70)
    print(" AgentGuard -- Multi-Agent Simulation Demo (Phase 2)")
    print("=" * 70)

    # 1. Initialize database tables
    init_db()

    # 2. Configure 5 agents
    agents = [
        Planner(agent_id="agent_planner_0", name="Planner Alpha"),
        Researcher(agent_id="agent_researcher_0", name="Researcher Beta"),
        Analyst(agent_id="agent_analyst_0", name="Analyst Gamma"),
        Verifier(agent_id="agent_verifier_0", name="Verifier Delta"),
        DecisionAgent(agent_id="agent_decision_0", name="Decision Epsilon"),
    ]

    # 3. Create and execute simulation run
    sim_run = SimulationRun(
        run_id="run_demo_phase2_001",
        task_type="research",
        topology="pipeline",
        agents=agents,
        random_seed=42,
        task_input={"topic": "Cascading Failures in Distributed Agent Swarms"},
    )

    print("\n[+] Executing simulation...")
    sim_run.execute()

    # 4. Print requested details
    print(f"\n[RUN ID]: {sim_run.run_id}")
    print(f"[TASK TYPE]: {sim_run.task_type}")
    print(f"[TOPOLOGY]: {sim_run.topology}")
    print(f"[SEED]: {sim_run.random_seed}")
    print(f"[DURATION]: {sim_run.duration_seconds:.4f}s")
    print(f"[FINAL STATUS]: {sim_run.final_status}")

    print("\n--- PARTICIPATING AGENTS ---")
    for agent in sim_run.agents:
        role_name = agent.role.value if hasattr(agent.role, "value") else str(agent.role)
        print(f"  * ID: {agent.agent_id:<20} Name: {agent.name:<18} Role: {role_name:<12} State: {agent.state}")

    print(f"\n--- RECORDED EVENTS ({len(sim_run.events)} events) ---")
    for i, ev in enumerate(sim_run.events):
        print(f"  Step {ev.step_idx} | t={ev.timestamp:.3f}s | {ev.source_agent} -> {ev.target_agent} [{ev.event_type}]")
        print(f"    Payload: {ev.message}")
        print(f"    Latency: {ev.latency:.3f}s | Confidence: {ev.confidence:.3f} | Quality: {ev.output_quality:.3f}")

    print("\n--- TASK OUTPUT SUMMARY ---")
    if sim_run.task_output:
        for k, v in sim_run.task_output.items():
            print(f"  {k}: {v}")

    # 5. Database Integration
    db = SessionLocal()
    try:
        print("\n[+] Persisting run to database...")
        db_run = sim_run.save_to_db(db)
        print(f"  [OK] Run stored in DB with ID: {db_run.id}")

        # Verification query
        stored_run = db.query(Run).filter(Run.id == sim_run.run_id).first()
        stored_agents = db.query(DBAgent).filter(DBAgent.run_id == sim_run.run_id).all()
        stored_events = db.query(DBEvent).filter(DBEvent.run_id == sim_run.run_id).all()

        print(f"  [VERIFICATION] Database query confirmed:")
        print(f"    - DB Run: id={stored_run.id}, task={stored_run.task_type}, topology={stored_run.topology}")
        print(f"    - DB Agents stored: {len(stored_agents)}")
        print(f"    - DB Events stored: {len(stored_events)}")

    finally:
        db.close()

    print("\n" + "=" * 70)
    print(" Simulation Completed Successfully")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
