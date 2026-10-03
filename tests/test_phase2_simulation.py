"""Tests for SimulationRun, Reproducibility, and Database Integration (Phase 2)."""

import pytest
from backend.app.database.models import Run, Agent as DBAgent, Event as DBEvent
from ml.simulation.run import SimulationRun
from ml.simulation.agents import (
    Planner,
    Researcher,
    Analyst,
    Verifier,
    DecisionAgent,
    Coder,
)
from ml.simulation.environment.orchestrator import SimulationEnvironment
from ml.simulation.topologies import PipelineTopology, StarTopology, MeshTopology


def test_simulation_run_creation_and_execution():
    """Verify SimulationRun runs to completion and records mandatory attributes."""
    agents = [
        Planner(agent_id="p1", name="Planner"),
        Researcher(agent_id="r1", name="Researcher"),
        Analyst(agent_id="a1", name="Analyst"),
        Verifier(agent_id="v1", name="Verifier"),
        DecisionAgent(agent_id="d1", name="Decision"),
    ]
    sim = SimulationRun(
        run_id="run_test_phase2_01",
        task_type="research",
        topology="pipeline",
        agents=agents,
        random_seed=42,
    )

    assert sim.run_id == "run_test_phase2_01"
    assert sim.task_type == "research"
    assert sim.topology == "pipeline"
    assert len(sim.agents) == 5
    assert sim.random_seed == 42
    assert sim.final_status == "INITIALIZED"

    sim.execute()

    assert sim.final_status == "COMPLETED"
    assert sim.start_time == 0.0
    assert sim.end_time is not None
    assert sim.duration_seconds > 0.0
    # In a 5-agent pipeline: 4 communication events (A->B, B->C, C->D, D->E)
    assert len(sim.events) == 4
    assert sim.task_output is not None
    assert sim.task_output["status"] == "COMPLETED"

    # Verify event properties
    for i, ev in enumerate(sim.events):
        assert ev.step_idx == i
        assert ev.source_agent == agents[i].agent_id
        assert ev.target_agent == agents[i + 1].agent_id
        assert ev.message != ""
        assert ev.confidence > 0.0


def test_simulation_run_star_topology():
    """Verify SimulationRun under Star topology (hub <-> leaves)."""
    agents = [
        Planner(agent_id="hub_0", name="Hub Coordinator"),
        Researcher(agent_id="leaf_1", name="Researcher"),
        Analyst(agent_id="leaf_2", name="Analyst"),
        Verifier(agent_id="leaf_3", name="Verifier"),
    ]
    sim = SimulationRun(
        run_id="run_star_test",
        task_type="research",
        topology="star",
        agents=agents,
        random_seed=100,
    )
    sim.execute()
    assert sim.final_status == "COMPLETED"
    # 3 leaves * 2 (Hub->Leaf, Leaf->Hub) = 6 events
    assert len(sim.events) == 6
    for ev in sim.events:
        # Every event must involve the hub
        assert ev.source_agent == "hub_0" or ev.target_agent == "hub_0"


def test_simulation_run_mesh_topology():
    """Verify SimulationRun under Mesh topology."""
    agents = [
        Planner(agent_id="m1", name="Planner"),
        Coder(agent_id="m2", name="Coder"),
        Verifier(agent_id="m3", name="Verifier"),
        DecisionAgent(agent_id="m4", name="Decision"),
    ]
    sim = SimulationRun(
        run_id="run_mesh_test",
        task_type="coding",
        topology="mesh",
        agents=agents,
        random_seed=77,
    )
    sim.execute()
    assert sim.final_status == "COMPLETED"
    assert len(sim.events) == 3


def test_simulation_reproducibility():
    """Verify that given same configuration + same random seed, simulator produces equivalent execution structure."""
    agents_run1 = [
        Planner(agent_id="p1", name="Planner Alpha"),
        Researcher(agent_id="r1", name="Researcher Beta"),
        Analyst(agent_id="a1", name="Analyst Gamma"),
        Verifier(agent_id="v1", name="Verifier Delta"),
        DecisionAgent(agent_id="d1", name="Decision Epsilon"),
    ]
    agents_run2 = [
        Planner(agent_id="p1", name="Planner Alpha"),
        Researcher(agent_id="r1", name="Researcher Beta"),
        Analyst(agent_id="a1", name="Analyst Gamma"),
        Verifier(agent_id="v1", name="Verifier Delta"),
        DecisionAgent(agent_id="d1", name="Decision Epsilon"),
    ]

    run1 = SimulationRun(
        run_id="run_repro_01",
        task_type="research",
        topology="pipeline",
        agents=agents_run1,
        random_seed=12345,
    ).execute()

    run2 = SimulationRun(
        run_id="run_repro_02",
        task_type="research",
        topology="pipeline",
        agents=agents_run2,
        random_seed=12345,
    ).execute()

    # Exact event structure equivalence
    assert len(run1.events) == len(run2.events)
    for ev1, ev2 in zip(run1.events, run2.events):
        assert ev1.source_agent == ev2.source_agent
        assert ev1.target_agent == ev2.target_agent
        assert ev1.event_type == ev2.event_type
        assert ev1.step_idx == ev2.step_idx
        assert ev1.message == ev2.message
        assert ev1.confidence == ev2.confidence
        assert ev1.latency == ev2.latency
        assert ev1.output_quality == ev2.output_quality
        assert ev1.timestamp == ev2.timestamp

    # Duration and structured output equivalence
    assert run1.duration_seconds == run2.duration_seconds
    assert run1.task_output == run2.task_output
    assert run1.final_status == run2.final_status


def test_database_persistence_integration(db_session):
    """Verify completed runs, agents, and events are stored in Phase 1 database."""
    agents = [
        Researcher(agent_id="db_r1", name="DB Researcher"),
        Analyst(agent_id="db_a1", name="DB Analyst"),
        Verifier(agent_id="db_v1", name="DB Verifier"),
    ]
    sim = SimulationRun(
        run_id="run_db_persisted_01",
        task_type="analysis",
        topology="pipeline",
        agents=agents,
        random_seed=999,
    ).execute()

    # Persist to database
    db_run = sim.save_to_db(db_session)
    assert db_run.id == "run_db_persisted_01"

    # Query back from DB
    queried_run = db_session.query(Run).filter(Run.id == "run_db_persisted_01").first()
    assert queried_run is not None
    assert queried_run.task_type == "analysis"
    assert queried_run.topology == "pipeline"
    assert queried_run.num_agents == 3
    assert queried_run.random_seed == 999
    assert queried_run.duration_seconds > 0

    queried_agents = db_session.query(DBAgent).filter(DBAgent.run_id == "run_db_persisted_01").all()
    assert len(queried_agents) == 3
    roles = {a.role for a in queried_agents}
    assert "researcher" in roles
    assert "analyst" in roles
    assert "verifier" in roles

    queried_events = db_session.query(DBEvent).filter(DBEvent.run_id == "run_db_persisted_01").all()
    assert len(queried_events) == 2  # 3 agents in pipeline -> 2 events


def test_forbidden_communication_raises_error():
    """Verify that communication forbidden by topology raises an error in the environment."""
    env = SimulationEnvironment(seed=42)
    a1 = Researcher(agent_id="node_a", name="A")
    a2 = Analyst(agent_id="node_b", name="B")
    a3 = Verifier(agent_id="node_c", name="C")

    env.register_agent(a1)
    env.register_agent(a2)
    env.register_agent(a3)

    # In pipeline A -> B -> C: A cannot talk directly to C!
    pipe = PipelineTopology(agent_ids=["node_a", "node_b", "node_c"])
    env.set_topology(pipe)

    # Valid step
    msg = env.step("node_a", "node_b", step_idx=0, run_id="run_err_test")
    assert msg.source_agent == "node_a"

    # Invalid jump step (prohibited by pipeline)
    with pytest.raises(ValueError, match="prohibits direct communication"):
        env.step("node_a", "node_c", step_idx=1, run_id="run_err_test")

    # Invalid reverse step (prohibited by strict forward pipeline)
    with pytest.raises(ValueError, match="prohibits direct communication"):
        env.step("node_b", "node_a", step_idx=2, run_id="run_err_test")


def test_invalid_agent_in_environment_raises_error():
    """Verify environment raises error when referenced agents do not exist."""
    env = SimulationEnvironment(seed=42)
    a1 = Researcher(agent_id="node_a", name="A")
    env.register_agent(a1)

    pipe = PipelineTopology(agent_ids=["node_a", "non_existent_node"])
    env.set_topology(pipe)

    with pytest.raises(ValueError, match="Target agent 'non_existent_node' not found"):
        env.step("node_a", "non_existent_node", step_idx=0, run_id="run_agent_err")
