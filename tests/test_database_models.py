"""Tests for database tables, foreign keys, and ORM operations."""

from backend.app.database.models import (
    Experiment,
    Run,
    Agent,
    Event,
    AgentInteraction,
    FaultInjection,
    Failure,
    Prediction,
    ModelResult,
)


def test_database_crud_and_relationships(db_session):
    """Verify relational integrity between Experiment, Run, Agents, Events, and Failures."""
    # 1. Create Experiment
    exp = Experiment(
        id="exp_test_db",
        name="Unit Test Experiment",
        config={"topology": "pipeline"},
        status="created",
    )
    db_session.add(exp)
    db_session.commit()

    # 2. Create Run linked to Experiment
    run = Run(
        id="run_test_db",
        experiment_id=exp.id,
        task_type="research",
        topology="pipeline",
        num_agents=3,
        has_cascading_failure=True,
        cascading_failure_step=12,
    )
    db_session.add(run)
    db_session.commit()

    # 3. Add Agents to Run
    agent1 = Agent(id="agent_1", run_id=run.id, agent_name="planner_1", role="planner")
    agent2 = Agent(id="agent_2", run_id=run.id, agent_name="coder_1", role="coder")
    db_session.add_all([agent1, agent2])
    db_session.commit()

    # 4. Add Event
    event = Event(
        id="evt_01",
        run_id=run.id,
        step_idx=0,
        timestamp=0.1,
        source_agent="planner_1",
        target_agent="coder_1",
        event_type="message",
        message_length=40,
        token_count=10,
        latency=0.3,
        confidence=0.9,
    )
    db_session.add(event)
    db_session.commit()

    # 5. Add Fault Injection & Failure
    fault = FaultInjection(
        id="flt_01",
        run_id=run.id,
        step_idx=5,
        target_agent="planner_1",
        fault_type="hallucinated_output",
    )
    failure = Failure(
        id="fail_01",
        run_id=run.id,
        step_idx=12,
        failure_level=3,
        originating_agent="planner_1",
        failure_type="cascading_hallucination",
    )
    db_session.add_all([fault, failure])
    db_session.commit()

    # 6. Verify associations
    queried_run = db_session.query(Run).filter(Run.id == "run_test_db").first()
    assert queried_run is not None
    assert len(queried_run.agents) == 2
    assert len(queried_run.events) == 1
    assert len(queried_run.failures) == 1
    assert queried_run.failures[0].failure_level == 3
    assert queried_run.experiment.name == "Unit Test Experiment"
