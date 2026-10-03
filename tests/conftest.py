"""Pytest configuration and shared test fixtures for AgentGuard."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database.session import Base
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.config.experiment_config import ExperimentConfig, SimulationConfig, TopologyType, TaskType


@pytest.fixture
def db_session():
    """In-memory SQLite database session for unit and integration testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_telemetry_event():
    """Sample valid telemetry event fixture."""
    return AgentTelemetryEvent(
        run_id="run_test_001",
        step_idx=0,
        timestamp=0.15,
        source_agent="researcher",
        target_agent="planner",
        event_type="message",
        message="Preliminary research summary regarding optimization bounds.",
        message_length=56,
        token_count=14,
        latency=0.45,
        confidence=0.88,
        output_quality=0.92,
        contradiction_score=0.05,
        topology="pipeline",
    )


@pytest.fixture
def sample_experiment_config():
    """Sample valid experiment configuration fixture."""
    return ExperimentConfig(
        experiment_id="exp_test_001",
        name="Test Experiment",
        simulation=SimulationConfig(
            num_runs=10,
            num_agents=5,
            topology=TopologyType.PIPELINE,
            task_type=TaskType.RESEARCH,
        ),
    )
