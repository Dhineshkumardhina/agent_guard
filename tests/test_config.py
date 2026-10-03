"""Tests for configuration management and settings."""

from pathlib import Path
from backend.app.core.config import Settings
from ml.config.experiment_config import (
    ExperimentConfig,
    SimulationConfig,
    TopologyType,
    TaskType,
    FaultType,
)


def test_core_settings_defaults():
    """Verify application settings initialize with sane defaults."""
    settings = Settings()
    assert settings.PROJECT_NAME == "AgentGuard"
    assert settings.RANDOM_SEED == 42
    assert isinstance(settings.BASE_DIR, Path)
    assert len(settings.ALLOWED_ORIGINS) >= 1


def test_experiment_config_validation():
    """Test experiment configuration parameters and validation."""
    sim = SimulationConfig(
        num_runs=25,
        num_agents=5,
        topology=TopologyType.STAR,
        task_type=TaskType.CODING,
    )
    exp = ExperimentConfig(
        experiment_id="exp_01",
        simulation=sim,
    )
    assert exp.simulation.num_runs == 25
    assert exp.simulation.topology == TopologyType.STAR
    assert FaultType.HALLUCINATED_OUTPUT in exp.simulation.active_fault_types
    assert exp.prediction_horizons == [1, 3, 5, 10, 20]
