"""Experiment and Simulation Configuration.

Defines schemas, enumerations, and parameters for scientific experiments,
data generation, fault injection, and model evaluation protocols.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Configurable agent roles within the multi-agent system."""
    PLANNER = "planner"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CODER = "coder"
    VERIFIER = "verifier"
    CRITIC = "critic"
    DECISION = "decision"


class TopologyType(str, Enum):
    """Communication topology patterns."""
    PIPELINE = "pipeline"
    STAR = "star"
    MESH = "mesh"
    CUSTOM = "custom"


class TaskType(str, Enum):
    """Controlled multi-agent benchmark task categories."""
    RESEARCH = "research"
    CODING = "coding"
    DATA_ANALYSIS = "data_analysis"
    PLANNING = "planning"


class FailureLevel(int, Enum):
    """Three-level failure taxonomy for multi-agent systems."""
    LEVEL_1_AGENT = 1        # Local agent failure (e.g., hallucination, tool error, timeout)
    LEVEL_2_INTERACTION = 2  # Pairwise interaction failure (e.g., conflicting info, communication loop)
    LEVEL_3_CASCADING = 3    # System-level cascading failure propagating across dependencies


class FaultType(str, Enum):
    """Specific fault modes supported by the fault injection engine."""
    HALLUCINATED_OUTPUT = "hallucinated_output"
    INCORRECT_INFORMATION = "incorrect_information"
    TOOL_FAILURE = "tool_failure"
    TOOL_TIMEOUT = "tool_timeout"
    DELAYED_RESPONSE = "delayed_response"
    MALFORMED_OUTPUT = "malformed_output"
    LOW_CONFIDENCE_OUTPUT = "low_confidence_output"
    CONTRADICTORY_INFORMATION = "contradictory_information"
    COMMUNICATION_LOOP = "communication_loop"
    INCORRECT_DELEGATION = "incorrect_delegation"
    STALE_CONTEXT = "stale_context"
    AGENT_DROPOUT = "agent_dropout"


class ModelArchitecture(str, Enum):
    """Model architectures for comparative hypothesis testing."""
    RULE_BASED = "rule_based"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LSTM = "lstm"
    STATIC_GNN = "static_gnn"
    TEMPORAL_GNN = "temporal_gnn"


class DataSplitConfig(BaseModel):
    """Trajectory-level dataset split configuration avoiding temporal/trajectory leakage."""
    train_ratio: float = Field(default=0.70, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    split_strategy: str = Field(default="trajectory_level", description="Must be trajectory_level or task_level")


class SimulationConfig(BaseModel):
    """Parameters for running multi-agent simulations."""
    num_runs: int = Field(default=100, ge=1, description="Number of trajectories to generate")
    num_agents: int = Field(default=5, ge=3, le=20, description="Agent count (3, 5, 8, 12, etc.)")
    topology: TopologyType = Field(default=TopologyType.PIPELINE)
    custom_topology_edges: Optional[List[tuple[str, str]]] = Field(
        default=None, description="List of (source, target) directed edges if topology is CUSTOM"
    )
    task_type: TaskType = Field(default=TaskType.RESEARCH)
    fault_injection_rate: float = Field(default=0.35, ge=0.0, le=1.0, description="Probability of injecting a fault")
    active_fault_types: List[FaultType] = Field(
        default_factory=lambda: [
            FaultType.HALLUCINATED_OUTPUT,
            FaultType.INCORRECT_INFORMATION,
            FaultType.TOOL_FAILURE,
            FaultType.TOOL_TIMEOUT,
            FaultType.CONTRADICTORY_INFORMATION,
            FaultType.COMMUNICATION_LOOP,
        ]
    )
    max_steps_per_run: int = Field(default=30, ge=5, le=200)
    random_seed: int = Field(default=42)


class ExperimentConfig(BaseModel):
    """Complete specification for a reproducible research experiment."""
    experiment_id: str = Field(..., description="Unique deterministic or generated experiment identifier")
    name: str = Field(default="Default Hypothesis Experiment")
    description: str = Field(default="Testing H0, H1, H2, and H3 for cascading failure prediction")
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    prediction_horizons: List[int] = Field(default=[1, 3, 5, 10, 20], description="Evaluation horizons K")
    models_to_evaluate: List[ModelArchitecture] = Field(
        default_factory=lambda: [
            ModelArchitecture.RULE_BASED,
            ModelArchitecture.LOGISTIC_REGRESSION,
            ModelArchitecture.RANDOM_FOREST,
            ModelArchitecture.XGBOOST,
            ModelArchitecture.LSTM,
            ModelArchitecture.STATIC_GNN,
            ModelArchitecture.TEMPORAL_GNN,
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
