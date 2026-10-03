"""Simulation package for AgentGuard.

Exposes Agent abstractions, Topologies, Tasks, Orchestration Environment,
and the SimulationRun execution system.
"""

from ml.simulation.interfaces import (
    AgentInterface,
    TopologyInterface,
    TaskInterface,
    FaultInjectorInterface,
    EnvironmentInterface,
)
from ml.simulation.events import SimulationMessage
from ml.simulation.agents import (
    Agent,
    Planner,
    Researcher,
    Analyst,
    Coder,
    Verifier,
    Critic,
    DecisionAgent,
    create_agent,
    register_role,
)
from ml.simulation.topologies import (
    BaseTopology,
    PipelineTopology,
    StarTopology,
    MeshTopology,
    create_topology,
    register_topology,
)
from ml.simulation.tasks import (
    BaseTask,
    ResearchTask,
    CodingTask,
    AnalysisTask,
    PlanningTask,
    create_task,
    register_task,
)
from ml.simulation.environment.orchestrator import SimulationEnvironment
from ml.simulation.run import SimulationRun
from ml.simulation.db import save_simulation_run

__all__ = [
    # Interfaces
    "AgentInterface",
    "TopologyInterface",
    "TaskInterface",
    "FaultInjectorInterface",
    "EnvironmentInterface",
    # Events
    "SimulationMessage",
    # Agents
    "Agent",
    "Planner",
    "Researcher",
    "Analyst",
    "Coder",
    "Verifier",
    "Critic",
    "DecisionAgent",
    "create_agent",
    "register_role",
    # Topologies
    "BaseTopology",
    "PipelineTopology",
    "StarTopology",
    "MeshTopology",
    "create_topology",
    "register_topology",
    # Tasks
    "BaseTask",
    "ResearchTask",
    "CodingTask",
    "AnalysisTask",
    "PlanningTask",
    "create_task",
    "register_task",
    # Environment & Runs
    "SimulationEnvironment",
    "SimulationRun",
    "save_simulation_run",
]
