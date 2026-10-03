"""Simulator Interfaces and Abstract Base Classes.

Provides rigorous abstract contracts for Agents, Communication Topologies,
Environments, Tasks, and Fault Injectors to ensure modularity and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import networkx as nx

from ml.telemetry.schemas import AgentTelemetryEvent
from ml.config.experiment_config import AgentRole, TopologyType, FaultType


class AgentInterface(ABC):
    """Abstract interface for an agent participating in multi-agent execution."""

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique agent identifier."""
        pass

    @property
    @abstractmethod
    def role(self) -> AgentRole:
        """Functional role assigned to the agent."""
        pass

    @abstractmethod
    def receive_event(self, event: AgentTelemetryEvent) -> None:
        """Receive an incoming interaction event from another agent."""
        pass

    @abstractmethod
    def act(self, context: Dict[str, Any]) -> Optional[AgentTelemetryEvent]:
        """Perform a computation or action, generating an outgoing telemetry event."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal agent memory/state between runs."""
        pass


class TopologyInterface(ABC):
    """Abstract interface for multi-agent communication topologies."""

    @property
    @abstractmethod
    def topology_type(self) -> TopologyType:
        """The topology category."""
        pass

    @abstractmethod
    def build_graph(self, agent_ids: List[str]) -> nx.DiGraph:
        """Construct the NetworkX directed graph defining permitted communication paths."""
        pass

    @abstractmethod
    def can_communicate(self, source: str, target: str) -> bool:
        """Check whether source agent is permitted to send a direct message to target agent."""
        pass


class TaskInterface(ABC):
    """Abstract interface for a simulated task workflow."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Name of the task."""
        pass

    @abstractmethod
    def get_initial_prompt(self) -> Dict[str, Any]:
        """Provide initial task specification and inputs."""
        pass

    @abstractmethod
    def is_complete(self, trajectory: List[AgentTelemetryEvent]) -> bool:
        """Check if task objectives have been satisfied."""
        pass

    @abstractmethod
    def evaluate_success(self, trajectory: List[AgentTelemetryEvent]) -> bool:
        """Evaluate if final system execution was successful and free from cascading failure."""
        pass


class FaultInjectorInterface(ABC):
    """Abstract interface for injecting faults into agent interactions."""

    @abstractmethod
    def should_inject(self, step_idx: int, event: AgentTelemetryEvent) -> bool:
        """Determine whether a fault should be injected at the current step."""
        pass

    @abstractmethod
    def apply_fault(self, event: AgentTelemetryEvent, fault_type: Optional[FaultType] = None) -> AgentTelemetryEvent:
        """Mutate the event to introduce the specified fault mode."""
        pass

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal pseudo-random state with a designated seed."""
        pass


class EnvironmentInterface(ABC):
    """Abstract interface for the multi-agent simulation orchestrator."""

    @abstractmethod
    def register_agent(self, agent: AgentInterface) -> None:
        """Add an agent to the environment."""
        pass

    @abstractmethod
    def set_topology(self, topology: TopologyInterface) -> None:
        """Configure the communication topology."""
        pass

    @abstractmethod
    def set_task(self, task: TaskInterface) -> None:
        """Assign the task to be solved."""
        pass

    @abstractmethod
    def set_fault_injector(self, injector: FaultInjectorInterface) -> None:
        """Attach fault injection engine."""
        pass

    @abstractmethod
    def run_trajectory(self, run_id: str, max_steps: int = 50) -> List[AgentTelemetryEvent]:
        """Execute a full simulation run, returning the chronological sequence of events."""
        pass
