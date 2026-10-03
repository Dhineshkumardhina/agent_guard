"""Base Task Workflow Abstraction for Controlled Multi-Agent Benchmarks.

Provides deterministic step generation, role routing, and success validation.
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional, Union
import random

from ml.simulation.interfaces import TaskInterface
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.events import SimulationMessage


class BaseTask(TaskInterface):
    """Abstract base class for deterministic multi-agent benchmark tasks."""

    def __init__(
        self,
        task_name: str,
        task_type: TaskType,
        initial_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self._task_name = task_name
        self._task_type = task_type
        self._initial_prompt = initial_prompt or {"topic": "Multi-Agent System Reliability"}
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def task_type(self) -> TaskType:
        return self._task_type

    def get_initial_prompt(self) -> Dict[str, Any]:
        return dict(self._initial_prompt)

    @property
    @abstractmethod
    def required_roles(self) -> List[Union[AgentRole, str]]:
        """List of functional agent roles required to complete the task workflow."""
        pass

    @property
    @abstractmethod
    def workflow_stages(self) -> List[Dict[str, Any]]:
        """Ordered sequence of interaction stages defining the task workflow."""
        pass

    def is_complete(self, trajectory: List[Any]) -> bool:
        """Check whether all stages of the workflow have completed."""
        return len(trajectory) >= len(self.workflow_stages)

    def evaluate_success(self, trajectory: List[Any]) -> bool:
        """Evaluate if trajectory reached valid completion without failure."""
        if not self.is_complete(trajectory):
            return False
        # Verify no failure flags were raised
        for ev in trajectory:
            if getattr(ev, "failure_label", 0) > 0 or getattr(ev, "tool_error", False):
                return False
        return True

    @abstractmethod
    def generate_final_output(self, trajectory: List[SimulationMessage]) -> Dict[str, Any]:
        """Produce deterministic, structured final output summary from completed trajectory."""
        pass

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset internal pseudo-random state for deterministic replay."""
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
