"""Multi-agent task implementations and workflow registry."""

from typing import Dict, Type, Union, Optional, Any
from ml.config.experiment_config import TaskType
from ml.simulation.tasks.base import BaseTask
from ml.simulation.tasks.research import ResearchTask
from ml.simulation.tasks.coding import CodingTask
from ml.simulation.tasks.analysis import AnalysisTask
from ml.simulation.tasks.planning import PlanningTask

TASK_REGISTRY: Dict[str, Type[BaseTask]] = {
    "research": ResearchTask,
    "coding": CodingTask,
    "analysis": AnalysisTask,
    "data_analysis": AnalysisTask,
    "planning": PlanningTask,
}


def register_task(name: str):
    """Decorator to register a custom task workflow."""
    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        TASK_REGISTRY[name.lower().strip()] = cls
        return cls
    return decorator


def create_task(
    task_type: Union[TaskType, str],
    initial_prompt: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    **kwargs,
) -> BaseTask:
    """Factory method to instantiate a task workflow by type with validation.
    
    Args:
        task_type: Name or TaskType enum of workflow.
        initial_prompt: Optional task input parameters.
        seed: Random seed for deterministic simulation.
        
    Raises:
        ValueError: If task_type is unrecognized.
    """
    clean_name = task_type.value if isinstance(task_type, TaskType) else str(task_type).lower().strip()

    if clean_name not in TASK_REGISTRY:
        available = sorted(list(TASK_REGISTRY.keys()))
        raise ValueError(
            f"Invalid or unsupported task type: '{clean_name}'. Available: {available}"
        )

    cls = TASK_REGISTRY[clean_name]
    return cls(initial_prompt=initial_prompt, seed=seed, **kwargs)


__all__ = [
    "BaseTask",
    "ResearchTask",
    "CodingTask",
    "AnalysisTask",
    "PlanningTask",
    "TASK_REGISTRY",
    "register_task",
    "create_task",
]
