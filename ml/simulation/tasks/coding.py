"""Coding Task Workflow Implementation.

Workflow sequence:
Planner → Coder → Tester/Verifier → Critic
"""

from typing import Dict, Any, List, Optional
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.tasks.base import BaseTask
from ml.simulation.events import SimulationMessage


class CodingTask(BaseTask):
    """Coding workflow: Planner designs architecture -> Coder writes code -> Verifier tests -> Critic reviews."""

    def __init__(
        self,
        initial_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            task_name="Software Module Implementation & Audit",
            task_type=TaskType.CODING,
            initial_prompt=initial_prompt or {
                "module": "TemporalAttentionLayer",
                "specification": "Implement dynamic attention over agent communication graph edges",
            },
            seed=seed,
        )

    @property
    def required_roles(self) -> List[AgentRole]:
        return [
            AgentRole.PLANNER,
            AgentRole.CODER,
            AgentRole.VERIFIER,
            AgentRole.CRITIC,
        ]

    @property
    def workflow_stages(self) -> List[Dict[str, Any]]:
        return [
            {"from_role": AgentRole.PLANNER, "to_role": AgentRole.CODER, "action": "architectural_spec"},
            {"from_role": AgentRole.CODER, "to_role": AgentRole.VERIFIER, "action": "code_delivery"},
            {"from_role": AgentRole.VERIFIER, "to_role": AgentRole.CRITIC, "action": "test_verification"},
            {"from_role": AgentRole.CRITIC, "to_role": AgentRole.PLANNER, "action": "code_review_signoff"},
        ]

    def generate_final_output(self, trajectory: List[SimulationMessage]) -> Dict[str, Any]:
        """Produce deterministic structured code review and test report."""
        module = self._initial_prompt.get("module", "unnamed_module")
        avg_confidence = (
            sum(e.confidence for e in trajectory) / len(trajectory) if trajectory else 1.0
        )
        total_tokens = sum(e.token_count for e in trajectory)

        return {
            "task_type": "coding",
            "module": module,
            "status": "COMPLETED",
            "stages_executed": len(trajectory),
            "code_artifact": {
                "name": f"{module}.py",
                "tests_passed": 12,
                "code_coverage": 0.98,
                "lint_errors": 0,
            },
            "metrics": {
                "mean_confidence": round(avg_confidence, 4),
                "total_tokens": total_tokens,
                "verified": True,
            },
        }
