"""Strategic Planning Task Workflow Implementation.

Workflow sequence:
Planner → Specialists → Critic → Decision
"""

from typing import Dict, Any, List, Optional
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.tasks.base import BaseTask
from ml.simulation.events import SimulationMessage


class PlanningTask(BaseTask):
    """Planning workflow: Planner creates plan -> Specialists review -> Critic stress-tests -> Decision approves."""

    def __init__(
        self,
        initial_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            task_name="Strategic System Architecture Planning",
            task_type=TaskType.PLANNING,
            initial_prompt=initial_prompt or {
                "mission": "Deploy Fault-Tolerant Multi-Agent Orchestration",
                "risk_tolerance": "low",
            },
            seed=seed,
        )

    @property
    def required_roles(self) -> List[AgentRole]:
        return [
            AgentRole.PLANNER,
            AgentRole.ANALYST,
            AgentRole.CRITIC,
            AgentRole.DECISION,
        ]

    @property
    def workflow_stages(self) -> List[Dict[str, Any]]:
        return [
            {"from_role": AgentRole.PLANNER, "to_role": AgentRole.ANALYST, "action": "draft_architecture"},
            {"from_role": AgentRole.ANALYST, "to_role": AgentRole.CRITIC, "action": "feasibility_assessment"},
            {"from_role": AgentRole.CRITIC, "to_role": AgentRole.DECISION, "action": "risk_audit"},
            {"from_role": AgentRole.DECISION, "to_role": AgentRole.PLANNER, "action": "charter_ratification"},
        ]

    def generate_final_output(self, trajectory: List[SimulationMessage]) -> Dict[str, Any]:
        """Produce deterministic structured strategic plan output."""
        mission = self._initial_prompt.get("mission", "default_mission")
        avg_confidence = (
            sum(e.confidence for e in trajectory) / len(trajectory) if trajectory else 1.0
        )
        total_tokens = sum(e.token_count for e in trajectory)

        return {
            "task_type": "planning",
            "mission": mission,
            "status": "COMPLETED",
            "stages_executed": len(trajectory),
            "plan_approved": True,
            "strategic_charter": {
                "phases": 4,
                "contingency_buffers": "included",
                "consensus_level": 1.0,
            },
            "metrics": {
                "mean_confidence": round(avg_confidence, 4),
                "total_tokens": total_tokens,
                "verified": True,
            },
        }
