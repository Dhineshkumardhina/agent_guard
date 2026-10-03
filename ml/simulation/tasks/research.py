"""Research Task Workflow Implementation.

Workflow sequence:
Researcher → Analyst → Verifier → Decision
"""

from typing import Dict, Any, List, Optional
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.tasks.base import BaseTask
from ml.simulation.events import SimulationMessage


class ResearchTask(BaseTask):
    """Research workflow: Researcher gathers facts -> Analyst analyzes -> Verifier checks -> Decision concludes."""

    def __init__(
        self,
        initial_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            task_name="Empirical Research Investigation",
            task_type=TaskType.RESEARCH,
            initial_prompt=initial_prompt or {
                "topic": "Temporal Cascading Failures in Agent Swarms",
                "hypothesis": "Temporal interaction graphs predict failures k steps earlier than baselines",
            },
            seed=seed,
        )

    @property
    def required_roles(self) -> List[AgentRole]:
        return [
            AgentRole.RESEARCHER,
            AgentRole.ANALYST,
            AgentRole.VERIFIER,
            AgentRole.DECISION,
        ]

    @property
    def workflow_stages(self) -> List[Dict[str, Any]]:
        return [
            {"from_role": AgentRole.RESEARCHER, "to_role": AgentRole.ANALYST, "action": "gather_evidence"},
            {"from_role": AgentRole.ANALYST, "to_role": AgentRole.VERIFIER, "action": "synthesize_statistics"},
            {"from_role": AgentRole.VERIFIER, "to_role": AgentRole.DECISION, "action": "verify_findings"},
            {"from_role": AgentRole.DECISION, "to_role": AgentRole.RESEARCHER, "action": "deliver_conclusion"},
        ]

    def generate_final_output(self, trajectory: List[SimulationMessage]) -> Dict[str, Any]:
        """Produce deterministic structured research findings summary."""
        topic = self._initial_prompt.get("topic", "")
        # Compute deterministic metrics based on trajectory event metrics
        avg_confidence = (
            sum(e.confidence for e in trajectory) / len(trajectory) if trajectory else 1.0
        )
        avg_quality = (
            sum(e.output_quality for e in trajectory) / len(trajectory) if trajectory else 1.0
        )
        total_tokens = sum(e.token_count for e in trajectory)

        return {
            "task_type": "research",
            "topic": topic,
            "status": "COMPLETED",
            "stages_executed": len(trajectory),
            "findings_summary": (
                f"Completed empirical research on '{topic}'. Interaction patterns "
                "confirm structural failure signals across stages."
            ),
            "metrics": {
                "mean_confidence": round(avg_confidence, 4),
                "mean_quality": round(avg_quality, 4),
                "total_tokens": total_tokens,
                "verified": True,
            },
        }
