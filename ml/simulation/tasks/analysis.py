"""Data Analysis Task Workflow Implementation.

Workflow sequence:
Researcher → Analyst → Verifier → Decision
"""

from typing import Dict, Any, List, Optional
from ml.config.experiment_config import TaskType, AgentRole
from ml.simulation.tasks.base import BaseTask
from ml.simulation.events import SimulationMessage


class AnalysisTask(BaseTask):
    """Analysis workflow: Researcher retrieves data -> Analyst analyzes -> Verifier checks -> Decision concludes."""

    def __init__(
        self,
        initial_prompt: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__(
            task_name="Empirical Trajectory Distribution Analysis",
            task_type=TaskType.DATA_ANALYSIS,
            initial_prompt=initial_prompt or {
                "dataset": "trajectory_telemetry_v1",
                "target_metric": "early_warning_lead_time",
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
            {"from_role": AgentRole.RESEARCHER, "to_role": AgentRole.ANALYST, "action": "extract_distributions"},
            {"from_role": AgentRole.ANALYST, "to_role": AgentRole.VERIFIER, "action": "fit_models"},
            {"from_role": AgentRole.VERIFIER, "to_role": AgentRole.DECISION, "action": "audit_leakage"},
            {"from_role": AgentRole.DECISION, "to_role": AgentRole.RESEARCHER, "action": "authorize_findings"},
        ]

    def generate_final_output(self, trajectory: List[SimulationMessage]) -> Dict[str, Any]:
        """Produce deterministic structured analysis output."""
        dataset = self._initial_prompt.get("dataset", "dataset_default")
        avg_confidence = (
            sum(e.confidence for e in trajectory) / len(trajectory) if trajectory else 1.0
        )
        total_tokens = sum(e.token_count for e in trajectory)

        return {
            "task_type": "data_analysis",
            "dataset": dataset,
            "status": "COMPLETED",
            "stages_executed": len(trajectory),
            "statistical_summary": {
                "sample_size": 1000,
                "auroc": 0.892,
                "lead_time_steps": 3.8,
                "p_value": 0.002,
            },
            "metrics": {
                "mean_confidence": round(avg_confidence, 4),
                "total_tokens": total_tokens,
                "verified": True,
            },
        }
