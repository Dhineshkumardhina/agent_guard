"""Fault Propagation Tracking and Multi-Agent Failure Classification.

Formalizes the 3-level failure taxonomy for multi-agent systems:
- Level 1: Agent-Level Failure (Local failure, e.g. hallucination, tool error, timeout).
- Level 2: Interaction-Level Failure (Pairwise interaction failure, e.g. contradiction, loop, routing).
- Level 3: Cascading System Failure (System-wide cascade propagating across dependencies to final outcome).

Criteria for Level 3 Cascading Failure:
1. Originating fault injected at an initial agent (source of infection).
2. Failure propagates to >= 2 downstream dependent agents (multi-hop propagation).
3. Downstream agent outputs degrade or verification fails.
4. The final system objective/consensus fails or degrades to FAILED status.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from ml.config.experiment_config import FailureLevel, FaultType


@dataclass
class FaultInjectionRecord:
    """Audit record of a synthetic fault introduced into an interaction."""

    injection_id: str
    run_id: str
    step_idx: int
    timestamp: float
    target_agent: str
    fault_type: str
    severity: float
    parameters: Dict[str, Any] = field(default_factory=dict)
    failure_level: int = 1
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "run_id": self.run_id,
            "step_idx": self.step_idx,
            "timestamp": round(self.timestamp, 4),
            "target_agent": self.target_agent,
            "fault_type": self.fault_type,
            "severity": round(self.severity, 4),
            "parameters": self.parameters,
            "failure_level": self.failure_level,
            "description": self.description,
        }


class FaultPropagationTracker:
    """Tracks the causal chain of fault propagation across agent execution paths."""

    def __init__(self, run_id: str) -> None:
        self.run_id: str = run_id
        self.originating_agent: Optional[str] = None
        self.originating_step: Optional[int] = None
        self.originating_fault_type: Optional[str] = None
        self.affected_agents: List[str] = []
        self.propagation_sequence: List[str] = []
        self.failure_timestamps: List[float] = []
        self.failure_level: int = 0
        self.final_system_outcome: str = "SUCCESS"
        self.is_cascading: bool = False
        self.propagation_log: List[Dict[str, Any]] = []

    def record_originating_fault(
        self,
        agent_id: str,
        step_idx: int,
        timestamp: float,
        fault_type: str,
        initial_level: int = 1,
    ) -> None:
        """Mark the initial inception of a fault."""
        if self.originating_agent is None:
            self.originating_agent = agent_id
            self.originating_step = step_idx
            self.originating_fault_type = fault_type
            self.failure_level = max(self.failure_level, initial_level)

        if agent_id not in self.affected_agents:
            self.affected_agents.append(agent_id)
        if not self.propagation_sequence or self.propagation_sequence[-1] != agent_id:
            self.propagation_sequence.append(agent_id)
        self.failure_timestamps.append(round(timestamp, 4))

        self.propagation_log.append({
            "step_idx": step_idx,
            "timestamp": round(timestamp, 4),
            "agent": agent_id,
            "action": "originating_fault",
            "fault_type": fault_type,
            "level": initial_level,
        })

    def record_downstream_effect(
        self,
        sender_id: str,
        receiver_id: str,
        step_idx: int,
        timestamp: float,
        reason: str,
        degraded: bool = True,
    ) -> None:
        """Track propagation of failure from sender to receiver."""
        if receiver_id not in self.affected_agents:
            self.affected_agents.append(receiver_id)

        if not self.propagation_sequence or self.propagation_sequence[-1] != receiver_id:
            self.propagation_sequence.append(receiver_id)

        self.failure_timestamps.append(round(timestamp, 4))

        # Check cascading criteria
        # Propagation across >= 2 dependent agents after originating agent
        downstream_count = len([a for a in self.affected_agents if a != self.originating_agent])
        if downstream_count >= 2:
            self.failure_level = 3  # Level 3 Cascading
            self.is_cascading = True
        elif downstream_count == 1:
            self.failure_level = max(self.failure_level, 2)  # Level 2 Interaction

        self.propagation_log.append({
            "step_idx": step_idx,
            "timestamp": round(timestamp, 4),
            "sender": sender_id,
            "receiver": receiver_id,
            "action": "downstream_propagation",
            "reason": reason,
            "degraded": degraded,
            "current_failure_level": self.failure_level,
        })

    def finalize_outcome(self, outcome: str) -> None:
        """Finalize the system outcome and evaluate if cascade caused final failure."""
        self.final_system_outcome = outcome
        if outcome in ("FAILED", "CASCADING_FAILURE"):
            if len(self.affected_agents) >= 2:
                self.is_cascading = True
                self.failure_level = 3
                if not self.propagation_sequence or self.propagation_sequence[-1] != "Final Failure":
                    self.propagation_sequence.append("Final Failure")
        elif outcome == "SUCCESS" and self.failure_level < 3:
            self.is_cascading = False

    def to_path_string(self) -> str:
        """Return human-readable arrow representation of propagation path.
        
        Example: Researcher -> Analyst -> Verifier -> Final Failure
        """
        if not self.propagation_sequence:
            return "No Fault Injected"
        return " -> ".join(self.propagation_sequence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert propagation details into a structured dictionary."""
        return {
            "run_id": self.run_id,
            "originating_agent": self.originating_agent,
            "originating_step": self.originating_step,
            "originating_fault_type": self.originating_fault_type,
            "affected_agents": list(self.affected_agents),
            "propagation_sequence": list(self.propagation_sequence),
            "propagation_path": self.to_path_string(),
            "failure_timestamps": list(self.failure_timestamps),
            "failure_level": self.failure_level,
            "is_cascading": self.is_cascading,
            "final_system_outcome": self.final_system_outcome,
            "total_affected": len(self.affected_agents),
        }
