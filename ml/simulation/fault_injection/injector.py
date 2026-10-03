"""Controlled Fault Injection Engine for Multi-Agent Systems.

Implements all 12 canonical failure modes:
1. hallucinated_output
2. incorrect_information
3. tool_failure
4. tool_timeout
5. delayed_response
6. malformed_output
7. low_confidence_output
8. contradictory_output
9. communication_loop
10. incorrect_delegation
11. stale_context
12. agent_dropout
"""

from typing import Optional, Dict, Any, List, Union
import random
from uuid import uuid4

from ml.simulation.interfaces import FaultInjectorInterface
from ml.config.experiment_config import FaultType, FailureLevel
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.simulation.fault_injection.propagation import FaultInjectionRecord


ALL_FAULT_TYPES: List[str] = [
    "hallucinated_output",
    "incorrect_information",
    "tool_failure",
    "tool_timeout",
    "delayed_response",
    "malformed_output",
    "low_confidence_output",
    "contradictory_output",
    "communication_loop",
    "incorrect_delegation",
    "stale_context",
    "agent_dropout",
]


class FaultInjector(FaultInjectorInterface):
    """Engine for introducing reproducible, controlled faults into multi-agent workflows."""

    def __init__(
        self,
        fault_type: Union[FaultType, str] = FaultType.HALLUCINATED_OUTPUT,
        target_agent: Optional[str] = None,
        injection_step: Optional[int] = None,
        injection_time: Optional[float] = None,
        probability: float = 1.0,
        severity: float = 0.8,
        random_seed: int = 42,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize fault injector.
        
        Args:
            fault_type: Mode of failure to inject.
            target_agent: Optional ID/role of agent to infect (if None, targets any).
            injection_step: Specific trajectory step index to target.
            injection_time: Specific simulated timestamp after which to inject.
            probability: Likelihood of injection when step/agent matches (0.0 to 1.0).
            severity: Impact intensity (0.0 to 1.0).
            random_seed: Seed for reproducible pseudo-random behavior.
            metadata: Supplemental parameters.
        """
        # Normalize fault type
        if isinstance(fault_type, FaultType):
            self.fault_type: str = fault_type.value
        else:
            clean_type = str(fault_type).lower().strip()
            # Normalize contradictory_information alias
            if clean_type == "contradictory_information":
                clean_type = "contradictory_output"
            self.fault_type = clean_type

        if self.fault_type not in ALL_FAULT_TYPES and self.fault_type != "contradictory_information":
            raise ValueError(f"Unknown fault_type '{fault_type}'. Available: {ALL_FAULT_TYPES}")

        self.target_agent: Optional[str] = target_agent
        self.injection_step: Optional[int] = injection_step
        self.injection_time: Optional[float] = injection_time
        self.probability: float = max(0.0, min(1.0, probability))
        self.severity: float = max(0.0, min(1.0, severity))
        self.random_seed: int = random_seed
        self.metadata: Dict[str, Any] = metadata or {}

        self.rng = random.Random(random_seed)
        self.injected_records: List[FaultInjectionRecord] = []

    def should_inject(self, step_idx: int, event: AgentTelemetryEvent) -> bool:
        """Evaluate whether a fault should trigger on the current event."""
        # 1. Check target agent match
        if self.target_agent is not None:
            if event.source_agent != self.target_agent and event.target_agent != self.target_agent:
                return False

        # 2. Check injection step match
        if self.injection_step is not None:
            if step_idx != self.injection_step:
                return False

        # 3. Check injection time match
        if self.injection_time is not None:
            if event.timestamp is not None and event.timestamp < self.injection_time:
                return False

        # 4. Check probability
        roll = self.rng.random()
        return roll < self.probability

    def apply_fault(
        self,
        event: AgentTelemetryEvent,
        fault_type: Optional[Union[FaultType, str]] = None,
    ) -> AgentTelemetryEvent:
        """Mutate the telemetry event to inject the specified synthetic fault.
        
        Records every injection in an audit log.
        """
        active_type = self.fault_type
        if fault_type is not None:
            active_type = fault_type.value if isinstance(fault_type, FaultType) else str(fault_type).lower().strip()

        initial_level = 1
        description = ""

        # Apply specific fault mutations
        if active_type == "hallucinated_output":
            initial_level = 1
            event.message = (
                f"[HALLUCINATED_OUTPUT]: Asserts unverified claim: '100% convergence in zero steps'. "
                f"Fabricated non-existent paper citation (Smith et al. 2099)."
            )
            event.output_quality = max(0.05, round(1.0 - (0.8 * self.severity), 4))
            event.confidence = min(1.0, round(0.95 + (0.05 * self.severity), 4))  # False overconfidence
            event.injected_fault = "hallucinated_output"
            event.error_type = "hallucination"
            event.failure_label = 1
            description = "Injected false claim with high artificial confidence."

        elif active_type == "incorrect_information":
            initial_level = 1
            event.message = (
                f"[INCORRECT_INFORMATION]: Numerical values inverted: mean error reported as -42.8% (impossible), "
                f"statistical significance p=0.99 reported as validated."
            )
            event.output_quality = max(0.1, round(1.0 - (0.75 * self.severity), 4))
            event.injected_fault = "incorrect_information"
            event.error_type = "factual_inaccuracy"
            event.failure_label = 1
            description = "Factual numerical inversion injected."

        elif active_type == "tool_failure":
            initial_level = 1
            event.tool_used = "database_query_tool"
            event.tool_success = False
            event.tool_error = True
            event.message += " | [TOOL_FAILURE]: Fatal DatabaseConnectionTimeout: query aborted."
            event.output_quality = 0.1
            event.injected_fault = "tool_failure"
            event.error_type = "tool_execution_error"
            event.failure_label = 1
            description = "Tool call failed with fatal error."

        elif active_type == "tool_timeout":
            initial_level = 1
            event.tool_used = "vector_retrieval_tool"
            event.tool_success = False
            event.tool_error = True
            added_lat = round(5.0 * self.severity, 4)
            event.latency = round((event.latency or 0.0) + added_lat, 4)
            event.message += f" | [TOOL_TIMEOUT]: Execution exceeded deadline of 5000ms (+{added_lat}s)."
            event.output_quality = 0.2
            event.injected_fault = "tool_timeout"
            event.error_type = "timeout_error"
            event.failure_label = 1
            description = f"Tool operation timed out, adding {added_lat}s latency."

        elif active_type == "delayed_response":
            initial_level = 2
            added_lat = round(3.5 * self.severity, 4)
            event.latency = round((event.latency or 0.0) + added_lat, 4)
            event.message += f" | [DELAYED_RESPONSE]: Network stall caused {added_lat}s lag."
            event.injected_fault = "delayed_response"
            event.error_type = "latency_spike"
            event.failure_label = 2
            description = f"Injected delay latency spike of {added_lat}s."

        elif active_type == "malformed_output":
            initial_level = 1
            event.message = "{'status': 'error', 'data': <<<MALFORMED_UNPARSEABLE_JSON_SYNTAX_ERR>>>}"
            event.output_quality = 0.05
            event.tool_error = True
            event.injected_fault = "malformed_output"
            event.error_type = "syntax_corruption"
            event.failure_label = 1
            description = "Corrupted message payload syntax into unparseable tokens."

        elif active_type == "low_confidence_output":
            initial_level = 1
            event.confidence = max(0.05, round(0.25 - (0.20 * self.severity), 4))
            event.message += f" | [LOW_CONFIDENCE]: Agent flagged severe uncertainty (conf={event.confidence})."
            event.injected_fault = "low_confidence_output"
            event.error_type = "model_uncertainty"
            event.failure_label = 1
            description = "Drastically degraded self-reported confidence."

        elif active_type in ("contradictory_output", "contradictory_information"):
            initial_level = 2
            event.contradiction_score = min(1.0, round(0.85 + (0.15 * self.severity), 4))
            event.message = (
                f"[CONTRADICTION]: Directly contradicts previous consensus! "
                f"Previous hypothesis negated without empirical evidence."
            )
            event.output_quality = 0.2
            event.injected_fault = "contradictory_output"
            event.error_type = "semantic_contradiction"
            event.failure_label = 2
            description = "Injected contradictory statement with high contradiction score."

        elif active_type == "communication_loop":
            initial_level = 2
            event.retry_count = (event.retry_count or 0) + int(3 * self.severity) + 1
            event.event_type = "retry"
            event.message = f"[COMMUNICATION_LOOP]: Duplicate repeated query loop detected (retry={event.retry_count})."
            event.injected_fault = "communication_loop"
            event.error_type = "infinite_communication_loop"
            event.failure_label = 2
            description = "Injected cyclical communication loop with elevated retry count."

        elif active_type == "incorrect_delegation":
            initial_level = 2
            event.message = f"[INCORRECT_DELEGATION]: Misrouted task to incompatible specialist."
            event.injected_fault = "incorrect_delegation"
            event.error_type = "routing_failure"
            event.failure_label = 2
            description = "Routing misdirected to inappropriate agent."

        elif active_type == "stale_context":
            initial_level = 2
            event.message = (
                "[STALE_CONTEXT]: Using cached state from iteration -5. "
                "Current updates and intermediate corrections ignored."
            )
            event.output_quality = 0.3
            event.injected_fault = "stale_context"
            event.error_type = "cache_staleness"
            event.failure_label = 2
            description = "Injected outdated context ignoring fresh state."

        elif active_type == "agent_dropout":
            initial_level = 1
            event.message = "[AGENT_DROPOUT]: Heartbeat failed. Agent crashed or disconnected abruptly."
            event.output_quality = 0.0
            event.tool_error = True
            event.failure_label = 1
            event.injected_fault = "agent_dropout"
            event.error_type = "agent_crash"
            description = "Simulated agent node sudden dropout / crash."

        # Audit record
        record = FaultInjectionRecord(
            injection_id=str(uuid4()),
            run_id=event.run_id or "unknown_run",
            step_idx=event.step_idx or 0,
            timestamp=event.timestamp or 0.0,
            target_agent=event.source_agent,
            fault_type=active_type,
            severity=self.severity,
            parameters={
                "target_agent": self.target_agent,
                "severity": self.severity,
                "probability": self.probability,
            },
            failure_level=initial_level,
            description=description,
        )
        self.injected_records.append(record)
        return event

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset pseudo-random state and clear injection audit logs."""
        if seed is not None:
            self.random_seed = seed
        self.rng = random.Random(self.random_seed)
        self.injected_records.clear()
