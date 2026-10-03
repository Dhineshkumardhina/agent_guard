"""Specialized Agent Roles and Registry for Multi-Agent Simulation.

Implements initial roles:
- Planner: Goal decomposition, workflow orchestration, architectural planning.
- Researcher: Information gathering, context synthesis, fact finding.
- Analyst: Statistical evaluation, pattern discovery, trend quantification.
- Coder: Algorithm implementation, software logic, syntax construction.
- Verifier: Constraint verification, quality assertion, boundary checking.
- Critic: Adversarial critique, flaw detection, contradiction analysis.
- DecisionAgent: Multi-criteria synthesis, consensus resolution, final decision.

Provides an extensible registry to allow new roles to be registered dynamically.
"""

from typing import Dict, Any, List, Optional, Type, Union
import random

from ml.config.experiment_config import AgentRole
from ml.simulation.agents.base import Agent
from ml.simulation.events import SimulationMessage


# Extensible Agent Role Registry
AGENT_ROLE_REGISTRY: Dict[str, Type[Agent]] = {}


def register_role(role_name: str):
    """Decorator to register a new agent role in the simulation ecosystem."""
    def decorator(cls: Type[Agent]) -> Type[Agent]:
        clean_name = role_name.lower().strip()
        AGENT_ROLE_REGISTRY[clean_name] = cls
        return cls
    return decorator


@register_role("planner")
class Planner(Agent):
    """Planner agent: decomposes goals into structured milestones and plans."""

    def __init__(
        self,
        agent_id: str = "agent_planner_0",
        name: str = "Planner",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.PLANNER,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "researcher")
        topic = context.get("task_input", {}).get("topic", "System Optimization")
        rng: random.Random = context.get("rng", random.Random(42))

        plan_items = [
            f"Phase 1: Formulate hypotheses for {topic}",
            "Phase 2: Partition empirical sub-objectives across specialists",
            "Phase 3: Establish verification thresholds and acceptance criteria",
        ]
        message = f"Execution Plan formulated for [{topic}]: " + " | ".join(plan_items)
        latency = round(rng.uniform(0.10, 0.35), 4)
        confidence = round(rng.uniform(0.90, 0.99), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="plan_specification",
            timestamp=context.get("timestamp", 0.0),
            metadata={"topic": topic, "milestones_count": len(plan_items)},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.95,
        )


@register_role("researcher")
class Researcher(Agent):
    """Researcher agent: gathers empirical facts and domain knowledge."""

    def __init__(
        self,
        agent_id: str = "agent_researcher_0",
        name: str = "Researcher",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.RESEARCHER,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "analyst")
        topic = context.get("task_input", {}).get("topic", "Multi-Agent Cascading Dynamics")
        rng: random.Random = context.get("rng", random.Random(42))

        findings = [
            f"Observation 1: Cascading failures exhibit non-linear propagation delays under {topic}",
            "Observation 2: Interaction bottlenecks correlate with elevated contradiction indices",
            "Observation 3: High centrality nodes show 2.4x higher failure transmission rates",
        ]
        message = f"Research Report on [{topic}]: " + " ; ".join(findings)
        latency = round(rng.uniform(0.15, 0.45), 4)
        confidence = round(rng.uniform(0.88, 0.98), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="research_findings",
            timestamp=context.get("timestamp", 0.0),
            metadata={"topic": topic, "num_observations": len(findings)},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.92,
        )


@register_role("analyst")
class Analyst(Agent):
    """Analyst agent: performs quantitative synthesis and statistical evaluations."""

    def __init__(
        self,
        agent_id: str = "agent_analyst_0",
        name: str = "Analyst",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.ANALYST,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "verifier")
        rng: random.Random = context.get("rng", random.Random(42))

        mean_val = round(rng.uniform(0.75, 0.92), 4)
        variance = round(rng.uniform(0.01, 0.05), 4)
        message = (
            f"Statistical Analysis: Evaluated signal metrics across memory trajectory. "
            f"Signal mean={mean_val}, variance={variance}. Hypothesis H1 supported (p < 0.01)."
        )
        latency = round(rng.uniform(0.12, 0.38), 4)
        confidence = round(rng.uniform(0.91, 0.99), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="analysis_summary",
            timestamp=context.get("timestamp", 0.0),
            metadata={"mean_signal": mean_val, "variance": variance, "significance": 0.01},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.94,
        )


@register_role("coder")
class Coder(Agent):
    """Coder agent: generates executable routines and software modules."""

    def __init__(
        self,
        agent_id: str = "agent_coder_0",
        name: str = "Coder",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.CODER,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "verifier")
        rng: random.Random = context.get("rng", random.Random(42))

        module_name = context.get("task_input", {}).get("module", "failure_detector")
        code_stub = f"def detect_cascades(graph, k=5):\n    return graph.compute_temporal_risk(k)\n"
        message = f"Implemented module [{module_name}] with standard interface contract:\n{code_stub}"
        latency = round(rng.uniform(0.20, 0.50), 4)
        confidence = round(rng.uniform(0.92, 0.99), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="code_implementation",
            timestamp=context.get("timestamp", 0.0),
            metadata={"module": module_name, "loc": 2},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.96,
        )


@register_role("verifier")
class Verifier(Agent):
    """Verifier agent: checks constraints, validates assertions, and verifies test suites."""

    def __init__(
        self,
        agent_id: str = "agent_verifier_0",
        name: str = "Verifier",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.VERIFIER,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "decision")
        rng: random.Random = context.get("rng", random.Random(42))

        checks_passed = 8
        total_checks = 8
        message = (
            f"Verification Audit: All {checks_passed}/{total_checks} invariant tests passed. "
            "Zero memory leaks, type safety verified, zero temporal future leakage detected."
        )
        latency = round(rng.uniform(0.10, 0.30), 4)
        confidence = round(rng.uniform(0.95, 1.0), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="verification_report",
            timestamp=context.get("timestamp", 0.0),
            metadata={"checks_passed": checks_passed, "total_checks": total_checks, "verified": True},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.98,
        )


@register_role("critic")
class Critic(Agent):
    """Critic agent: performs stress testing, identifies edge-case risks and logical inconsistencies."""

    def __init__(
        self,
        agent_id: str = "agent_critic_0",
        name: str = "Critic",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.CRITIC,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "decision")
        rng: random.Random = context.get("rng", random.Random(42))

        critique_points = [
            "Sensitivity analysis indicates vulnerability to high-latency star topologies",
            "Edge-case degradation bound meets safety margin (>0.85)",
        ]
        message = "Critic Review: Evaluated system risks and failure pathways. " + " | ".join(critique_points)
        latency = round(rng.uniform(0.12, 0.32), 4)
        confidence = round(rng.uniform(0.89, 0.97), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="critic_evaluation",
            timestamp=context.get("timestamp", 0.0),
            metadata={"risk_score": 0.12, "edge_cases_checked": 4},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.91,
        )


@register_role("decision")
@register_role("decisionagent")
class DecisionAgent(Agent):
    """DecisionAgent: aggregates specialist inputs and issues final deterministic consensus."""

    def __init__(
        self,
        agent_id: str = "agent_decision_0",
        name: str = "DecisionAgent",
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            name=name,
            role=AgentRole.DECISION,
            state=state,
            memory=memory,
            configuration=configuration,
        )

    def act(self, context: Dict[str, Any]) -> SimulationMessage:
        target = context.get("target_agent", "coordinator")
        rng: random.Random = context.get("rng", random.Random(42))

        message = (
            "Final Decision: Consensus established. All verification criteria met. "
            "Action approved for execution: STATUS=APPROVED."
        )
        latency = round(rng.uniform(0.08, 0.25), 4)
        confidence = round(rng.uniform(0.96, 1.0), 4)

        return self.send_message(
            target_agent=target,
            message=message,
            event_type="decision_output",
            timestamp=context.get("timestamp", 0.0),
            metadata={"decision": "APPROVED", "consensus_reached": True},
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
            latency=latency,
            confidence=confidence,
            output_quality=0.99,
        )


# Factory for creating agents
def create_agent(
    role: Union[AgentRole, str],
    agent_id: Optional[str] = None,
    name: Optional[str] = None,
    state: Optional[Dict[str, Any]] = None,
    memory: Optional[List[Any]] = None,
    configuration: Optional[Dict[str, Any]] = None,
) -> Agent:
    """Instantiate an agent by role name with validation.
    
    Args:
        role: Role name (string or AgentRole enum).
        agent_id: Optional explicit agent ID.
        name: Optional human-readable agent name.
        state: Optional initial state.
        memory: Optional initial memory.
        configuration: Optional configuration dictionary.
        
    Raises:
        ValueError: If the role is unknown or invalid.
    """
    clean_role = role.value if isinstance(role, AgentRole) else str(role).lower().strip()
    
    if clean_role not in AGENT_ROLE_REGISTRY:
        available = sorted(list(AGENT_ROLE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown agent role: '{clean_role}'. Available roles: {available}"
        )
    
    cls = AGENT_ROLE_REGISTRY[clean_role]
    chosen_id = agent_id or f"agent_{clean_role}_{random.randint(100, 999)}"
    chosen_name = name or cls.__name__
    
    return cls(
        agent_id=chosen_id,
        name=chosen_name,
        state=state,
        memory=memory,
        configuration=configuration,
    )
