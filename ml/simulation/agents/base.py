"""Base Agent Abstraction for Multi-Agent Simulation.

Every agent maintains identity, functional role, mutable internal state,
interaction memory, and static configuration.
"""

from typing import Dict, Any, List, Optional, Union
import copy

from ml.simulation.interfaces import AgentInterface
from ml.simulation.events import SimulationMessage
from ml.config.experiment_config import AgentRole
from ml.telemetry.schemas import AgentTelemetryEvent


class Agent(AgentInterface):
    """Base class for all multi-agent simulation agents."""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: Union[AgentRole, str],
        state: Optional[Dict[str, Any]] = None,
        memory: Optional[List[Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize base agent attributes.
        
        Args:
            agent_id: Unique string identifier for the agent.
            name: Human-readable name.
            role: Functional role assigned to agent.
            state: Initial mutable state dictionary.
            memory: Initial list of remembered events/interactions.
            configuration: Static hyperparameters and capabilities.
        """
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError(f"agent_id must be a non-empty string, got: {agent_id!r}")
        if not name or not isinstance(name, str):
            raise ValueError(f"name must be a non-empty string, got: {name!r}")

        self._agent_id: str = agent_id
        self._name: str = name
        
        # Handle string or Enum roles cleanly
        if isinstance(role, AgentRole):
            self._role: AgentRole = role
        elif isinstance(role, str):
            clean_role = role.lower().strip()
            matched = False
            for r in AgentRole:
                if r.value == clean_role:
                    self._role = r
                    matched = True
                    break
            if not matched:
                # Custom or newly registered role represented as string
                self._role = clean_role  # type: ignore
        else:
            raise ValueError(f"Invalid role type: {type(role)}")

        self._state: Dict[str, Any] = state if state is not None else {"status": "active", "step_count": 0}
        self._memory: List[Any] = memory if memory is not None else []
        self._configuration: Dict[str, Any] = configuration if configuration is not None else {}

        # Cache baseline copies for reset() reproducibility
        self._initial_state: Dict[str, Any] = copy.deepcopy(self._state)
        self._initial_memory: List[Any] = copy.deepcopy(self._memory)

    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._agent_id

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self._name

    @property
    def role(self) -> Union[AgentRole, str]:
        """Functional role assigned to the agent."""
        return self._role

    @property
    def state(self) -> Dict[str, Any]:
        """Mutable internal state dictionary."""
        return self._state

    @state.setter
    def state(self, new_state: Dict[str, Any]) -> None:
        if not isinstance(new_state, dict):
            raise TypeError("Agent state must be a dictionary")
        self._state = new_state

    @property
    def memory(self) -> List[Any]:
        """Sequence of received messages and telemetry records."""
        return self._memory

    @property
    def configuration(self) -> Dict[str, Any]:
        """Immutable agent configuration."""
        return self._configuration

    def receive_event(self, event: Union[SimulationMessage, AgentTelemetryEvent, Dict[str, Any]]) -> None:
        """Process and append an incoming interaction event to agent memory.
        
        Args:
            event: An incoming SimulationMessage, AgentTelemetryEvent, or dict.
        """
        self._memory.append(event)
        self._state["step_count"] = self._state.get("step_count", 0) + 1
        self._state["last_received_from"] = getattr(event, "source_agent", None) if not isinstance(event, dict) else event.get("source_agent")

    def send_message(
        self,
        target_agent: str,
        message: str,
        event_type: str = "message",
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        step_idx: int = 0,
        run_id: str = "",
        **kwargs: Any,
    ) -> SimulationMessage:
        """Construct a structured outgoing interaction message.
        
        Args:
            target_agent: Recipient agent identifier.
            message: Message payload content.
            event_type: Type classification of the event.
            timestamp: Simulation time offset in seconds.
            metadata: Supplemental structured dictionary.
            step_idx: Trajectory step index.
            run_id: Associated trajectory run identifier.
        """
        if not target_agent or not isinstance(target_agent, str):
            raise ValueError(f"target_agent must be a non-empty string, got: {target_agent!r}")

        event = SimulationMessage(
            source_agent=self.agent_id,
            target_agent=target_agent,
            timestamp=timestamp,
            message=message,
            event_type=event_type,
            metadata=metadata or {},
            step_idx=step_idx,
            run_id=run_id,
            **kwargs,
        )
        self._state["last_sent_to"] = target_agent
        self._state["last_sent_step"] = step_idx
        return event

    def act(self, context: Dict[str, Any]) -> Optional[SimulationMessage]:
        """Execute agent cognition or tool computation given task context.
        
        Should be overridden by specialized subclasses.
        """
        target = context.get("target_agent", "coordinator")
        return self.send_message(
            target_agent=target,
            message=f"Agent {self.name} acknowledged context.",
            event_type="acknowledgment",
            timestamp=context.get("timestamp", 0.0),
            step_idx=context.get("step_idx", 0),
            run_id=context.get("run_id", ""),
        )

    def reset(self) -> None:
        """Reset internal agent memory and mutable state to initial baseline."""
        self._state = copy.deepcopy(self._initial_state)
        self._memory = copy.deepcopy(self._initial_memory)

    def __repr__(self) -> str:
        role_str = self.role.value if isinstance(self.role, AgentRole) else str(self.role)
        return f"<Agent id='{self.agent_id}' name='{self.name}' role='{role_str}'>"
