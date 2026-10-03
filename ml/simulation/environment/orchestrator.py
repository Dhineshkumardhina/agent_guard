"""Multi-Agent Simulation Orchestrator and Environment.

Coordinates agent lifecycles, communication topology constraints,
task workflow execution, and deterministic event telemetry collection.
"""

from typing import List, Dict, Any, Optional
import random

from ml.simulation.interfaces import EnvironmentInterface, FaultInjectorInterface
from ml.simulation.agents.base import Agent
from ml.simulation.topologies.base import BaseTopology
from ml.simulation.tasks.base import BaseTask
from ml.simulation.events import SimulationMessage
from ml.telemetry.schemas import AgentTelemetryEvent


class SimulationEnvironment(EnvironmentInterface):
    """Execution environment that enforces topology routing and executes task workflows."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._agents: Dict[str, Agent] = {}
        self._topology: Optional[BaseTopology] = None
        self._task: Optional[BaseTask] = None
        self._fault_injector: Optional[FaultInjectorInterface] = None
        self._history: List[SimulationMessage] = []
        self._sim_time: float = 0.0

    @property
    def agents(self) -> Dict[str, Agent]:
        return self._agents

    @property
    def topology(self) -> Optional[BaseTopology]:
        return self._topology

    @property
    def task(self) -> Optional[BaseTask]:
        return self._task

    @property
    def history(self) -> List[SimulationMessage]:
        return list(self._history)

    def register_agent(self, agent: Agent) -> None:
        """Register an agent into the simulation environment."""
        if not isinstance(agent, Agent):
            raise TypeError(f"Expected Agent instance, got {type(agent)}")
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent with ID '{agent.agent_id}' already registered")
        self._agents[agent.agent_id] = agent

    def set_topology(self, topology: BaseTopology) -> None:
        """Configure the communication topology."""
        if not isinstance(topology, BaseTopology):
            raise TypeError(f"Expected BaseTopology instance, got {type(topology)}")
        self._topology = topology

    def set_task(self, task: BaseTask) -> None:
        """Assign the task workflow to be solved."""
        if not isinstance(task, BaseTask):
            raise TypeError(f"Expected BaseTask instance, got {type(task)}")
        self._task = task

    def set_fault_injector(self, injector: FaultInjectorInterface) -> None:
        """Attach fault injection engine (extensible for Phase 3)."""
        self._fault_injector = injector

    def step(
        self,
        source_agent_id: str,
        target_agent_id: str,
        step_idx: int,
        run_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> SimulationMessage:
        """Execute a single communication step between two agents under topology enforcement.
        
        Raises:
            ValueError: If agents do not exist or communication is forbidden by topology.
        """
        if source_agent_id not in self._agents:
            raise ValueError(f"Source agent '{source_agent_id}' not found in registered agents")
        if target_agent_id not in self._agents:
            raise ValueError(f"Target agent '{target_agent_id}' not found in registered agents")

        if self._topology is not None:
            if not self._topology.can_communicate(source_agent_id, target_agent_id):
                raise ValueError(
                    f"Topology '{self._topology.topology_type.value}' prohibits direct communication "
                    f"from '{source_agent_id}' to '{target_agent_id}'"
                )

        source_agent = self._agents[source_agent_id]
        target_agent = self._agents[target_agent_id]

        step_context = {
            "target_agent": target_agent_id,
            "step_idx": step_idx,
            "run_id": run_id,
            "timestamp": round(self._sim_time, 4),
            "rng": self._rng,
            "task_input": self._task.get_initial_prompt() if self._task else {},
        }
        if context:
            step_context.update(context)

        # Source agent acts
        msg = source_agent.act(step_context)
        if msg is None:
            msg = source_agent.send_message(
                target_agent=target_agent_id,
                message=f"Agent {source_agent.name} completed step {step_idx}",
                timestamp=round(self._sim_time, 4),
                step_idx=step_idx,
                run_id=run_id,
            )

        # Apply fault injection hook if active (Phase 3 extension point)
        if self._fault_injector and self._fault_injector.should_inject(step_idx, msg.to_telemetry_event()):
            mutated = self._fault_injector.apply_fault(msg.to_telemetry_event())
            msg.injected_fault = mutated.injected_fault
            msg.error_type = mutated.error_type
            msg.failure_label = mutated.failure_label
            msg.tool_error = mutated.tool_error

        # Advance simulated time deterministically by message latency
        self._sim_time += max(0.05, getattr(msg, "latency", 0.1))

        # Target agent receives message
        target_agent.receive_event(msg)
        self._history.append(msg)
        return msg

    def run_trajectory(self, run_id: str, max_steps: int = 50) -> List[AgentTelemetryEvent]:
        """Execute simulation trajectory and return telemetry events."""
        telemetry_events = [m.to_telemetry_event() for m in self._history]
        return telemetry_events

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset environment, clocks, and all registered agents."""
        if seed is not None:
            self._seed = seed
        self._rng = random.Random(self._seed)
        self._sim_time = 0.0
        self._history.clear()
        for agent in self._agents.values():
            agent.reset()
        if self._task:
            self._task.reset(self._seed)
