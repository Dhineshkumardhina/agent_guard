"""SimulationRun Abstraction for Controlled Multi-Agent Executions.

Coordinates agents, topology, task workflows, deterministic random state,
telemetry recording, and controlled fault injection / propagation tracking.
"""

from typing import List, Dict, Any, Optional, Union
from uuid import uuid4
import random

from sqlalchemy.orm import Session

from ml.config.experiment_config import TaskType, TopologyType
from ml.simulation.agents.base import Agent
from ml.simulation.agents.roles import (
    Planner,
    Researcher,
    Analyst,
    Coder,
    Verifier,
    Critic,
    DecisionAgent,
    create_agent_roster,
)
from ml.simulation.topologies import (
    BaseTopology,
    PipelineTopology,
    StarTopology,
    MeshTopology,
    create_topology,
)
from ml.simulation.tasks import BaseTask, create_task
from ml.simulation.environment.orchestrator import SimulationEnvironment
from ml.simulation.events import SimulationMessage
from ml.simulation.db import save_simulation_run
from ml.simulation.fault_injection.injector import FaultInjector
from ml.simulation.fault_injection.propagation import FaultPropagationTracker


class SimulationRun:
    """Manages the full lifecycle of a single multi-agent simulation run."""

    def __init__(
        self,
        run_id: Optional[str] = None,
        task_type: Union[TaskType, str] = TaskType.RESEARCH,
        topology: Union[TopologyType, str, BaseTopology] = TopologyType.PIPELINE,
        agents: Optional[List[Agent]] = None,
        random_seed: int = 42,
        task_input: Optional[Dict[str, Any]] = None,
        max_steps: int = 50,
        fault_injector: Optional[FaultInjector] = None,
        fault_config: Optional[Dict[str, Any]] = None,
        num_agents: Optional[int] = None,
    ) -> None:
        """Initialize simulation run parameters.
        
        Args:
            run_id: Optional unique identifier. Generated automatically if omitted.
            task_type: Type of benchmark task workflow to execute.
            topology: Topology pattern name, enum, or BaseTopology instance.
            agents: Optional list of Agent instances. Created automatically if omitted.
            random_seed: Seed for reproducible pseudo-random behavior.
            task_input: Custom dictionary of parameters passed to task.
            max_steps: Maximum allowable steps before safety cutoff.
            fault_injector: Optional FaultInjector instance for controlled failure injection.
            fault_config: Optional dictionary to instantiate a FaultInjector.
            num_agents: Optional number of agents to initialize if agents list is not provided.
        """
        self.run_id: str = run_id or f"run_{uuid4().hex[:12]}"
        
        # Normalize task_type
        if isinstance(task_type, TaskType):
            self.task_type: str = task_type.value
        else:
            self.task_type = str(task_type).lower().strip()

        # Normalize topology
        if isinstance(topology, BaseTopology):
            self._topology_instance: Optional[BaseTopology] = topology
            self.topology: str = topology.topology_type.value
        elif isinstance(topology, TopologyType):
            self._topology_instance = None
            self.topology = topology.value
        else:
            self._topology_instance = None
            self.topology = str(topology).lower().strip()

        self.random_seed: int = random_seed
        self.max_steps: int = max_steps
        self.task_input: Dict[str, Any] = task_input or {}

        # Lifecycle tracking
        self.start_time: float = 0.0
        self.end_time: Optional[float] = None
        self.duration_seconds: float = 0.0
        self.final_status: str = "INITIALIZED"
        self.has_cascading_failure: bool = False
        self.cascading_failure_step: Optional[int] = None
        self.events: List[SimulationMessage] = []
        self.task_output: Optional[Dict[str, Any]] = None

        # Fault injection & propagation tracking
        if fault_injector is not None:
            self.fault_injector: Optional[FaultInjector] = fault_injector
        elif fault_config is not None:
            config = dict(fault_config)
            if "random_seed" not in config:
                config["random_seed"] = self.random_seed
            self.fault_injector = FaultInjector(**config)
        else:
            self.fault_injector = None

        self.propagation_tracker: FaultPropagationTracker = FaultPropagationTracker(run_id=self.run_id)

        # Seeded local PRNG
        self._rng = random.Random(random_seed)

        # Setup agents
        if agents is not None:
            self.agents: List[Agent] = list(agents)
        elif num_agents is not None:
            self.agents = create_agent_roster(num_agents=num_agents)
        else:
            self.agents = self._create_default_agents()

        # Build topology instance
        if self._topology_instance is None:
            agent_ids = [a.agent_id for a in self.agents]
            self._topology_instance = create_topology(self.topology, agent_ids=agent_ids)
        else:
            if not self._topology_instance.agent_ids:
                self._topology_instance.build_graph([a.agent_id for a in self.agents])

        # Task instance
        self.task: BaseTask = create_task(self.task_type, initial_prompt=self.task_input, seed=self.random_seed)

    def _create_default_agents(self) -> List[Agent]:
        """Construct standard default set of 5 agents for simulation."""
        return [
            Planner(agent_id="planner_1", name="Planner Alpha"),
            Researcher(agent_id="researcher_1", name="Researcher Beta"),
            Analyst(agent_id="analyst_1", name="Analyst Gamma"),
            Verifier(agent_id="verifier_1", name="Verifier Delta"),
            DecisionAgent(agent_id="decision_1", name="Decision Epsilon"),
        ]

    def execute(self) -> "SimulationRun":
        """Execute the multi-agent task under topology routing rules and fault injection.
        
        Guaranteed to produce identical event sequences and outputs when given
        identical configuration and random seed.
        """
        self.final_status = "RUNNING"
        self.start_time = 0.0
        self.events.clear()
        self.has_cascading_failure = False
        self.cascading_failure_step = None
        self.propagation_tracker = FaultPropagationTracker(run_id=self.run_id)

        if self.fault_injector:
            self.fault_injector.reset(self.random_seed)

        # Reset components with seeded state
        env = SimulationEnvironment(seed=self.random_seed)
        for agent in self.agents:
            agent.reset()
            env.register_agent(agent)

        env.set_topology(self._topology_instance)
        env.set_task(self.task)
        if self.fault_injector:
            env.set_fault_injector(self.fault_injector)

        sim_time = 0.0
        step_idx = 0

        def process_step(src_agent: Agent, dst_agent: Agent, context: Optional[Dict[str, Any]] = None) -> SimulationMessage:
            nonlocal sim_time, step_idx
            
            # Check if source agent was previously infected by an upstream fault
            is_downstream = (
                self.propagation_tracker.originating_agent is not None
                and src_agent.agent_id in self.propagation_tracker.affected_agents
            )

            msg = env.step(
                source_agent_id=src_agent.agent_id,
                target_agent_id=dst_agent.agent_id,
                step_idx=step_idx,
                run_id=self.run_id,
                context=context,
            )

            # If fault was injected at this step
            if msg.injected_fault:
                self.propagation_tracker.record_originating_fault(
                    agent_id=src_agent.agent_id,
                    step_idx=step_idx,
                    timestamp=msg.timestamp,
                    fault_type=msg.injected_fault,
                    initial_level=msg.failure_label,
                )
                # Degrade receiver memory state
                self.propagation_tracker.record_downstream_effect(
                    sender_id=src_agent.agent_id,
                    receiver_id=dst_agent.agent_id,
                    step_idx=step_idx,
                    timestamp=msg.timestamp,
                    reason=f"Received {msg.injected_fault} from {src_agent.name}",
                    degraded=True,
                )
            elif is_downstream:
                # Downstream cascading propagation
                msg.output_quality = max(0.1, round(msg.output_quality * 0.5, 4))
                msg.confidence = max(0.2, round(msg.confidence * 0.6, 4))
                msg.downstream_failure = True
                self.propagation_tracker.record_downstream_effect(
                    sender_id=src_agent.agent_id,
                    receiver_id=dst_agent.agent_id,
                    step_idx=step_idx,
                    timestamp=msg.timestamp,
                    reason=f"Compounded degradation propagated from {src_agent.name}",
                    degraded=True,
                )

            self.events.append(msg)
            sim_time += msg.latency
            step_idx += 1
            return msg

        # Execute according to topology category
        if self.topology == "pipeline":
            for i in range(len(self.agents) - 1):
                if step_idx >= self.max_steps:
                    break
                process_step(self.agents[i], self.agents[i + 1])

        elif self.topology == "star":
            hub_id = getattr(self._topology_instance, "hub_agent_id", self.agents[0].agent_id)
            hub_agent = next(a for a in self.agents if a.agent_id == hub_id)
            leaves = [a for a in self.agents if a.agent_id != hub_id]

            for leaf in leaves:
                if step_idx >= self.max_steps:
                    break
                # Hub -> Leaf
                process_step(hub_agent, leaf, context={"directive": f"Delegate subtask to {leaf.name}"})
                if step_idx >= self.max_steps:
                    break
                # Leaf -> Hub
                process_step(leaf, hub_agent)

        elif self.topology == "mesh":
            for i in range(len(self.agents) - 1):
                if step_idx >= self.max_steps:
                    break
                process_step(self.agents[i], self.agents[i + 1])

        elif self.topology == "custom":
            agent_map = {a.agent_id: a for a in self.agents}
            edges = list(self._topology_instance.graph.edges()) if self._topology_instance else []
            if not edges:
                edges = [(self.agents[i].agent_id, self.agents[i + 1].agent_id) for i in range(len(self.agents) - 1)]
            for src_id, dst_id in edges:
                if step_idx >= self.max_steps:
                    break
                if src_id in agent_map and dst_id in agent_map:
                    process_step(agent_map[src_id], agent_map[dst_id])

        self.end_time = round(sim_time, 4)
        self.duration_seconds = self.end_time - self.start_time

        # Determine final status and cascading evaluation
        task_success = self.task.evaluate_success(self.events)
        
        # Criteria for Level 3 Cascading Failure:
        # 1. Fault originated at an agent
        # 2. Corrupted state propagated to >= 2 downstream dependent agents
        # 3. Overall task evaluation degraded or failed
        if self.propagation_tracker.originating_agent is not None:
            downstream_count = len([a for a in self.propagation_tracker.affected_agents if a != self.propagation_tracker.originating_agent])
            if downstream_count >= 2 or not task_success or any(e.failure_label > 0 for e in self.events):
                self.has_cascading_failure = True
                self.cascading_failure_step = self.propagation_tracker.originating_step
                self.final_status = "FAILED"
                self.propagation_tracker.finalize_outcome("CASCADING_FAILURE")
            else:
                self.final_status = "COMPLETED"
                self.propagation_tracker.finalize_outcome("DEGRADED")
        else:
            self.final_status = "COMPLETED"
            self.propagation_tracker.finalize_outcome("SUCCESS")

        self.task_output = self.task.generate_final_output(self.events)
        if self.final_status == "FAILED":
            self.task_output["status"] = "FAILED"

        return self

    def save_to_db(self, session: Session) -> Any:
        """Persist the completed run, participating agents, events, fault injections, and failures."""
        fault_records = self.fault_injector.injected_records if self.fault_injector else []
        failure_records = []
        if self.propagation_tracker.originating_agent:
            failure_records.append({
                "failure_id": f"{self.run_id}_fail_0",
                "step_idx": self.propagation_tracker.originating_step or 0,
                "failure_level": self.propagation_tracker.failure_level,
                "originating_agent": self.propagation_tracker.originating_agent,
                "affected_agents": self.propagation_tracker.affected_agents,
                "failure_type": self.propagation_tracker.originating_fault_type or "injected_fault",
                "description": f"Propagation path: {self.propagation_tracker.to_path_string()}",
            })

        return save_simulation_run(
            run_id=self.run_id,
            task_type=self.task_type,
            topology=self.topology,
            agents=self.agents,
            events=self.events,
            random_seed=self.random_seed,
            duration_seconds=self.duration_seconds,
            session=session,
            has_cascading_failure=self.has_cascading_failure,
            cascading_failure_step=self.cascading_failure_step,
            metadata={
                "final_status": self.final_status,
                "propagation_path": self.propagation_tracker.to_path_string(),
            },
            fault_injections=fault_records,
            failures=failure_records,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert simulation run into structured dictionary."""
        return {
            "run_id": self.run_id,
            "task_type": self.task_type,
            "topology": self.topology,
            "num_agents": len(self.agents),
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "role": a.role.value if hasattr(a.role, "value") else str(a.role),
                    "status": a.state.get("status", "active"),
                }
                for a in self.agents
            ],
            "random_seed": self.random_seed,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_events": len(self.events),
            "final_status": self.final_status,
            "has_cascading_failure": self.has_cascading_failure,
            "cascading_failure_step": self.cascading_failure_step,
            "fault_propagation": self.propagation_tracker.to_dict(),
            "task_output": self.task_output,
        }
