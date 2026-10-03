"""SimulationRun Abstraction for Controlled Multi-Agent Executions.

Coordinates agents, topology, task workflows, deterministic random state,
and telemetry recording to generate reproducible multi-agent trajectory runs.
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
        self.events: List[SimulationMessage] = []
        self.task_output: Optional[Dict[str, Any]] = None

        # Seeded local PRNG
        self._rng = random.Random(random_seed)

        # Setup agents
        if agents is not None:
            self.agents: List[Agent] = list(agents)
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
        """Execute the multi-agent task under topology routing rules.
        
        Guaranteed to produce identical event sequences and outputs when given
        identical configuration and random seed.
        """
        self.final_status = "RUNNING"
        self.start_time = 0.0
        self.events.clear()

        # Reset components with seeded state
        env = SimulationEnvironment(seed=self.random_seed)
        for agent in self.agents:
            agent.reset()
            env.register_agent(agent)

        env.set_topology(self._topology_instance)
        env.set_task(self.task)

        sim_time = 0.0
        step_idx = 0

        # Execute according to topology category
        if self.topology == "pipeline":
            # Linear pipeline: A[i] -> A[i+1]
            for i in range(len(self.agents) - 1):
                if step_idx >= self.max_steps:
                    break
                src_agent = self.agents[i]
                dst_agent = self.agents[i + 1]

                msg = env.step(
                    source_agent_id=src_agent.agent_id,
                    target_agent_id=dst_agent.agent_id,
                    step_idx=step_idx,
                    run_id=self.run_id,
                )
                self.events.append(msg)
                sim_time += msg.latency
                step_idx += 1

        elif self.topology == "star":
            # Hub coordinates with leaves: Hub -> Leaf_i -> Hub
            hub_id = getattr(self._topology_instance, "hub_agent_id", self.agents[0].agent_id)
            leaves = [a for a in self.agents if a.agent_id != hub_id]

            for leaf in leaves:
                if step_idx >= self.max_steps:
                    break
                # Hub -> Leaf
                msg_out = env.step(
                    source_agent_id=hub_id,
                    target_agent_id=leaf.agent_id,
                    step_idx=step_idx,
                    run_id=self.run_id,
                    context={"directive": f"Delegate subtask to {leaf.name}"},
                )
                self.events.append(msg_out)
                sim_time += msg_out.latency
                step_idx += 1

                if step_idx >= self.max_steps:
                    break
                # Leaf -> Hub
                msg_in = env.step(
                    source_agent_id=leaf.agent_id,
                    target_agent_id=hub_id,
                    step_idx=step_idx,
                    run_id=self.run_id,
                )
                self.events.append(msg_in)
                sim_time += msg_in.latency
                step_idx += 1

        elif self.topology == "mesh":
            # Peer-to-peer mesh execution: agents execute workflow stages directly
            for i in range(len(self.agents) - 1):
                if step_idx >= self.max_steps:
                    break
                src_agent = self.agents[i]
                dst_agent = self.agents[i + 1]

                msg = env.step(
                    source_agent_id=src_agent.agent_id,
                    target_agent_id=dst_agent.agent_id,
                    step_idx=step_idx,
                    run_id=self.run_id,
                )
                self.events.append(msg)
                sim_time += msg.latency
                step_idx += 1

        self.end_time = round(sim_time, 4)
        self.duration_seconds = self.end_time - self.start_time
        self.task_output = self.task.generate_final_output(self.events)
        self.final_status = "COMPLETED"
        return self

    def save_to_db(self, session: Session) -> Any:
        """Persist the completed run, its participating agents, and its events to the database."""
        return save_simulation_run(
            run_id=self.run_id,
            task_type=self.task_type,
            topology=self.topology,
            agents=self.agents,
            events=self.events,
            random_seed=self.random_seed,
            duration_seconds=self.duration_seconds,
            session=session,
            has_cascading_failure=False,
            metadata={"final_status": self.final_status},
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
            "task_output": self.task_output,
        }
