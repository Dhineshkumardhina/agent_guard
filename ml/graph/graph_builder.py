"""Temporal Graph Builder: Converting Multi-Agent Telemetry into Dynamic Interaction Graphs.

Constructs G(t) = (V(t), E(t), X(t)) snapshots and historical sequences G(t-n)...G(t)
with strict causal data leakage guarantees.
"""

from typing import List, Dict, Any, Optional, Union, Set
from ml.telemetry.schemas import AgentTelemetryEvent
from ml.graph.graph_snapshot import GraphSnapshot
from ml.graph.node_features import extract_node_features
from ml.graph.edge_features import extract_edge_features
from ml.graph.graph_window import WindowStrategy, filter_events_by_window
from ml.utils.reproducibility import verify_no_future_leakage


class TemporalGraphBuilder:
    """Builder for constructing causal dynamic interaction graphs from agent telemetry."""

    def __init__(
        self,
        default_window_strategy: WindowStrategy = WindowStrategy.CUMULATIVE,
        default_window_size: Optional[int] = None,
        default_time_window: Optional[float] = None,
    ) -> None:
        self.default_window_strategy = default_window_strategy
        self.default_window_size = default_window_size
        self.default_time_window = default_time_window

    def build_snapshot(
        self,
        events: List[Union[AgentTelemetryEvent, Any]],
        timestamp: float,
        step_idx: Optional[int] = None,
        run_id: Optional[str] = None,
        window_size: Optional[int] = None,
        time_window: Optional[float] = None,
        strategy: Optional[WindowStrategy] = None,
        known_agents: Optional[List[str]] = None,
        agent_roles: Optional[Dict[str, str]] = None,
    ) -> GraphSnapshot:
        """Construct a single graph snapshot G(t) = (V(t), E(t), X(t)) at timestamp t.
        
        Args:
            events: Sequence of execution events.
            timestamp: Snapshot evaluation time t.
            step_idx: Optional step index cutoff.
            run_id: Run identifier.
            window_size: Event count window size.
            time_window: Time span window duration.
            strategy: Windowing policy.
            known_agents: Optional list of all agents in the environment.
            agent_roles: Optional dictionary mapping agent_id -> role name.
            
        Returns:
            A validated GraphSnapshot instance representing G(t).
        """
        # Normalize events to AgentTelemetryEvent
        norm_events: List[AgentTelemetryEvent] = []
        for e in events:
            if isinstance(e, AgentTelemetryEvent):
                norm_events.append(e)
            elif hasattr(e, "to_telemetry_event"):
                norm_events.append(e.to_telemetry_event())
            elif isinstance(e, dict):
                norm_events.append(AgentTelemetryEvent(**e))

        # Sort chronologically
        norm_events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

        # Apply causal window filtering
        active_strategy = strategy or self.default_window_strategy
        w_size = window_size if window_size is not None else self.default_window_size
        t_win = time_window if time_window is not None else self.default_time_window

        window_events = filter_events_by_window(
            events=norm_events,
            target_timestamp=timestamp,
            target_step_idx=step_idx,
            strategy=active_strategy,
            window_size=w_size,
            time_window=t_win,
        )

        # Enforce no future leakage
        used_steps = [e.step_idx if e.step_idx is not None else 0 for e in window_events]
        used_times = [e.timestamp if e.timestamp is not None else 0.0 for e in window_events]
        verify_no_future_leakage(
            current_step=step_idx if step_idx is not None else (max(used_steps) if used_steps else 0),
            current_timestamp=timestamp,
            used_event_steps=used_steps,
            used_event_timestamps=used_times,
        )

        # 1. Identify participating nodes V(t)
        agents_set: Set[str] = set(known_agents or [])
        for ev in window_events:
            if ev.source_agent:
                agents_set.add(ev.source_agent)
            if ev.target_agent:
                agents_set.add(ev.target_agent)

        roles_map = agent_roles or {}
        nodes_dict: Dict[str, Dict[str, Any]] = {}
        for aid in sorted(list(agents_set)):
            nodes_dict[aid] = extract_node_features(
                agent_id=aid,
                events=window_events,
                timestamp=timestamp,
                step_idx=step_idx,
                agent_role=roles_map.get(aid),
                is_active=True,
            )

        # 2. Identify directed interaction edges E(t)
        edges_set: Set[tuple[str, str]] = set()
        for ev in window_events:
            # Interaction events establish directed edges
            if ev.source_agent and ev.target_agent and ev.source_agent != ev.target_agent:
                edges_set.add((ev.source_agent, ev.target_agent))

        edges_dict: Dict[tuple[str, str], Dict[str, Any]] = {}
        for (src, tgt) in sorted(list(edges_set)):
            edges_dict[(src, tgt)] = extract_edge_features(
                source_agent=src,
                target_agent=tgt,
                events=window_events,
                timestamp=timestamp,
                step_idx=step_idx,
                total_trajectory_time=timestamp,
            )

        inferred_run_id = run_id
        if not inferred_run_id and window_events:
            inferred_run_id = window_events[0].run_id
        inferred_run_id = inferred_run_id or "run_snapshot"

        metadata = {
            "window_strategy": active_strategy.value,
            "window_size": w_size,
            "time_window": t_win,
            "events_in_window": len(window_events),
        }

        return GraphSnapshot(
            timestamp=timestamp,
            run_id=inferred_run_id,
            step_idx=step_idx,
            nodes=nodes_dict,
            edges=edges_dict,
            metadata=metadata,
        )

    def build_snapshots_over_run(
        self,
        events: List[Union[AgentTelemetryEvent, Any]],
        snapshot_strategy: str = "event",
        window_size: Optional[int] = None,
        stride: int = 1,
        known_agents: Optional[List[str]] = None,
        agent_roles: Optional[Dict[str, str]] = None,
    ) -> List[GraphSnapshot]:
        """Generate a temporal sequence of snapshots across a full trajectory.
        
        Args:
            events: Chronological sequence of all events in the trajectory.
            snapshot_strategy: 'event' (snapshot at each event step) or 'periodic'.
            window_size: Window size in event steps for rolling graph snapshots.
            stride: Step stride between consecutive snapshots.
            known_agents: List of agent IDs.
            agent_roles: Dictionary mapping agent_id -> role name.
            
        Returns:
            List of chronologically ordered GraphSnapshot instances.
        """
        if not events:
            return []

        # Normalize events
        norm_events: List[AgentTelemetryEvent] = []
        for e in events:
            if isinstance(e, AgentTelemetryEvent):
                norm_events.append(e)
            elif hasattr(e, "to_telemetry_event"):
                norm_events.append(e.to_telemetry_event())
            elif isinstance(e, dict):
                norm_events.append(AgentTelemetryEvent(**e))

        norm_events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

        snapshots: List[GraphSnapshot] = []
        for i in range(0, len(norm_events), max(1, stride)):
            ev = norm_events[i]
            snap = self.build_snapshot(
                events=norm_events[: i + 1],
                timestamp=ev.timestamp or 0.0,
                step_idx=ev.step_idx or i,
                run_id=ev.run_id,
                window_size=window_size,
                known_agents=known_agents,
                agent_roles=agent_roles,
            )
            snapshots.append(snap)

        return snapshots

    def build_temporal_history(
        self,
        events: List[Union[AgentTelemetryEvent, Any]],
        current_timestamp: float,
        history_length: int = 5,
        step_interval: int = 1,
        window_size: Optional[int] = None,
        known_agents: Optional[List[str]] = None,
        agent_roles: Optional[Dict[str, str]] = None,
    ) -> List[GraphSnapshot]:
        """Construct sequence of temporal history snapshots G(t-n), ..., G(t-1), G(t) for prediction point t.
        
        All snapshots in the history contain ONLY information <= current_timestamp.
        
        Args:
            events: Sequence of all events.
            current_timestamp: Target time t.
            history_length: Number of historical graph snapshots n to return.
            step_interval: Step stride between historical snapshots.
            window_size: Window size in events for each snapshot.
            known_agents: List of agent IDs.
            agent_roles: Mapping of agent_id -> role.
            
        Returns:
            Chronological list of GraphSnapshots [G(t-n), ..., G(t)].
        """
        # Filter all events up to current_timestamp
        causal_events = [
            e for e in events
            if (getattr(e, "timestamp", None) or 0.0) <= current_timestamp
        ]
        if not causal_events:
            return []

        all_snaps = self.build_snapshots_over_run(
            events=causal_events,
            window_size=window_size,
            stride=step_interval,
            known_agents=known_agents,
            agent_roles=agent_roles,
        )

        # Slice the most recent history_length snapshots up to t
        return all_snaps[-history_length:] if len(all_snaps) >= history_length else all_snaps

    def attach_prediction_targets(
        self,
        snapshots: List[GraphSnapshot],
        all_events: List[Union[AgentTelemetryEvent, Any]],
        horizons: Optional[List[int]] = None,
    ) -> None:
        """Annotate snapshots with ground-truth binary failure labels for future horizons K.
        
        Prediction Task:
        P(Failure in (t, t + K] | G(<= t))
        
        Labels are stored in snapshot.metadata['targets'][K] without contaminating
        node features or edge features X(t).
        
        Args:
            snapshots: List of GraphSnapshots G(t).
            all_events: Full sequence of trajectory events.
            horizons: Prediction horizon steps K (default: [1, 3, 5, 10, 20]).
        """
        k_values = horizons or [1, 3, 5, 10, 20]
        norm_events = [
            e if isinstance(e, AgentTelemetryEvent) else (
                e.to_telemetry_event() if hasattr(e, "to_telemetry_event") else AgentTelemetryEvent(**e)
            )
            for e in all_events
        ]
        norm_events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

        for snap in snapshots:
            snap_step = snap.step_idx if snap.step_idx is not None else 0
            if "targets" not in snap.metadata:
                snap.metadata["targets"] = {}

            for k in k_values:
                # Horizon window: steps in (snap_step, snap_step + k]
                horizon_events = [
                    e for e in norm_events
                    if e.step_idx is not None and snap_step < e.step_idx <= (snap_step + k)
                ]
                has_failure = any(
                    (e.failure_label is not None and e.failure_label > 0)
                    or e.tool_error
                    or e.event_type in ("error", "failure")
                    for e in horizon_events
                )
                snap.metadata["targets"][f"failure_k_{k}"] = 1 if has_failure else 0
