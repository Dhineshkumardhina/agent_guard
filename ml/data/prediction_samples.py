"""Prediction Point and Sample Generation for Multi-Agent Trajectories.

Generates validated prediction samples:
P(F(t+k) | observations <= t) for k in {1, 3, 5, 10, 20}.

Rejects prediction points where the required future horizon is truncated or does not exist.
Ensures zero data leakage between future observation window and past input features.
"""

from typing import List, Dict, Any, Optional
from ml.data.schema import PredictionSample
from ml.data.labeling import compute_prediction_label
from ml.data.feature_windows import extract_agent_level_features, extract_graph_features
from ml.graph.graph_builder import TemporalGraphBuilder
from ml.simulation.run import SimulationRun
from ml.telemetry.schemas import AgentTelemetryEvent


class PredictionSampleGenerator:
    """Extracts valid, causal prediction samples across multi-agent simulation runs."""

    def __init__(
        self,
        prediction_horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
        history_length: int = 5,
    ) -> None:
        self.prediction_horizons = prediction_horizons or [1, 3, 5, 10, 20]
        self.dataset_version = dataset_version
        self.history_length = history_length
        self._graph_builder = TemporalGraphBuilder()

    def generate_from_run(
        self,
        run: SimulationRun,
    ) -> List[PredictionSample]:
        """Extract all valid prediction samples across all supported horizons from a completed run.
        
        Args:
            run: Executed SimulationRun instance with events and propagation tracker.
            
        Returns:
            List of validated PredictionSample instances.
        """
        events = run.events
        if not events:
            return []

        # Convert to AgentTelemetryEvent if needed
        norm_events = [
            e if isinstance(e, AgentTelemetryEvent) else (
                e.to_telemetry_event() if hasattr(e, "to_telemetry_event") else AgentTelemetryEvent(**(e.to_dict() if hasattr(e, "to_dict") else e))
            )
            for e in events
        ]
        norm_events.sort(key=lambda e: (e.timestamp if e.timestamp is not None else 0.0, e.step_idx or 0))

        total_steps = len(norm_events)
        max_step_idx = max((e.step_idx or 0) for e in norm_events)

        samples: List[PredictionSample] = []

        # Iterate over prediction points t (step_idx i)
        for i in range(total_steps):
            current_event = norm_events[i]
            current_step = current_event.step_idx if current_event.step_idx is not None else i
            current_timestamp = current_event.timestamp if current_event.timestamp is not None else 0.0

            # Historical events strictly <= current_step
            causal_history = norm_events[: i + 1]

            # Compute causal features once per prediction point
            agent_feats = extract_agent_level_features(
                events=causal_history,
                current_step_idx=current_step,
                current_timestamp=current_timestamp,
            )

            node_feats, edge_feats, graph_history = extract_graph_features(
                events=causal_history,
                current_step_idx=current_step,
                current_timestamp=current_timestamp,
                run_id=run.run_id,
                builder=self._graph_builder,
                history_length=self.history_length,
            )

            for k in self.prediction_horizons:
                # Validation: avoid generating samples where the required future horizon does not exist
                if (current_step + k) > max_step_idx:
                    continue

                # Compute ground truth future label strictly within (current_step, current_step + k]
                label, failure_type, failure_level = compute_prediction_label(
                    events=norm_events,
                    current_step_idx=current_step,
                    prediction_horizon=k,
                    propagation_tracker=run.propagation_tracker,
                )

                sample_id = f"{run.run_id}_s{current_step}_k{k}"
                sample = PredictionSample(
                    sample_id=sample_id,
                    run_id=run.run_id,
                    timestamp=current_timestamp,
                    step_idx=current_step,
                    task_type=run.task_type,
                    topology=run.topology,
                    number_of_agents=len(run.agents),
                    prediction_horizon=k,
                    node_features=node_feats,
                    edge_features=edge_feats,
                    temporal_graph_history=graph_history,
                    agent_level_features=agent_feats,
                    label=label,
                    failure_type=failure_type,
                    failure_level=failure_level,
                    source_event_id=getattr(current_event, "event_id", None),
                    random_seed=run.random_seed,
                    dataset_version=self.dataset_version,
                )
                samples.append(sample)

        return samples

    def generate_from_runs(
        self,
        runs: List[SimulationRun],
    ) -> List[PredictionSample]:
        """Generate prediction samples across a collection of simulation runs."""
        all_samples: List[PredictionSample] = []
        for r in runs:
            all_samples.extend(self.generate_from_run(r))
        return all_samples
