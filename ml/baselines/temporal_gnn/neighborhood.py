"""Temporal Neighborhood Tracker for Temporal Graph Networks - Phase 11.

Maintains historical interaction records per agent and samples the most recent
temporal neighbors up to the current prediction cutoff t.

STRICT CAUSALITY GUARANTEE:
Only interactions observed strictly <= current_timestamp may enter temporal
neighborhood sampling. No future interactions are ever returned.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import torch


class TemporalNeighborhoodTracker:
    """Maintains and samples chronological interaction history per agent."""

    def __init__(self, max_history: int = 20) -> None:
        """Initialize tracker.
        
        Args:
            max_history: Maximum number of recent historical interactions to retain per agent.
        """
        self.max_history = max_history
        # Map agent_id -> list of (neighbor_id, timestamp, edge_features_tensor)
        self._history: Dict[str, List[Tuple[str, float, torch.Tensor]]] = defaultdict(list)

    def add_interaction(
        self,
        source_agent: str,
        target_agent: str,
        timestamp: float,
        edge_features: torch.Tensor,
    ) -> None:
        """Record directed interaction between source and target at timestamp."""
        t = float(timestamp)
        # Record for source agent (interacting with target)
        self._history[source_agent].append((target_agent, t, edge_features))
        # Record for target agent (interacting with source)
        self._history[target_agent].append((source_agent, t, edge_features))

        # Enforce history limit
        if len(self._history[source_agent]) > self.max_history * 2:
            self._history[source_agent] = self._history[source_agent][-self.max_history:]
        if len(self._history[target_agent]) > self.max_history * 2:
            self._history[target_agent] = self._history[target_agent][-self.max_history:]

    def get_recent_neighbors(
        self,
        agent_id: str,
        current_timestamp: float,
        k_neighbors: int = 10,
    ) -> List[Tuple[str, float, torch.Tensor]]:
        """Retrieve up to k_neighbors most recent interactions involving agent_id <= current_timestamp.
        
        Args:
            agent_id: Identifier of target agent node.
            current_timestamp: Upper time cutoff t.
            k_neighbors: Maximum number of historical neighbor interactions to retrieve.
            
        Returns:
            List of (neighbor_id, interaction_time, edge_features) ordered chronologically.
        """
        records = self._history.get(agent_id, [])
        # Strictly causal filtering: timestamp <= current_timestamp
        causal_records = [r for r in records if r[1] <= current_timestamp]
        # Take the most recent k_neighbors
        recent = causal_records[-k_neighbors:] if len(causal_records) > k_neighbors else causal_records
        return recent

    def clear(self) -> None:
        """Clear all historical interaction tracking across all agents between runs."""
        self._history.clear()
