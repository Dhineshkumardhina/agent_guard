"""Node Memory and Memory Updater Modules for Temporal Graph Networks - Phase 11.

Maintains dynamic agent-level temporal memory states m_v(t) that update
as interactions occur chronologically.

STRICT RUN ISOLATION:
Node memory must be completely reset between independent simulation trajectories
so that no agent state leaks across different runs.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class NodeMemory(nn.Module):
    """Stores and manages per-agent dynamic memory vectors m_v(t) and interaction timestamps."""

    def __init__(
        self,
        memory_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize NodeMemory.
        
        Args:
            memory_dim: Dimensionality of memory vector for each agent.
            device: Compute device.
        """
        super().__init__()
        self.memory_dim = memory_dim
        self.device = device or torch.device("cpu")

        # Map agent_id -> memory tensor [memory_dim]
        self._memory: Dict[str, torch.Tensor] = {}
        # Map agent_id -> last interaction timestamp float
        self._last_update: Dict[str, float] = {}

    def reset_memory(self) -> None:
        """Reset all node memory states and timestamp records.
        
        Mandatory call before processing any new simulation trajectory to prevent
        cross-run state leakage.
        """
        self._memory.clear()
        self._last_update.clear()

    def get_memory(self, agent_id: str) -> torch.Tensor:
        """Retrieve current memory vector m_v for a specific agent.
        
        If agent has not yet participated in interactions, initializes to zeros.
        """
        if agent_id not in self._memory:
            self._memory[agent_id] = torch.zeros(self.memory_dim, dtype=torch.float32, device=self.device)
            self._last_update[agent_id] = 0.0
        return self._memory[agent_id]

    def set_memory(self, agent_id: str, new_memory: torch.Tensor, timestamp: float) -> None:
        """Update memory vector and last interaction timestamp for an agent."""
        self._memory[agent_id] = new_memory.detach()
        self._last_update[agent_id] = float(timestamp)

    def get_last_timestamp(self, agent_id: str) -> float:
        """Retrieve timestamp of the most recent interaction involving this agent."""
        return self._last_update.get(agent_id, 0.0)

    def get_all_memories(self, agent_ids: List[str]) -> torch.Tensor:
        """Retrieve stacked memory tensor for a list of agent IDs.
        
        Returns:
            Tensor of shape [len(agent_ids), memory_dim].
        """
        memories = [self.get_memory(aid) for aid in agent_ids]
        if not memories:
            return torch.zeros((0, self.memory_dim), dtype=torch.float32, device=self.device)
        return torch.stack(memories, dim=0)

    def to(self, device: torch.device) -> "NodeMemory":
        self.device = device
        for k in list(self._memory.keys()):
            self._memory[k] = self._memory[k].to(device)
        return super().to(device)


class MemoryUpdater(nn.Module):
    """GRU-based memory updater mapping incoming message and previous memory to new memory."""

    def __init__(self, message_dim: int, memory_dim: int) -> None:
        """Initialize MemoryUpdater.
        
        Args:
            message_dim: Dimensionality of temporal interaction message.
            memory_dim: Dimensionality of agent memory vector.
        """
        super().__init__()
        self.message_dim = message_dim
        self.memory_dim = memory_dim
        self.gru = nn.GRUCell(input_size=message_dim, hidden_size=memory_dim)

    def forward(self, message: torch.Tensor, prev_memory: torch.Tensor) -> torch.Tensor:
        """Update node memory using GRU cell.
        
        Args:
            message: Incoming interaction message tensor.
            prev_memory: Previous memory tensor.
            
        Returns:
            Updated memory tensor with identical dimensional layout.
        """
        is_1d = (message.dim() == 1)
        if is_1d:
            message = message.unsqueeze(0)
        if prev_memory.dim() == 1:
            prev_memory = prev_memory.unsqueeze(0)

        new_memory = self.gru(message, prev_memory)
        return new_memory.squeeze(0) if is_1d else new_memory
