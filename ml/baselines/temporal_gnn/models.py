r"""Temporal Graph Neural Network (TGN-Style) Architecture - Phase 11.

Implements the core research model:
TemporalGraphFailurePredictor

Components:
1. TimeEncoder: Continuous sinusoidal Fourier temporal encoding of elapsed intervals \Delta t
2. MessageFunction: Interaction message constructor combining memories, edge features, and time encodings
3. NodeMemory: Dynamic per-agent temporal state m_v(t) with strict run-level reset
4. MemoryUpdater: GRU-based memory state transition function
5. TemporalEmbedding: Combines node memory, static/current node features, and multi-head temporal attention over recent neighbors
6. GraphReadout: Multi-pooling (mean + max) system-level representation
7. FailurePredictionHead: Multi-layer perceptron mapping graph embedding to impending failure probability
8. AblationConfig: Modular switches allowing targeted ablation of individual components in Phase 13
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.baselines.temporal_gnn.time_encoding import TimeEncoder
from ml.baselines.temporal_gnn.memory import NodeMemory, MemoryUpdater
from ml.baselines.temporal_gnn.neighborhood import TemporalNeighborhoodTracker


def get_device(verbose: bool = True) -> torch.device:
    """Detect available compute device (CUDA GPU or CPU) and report details."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[Device] PyTorch {torch.__version__} | Selected Device: CUDA | GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"[Device] PyTorch {torch.__version__} | Selected Device: CPU")
    return device


@dataclass
class AblationConfig:
    """Configuration hooks to selectively disable components for ablation studies."""
    enable_time_encoding: bool = True
    enable_memory: bool = True
    enable_node_features: bool = True
    enable_edge_features: bool = True
    enable_neighborhood: bool = True
    enable_graph_structure: bool = True
    enable_interaction_frequency: bool = True
    enable_contradiction_features: bool = True
    enable_confidence_features: bool = True
    enable_failure_history: bool = True



class MessageFunction(nn.Module):
    """Constructs directional temporal interaction messages between source and target agents."""

    def __init__(
        self,
        memory_dim: int = 64,
        edge_dim: int = 10,
        time_dim: int = 16,
        message_dim: int = 64,
        ablation_config: Optional[AblationConfig] = None,
    ) -> None:
        super().__init__()
        self.ablation = ablation_config or AblationConfig()
        
        in_dim = (memory_dim * 2) + edge_dim + time_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(
        self,
        src_memory: torch.Tensor,
        tgt_memory: torch.Tensor,
        edge_feat: torch.Tensor,
        time_enc: torch.Tensor,
    ) -> torch.Tensor:
        # Apply ablation masks if configured
        if not self.ablation.enable_memory:
            src_memory = torch.zeros_like(src_memory)
            tgt_memory = torch.zeros_like(tgt_memory)
        if not self.ablation.enable_edge_features:
            edge_feat = torch.zeros_like(edge_feat)
        if not self.ablation.enable_time_encoding:
            time_enc = torch.zeros_like(time_enc)

        raw_input = torch.cat([src_memory, tgt_memory, edge_feat, time_enc], dim=-1)
        return self.mlp(raw_input)


class TemporalEmbedding(nn.Module):
    """Generates node-level temporal embeddings combining memory, static features, and neighbor context."""

    def __init__(
        self,
        memory_dim: int = 64,
        node_dim: int = 14,
        edge_dim: int = 10,
        time_dim: int = 16,
        embed_dim: int = 64,
        ablation_config: Optional[AblationConfig] = None,
    ) -> None:
        super().__init__()
        self.ablation = ablation_config or AblationConfig()
        self.memory_dim = memory_dim
        self.embed_dim = embed_dim

        # Temporal attention over neighbor interactions
        self.query_proj = nn.Linear(memory_dim + node_dim, embed_dim)
        self.key_proj = nn.Linear(memory_dim + edge_dim + time_dim, embed_dim)
        self.value_proj = nn.Linear(memory_dim + edge_dim + time_dim, embed_dim)

        # Output projection
        self.out_mlp = nn.Sequential(
            nn.Linear(memory_dim + node_dim + embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        node_memory: torch.Tensor,
        node_features: torch.Tensor,
        neighbor_records: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute temporal embedding for a single node.
        
        Args:
            node_memory: Current agent memory m_v(t) [memory_dim].
            node_features: Current node feature vector x_v(t) [node_dim].
            neighbor_records: List of tuples (nbr_memory, edge_features, time_encoding).
            device: Compute device.
            
        Returns:
            Embedding vector z_v(t) of shape [embed_dim].
        """
        if not self.ablation.enable_memory:
            node_memory = torch.zeros_like(node_memory)
        if not self.ablation.enable_node_features:
            node_features = torch.zeros_like(node_features)

        combined_self = torch.cat([node_memory, node_features], dim=-1).unsqueeze(0)  # [1, memory_dim + node_dim]
        
        # If neighborhood or graph structure is disabled or empty, use zero neighborhood context
        if not self.ablation.enable_neighborhood or not self.ablation.enable_graph_structure or not neighbor_records:
            nbr_context = torch.zeros(1, self.embed_dim, device=device)
        else:
            nbr_keys_list = []
            nbr_vals_list = []
            for nbr_mem, nbr_edge, nbr_t_enc in neighbor_records:
                if not self.ablation.enable_memory:
                    nbr_mem = torch.zeros_like(nbr_mem)
                if not self.ablation.enable_edge_features:
                    nbr_edge = torch.zeros_like(nbr_edge)
                if not self.ablation.enable_time_encoding:
                    nbr_t_enc = torch.zeros_like(nbr_t_enc)

                ctx_cat = torch.cat([nbr_mem, nbr_edge, nbr_t_enc], dim=-1)
                nbr_keys_list.append(ctx_cat)
                nbr_vals_list.append(ctx_cat)

            keys_tensor = torch.stack(nbr_keys_list, dim=0)    # [N_nbr, ctx_dim]
            vals_tensor = torch.stack(nbr_vals_list, dim=0)    # [N_nbr, ctx_dim]

            # Attention scores
            query = self.query_proj(combined_self)              # [1, embed_dim]
            keys = self.key_proj(keys_tensor)                   # [N_nbr, embed_dim]
            vals = self.value_proj(vals_tensor)                 # [N_nbr, embed_dim]

            scores = torch.matmul(query, keys.t()) / (self.embed_dim ** 0.5)  # [1, N_nbr]
            attn_weights = F.softmax(scores, dim=-1)                          # [1, N_nbr]
            nbr_context = torch.matmul(attn_weights, vals)                     # [1, embed_dim]

        full_cat = torch.cat([combined_self, nbr_context], dim=-1)  # [1, memory_dim + node_dim + embed_dim]
        out_embed = self.out_mlp(full_cat).squeeze(0)
        return out_embed


class GraphReadout(nn.Module):
    """Multi-pooling (mean + max) graph representation from agent temporal embeddings."""

    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into a single system-level representation.
        
        Args:
            node_embeddings: Tensor of shape [num_nodes, embed_dim].
            
        Returns:
            Graph embedding tensor of shape [embed_dim * 2].
        """
        if node_embeddings.size(0) == 0:
            return torch.zeros(self.embed_dim * 2, device=node_embeddings.device)
        mean_rep = torch.mean(node_embeddings, dim=0)
        max_rep, _ = torch.max(node_embeddings, dim=0)
        return torch.cat([mean_rep, max_rep], dim=-1)


class FailurePredictionHead(nn.Module):
    """Classification head mapping system-level graph embedding to impending failure logit."""

    def __init__(self, input_dim: int = 128, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """Output single logit for failure prediction.
        
        Args:
            graph_embedding: Tensor of shape [..., input_dim].
            
        Returns:
            Logit tensor of shape [..., 1] or scalar.
        """
        return self.mlp(graph_embedding).squeeze(-1)


class TemporalGraphFailurePredictor(nn.Module):
    """Core Temporal Graph Neural Network (TGN-style) for multi-agent failure forecasting."""

    def __init__(
        self,
        node_in_dim: int = 14,
        edge_in_dim: int = 10,
        memory_dim: int = 64,
        time_dim: int = 16,
        embed_dim: int = 64,
        neighbor_history: int = 10,
        dropout: float = 0.2,
        ablation_config: Optional[AblationConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embed_dim = embed_dim
        self.neighbor_history = neighbor_history
        self.dropout_rate = dropout
        self.ablation = ablation_config or AblationConfig()
        self.device = device or torch.device("cpu")
        self.model_name = "temporal_gnn"

        # Modular components
        self.time_encoder = TimeEncoder(dimension=time_dim, encoding_type="sinusoidal")
        self.message_function = MessageFunction(
            memory_dim=memory_dim,
            edge_dim=edge_in_dim,
            time_dim=time_dim,
            message_dim=memory_dim,
            ablation_config=self.ablation,
        )
        self.node_memory = NodeMemory(memory_dim=memory_dim, device=self.device)
        self.memory_updater = MemoryUpdater(message_dim=memory_dim, memory_dim=memory_dim)
        self.neighborhood_tracker = TemporalNeighborhoodTracker(max_history=neighbor_history * 2)
        self.temporal_embedding = TemporalEmbedding(
            memory_dim=memory_dim,
            node_dim=node_in_dim,
            edge_dim=edge_in_dim,
            time_dim=time_dim,
            embed_dim=embed_dim,
            ablation_config=self.ablation,
        )
        self.readout = GraphReadout(embed_dim=embed_dim)
        self.head = FailurePredictionHead(input_dim=embed_dim * 2, hidden_dim=32, dropout=dropout)

    def reset_memory(self) -> None:
        """Reset dynamic agent memories and neighbor trackers between simulation trajectories.
        
        Strictly guarantees run isolation and prevents cross-run information leakage.
        """
        self.node_memory.reset_memory()
        self.neighborhood_tracker.clear()

    def process_interaction(
        self,
        source_agent: str,
        target_agent: str,
        timestamp: float,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single chronological interaction event u -> v at timestamp t.
        
        Updates node memories for both source and target agents.
        
        Returns:
            Tuple of (source_new_memory, target_new_memory).
        """
        edge_features = edge_features.to(self.device)
        if edge_features.dim() == 0:
            edge_features = edge_features.unsqueeze(0)

        # 1. Retrieve prior states
        src_mem = self.node_memory.get_memory(source_agent)
        tgt_mem = self.node_memory.get_memory(target_agent)

        # If graph structure is disabled, do not process edge interactions across agents
        if not self.ablation.enable_graph_structure:
            return src_mem, tgt_mem

        # 2. Elapsed intervals \Delta t
        last_t_src = self.node_memory.get_last_timestamp(source_agent)
        last_t_tgt = self.node_memory.get_last_timestamp(target_agent)
        dt_src = torch.tensor([max(0.0, timestamp - last_t_src)], dtype=torch.float32, device=self.device)
        dt_tgt = torch.tensor([max(0.0, timestamp - last_t_tgt)], dtype=torch.float32, device=self.device)

        if not self.ablation.enable_time_encoding:
            dt_src = torch.zeros_like(dt_src)
            dt_tgt = torch.zeros_like(dt_tgt)

        # 3. Continuous time encodings \phi(\Delta t)
        t_enc_src = self.time_encoder(dt_src).squeeze(0)
        t_enc_tgt = self.time_encoder(dt_tgt).squeeze(0)

        # 4. Message generation
        src_msg = self.message_function(src_mem, tgt_mem, edge_features, t_enc_src)
        tgt_msg = self.message_function(tgt_mem, src_mem, edge_features, t_enc_tgt)

        # 5. Memory updates via GRU
        if not self.ablation.enable_memory:
            new_src_mem = torch.zeros_like(src_mem)
            new_tgt_mem = torch.zeros_like(tgt_mem)
        else:
            new_src_mem = self.memory_updater(src_msg, src_mem)
            new_tgt_mem = self.memory_updater(tgt_msg, tgt_mem)

        # 6. Save updated memory and timestamps
        self.node_memory.set_memory(source_agent, new_src_mem, timestamp)
        self.node_memory.set_memory(target_agent, new_tgt_mem, timestamp)

        # 7. Record interaction in temporal neighborhood
        self.neighborhood_tracker.add_interaction(
            source_agent=source_agent,
            target_agent=target_agent,
            timestamp=timestamp,
            edge_features=edge_features.detach(),
        )

        return new_src_mem, new_tgt_mem

    def predict_at_timestamp(
        self,
        agent_ids: List[str],
        node_features_dict: Dict[str, List[float]],
        current_timestamp: float,
    ) -> Tuple[torch.Tensor, float]:
        """Compute failure prediction at cutoff timestamp t using data observed strictly <= t.
        
        Args:
            agent_ids: List of active agent IDs at time t.
            node_features_dict: Dict mapping agent_id -> 14-dim node feature vector.
            current_timestamp: Upper time cutoff t.
            
        Returns:
            Tuple of (logit_tensor, failure_probability_float in [0.0, 1.0]).
        """
        node_embeddings = []
        for aid in agent_ids:
            mem = self.node_memory.get_memory(aid)
            if not self.ablation.enable_memory:
                mem = torch.zeros_like(mem)
            n_feats = node_features_dict.get(aid, [0.0] * self.node_in_dim)
            if not self.ablation.enable_node_features:
                n_feats = [0.0] * self.node_in_dim
            n_feat_tensor = torch.tensor(n_feats, dtype=torch.float32, device=self.device)

            # Sample temporal neighbors strictly <= current_timestamp
            if self.ablation.enable_neighborhood and self.ablation.enable_graph_structure:
                nbr_records = self.neighborhood_tracker.get_recent_neighbors(
                    agent_id=aid,
                    current_timestamp=current_timestamp,
                    k_neighbors=self.neighbor_history,
                )
            else:
                nbr_records = []

            # Format neighbor context
            nbr_inputs = []
            for nbr_id, nbr_t, nbr_e_feat in nbr_records:
                nbr_mem = self.node_memory.get_memory(nbr_id)
                if not self.ablation.enable_memory:
                    nbr_mem = torch.zeros_like(nbr_mem)
                dt = torch.tensor([max(0.0, current_timestamp - nbr_t)], dtype=torch.float32, device=self.device)
                if not self.ablation.enable_time_encoding:
                    dt = torch.zeros_like(dt)
                t_enc = self.time_encoder(dt).squeeze(0)
                nbr_inputs.append((nbr_mem, nbr_e_feat.to(self.device), t_enc))

            # Compute agent temporal embedding
            z_v = self.temporal_embedding(
                node_memory=mem,
                node_features=n_feat_tensor,
                neighbor_records=nbr_inputs,
                device=self.device,
            )
            node_embeddings.append(z_v)

        if node_embeddings:
            stacked_z = torch.stack(node_embeddings, dim=0)  # [num_nodes, embed_dim]
        else:
            stacked_z = torch.zeros(1, self.embed_dim, device=self.device)

        # Graph-level multi-pooling readout
        graph_rep = self.readout(stacked_z)
        logit = self.head(graph_rep)
        prob = torch.sigmoid(logit).item()
        return logit, prob

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save model checkpoint with complete temporal architecture configuration."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "architecture_config": {
                "node_in_dim": self.node_in_dim,
                "edge_in_dim": self.edge_in_dim,
                "memory_dim": self.memory_dim,
                "time_dim": self.time_dim,
                "embed_dim": self.embed_dim,
                "neighbor_history": self.neighbor_history,
                "dropout": self.dropout_rate,
            },
            "ablation_config": self.ablation.__dict__,
            "pytorch_version": torch.__version__,
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, filepath: Union[str, Path], device: Optional[torch.device] = None) -> "TemporalGraphFailurePredictor":
        """Load model state and configuration from a saved checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        try:
            payload = torch.load(path, map_location=dev, weights_only=True)
        except Exception:
            try:
                payload = torch.load(path, map_location=dev, weights_only=False)  # nosec
            except TypeError:
                payload = torch.load(path, map_location=dev)  # nosec

        self.load_state_dict(payload["model_state_dict"])
        self.to(dev)
        self.device = dev
        return self
