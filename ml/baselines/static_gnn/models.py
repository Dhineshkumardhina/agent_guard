"""PyTorch Geometric Static GNN Baseline Architectures - Phase 10.

Implements static Graph Neural Network baselines for multi-agent failure forecasting:
1. GCNBaseline (Graph Convolutional Network)
2. GATBaseline (Graph Attention Network with multi-head attention and edge attributes)

Consumes static snapshot G(t) = (V, E, X) available at prediction cutoff t.
Maps node and edge representations through message-passing layers,
pools node embeddings to graph level, and predicts impending failure probability in [0, 1].
"""

from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    nn = object
    Data = object
    DataLoader = object


def get_device(verbose: bool = True) -> Any:
    """Detect available compute device (CUDA GPU or CPU) and report details."""
    if not HAS_PYG:
        raise ImportError("PyTorch Geometric is required for static GNN models.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        if verbose:
            print(f"[Device] PyTorch {torch.__version__} | PyG {torch_geometric.__version__} | Selected Device: CUDA | GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        if verbose:
            print(f"[Device] PyTorch {torch.__version__} | PyG {torch_geometric.__version__} | Selected Device: CPU")

    return device


class BaseStaticGNN(nn.Module if HAS_PYG else object):
    """Abstract base class for all static Graph Neural Network baselines."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name

    def forward(
        self,
        x: Any,
        edge_index: Any,
        batch: Any,
        edge_attr: Optional[Any] = None,
        edge_weight: Optional[Any] = None,
    ) -> Any:
        """Forward pass through graph layers, pooling, and classification head.
        
        Args:
            x: Node feature tensor of shape [num_nodes, node_in_dim].
            edge_index: Graph connectivity tensor of shape [2, num_edges].
            batch: Batch assignment vector of shape [num_nodes].
            edge_attr: Optional multi-dimensional edge features [num_edges, edge_dim].
            edge_weight: Optional scalar edge weights [num_edges].
            
        Returns:
            Logit tensor of shape [batch_size].
        """
        raise NotImplementedError

    def predict_proba(self, loader: Any, device: Optional[Any] = None) -> np.ndarray:
        """Predict failure probability scores P(y=1) in [0.0, 1.0] for a PyG DataLoader."""
        dev = device or next(self.parameters()).device
        self.eval()
        all_probs: List[float] = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(dev)
                edge_attr = getattr(batch, "edge_attr", None)
                edge_weight = getattr(batch, "edge_weight", None)
                logits = self(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    batch=batch.batch,
                    edge_attr=edge_attr,
                    edge_weight=edge_weight,
                )
                probs = torch.sigmoid(logits).cpu().numpy().tolist()
                if isinstance(probs, list):
                    all_probs.extend(probs)
                else:
                    all_probs.append(float(probs))

        return np.array(all_probs, dtype=np.float32)

    def predict(self, loader: Any, threshold: float = 0.50, device: Optional[Any] = None) -> np.ndarray:
        """Predict binary failure warning labels (0 or 1) using calibrated decision threshold."""
        probs = self.predict_proba(loader, device=device)
        return (probs >= threshold).astype(int)

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        optimizer: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save complete, verifiable model checkpoint with architecture and training metadata."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "architecture_config": getattr(self, "config", {}),
            "pytorch_version": torch.__version__,
            "pyg_version": torch_geometric.__version__,
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        return path

    def load_checkpoint(self, filepath: Union[str, Path], device: Optional[Any] = None) -> "BaseStaticGNN":
        """Load model state and configuration from a saved checkpoint."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

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
        return self


class GCNBaseline(BaseStaticGNN):
    r"""Static Graph Convolutional Network (GCN) Baseline.
    
    Standard Kipf-Welling formulation:
    H^{(l+1)} = \sigma( \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)} )
    
    Edge attributes note:
    Standard GCN operates natively on graph adjacency and scalar edge weights.
    Edge attributes are incorporated as scalar interaction weights via edge_weight.
    """

    def __init__(
        self,
        node_in_dim: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        """Initialize GCN baseline.
        
        Args:
            node_in_dim: Input node feature dimension (default 14).
            hidden_dim: Hidden dimension across graph convolutional layers.
            num_layers: Number of stacked GCNConv layers (min 1).
            dropout: Dropout probability between graph convolutions and head.
            pooling: Graph readout pooling strategy ("mean", "max", or "both").
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GCNBaseline.")
        super().__init__(model_name="gcn")

        self.node_in_dim = node_in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)
        self.dropout_rate = dropout
        self.pooling = pooling

        self.config = {
            "model_name": "gcn",
            "node_in_dim": node_in_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "pooling": pooling,
        }

        # GCN convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_in_dim, hidden_dim))
        for _ in range(self.num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Graph pooling dimension
        pool_dim = hidden_dim * 2 if pooling == "both" else hidden_dim

        # Graph-level classification head
        self.head = nn.Sequential(
            nn.Linear(pool_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: Any,
        edge_index: Any,
        batch: Any,
        edge_attr: Optional[Any] = None,
        edge_weight: Optional[Any] = None,
    ) -> Any:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = F.relu(h)
            h = self.dropout(h)

        # Graph readout pooling
        if self.pooling == "mean":
            graph_rep = global_mean_pool(h, batch)
        elif self.pooling == "max":
            graph_rep = global_max_pool(h, batch)
        elif self.pooling == "both":
            mean_rep = global_mean_pool(h, batch)
            max_rep = global_max_pool(h, batch)
            graph_rep = torch.cat([mean_rep, max_rep], dim=-1)
        else:
            graph_rep = global_mean_pool(h, batch)

        # Prediction head
        out = self.head(graph_rep).squeeze(-1)
        return out


class GATBaseline(BaseStaticGNN):
    """Static Graph Attention Network (GAT) Baseline.
    
    Veličković et al. formulation with multi-head attention:
    \alpha_{ij} = \text{Softmax}_j( \text{LeakyReLU}( a^T [Wh_i || Wh_j || W_e e_{ij}] ) )
    
    Edge attributes note:
    GAT natively incorporates multi-dimensional edge features into attention computation
    via the edge_dim parameter in GATConv.
    """

    def __init__(
        self,
        node_in_dim: int = 14,
        edge_dim: Optional[int] = 10,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        pooling: str = "mean",
    ) -> None:
        """Initialize GAT baseline.
        
        Args:
            node_in_dim: Input node feature dimension (default 14).
            edge_dim: Multi-dimensional edge feature dimension (default 10).
            hidden_dim: Total hidden dimension (split across attention heads).
            num_layers: Number of stacked GATConv layers (min 1).
            heads: Number of attention heads.
            dropout: Dropout probability for attention coefficients and features.
            pooling: Graph readout pooling strategy ("mean", "max", or "both").
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for GATBaseline.")
        super().__init__(model_name="gat")

        self.node_in_dim = node_in_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = max(1, num_layers)
        self.heads = max(1, heads)
        self.dropout_rate = dropout
        self.pooling = pooling

        # Ensure hidden_dim is divisible by heads
        self.head_dim = max(1, hidden_dim // self.heads)
        self.effective_hidden_dim = self.head_dim * self.heads

        self.config = {
            "model_name": "gat",
            "node_in_dim": node_in_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
            "effective_hidden_dim": self.effective_hidden_dim,
            "num_layers": num_layers,
            "heads": heads,
            "dropout": dropout,
            "pooling": pooling,
        }

        # GAT convolution layers
        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(
            GATConv(
                in_channels=node_in_dim,
                out_channels=self.head_dim,
                heads=self.heads,
                edge_dim=edge_dim,
                dropout=dropout,
                concat=True,
            )
        )
        # Additional layers
        for _ in range(self.num_layers - 1):
            self.convs.append(
                GATConv(
                    in_channels=self.effective_hidden_dim,
                    out_channels=self.head_dim,
                    heads=self.heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    concat=True,
                )
            )

        self.dropout = nn.Dropout(dropout)

        # Graph pooling dimension
        pool_dim = self.effective_hidden_dim * 2 if pooling == "both" else self.effective_hidden_dim

        # Graph-level classification head
        self.head = nn.Sequential(
            nn.Linear(pool_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: Any,
        edge_index: Any,
        batch: Any,
        edge_attr: Optional[Any] = None,
        edge_weight: Optional[Any] = None,
    ) -> Any:
        h = x
        for i, conv in enumerate(self.convs):
            # Check edge_attr availability and shape compatibility
            if edge_attr is not None and edge_attr.size(0) > 0 and self.edge_dim is not None:
                h = conv(h, edge_index, edge_attr=edge_attr)
            else:
                h = conv(h, edge_index)
            h = F.elu(h)
            h = self.dropout(h)

        # Graph readout pooling
        if self.pooling == "mean":
            graph_rep = global_mean_pool(h, batch)
        elif self.pooling == "max":
            graph_rep = global_max_pool(h, batch)
        elif self.pooling == "both":
            mean_rep = global_mean_pool(h, batch)
            max_rep = global_max_pool(h, batch)
            graph_rep = torch.cat([mean_rep, max_rep], dim=-1)
        else:
            graph_rep = global_mean_pool(h, batch)

        # Prediction head
        out = self.head(graph_rep).squeeze(-1)
        return out
