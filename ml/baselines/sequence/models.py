"""PyTorch LSTM and GRU Temporal Baseline Architectures.

Implements causal temporal baseline networks for failure forecasting:
1. LSTMSequenceBaseline
2. GRUSequenceBaseline

CAUSAL ARCHITECTURE DESIGN:
Models are strictly unidirectional (bidirectional=False) to ensure no future
leakage from later sequence positions into earlier temporal states.
Input sequences of shape (Batch, Seq_Len, Features) are mapped to failure logits
and probabilities in [0.0, 1.0].
"""

from typing import Dict, Any, Optional, Tuple
import sys

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object


def get_device(verbose: bool = True) -> Any:
    """Detect available compute device (CUDA GPU or CPU) and report details."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for sequence baselines. Install torch.")

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


class LSTMSequenceBaseline(nn.Module if HAS_TORCH else object):
    """Causal unidirectional LSTM baseline for multi-agent failure forecasting."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        """Initialize LSTM sequence baseline.
        
        Args:
            input_size: Feature vector dimension D (default 28).
            hidden_size: Hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability between recurrent layers.
            bidirectional: Must remain False for causal prediction.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LSTMSequenceBaseline.")
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.model_name = "lstm"

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        fc_in_dim = hidden_size * num_directions

        # Classification head mapping final temporal state to single failure logit
        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: Any) -> Any:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
            
        Returns:
            Logits tensor of shape (batch_size,).
        """
        # lstm_out: (batch_size, seq_len, hidden_size * directions)
        lstm_out, _ = self.lstm(x)
        # Extract the representation at the final sequence step t
        last_step_repr = lstm_out[:, -1, :]
        logits = self.head(last_step_repr).squeeze(-1)
        return logits

    def predict_proba(self, x: Any) -> Any:
        """Compute sigmoid failure probabilities in [0.0, 1.0]."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs


class GRUSequenceBaseline(nn.Module if HAS_TORCH else object):
    """Causal unidirectional GRU baseline for multi-agent failure forecasting."""

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        """Initialize GRU sequence baseline."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required for GRUSequenceBaseline.")
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.model_name = "gru"

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        num_directions = 2 if bidirectional else 1
        fc_in_dim = hidden_size * num_directions

        self.head = nn.Sequential(
            nn.Linear(fc_in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: Any) -> Any:
        gru_out, _ = self.gru(x)
        last_step_repr = gru_out[:, -1, :]
        logits = self.head(last_step_repr).squeeze(-1)
        return logits

    def predict_proba(self, x: Any) -> Any:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
        return probs
