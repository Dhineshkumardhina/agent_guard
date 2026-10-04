r"""Time Encoding Module for Temporal Graph Networks - Phase 11.

Mathematical Representation:
Given an elapsed time interval between interactions:
    \Delta t = t_{\text{current}} - t_{\text{previous}} \ge 0

The continuous sinusoidal Fourier temporal encoding is defined as:
    \phi_k(\Delta t) = \cos(\Delta t \cdot \omega_k + \psi_k) \quad \text{for } k = 1, \dots, D_{\text{time}}

where the frequencies are logarithmically spaced:
    \omega_k = \frac{1}{10000^{2k / D_{\text{time}}}}

Alternatively, a learnable Fourier feature or linear projection maps:
    \phi(\Delta t) = \text{Linear}(\text{FourierFeatures}(\Delta t))

This ensures smooth, continuous representation of interaction delays, burstiness,
and temporal idle periods between multi-agent communications.
"""

from typing import Optional
import math
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Continuous sinusoidal and learnable time encoding for interaction timestamps."""

    def __init__(
        self,
        dimension: int = 16,
        encoding_type: str = "sinusoidal",
        trainable: bool = False,
    ) -> None:
        """Initialize TimeEncoder.
        
        Args:
            dimension: Dimension of the output temporal embedding vector D_time.
            encoding_type: 'sinusoidal' or 'learnable'.
            trainable: Whether frequency weights are learnable.
        """
        super().__init__()
        self.dimension = dimension
        self.encoding_type = encoding_type.lower()
        self.trainable = trainable

        if self.encoding_type == "sinusoidal":
            half_dim = dimension // 2
            # Logarithmically spaced frequencies
            freqs = torch.exp(
                -math.log(10000.0) * torch.arange(0, half_dim, dtype=torch.float32) / max(1, half_dim)
            )
            if trainable:
                self.frequencies = nn.Parameter(freqs)
            else:
                self.register_buffer("frequencies", freqs)
        elif self.encoding_type == "learnable":
            self.linear = nn.Linear(1, dimension)
        else:
            raise ValueError(f"Unknown time encoding type: {encoding_type}. Choose 'sinusoidal' or 'learnable'.")

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        r"""Encode time intervals \Delta t into continuous vector representation.
        
        Args:
            delta_t: Tensor of shape (...,) or (..., 1) containing non-negative time intervals.
            
        Returns:
            Tensor of shape (..., dimension) containing time embeddings.
        """
        # Ensure delta_t is at least 1D float
        if delta_t.dim() == 0:
            delta_t = delta_t.unsqueeze(0)
        if delta_t.dim() > 1 and delta_t.size(-1) == 1:
            delta_t = delta_t.squeeze(-1)

        # Enforce non-negativity
        delta_t = torch.clamp(delta_t, min=0.0)

        if self.encoding_type == "sinusoidal":
            # delta_t shape: [B], frequencies shape: [D/2] -> [B, D/2]
            angles = delta_t.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            sin_part = torch.sin(angles)
            cos_part = torch.cos(angles)
            encoded = torch.cat([sin_part, cos_part], dim=-1)

            # Pad by 1 if dimension is odd
            if encoded.size(-1) < self.dimension:
                pad = torch.zeros(*encoded.shape[:-1], self.dimension - encoded.size(-1), device=encoded.device)
                encoded = torch.cat([encoded, pad], dim=-1)
            return encoded
        else:
            dt_in = delta_t.unsqueeze(-1)
            return torch.sin(self.linear(dt_in))
