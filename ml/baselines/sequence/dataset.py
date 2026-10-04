"""Temporal Sequence Dataset Builder for Multi-Agent Failure Prediction.

Constructs historical feature sequences:
X(t-L+1), ..., X(t-1), X(t)
for predicting failure within forward horizon k:
P(F(t+k) | X(<= t))

Supports configurable sequence lengths:
L in {5, 10, 20, 50}

STRICT CAUSALITY GUARANTEE:
Sequences are constructed using strictly causal left-padding (pre-padding with zeros)
so that the final sequence step always corresponds to the observation at time t.
No future events (steps > t or timestamps > t) are ever included.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import numpy as np

from ml.baselines.classical_ml.features import TabularFeatureExtractor, FEATURE_NAMES

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object


class MultiAgentSequenceDataset(Dataset):
    """PyTorch Dataset yielding (sequence_tensor, label, sample_metadata)."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Initialize Sequence Dataset.
        
        Args:
            sequences: NumPy array of shape (N, L, D).
            labels: NumPy array of shape (N,).
            metadata: List of N metadata dictionaries.
        """
        self.sequences = sequences
        self.labels = labels
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Dict[str, Any]]:
        seq = self.sequences[idx]
        lbl = self.labels[idx]
        meta = self.metadata[idx]

        if HAS_TORCH:
            seq_tensor = torch.tensor(seq, dtype=torch.float32)
            lbl_tensor = torch.tensor(lbl, dtype=torch.float32)
            return seq_tensor, lbl_tensor, meta

        return seq, lbl, meta


class SequenceDatasetBuilder:
    """Constructs causal sequence datasets from multi-agent prediction samples."""

    def __init__(
        self,
        sequence_length: int = 10,
        feature_extractor: Optional[TabularFeatureExtractor] = None,
    ) -> None:
        """Initialize builder.
        
        Args:
            sequence_length: Temporal sequence window length L (e.g. 5, 10, 20, 50).
            feature_extractor: TabularFeatureExtractor instance (Phase 8 feature spec).
        """
        self.sequence_length = sequence_length
        self.extractor = feature_extractor or TabularFeatureExtractor()
        self.feature_dim = len(self.extractor.feature_names)

    def build_sequences_for_horizon(
        self,
        samples: List[Union[Dict[str, Any], Any]],
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Construct sequence matrix X of shape (N, L, D), targets y, and metadata for horizon K.
        
        Args:
            samples: List of PredictionSample instances or tabular rows.
            horizon: Target prediction horizon K (e.g. 1, 3, 5, 10, 20).
            
        Returns:
            Tuple of:
                - sequences: np.ndarray of shape (N, L, D)
                - labels: np.ndarray of shape (N,)
                - metadata: List of dicts (sample_id, run_id, timestamp, step_idx, horizon)
        """
        # 1. Group all samples by run_id
        runs_map = defaultdict(list)
        for s in samples:
            rid = getattr(s, "run_id", None) or (s.get("run_id") if isinstance(s, dict) else "")
            runs_map[rid].append(s)

        sequences_list = []
        labels_list = []
        metadata_list = []

        # 2. Process each trajectory chronologically
        for rid, run_samples in runs_map.items():
            # Sort chronologically by timestamp and step_idx
            run_samples.sort(
                key=lambda x: (
                    getattr(x, "timestamp", 0.0) if hasattr(x, "timestamp") else x.get("timestamp", 0.0),
                    getattr(x, "step_idx", 0) if hasattr(x, "step_idx") else x.get("step_idx", 0),
                )
            )

            # Pre-extract tabular feature vectors for all steps in this run
            # Note: every sample's features were computed causally <= its step_idx
            step_to_feat = {}
            for s in run_samples:
                s_step = getattr(s, "step_idx", None) if hasattr(s, "step_idx") else s.get("step_idx", 0)
                if s_step not in step_to_feat:
                    f_dict = self.extractor.extract_from_sample(s)
                    f_vec = np.array([f_dict[name] for name in self.extractor.feature_names], dtype=np.float32)
                    step_to_feat[s_step] = f_vec

            ordered_steps = sorted(list(step_to_feat.keys()))

            # 3. For each prediction point matching the requested horizon K
            for s in run_samples:
                s_h = getattr(s, "prediction_horizon", None) if hasattr(s, "prediction_horizon") else s.get("prediction_horizon")
                if s_h != horizon:
                    continue

                curr_step = getattr(s, "step_idx", None) if hasattr(s, "step_idx") else s.get("step_idx", 0)
                curr_t = getattr(s, "timestamp", None) if hasattr(s, "timestamp") else s.get("timestamp", 0.0)
                curr_label = getattr(s, "label", None) if hasattr(s, "label") else s.get("label", 0)
                curr_id = getattr(s, "sample_id", None) or (s.get("sample_id") if isinstance(s, dict) else "")

                # Historical steps strictly <= curr_step
                valid_past_steps = [st for st in ordered_steps if st <= curr_step]

                # Slice the most recent self.sequence_length steps
                sliced_steps = valid_past_steps[-self.sequence_length:]
                history_vectors = [step_to_feat[st] for st in sliced_steps]

                # Causal pre-padding with zeros if history length < L
                pad_len = self.sequence_length - len(history_vectors)
                if pad_len > 0:
                    pad_block = [np.zeros((self.feature_dim,), dtype=np.float32) for _ in range(pad_len)]
                    seq_matrix = np.vstack(pad_block + history_vectors)
                else:
                    seq_matrix = np.vstack(history_vectors)

                sequences_list.append(seq_matrix)
                labels_list.append(int(curr_label))
                metadata_list.append({
                    "sample_id": curr_id,
                    "run_id": rid,
                    "step_idx": curr_step,
                    "timestamp": curr_t,
                    "prediction_horizon": horizon,
                    "sequence_length": self.sequence_length,
                    "history_steps_available": len(sliced_steps),
                })

        if not sequences_list:
            return (
                np.zeros((0, self.sequence_length, self.feature_dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        X = np.stack(sequences_list, axis=0).astype(np.float32)
        y = np.array(labels_list, dtype=np.float32)

        return X, y, metadata_list

    def create_dataloader(
        self,
        samples: List[Union[Dict[str, Any], Any]],
        horizon: int,
        batch_size: int = 16,
        shuffle: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Create PyTorch DataLoader for a specific horizon K."""
        X, y, meta = self.build_sequences_for_horizon(samples, horizon=horizon)
        dataset = MultiAgentSequenceDataset(X, y, meta)

        info = {
            "num_samples": len(X),
            "sequence_length": self.sequence_length,
            "feature_dim": self.feature_dim,
            "pos_count": int(np.sum(y == 1.0)) if len(y) > 0 else 0,
            "neg_count": int(np.sum(y == 0.0)) if len(y) > 0 else 0,
        }

        if HAS_TORCH:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return loader, info

        return dataset, info
