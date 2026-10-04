"""Experiment Runner and Benchmark Orchestration for Temporal GNNs - Phase 11.

Coordinates full-scale, reproducible experiments across prediction horizons k in {1, 3, 5, 10, 20}.
Persists:
- Temporal diagnostics (temporal_diagnostics.json)
- Checkpoints with memory and temporal architecture metadata
- Training histories (training_history.csv)
- Test prediction tables (predictions.json)
- Summary JSON reports
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import time
import random
import numpy as np
import torch

from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor, get_device, AblationConfig
from ml.baselines.temporal_gnn.dataset import (
    TemporalDatasetBuilder,
    compute_temporal_diagnostics,
)
from ml.baselines.temporal_gnn.trainer import TemporalGNNTrainer
from ml.baselines.temporal_gnn.evaluator import evaluate_temporal_gnn


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TemporalGNNExperimentRunner:
    """Orchestrates training, evaluation, diagnostics, and result storage for Temporal GNNs."""

    def __init__(
        self,
        base_results_dir: Union[str, Path] = "results/baselines/temporal_gnn",
        random_seed: int = 42,
        device: Optional[torch.device] = None,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.random_seed = random_seed
        self.device = device or get_device(verbose=False)
        self.dataset_builder = TemporalDatasetBuilder()

    def run_benchmark(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        train_graph_sequences: Dict[str, List[Dict[str, Any]]],
        val_graph_sequences: Dict[str, List[Dict[str, Any]]],
        test_graph_sequences: Dict[str, List[Dict[str, Any]]],
        horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
        epochs: int = 15,
        lr: float = 0.001,
        memory_dim: int = 64,
        time_dim: int = 16,
        embed_dim: int = 64,
        neighbor_history: int = 10,
        dropout: float = 0.2,
        ablation_config: Optional[AblationConfig] = None,
    ) -> Dict[str, Any]:
        """Execute full Temporal GNN benchmark suite."""
        set_seed(self.random_seed)
        horizons = horizons or [1, 3, 5, 10, 20]

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        benchmark_summary: Dict[str, Any] = {
            "dataset_version": dataset_version,
            "feature_version": "v1_temporal_gnn_14node_10edge",
            "model_name": "temporal_gnn",
            "horizons": horizons,
            "random_seed": self.random_seed,
            "pytorch_version": torch.__version__,
            "device": str(self.device),
            "timestamp": timestamp_str,
            "results": {},
        }

        # ── 1. Temporal Diagnostics ──
        all_train_trajs = self.dataset_builder.build_trajectories(
            tabular_samples=train_samples,
            graph_sequences=train_graph_sequences,
            horizon_filter=None,
        )
        diagnostics = compute_temporal_diagnostics(all_train_trajs)
        diagnostics_path = self.base_results_dir / "temporal_diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        benchmark_summary["diagnostics"] = diagnostics

        # ── 2. Run Experiments Per Horizon ──
        for k in horizons:
            experiment_id = f"temporal_gnn_k{k}_{timestamp_str}"
            checkpoint_dir = self.base_results_dir / f"k_{k}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Build trajectory partitions for horizon k
            train_trajs = self.dataset_builder.build_trajectories(
                tabular_samples=train_samples,
                graph_sequences=train_graph_sequences,
                horizon_filter=k,
            )
            val_trajs = self.dataset_builder.build_trajectories(
                tabular_samples=val_samples,
                graph_sequences=val_graph_sequences,
                horizon_filter=k,
            )
            test_trajs = self.dataset_builder.build_trajectories(
                tabular_samples=test_samples,
                graph_sequences=test_graph_sequences,
                horizon_filter=k,
            )

            # Filter out empty prediction trajectories
            train_trajs = [t for t in train_trajs if t.prediction_points]
            val_trajs = [t for t in val_trajs if t.prediction_points]
            test_trajs = [t for t in test_trajs if t.prediction_points]

            if not train_trajs or not test_trajs:
                continue

            # Calculate class distribution strictly from training split
            train_labels = [pp.label for t in train_trajs for pp in t.prediction_points]
            pos_count = sum(train_labels)
            neg_count = len(train_labels) - pos_count
            pos_weight = (neg_count / float(max(1, pos_count))) if pos_count > 0 else 1.0

            # Instantiate TemporalGraphFailurePredictor
            model = TemporalGraphFailurePredictor(
                node_in_dim=14,
                edge_in_dim=10,
                memory_dim=memory_dim,
                time_dim=time_dim,
                embed_dim=embed_dim,
                neighbor_history=neighbor_history,
                dropout=dropout,
                ablation_config=ablation_config,
                device=self.device,
            )

            trainer = TemporalGNNTrainer(
                model=model,
                lr=lr,
                patience=5,
                device=self.device,
                random_seed=self.random_seed,
            )

            metadata_payload = {
                "experiment_id": experiment_id,
                "dataset_version": dataset_version,
                "feature_version": "v1_temporal_gnn_14node_10edge",
                "prediction_horizon": k,
                "model_name": "temporal_gnn",
                "memory_dim": memory_dim,
                "time_dim": time_dim,
                "embed_dim": embed_dim,
                "neighbor_history": neighbor_history,
                "dropout": dropout,
                "pos_weight": round(float(pos_weight), 4),
                "train_trajectories": len(train_trajs),
                "val_trajectories": len(val_trajs),
                "test_trajectories": len(test_trajs),
            }

            # Train with validation early stopping and threshold tuning
            train_result = trainer.train(
                train_trajectories=train_trajs,
                val_trajectories=val_trajs,
                epochs=epochs,
                pos_weight=pos_weight,
                checkpoint_dir=checkpoint_dir,
                checkpoint_metadata=metadata_payload,
            )

            frozen_threshold = train_result["frozen_threshold"]

            # Evaluate on held-out test split using frozen threshold
            eval_result = evaluate_temporal_gnn(
                model=model,
                test_trajectories=test_trajs,
                threshold=frozen_threshold,
            )

            # Persist test predictions JSON
            predictions_path = checkpoint_dir / "predictions.json"
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(eval_result["predictions"], f, indent=2)

            # Persist metrics JSON
            metrics_record = {
                "experiment_id": experiment_id,
                "model_name": "temporal_gnn",
                "prediction_horizon": k,
                "selected_threshold": frozen_threshold,
                "metrics": eval_result["metrics"],
                "lead_time": eval_result["lead_time"],
                "training": {
                    "epochs_trained": len(train_result["history"]),
                    "best_epoch": train_result["best_epoch"],
                    "best_val_loss": train_result["best_val_loss"],
                },
                "metadata": metadata_payload,
            }
            with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics_record, f, indent=2)

            benchmark_summary["results"][f"k_{k}"] = metrics_record

        # Persist summary
        summary_path = self.base_results_dir / f"summary_{timestamp_str}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        latest_path = self.base_results_dir / "latest_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        return benchmark_summary
