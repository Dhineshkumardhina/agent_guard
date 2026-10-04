"""Experiment Runner and Benchmark Orchestration for Static GNNs - Phase 10.

Coordinates full-scale, reproducible experiments across:
Models (GCN, GAT) x Prediction Horizons (k in {1, 3, 5, 10, 20}).
Persists:
- Machine-readable checkpoints with complete architecture and training configuration
- Training histories (training_history.csv)
- Structural graph diagnostics (graph_diagnostics.json)
- Full prediction tables (predictions.json)
- Comparative experiment summaries (summary.json)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import time
import random
import numpy as np

try:
    import torch
    import torch_geometric
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    DataLoader = object

from ml.baselines.static_gnn.models import GCNBaseline, GATBaseline, get_device
from ml.baselines.static_gnn.dataset import (
    StaticGraphDatasetBuilder,
    compute_graph_diagnostics,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
)
from ml.baselines.static_gnn.trainer import StaticGNNTrainer
from ml.baselines.static_gnn.evaluator import evaluate_static_gnn


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_PYG:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class StaticGNNExperimentRunner:
    """Orchestrates training, evaluation, diagnostics, and result storage for Static GNNs."""

    def __init__(
        self,
        base_results_dir: Union[str, Path] = "results/baselines/static_gnn",
        random_seed: int = 42,
        device: Optional[Any] = None,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.random_seed = random_seed
        self.device = device or (get_device(verbose=False) if HAS_PYG else None)
        self.dataset_builder = StaticGraphDatasetBuilder()

    def run_benchmark(
        self,
        train_samples: List[Dict[str, Any]],
        val_samples: List[Dict[str, Any]],
        test_samples: List[Dict[str, Any]],
        train_graph_sequences: Dict[str, List[Dict[str, Any]]],
        val_graph_sequences: Dict[str, List[Dict[str, Any]]],
        test_graph_sequences: Dict[str, List[Dict[str, Any]]],
        models_to_run: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 0.001,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        heads: int = 4,
        pooling: str = "mean",
    ) -> Dict[str, Any]:
        """Execute full Static GNN benchmark suite."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric required for StaticGNNExperimentRunner.")

        set_seed(self.random_seed)
        models_to_run = [m.lower().strip() for m in (models_to_run or ["gcn", "gat"])]
        horizons = horizons or [1, 3, 5, 10, 20]

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        benchmark_summary: Dict[str, Any] = {
            "dataset_version": dataset_version,
            "feature_version": "v1_static_gnn_14node_10edge",
            "models": models_to_run,
            "horizons": horizons,
            "random_seed": self.random_seed,
            "pytorch_version": torch.__version__,
            "pyg_version": torch_geometric.__version__,
            "device": str(self.device),
            "timestamp": timestamp_str,
            "results": {},
        }

        # ── 1. Structural Graph Diagnostics ──
        # Generate diagnostics on all available training graph snapshots
        all_train_graphs = []
        for k in horizons:
            all_train_graphs.extend(
                self.dataset_builder.build_dataset_for_horizon(
                    train_samples, train_graph_sequences, horizon=k
                )
            )

        diagnostics = compute_graph_diagnostics(all_train_graphs)
        diagnostics_path = self.base_results_dir / "graph_diagnostics.json"
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)
        benchmark_summary["diagnostics"] = diagnostics

        # ── 2. Run Experiments: Model x Horizon ──
        for model_name in models_to_run:
            benchmark_summary["results"][model_name] = {}

            for k in horizons:
                experiment_id = f"static_{model_name}_k{k}_{timestamp_str}"
                checkpoint_dir = self.base_results_dir / model_name / f"k_{k}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Prepare horizon datasets
                train_data = self.dataset_builder.build_dataset_for_horizon(
                    train_samples, train_graph_sequences, horizon=k
                )
                val_data = self.dataset_builder.build_dataset_for_horizon(
                    val_samples, val_graph_sequences, horizon=k
                )
                test_data = self.dataset_builder.build_dataset_for_horizon(
                    test_samples, test_graph_sequences, horizon=k
                )

                if not train_data or not test_data:
                    continue

                # Calculate class imbalance strictly from training split
                train_labels = [int(g.y.item()) if hasattr(g.y, "item") else int(g.y[0]) for g in train_data]
                pos_count = sum(train_labels)
                neg_count = len(train_labels) - pos_count
                pos_weight = (neg_count / float(max(1, pos_count))) if pos_count > 0 else 1.0

                # DataLoaders
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else train_loader
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                # Instantiate model
                if model_name == "gcn":
                    model = GCNBaseline(
                        node_in_dim=len(NODE_FEATURE_NAMES),
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        pooling=pooling,
                    )
                elif model_name == "gat":
                    model = GATBaseline(
                        node_in_dim=len(NODE_FEATURE_NAMES),
                        edge_dim=len(EDGE_FEATURE_NAMES),
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        heads=heads,
                        dropout=dropout,
                        pooling=pooling,
                    )
                else:
                    raise ValueError(f"Unknown model name: {model_name}")

                # Trainer
                trainer = StaticGNNTrainer(
                    model=model,
                    lr=lr,
                    patience=5,
                    device=self.device,
                    random_seed=self.random_seed,
                )

                metadata_payload = {
                    "experiment_id": experiment_id,
                    "dataset_version": dataset_version,
                    "feature_version": "v1_static_gnn_14node_10edge",
                    "graph_representation_version": "discrete_snapshot_G(t)",
                    "prediction_horizon": k,
                    "model_name": model_name,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "heads": heads if model_name == "gat" else None,
                    "pooling": pooling,
                    "batch_size": batch_size,
                    "pos_weight": round(float(pos_weight), 4),
                    "train_samples": len(train_data),
                    "val_samples": len(val_data),
                    "test_samples": len(test_data),
                }

                # Train with validation early stopping and threshold tuning
                train_result = trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    pos_weight=pos_weight,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_metadata=metadata_payload,
                )

                frozen_threshold = train_result["frozen_threshold"]

                # Evaluate on held-out test split using frozen threshold
                eval_result = evaluate_static_gnn(
                    model=model,
                    test_loader=test_loader,
                    threshold=frozen_threshold,
                    device=self.device,
                )

                # Persist test predictions JSON
                predictions_path = checkpoint_dir / "predictions.json"
                with open(predictions_path, "w", encoding="utf-8") as f:
                    json.dump(eval_result["predictions"], f, indent=2)

                # Persist metrics JSON
                metrics_record = {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
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

                benchmark_summary["results"][model_name][f"k_{k}"] = metrics_record

        # Persist summary
        summary_path = self.base_results_dir / f"summary_{timestamp_str}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        latest_path = self.base_results_dir / "latest_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        return benchmark_summary
