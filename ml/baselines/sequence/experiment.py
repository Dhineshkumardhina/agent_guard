"""Temporal Sequence Experiment Orchestrator and Checkpoint Manager.

Coordinates multi-model, multi-sequence-length, multi-horizon benchmarking for:
- LSTM
- GRU
across sequence lengths L in {5, 10, 20, 50} and horizons K in {1, 3, 5, 10, 20}.

Results are persisted systematically under:
results/baselines/sequence/
  ├── lstm/
  │    └── <experiment_id>/
  │         ├── checkpoint.pt
  │         ├── training_history.csv
  │         ├── metrics.json
  │         └── predictions.parquet
  ├── gru/
  └── summary_<experiment_id>.json
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import json
from uuid import uuid4
import numpy as np

from ml.baselines.sequence.dataset import SequenceDatasetBuilder
from ml.baselines.sequence.models import (
    LSTMSequenceBaseline,
    GRUSequenceBaseline,
    get_device,
    HAS_TORCH,
)
from ml.baselines.sequence.trainer import SequenceTrainer
from ml.baselines.sequence.evaluator import evaluate_sequence_model
from ml.data.storage import HAS_PYARROW

if HAS_PYARROW:
    import pyarrow as pa
    import pyarrow.parquet as pq


class SequenceExperimentRunner:
    """Manages training loops, validation calibration, testing, and persistence for sequence models."""

    def __init__(
        self,
        base_results_dir: Union[str, Path] = "results/baselines/sequence",
        random_seed: int = 42,
        device: Optional[Any] = None,
    ) -> None:
        self.base_results_dir = Path(base_results_dir)
        self.random_seed = random_seed
        self.device = device or (get_device(verbose=False) if HAS_TORCH else None)

    def run_benchmark(
        self,
        train_samples: List[Any],
        val_samples: List[Any],
        test_samples: List[Any],
        models_to_run: Optional[List[str]] = None,
        sequence_lengths: Optional[List[int]] = None,
        horizons: Optional[List[int]] = None,
        dataset_version: str = "agentguard_dataset_v1",
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 0.001,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """Execute full temporal sequence benchmark."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SequenceExperimentRunner.")

        models = models_to_run or ["lstm", "gru"]
        seq_lens = sequence_lengths or [5, 10, 20]
        eval_horizons = horizons or [1, 3, 5, 10, 20]

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        experiment_id = f"exp_seq_{timestamp_str}_{uuid4().hex[:6]}"

        benchmark_summary: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version,
            "feature_version": "v1 (28 tabular features)",
            "random_seed": self.random_seed,
            "device": str(self.device),
            "models": models,
            "sequence_lengths": seq_lens,
            "horizons": eval_horizons,
            "training_hyperparameters": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "patience": patience,
            },
            "results": {},
        }

        for model_name in models:
            benchmark_summary["results"][model_name] = {}
            model_base_dir = self.base_results_dir / model_name / experiment_id
            model_base_dir.mkdir(parents=True, exist_ok=True)

            for L in seq_lens:
                benchmark_summary["results"][model_name][f"seq_{L}"] = {}
                builder = SequenceDatasetBuilder(sequence_length=L)

                for k in eval_horizons:
                    # 1. Build DataLoaders for horizon K
                    train_loader, train_info = builder.create_dataloader(
                        train_samples, horizon=k, batch_size=batch_size, shuffle=True
                    )
                    val_loader, val_info = builder.create_dataloader(
                        val_samples, horizon=k, batch_size=batch_size, shuffle=False
                    )
                    test_loader, test_info = builder.create_dataloader(
                        test_samples, horizon=k, batch_size=batch_size, shuffle=False
                    )

                    if train_info["num_samples"] == 0 or test_info["num_samples"] == 0:
                        continue

                    # If validation split is empty, fallback to training set for validation threshold selection
                    if val_info["num_samples"] == 0:
                        val_loader = train_loader
                        val_info = train_info

                    # 2. Instantiate Model
                    if model_name.lower() == "lstm":
                        net = LSTMSequenceBaseline(
                            input_size=builder.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=False,
                        )
                    elif model_name.lower() == "gru":
                        net = GRUSequenceBaseline(
                            input_size=builder.feature_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=False,
                        )
                    else:
                        raise ValueError(f"Unknown sequence model: {model_name}")

                    # 3. Calculate positive class weight from TRAINING set only
                    pos_w = None
                    if train_info["pos_count"] > 0:
                        pos_w = float(train_info["neg_count"] / train_info["pos_count"])

                    # 4. Train model with validation early stopping and threshold freezing
                    run_subdir = model_base_dir / f"seq_{L}_k{k}"
                    trainer = SequenceTrainer(
                        model=net,
                        lr=lr,
                        patience=patience,
                        device=self.device,
                        random_seed=self.random_seed,
                    )
                    train_out = trainer.train(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        pos_weight=pos_w,
                        checkpoint_dir=run_subdir,
                        checkpoint_metadata={
                            "experiment_id": experiment_id,
                            "horizon": k,
                            "sequence_length": L,
                            "dataset_version": dataset_version,
                        },
                    )

                    frozen_tau = train_out["frozen_threshold"]

                    # 5. Evaluate on held-out TEST split
                    test_res = evaluate_sequence_model(
                        model=net,
                        test_loader=test_loader,
                        threshold=frozen_tau,
                        device=self.device,
                    )

                    # Save test predictions
                    self._save_predictions(test_res["predictions"], run_subdir / "predictions.parquet")

                    # Save metrics JSON
                    metrics_payload = {
                        "experiment_id": experiment_id,
                        "model_name": model_name,
                        "sequence_length": L,
                        "horizon": k,
                        "dataset_version": dataset_version,
                        "selected_threshold": frozen_tau,
                        "training_summary": {
                            "best_epoch": train_out["best_epoch"],
                            "best_val_loss": train_out["best_val_loss"],
                        },
                        "metrics": test_res["metrics"],
                        "lead_time": test_res["lead_time"],
                    }
                    with open(run_subdir / "metrics.json", "w", encoding="utf-8") as f:
                        json.dump(metrics_payload, f, indent=2)

                    benchmark_summary["results"][model_name][f"seq_{L}"][f"k_{k}"] = {
                        "metrics": test_res["metrics"],
                        "lead_time": test_res["lead_time"],
                        "selected_threshold": frozen_tau,
                    }

        # 6. Save overarching summary
        summary_path = self.base_results_dir / f"summary_{experiment_id}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, indent=2)

        return benchmark_summary

    def _save_predictions(self, predictions: List[Dict[str, Any]], filepath: Path) -> None:
        """Save prediction records to Parquet or JSONL."""
        if not predictions:
            return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if HAS_PYARROW:
            table = pa.Table.from_pylist(predictions)
            pq.write_table(table, filepath)
        else:
            with open(filepath.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
