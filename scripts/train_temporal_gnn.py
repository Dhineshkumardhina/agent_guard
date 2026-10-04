"""Training CLI Script for Temporal Graph Neural Network (TGN-Style) - Phase 11.

Usage:
    python scripts/train_temporal_gnn.py --horizon all --epochs 15
    python scripts/train_temporal_gnn.py --horizon 5 --debug
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage
from ml.baselines.temporal_gnn.models import get_device
from ml.baselines.temporal_gnn.experiment import TemporalGNNExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Temporal Graph Neural Network (TGN-style) failure prediction model."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="all",
        help="Prediction horizon K: '1', '3', '5', '10', '20', or 'all'.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Trajectory batch size.")
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--memory-dim", type=int, default=64, help="Node memory dimension.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--time-dim", type=int, default=16, help="Temporal encoding dimension.")
    parser.add_argument("--neighbor-history", type=int, default=10, help="Max recent neighbor interactions.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines/temporal_gnn",
        help="Directory to persist machine-readable results.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in rapid debug mode (3 epochs, single horizon).",
    )
    return parser.parse_args()


def format_table_row(cols, widths):
    return " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, widths))


def main():
    args = parse_args()
    device = get_device(verbose=True)

    dataset_path = Path(args.dataset)
    storage = DatasetStorage(dataset_path.parent)

    train_file = dataset_path / "train.parquet"
    val_file = dataset_path / "val.parquet"
    test_file = dataset_path / "test.parquet"

    train_samples = storage.load_tabular(train_file)
    val_samples = storage.load_tabular(val_file) if val_file.exists() else []
    test_samples = storage.load_tabular(test_file)

    train_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_train.jsonl")
    val_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_val.jsonl") if (dataset_path / "graph_sequences_val.jsonl").exists() else {}
    test_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_test.jsonl")

    manifest_obj = storage.load_manifest(dataset_path) if (dataset_path / "manifest.json").exists() else None
    dataset_version = manifest_obj.dataset_version if manifest_obj else dataset_path.name

    # Resolve horizons
    if args.debug:
        horizons = [1]
        epochs = 3
    elif args.horizon.lower().strip() == "all":
        horizons = [1, 3, 5, 10, 20]
        epochs = args.epochs
    else:
        horizons = [int(h.strip()) for h in args.horizon.split(",") if h.strip()]
        epochs = args.epochs

    print("=" * 90)
    print("AgentGuard: Temporal Graph Neural Network (TGN-Style) - Phase 11")
    print("=" * 90)
    print(f"Dataset Path      : {dataset_path}")
    print(f"Dataset Version   : {dataset_version}")
    print(f"Horizons (K)      : {horizons}")
    print(f"Max Epochs        : {epochs}")
    print(f"Learning Rate     : {args.learning_rate}")
    print(f"Memory Dimension  : {args.memory_dim}")
    print(f"Hidden Dimension  : {args.hidden_dim}")
    print(f"Time Dimension    : {args.time_dim}")
    print(f"Neighbor History  : {args.neighbor_history}")
    print(f"Random Seed       : {args.seed}")
    print(f"Train / Val / Test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)} samples")
    print("-" * 90)

    runner = TemporalGNNExperimentRunner(
        base_results_dir=args.results_dir,
        random_seed=args.seed,
        device=device,
    )

    print("\nTraining and evaluating Temporal GNN model...")
    summary = runner.run_benchmark(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_graph_sequences=train_seqs,
        val_graph_sequences=val_seqs,
        test_graph_sequences=test_seqs,
        horizons=horizons,
        dataset_version=dataset_version,
        epochs=epochs,
        lr=args.learning_rate,
        memory_dim=args.memory_dim,
        time_dim=args.time_dim,
        embed_dim=args.hidden_dim,
        neighbor_history=args.neighbor_history,
        dropout=args.dropout,
    )

    # Print temporal diagnostics
    diag = summary.get("diagnostics", {})
    print("\n" + "=" * 90)
    print("TEMPORAL INTERACTION DIAGNOSTICS")
    print("=" * 90)
    print(f"Total Trajectories      : {diag.get('number_of_runs', 0)}")
    print(f"Total Interaction Events: {diag.get('total_interaction_events', 0)}")
    print(f"Unique Agents Tracked   : {diag.get('unique_agents_count', 0)}")
    print(f"Avg Interactions / Run  : {diag.get('average_interactions_per_run', 0.0)}")
    print(r"Temporal Gap (\Delta t) : " + f"{diag.get('temporal_gap_distribution', {})}")
    print(f"Avg Memory Updates / Run: {diag.get('average_memory_updates_per_run', 0.0)}")
    print(f"Prediction Points       : {diag.get('total_prediction_points', 0)}")
    print(f"Class Distribution      : {diag.get('class_distribution', {})}")
    print("-" * 90)

    # Print comparative performance table
    print("\n" + "=" * 90)
    print("TEMPORAL GNN PERFORMANCE TABLE")
    print("=" * 90)
    headers = ["Model", "Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "Lead (s)"]
    widths = [14, 11, 9, 9, 9, 9, 9, 8, 9]
    print(format_table_row(headers, widths))
    print("-" * 90)

    for h_key, res in summary.get("results", {}).items():
        k_val = h_key.replace("k_", "")
        m = res.get("metrics", {})
        lt = res.get("lead_time", {})
        row = [
            "TEMPORAL GNN",
            f"k={k_val}",
            f"{m.get('precision', 0.0):.4f}",
            f"{m.get('recall', 0.0):.4f}",
            f"{m.get('f1', 0.0):.4f}",
            f"{m.get('auroc', 0.0):.4f}",
            f"{m.get('auprc', 0.0):.4f}",
            f"{m.get('false_positive_rate', 0.0):.4f}",
            f"{lt.get('mean_lead_time', 0.0):.4f}",
        ]
        print(format_table_row(row, widths))
    print("-" * 90)

    print("\nTemporal GNN benchmark completed successfully!")
    print(f"Results persisted under: {runner.base_results_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
