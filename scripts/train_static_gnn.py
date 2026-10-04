"""Training CLI Script for Static Graph Neural Networks (GCN / GAT) - Phase 10.

Usage:
    python scripts/train_static_gnn.py --model all --horizon all --epochs 20
    python scripts/train_static_gnn.py --model gcn --horizon 5 --debug
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage
from ml.baselines.static_gnn.models import get_device, HAS_PYG
from ml.baselines.static_gnn.experiment import StaticGNNExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Static Graph Neural Network baselines (GCN and GAT)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to processed dataset directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model architecture: 'gcn', 'gat', or 'all'.",
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="all",
        help="Prediction horizon K: '1', '3', '5', '10', '20', or 'all'.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="GNN hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GNN layers.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads for GAT.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument("--pooling", type=str, default="mean", help="Readout pooling: 'mean', 'max', or 'both'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines/static_gnn",
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

    if not HAS_PYG:
        print("Error: PyTorch Geometric is required to run static GNN baselines. Install torch_geometric.", file=sys.stderr)
        sys.exit(1)

    device = get_device(verbose=True)

    dataset_path = Path(args.dataset)
    storage = DatasetStorage(dataset_path.parent)

    train_file = dataset_path / "train.parquet"
    val_file = dataset_path / "val.parquet"
    test_file = dataset_path / "test.parquet"

    train_samples = storage.load_tabular(train_file)
    val_samples = storage.load_tabular(val_file) if val_file.exists() else []
    test_samples = storage.load_tabular(test_file)

    # Load graph sequences JSONL for each split
    train_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_train.jsonl")
    val_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_val.jsonl") if (dataset_path / "graph_sequences_val.jsonl").exists() else {}
    test_seqs = storage.load_graph_sequences(dataset_path / "graph_sequences_test.jsonl")

    manifest_obj = storage.load_manifest(dataset_path) if (dataset_path / "manifest.json").exists() else None
    dataset_version = manifest_obj.dataset_version if manifest_obj else dataset_path.name

    # Resolve models
    if args.model.lower().strip() == "all":
        models_to_run = ["gcn", "gat"]
    else:
        models_to_run = [m.strip().lower() for m in args.model.split(",") if m.strip()]

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
    print("AgentGuard: Static Graph Neural Network Baselines (GCN / GAT) - Phase 10")
    print("=" * 90)
    print(f"Dataset Path      : {dataset_path}")
    print(f"Dataset Version   : {dataset_version}")
    print(f"Models            : {models_to_run}")
    print(f"Horizons (K)      : {horizons}")
    print(f"Max Epochs        : {epochs}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Learning Rate     : {args.lr}")
    print(f"Hidden Dimension  : {args.hidden_dim}")
    print(f"GNN Layers        : {args.num_layers}")
    print(f"Readout Pooling   : {args.pooling}")
    print(f"Random Seed       : {args.seed}")
    print(f"Train / Val / Test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)} samples")
    print("-" * 90)

    runner = StaticGNNExperimentRunner(
        base_results_dir=args.results_dir,
        random_seed=args.seed,
        device=device,
    )

    print("\nTraining and evaluating Static GNN models...")
    summary = runner.run_benchmark(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_graph_sequences=train_seqs,
        val_graph_sequences=val_seqs,
        test_graph_sequences=test_seqs,
        models_to_run=models_to_run,
        horizons=horizons,
        dataset_version=dataset_version,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        heads=args.heads,
        pooling=args.pooling,
    )

    # Print structural diagnostics
    diag = summary.get("diagnostics", {})
    print("\n" + "=" * 90)
    print("STRUCTURAL GRAPH DIAGNOSTICS")
    print("=" * 90)
    print(f"Total Graphs Analyzed   : {diag.get('num_graphs', 0)}")
    print(f"Average Nodes per Graph : {diag.get('avg_nodes_per_graph', 0.0)}")
    print(f"Average Edges per Graph : {diag.get('avg_edges_per_graph', 0.0)}")
    print(f"Average Graph Density   : {diag.get('avg_graph_density', 0.0)}")
    print(f"Average Node Degree     : {diag.get('avg_node_degree', 0.0)}")
    print(f"Total Isolated Nodes    : {diag.get('total_isolated_nodes', 0)}")
    print(f"Topology Distribution   : {diag.get('topology_distribution', {})}")
    print(f"Class Distribution      : {diag.get('class_distribution', {})}")
    print("-" * 90)

    # Print comparative performance table
    print("\n" + "=" * 90)
    print("STATIC GNN BASELINE PERFORMANCE TABLE")
    print("=" * 90)
    headers = ["Model", "Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "Lead (s)"]
    widths = [8, 11, 9, 9, 9, 9, 9, 8, 9]
    print(format_table_row(headers, widths))
    print("-" * 90)

    for m_name, h_dict in summary.get("results", {}).items():
        for h_key, res in h_dict.items():
            k_val = h_key.replace("k_", "")
            m = res.get("metrics", {})
            lt = res.get("lead_time", {})
            row = [
                m_name.upper(),
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

    print("\nStatic GNN benchmark completed successfully!")
    print(f"Results persisted under: {runner.base_results_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
