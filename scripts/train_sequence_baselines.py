"""Training CLI Script for Temporal Sequence Baselines (LSTM / GRU) - Phase 9.

Usage:
    python scripts/train_sequence_baselines.py --model all --sequence-length 10 --horizon all --epochs 15
    python scripts/train_sequence_baselines.py --model lstm --sequence-length 10 --horizon 5 --debug
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage
from ml.baselines.sequence.models import get_device, HAS_TORCH
from ml.baselines.sequence.experiment import SequenceExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate LSTM and GRU temporal sequence baselines."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/processed/agentguard_dataset_v1",
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model type: 'lstm', 'gru', or 'all'.",
    )
    parser.add_argument(
        "--sequence-length",
        type=str,
        default="10",
        help="Sequence length L: '5', '10', '20', '50', or 'all'.",
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
    parser.add_argument("--hidden-size", type=int, default=64, help="Recurrent hidden size.")
    parser.add_argument("--dropout", type=float, default=0.2, help="Recurrent dropout rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines/sequence",
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

    if not HAS_TORCH:
        print("Error: PyTorch is required to run sequence baselines. Install torch.", file=sys.stderr)
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

    manifest_obj = storage.load_manifest(dataset_path) if (dataset_path / "manifest.json").exists() else None
    dataset_version = manifest_obj.dataset_version if manifest_obj else dataset_path.name

    # Resolve models
    if args.model.lower().strip() == "all":
        models_to_run = ["lstm", "gru"]
    else:
        models_to_run = [m.strip().lower() for m in args.model.split(",") if m.strip()]

    # Resolve sequence lengths
    if args.sequence_length.lower().strip() == "all":
        seq_lens = [5, 10, 20, 50]
    else:
        seq_lens = [int(l.strip()) for l in args.sequence_length.split(",") if l.strip()]

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
    print("AgentGuard: Temporal Sequence Baselines (LSTM / GRU) - Phase 9")
    print("=" * 90)
    print(f"Dataset Path      : {dataset_path}")
    print(f"Dataset Version   : {dataset_version}")
    print(f"Models            : {models_to_run}")
    print(f"Sequence Lengths  : {seq_lens}")
    print(f"Horizons (K)      : {horizons}")
    print(f"Max Epochs        : {epochs}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Learning Rate     : {args.lr}")
    print(f"Random Seed       : {args.seed}")
    print(f"Train / Val / Test: {len(train_samples)} / {len(val_samples)} / {len(test_samples)} samples")
    print("-" * 90)

    runner = SequenceExperimentRunner(
        base_results_dir=args.results_dir,
        random_seed=args.seed,
        device=device,
    )

    print("\nTraining and evaluating temporal sequence models...")
    summary = runner.run_benchmark(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        models_to_run=models_to_run,
        sequence_lengths=seq_lens,
        horizons=horizons,
        dataset_version=dataset_version,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )

    # Print comparative performance table
    print("\n" + "=" * 90)
    print("TEMPORAL SEQUENCE BASELINE PERFORMANCE TABLE")
    print("=" * 90)
    headers = ["Model", "Seq Len (L)", "Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "Lead (s)"]
    widths = [8, 12, 11, 9, 9, 9, 9, 9, 8, 9]
    print(format_table_row(headers, widths))
    print("-" * 90)

    for m_name, seq_dict in summary.get("results", {}).items():
        for seq_key, h_dict in seq_dict.items():
            l_val = seq_key.replace("seq_", "")
            for h_key, res in h_dict.items():
                k_val = h_key.replace("k_", "")
                m = res.get("metrics", {})
                lt = res.get("lead_time", {})
                row = [
                    m_name.upper(),
                    f"L={l_val}",
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

    print("\nExperiment completed successfully!")
    print(f"Results persisted under: {runner.base_results_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
