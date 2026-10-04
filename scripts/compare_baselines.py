"""Baseline Comparison CLI Script for AgentGuard.

Aggregates and formats comparative measurements across:
Model x Horizon x Metric
Including:
- Rule-Based Early Warning Detector
- Logistic Regression
- Random Forest
- XGBoost

Reports:
- Precision, Recall, F1, AUROC, AUPRC, FPR, and Lead Time
- Generates machine-readable JSON/Markdown outputs
- Adheres to scientific objectivity: does NOT label any model as 'best'.
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Rule-Based and Classical ML Baselines across prediction horizons."
    )
    parser.add_argument(
        "--rule-results",
        type=str,
        default="results/baselines/rule_based",
        help="Path to rule-based baseline results directory.",
    )
    parser.add_argument(
        "--ml-results",
        type=str,
        default="results/baselines/classical_ml",
        help="Path to classical ML baseline results directory.",
    )
    parser.add_argument(
        "--seq-results",
        type=str,
        default="results/baselines/sequence",
        help="Path to temporal sequence (LSTM/GRU) baseline results directory.",
    )
    parser.add_argument(
        "--gnn-results",
        type=str,
        default="results/baselines/static_gnn",
        help="Path to static GNN (GCN/GAT) baseline results directory.",
    )
    parser.add_argument(
        "--temporal-gnn-results",
        type=str,
        default="results/baselines/temporal_gnn",
        help="Path to temporal GNN baseline results directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save machine-readable comparison JSON.",
    )
    return parser.parse_args()


def format_table_row(cols, widths):
    return " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, widths))


def main():
    args = parse_args()
    rule_dir = Path(args.rule_results)
    ml_dir = Path(args.ml_results)
    seq_dir = Path(args.seq_results)

    print("=" * 95)
    print("AgentGuard: Baseline Performance Comparative Report")
    print("=" * 95)

    comparison_records = []

    # 1. Load latest rule-based baseline results if available
    if rule_dir.exists():
        rule_subdirs = sorted([d for d in rule_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
        if rule_subdirs:
            latest_rule_dir = rule_subdirs[0]
            metrics_file = latest_rule_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    rule_data = json.load(f)
                for k_str, h_data in rule_data.get("horizons", {}).items():
                    k_val = int(k_str)
                    comparison_records.append({
                        "model": "Rule-Based Heuristic",
                        "horizon": k_val,
                        "precision": h_data.get("precision", 0.0),
                        "recall": h_data.get("recall", 0.0),
                        "f1": h_data.get("f1", 0.0),
                        "auroc": h_data.get("auroc", 0.0),
                        "auprc": h_data.get("auprc", 0.0),
                        "fpr": h_data.get("false_positive_rate", 0.0),
                        "lead_time": rule_data.get("lead_time", {}).get("mean_lead_time", 0.0),
                    })

    # 2. Load latest classical ML results if available
    if ml_dir.exists():
        ml_summaries = sorted(list(ml_dir.glob("summary_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if ml_summaries:
            latest_summary_file = ml_summaries[0]
            with open(latest_summary_file, "r", encoding="utf-8") as f:
                ml_summary = json.load(f)
            
            model_map_names = {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
            }
            res_by_model = ml_summary.get("results_by_model_and_horizon", {})
            for m_key, h_dict in res_by_model.items():
                m_display = model_map_names.get(m_key, m_key.title())
                for k_str, m_metrics in h_dict.items():
                    comparison_records.append({
                        "model": m_display,
                        "horizon": int(k_str),
                        "precision": m_metrics.get("precision", 0.0),
                        "recall": m_metrics.get("recall", 0.0),
                        "f1": m_metrics.get("f1", 0.0),
                        "auroc": m_metrics.get("auroc", 0.0),
                        "auprc": m_metrics.get("auprc", 0.0),
                        "fpr": m_metrics.get("false_positive_rate", 0.0),
                        "lead_time": m_metrics.get("mean_lead_time", 0.0),
                    })

    # 3. Load latest sequence ML results (LSTM / GRU) if available
    if seq_dir.exists():
        seq_summaries = sorted(list(seq_dir.glob("summary_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if seq_summaries:
            latest_seq_file = seq_summaries[0]
            with open(latest_seq_file, "r", encoding="utf-8") as f:
                seq_summary = json.load(f)

            for m_name, seq_dict in seq_summary.get("results", {}).items():
                m_display = f"{m_name.upper()}"
                for seq_key, h_dict in seq_dict.items():
                    l_val = seq_key.replace("seq_", "")
                    for h_key, res in h_dict.items():
                        k_val = int(h_key.replace("k_", ""))
                        m = res.get("metrics", {})
                        lt = res.get("lead_time", {})
                        comparison_records.append({
                            "model": f"{m_display} (L={l_val})",
                            "horizon": k_val,
                            "precision": m.get("precision", 0.0),
                            "recall": m.get("recall", 0.0),
                            "f1": m.get("f1", 0.0),
                            "auroc": m.get("auroc", 0.0),
                            "auprc": m.get("auprc", 0.0),
                            "fpr": m.get("false_positive_rate", 0.0),
                            "lead_time": lt.get("mean_lead_time", 0.0),
                        })

    # 4. Load latest static GNN results (GCN / GAT) if available
    gnn_dir = Path(args.gnn_results)
    if gnn_dir.exists():
        gnn_summaries = sorted(list(gnn_dir.glob("summary_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if gnn_summaries:
            latest_gnn_file = gnn_summaries[0]
            with open(latest_gnn_file, "r", encoding="utf-8") as f:
                gnn_summary = json.load(f)

            for m_name, h_dict in gnn_summary.get("results", {}).items():
                m_display = f"Static {m_name.upper()}"
                for h_key, res in h_dict.items():
                    k_val = int(h_key.replace("k_", ""))
                    m = res.get("metrics", {})
                    lt = res.get("lead_time", {})
                    comparison_records.append({
                        "model": m_display,
                        "horizon": k_val,
                        "precision": m.get("precision", 0.0),
                        "recall": m.get("recall", 0.0),
                        "f1": m.get("f1", 0.0),
                        "auroc": m.get("auroc", 0.0),
                        "auprc": m.get("auprc", 0.0),
                        "fpr": m.get("false_positive_rate", 0.0),
                        "lead_time": lt.get("mean_lead_time", 0.0),
                    })

    # 5. Load latest temporal GNN results if available
    tgn_dir = Path(args.temporal_gnn_results)
    if tgn_dir.exists():
        tgn_summaries = sorted(list(tgn_dir.glob("summary_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if tgn_summaries:
            latest_tgn_file = tgn_summaries[0]
            with open(latest_tgn_file, "r", encoding="utf-8") as f:
                tgn_summary = json.load(f)

            for h_key, res in tgn_summary.get("results", {}).items():
                k_val = int(h_key.replace("k_", ""))
                m = res.get("metrics", {})
                lt = res.get("lead_time", {})
                comparison_records.append({
                    "model": "Temporal GNN (TGN)",
                    "horizon": k_val,
                    "precision": m.get("precision", 0.0),
                    "recall": m.get("recall", 0.0),
                    "f1": m.get("f1", 0.0),
                    "auroc": m.get("auroc", 0.0),
                    "auprc": m.get("auprc", 0.0),
                    "fpr": m.get("false_positive_rate", 0.0),
                    "lead_time": lt.get("mean_lead_time", 0.0),
                })

    if not comparison_records:
        print("No evaluation results found. Run scripts/evaluate_rule_baseline.py and scripts/train_classical_baselines.py first.")
        sys.exit(0)

    # Sort records: Horizon ascending, then Model name
    comparison_records.sort(key=lambda r: (r["horizon"], r["model"]))

    headers = ["Model", "Horizon (k)", "Precision", "Recall", "F1 Score", "AUROC", "AUPRC", "FPR", "Lead Time (s)"]
    widths = [24, 11, 9, 9, 9, 9, 9, 8, 13]
    print(format_table_row(headers, widths))
    print("-" * 95)

    current_k = None
    for r in comparison_records:
        if current_k is not None and r["horizon"] != current_k:
            print("-" * 95)
        current_k = r["horizon"]
        row = [
            r["model"],
            f"k={r['horizon']}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            f"{r['auroc']:.4f}",
            f"{r['auprc']:.4f}",
            f"{r['fpr']:.4f}",
            f"{r['lead_time']:.4f}",
        ]
        print(format_table_row(row, widths))

    print("=" * 95)
    print("Scientific Notice: Measurements reflect empirical metrics on held-out test splits.")
    print("Models are not ranked by a subjective 'best model' judgment.")
    print("=" * 95)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(comparison_records, f, indent=2)
        print(f"Saved machine-readable comparison table to: {out_path}")


if __name__ == "__main__":
    main()
