"""Script to inspect all baseline results from Phases 7-11."""

import os
import json
import glob
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.storage import DatasetStorage

def main():
    storage = DatasetStorage("results")
    inventory = []

    # 1. Rule-based
    for p in glob.glob("results/baselines/rule_based/*/metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        parent = Path(p).parent
        pred_path = parent / "predictions.parquet"
        preds = storage.load_tabular(pred_path) if pred_path.exists() else []
        inventory.append({
            "family": "Rule-Based",
            "model": "rule_based",
            "dir": str(parent),
            "horizons": list(d.get("horizons", {}).keys()),
            "pred_count": len(preds),
            "dataset_version": d.get("configuration", {}).get("dataset_version", "v1"),
            "seed": d.get("configuration", {}).get("random_seed", 42),
        })

    # 2. Classical ML
    for p in glob.glob("results/baselines/classical_ml/*/*/metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        parent = Path(p).parent
        model_name = parent.parent.name
        pred_path = parent / "predictions.parquet"
        preds = storage.load_tabular(pred_path) if pred_path.exists() else []
        inventory.append({
            "family": "Classical ML",
            "model": model_name,
            "dir": str(parent),
            "horizons": list(d.get("horizons", {}).keys()),
            "pred_count": len(preds),
            "dataset_version": d.get("metadata", {}).get("dataset_version", "agentguard_dataset_v1"),
            "seed": d.get("metadata", {}).get("random_seed", 42),
        })

    # 3. Sequence ML
    for p in glob.glob("results/baselines/sequence/*/*/*/metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        parent = Path(p).parent
        model_name = parent.parent.parent.name
        pred_path = parent / "predictions.parquet"
        preds = storage.load_tabular(pred_path) if pred_path.exists() else []
        inventory.append({
            "family": "Temporal Sequence",
            "model": model_name,
            "dir": str(parent),
            "horizon": d.get("prediction_horizon"),
            "seq_len": d.get("metadata", {}).get("sequence_length"),
            "pred_count": len(preds),
            "dataset_version": d.get("metadata", {}).get("dataset_version", "agentguard_dataset_v1"),
            "seed": d.get("metadata", {}).get("random_seed", 42),
        })

    # 4. Static GNN
    for p in glob.glob("results/baselines/static_gnn/*/*/metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        parent = Path(p).parent
        model_name = parent.parent.name
        pred_path = parent / "predictions.json"
        preds = []
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as pf:
                preds = json.load(pf)
        inventory.append({
            "family": "Static GNN",
            "model": model_name,
            "dir": str(parent),
            "horizon": d.get("prediction_horizon"),
            "pred_count": len(preds),
            "dataset_version": d.get("metadata", {}).get("dataset_version", "agentguard_dataset_v1"),
            "seed": d.get("metadata", {}).get("random_seed", 42),
        })

    # 5. Temporal GNN
    for p in glob.glob("results/baselines/temporal_gnn/*/metrics.json"):
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        parent = Path(p).parent
        pred_path = parent / "predictions.json"
        preds = []
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as pf:
                preds = json.load(pf)
        inventory.append({
            "family": "Temporal GNN",
            "model": "temporal_gnn",
            "dir": str(parent),
            "horizon": d.get("prediction_horizon"),
            "pred_count": len(preds),
            "dataset_version": d.get("metadata", {}).get("dataset_version", "agentguard_dataset_v1"),
            "seed": d.get("metadata", {}).get("random_seed", 42),
        })

    print(f"Total experiment runs found: {len(inventory)}")
    for item in inventory:
        h = item.get("horizon") if "horizon" in item else item.get("horizons")
        s = item.get("seq_len", "-")
        print(f"[{item['family']}] {item['model']} | Horizon: {h} | Seq: {s} | Preds: {item['pred_count']} | Seed: {item['seed']}")

if __name__ == "__main__":
    main()
