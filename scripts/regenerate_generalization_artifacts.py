"""Script to recompute gaps and breakdowns from existing predictions and regenerate artifacts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from ml.generalization.runner import GeneralizationExperimentRunner
from ml.generalization.schema import GeneralizationResultRecord, GeneralizationGapRecord
from ml.generalization.splits import GENERALIZATION_REGISTRY

def main():
    dataset_dir = Path("data/processed/agentguard_generalization_v1")
    output_dir = Path("results/generalization")

    runner = GeneralizationExperimentRunner(dataset_dir=dataset_dir, base_results_dir=output_dir)
    runner.load_dataset()

    # Load existing results records
    metrics_path = output_dir / "metrics" / "generalization_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        recs_data = json.load(f)
    runner.results = [GeneralizationResultRecord(**r) for r in recs_data]

    # Recompute gaps with two-sample block bootstrap using saved predictions
    runner.gaps = []
    print("[Artifacts] Recomputing generalization gaps with two-sample trajectory bootstrap...")
    pred_dirs = list((output_dir / "predictions").glob("*"))
    for pdir in pred_dirs:
        if not pdir.is_dir():
            continue
        parts = pdir.name.split("_")
        eid = parts[0]
        if eid not in GENERALIZATION_REGISTRY:
            continue
        cfg = GENERALIZATION_REGISTRY[eid]

        if "temporal_gnn" in pdir.name:
            model = "temporal_gnn"
        elif "logistic_regression" in pdir.name:
            model = "logistic_regression"
        elif "random_forest" in pdir.name:
            model = "random_forest"
        elif "xgboost" in pdir.name:
            model = "xgboost"
        else:
            continue

        h = 1
        s = 42
        for p in parts:
            if p.startswith("k") and p[1:].isdigit():
                h = int(p[1:])
            elif p.startswith("s") and p[1:].isdigit():
                s = int(p[1:])

        id_f = pdir / "test_id_predictions.json"
        ood_f = pdir / "test_ood_predictions.json"
        if not id_f.exists() or not ood_f.exists():
            continue

        with open(id_f, "r", encoding="utf-8") as f:
            id_preds = json.load(f)
        with open(ood_f, "r", encoding="utf-8") as f:
            ood_preds = json.load(f)

        id_rec = next((r for r in runner.results if r.experiment_id == eid and r.model == model and r.horizon == h and r.seed == s and r.split_type == "in_distribution"), None)
        ood_rec = next((r for r in runner.results if r.experiment_id == eid and r.model == model and r.horizon == h and r.seed == s and r.split_type == "out_of_distribution"), None)

        if id_rec and ood_rec:
            gap_f1 = round(id_rec.f1 - ood_rec.f1, 4)
            pct_change = round(((ood_rec.f1 - id_rec.f1) / max(1e-4, id_rec.f1)) * 100.0, 2)
            ci_lower, ci_upper, p_val = runner._bootstrap_generalization_gap(
                id_preds=id_preds,
                ood_preds=ood_preds,
                metric_name="f1",
                n_bootstraps=300,
                random_seed=s,
            )
            is_sig = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)
            interp = "Robust transfer" if abs(gap_f1) <= 0.05 else ("Moderate degradation" if gap_f1 > 0 else "Negative transfer")

            runner.gaps.append(
                GeneralizationGapRecord(
                    experiment_id=eid,
                    dimension=cfg.dimension,
                    model=model,
                    horizon=h,
                    metric="f1",
                    id_value=id_rec.f1,
                    ood_value=ood_rec.f1,
                    gap=gap_f1,
                    pct_change=pct_change,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    p_value=p_val,
                    statistically_significant=is_sig,
                    interpretation=interp,
                    seed=s,
                )
            )

    print(f"[Artifacts] Recomputed {len(runner.gaps)} gap records.")

    # Compute fine-grained subgroup breakdowns
    print("[Artifacts] Computing subgroup breakdowns across agent count, topology, task, and failure mode...")
    runner._compute_subgroup_breakdowns()
    print(f"[Artifacts] Computed {len(runner.breakdowns)} subgroup breakdown records.")

    # Save artifacts
    print("[Artifacts] Saving updated metrics, tables, plots, and report...")
    runner.save_artifacts()
    print("[Artifacts] Regeneration complete!")

if __name__ == "__main__":
    main()
