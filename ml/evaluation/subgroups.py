"""Subgroup Analysis Module - Phase 12 (Sections 17-21).

Breaks down predictive performance across:
1. Failure Levels (Level 1 agent, Level 2 interaction, Level 3 cascading/system)
2. Failure Types (hallucination, tool failure, timeout, delayed response, contradiction, etc.)
3. Topologies (Pipeline, Star, Mesh, Custom)
4. Tasks (Research, Coding, Analysis, Planning)
5. Agent Counts (3, 5, 8, 12 agents)
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np

from ml.evaluation.metrics import compute_comprehensive_metrics
from ml.evaluation.schema import SubgroupMetricRecord


class SubgroupEvaluator:
    """Evaluates model performance across stratified sub-populations."""

    def evaluate_subgroups(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_metadata: Dict[str, Dict[str, Any]],
        model_name: str,
        horizon: int,
    ) -> List[SubgroupMetricRecord]:
        """Compute subgroup metrics for a given model and prediction horizon.
        
        Args:
            predictions: List of test predictions for this horizon.
            ground_truth_metadata: Mapping sample_id -> ground truth dictionary with topology, task_type, etc.
            model_name: Name of evaluated model.
            horizon: Evaluated horizon k.
            
        Returns:
            List of SubgroupMetricRecord instances.
        """
        # Group prediction records by category and value
        subgroup_maps: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
            "failure_level": defaultdict(list),
            "failure_type": defaultdict(list),
            "topology": defaultdict(list),
            "task": defaultdict(list),
            "agent_count": defaultdict(list),
        }

        for pred in predictions:
            sid = pred.get("sample_id", "")
            meta = ground_truth_metadata.get(sid, {})

            f_level = str(meta.get("failure_level", "unknown"))
            f_type = str(meta.get("failure_type", "unknown"))
            topo = str(meta.get("topology", "unknown"))
            task = str(meta.get("task_type", "unknown"))
            n_agents = str(meta.get("number_of_agents", "unknown"))

            subgroup_maps["failure_level"][f_level].append(pred)
            subgroup_maps["failure_type"][f_type].append(pred)
            subgroup_maps["topology"][topo].append(pred)
            subgroup_maps["task"][task].append(pred)
            subgroup_maps["agent_count"][n_agents].append(pred)

        records: List[SubgroupMetricRecord] = []

        for category, val_dict in subgroup_maps.items():
            for val, preds_list in sorted(val_dict.items()):
                if not preds_list:
                    continue

                y_true = [int(p.get("true_label", p.get("ground_truth", 0))) for p in preds_list]
                y_pred = [int(p.get("predicted_label", p.get("prediction", 0))) for p in preds_list]
                y_prob = [float(p.get("predicted_probability", p.get("prediction", 0.0))) for p in preds_list]

                m = compute_comprehensive_metrics(y_true, y_pred, y_prob)

                rec = SubgroupMetricRecord(
                    subgroup_category=category,
                    subgroup_value=val,
                    model_name=model_name,
                    horizon=horizon,
                    sample_count=len(preds_list),
                    positive_count=int(sum(y_true)),
                    negative_count=int(len(y_true) - sum(y_true)),
                    precision=m.get("precision", 0.0),
                    recall=m.get("recall", 0.0),
                    f1=m.get("f1", 0.0),
                    auroc=m.get("auroc", 0.5),
                    auprc=m.get("auprc", 0.0),
                )
                records.append(rec)

        return records
