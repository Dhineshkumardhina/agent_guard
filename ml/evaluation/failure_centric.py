"""Failure-Centric Evaluation Module - Phase 12 (Sections 6 & 7).

Policy:
1. For each actual failure at timestamp t_fail:
   - Identify candidate warnings emitted by the model at t_warn <= t_fail.
   - Select the earliest valid warning timestamp t_warn^* within the evaluation window.
   - lead_time = t_fail - t_warn^* (non-negative).
2. Warnings occurring after failure (t_warn > t_fail) explicitly DO NOT count as early warnings.
3. Compute detection coverage: percentage of actual failures receiving at least one valid early warning.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

from ml.evaluation.schema import FailureCentricRecord


class FailureCentricEvaluator:
    """Evaluates early warning performance from an incident/failure-centric perspective."""

    def evaluate_failures(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth_test_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform failure-centric evaluation for a model's predictions on the test set.
        
        Args:
            predictions: List of prediction records (each having run_id, timestamp, predicted_label, etc.).
            ground_truth_test_rows: Raw test tabular rows containing ground-truth failure annotations.
            
        Returns:
            Dictionary containing failure records, detection coverage, and lead-time metrics.
        """
        # 1. Identify all actual failures from test set
        # A failure in a run is identified by actual positive labels (y=1) or failure events
        failures_by_run: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in ground_truth_test_rows:
            label = int(row.get("label", 0))
            if label == 1:
                rid = row["run_id"]
                t = float(row.get("timestamp", 0.0))
                failures_by_run[rid].append({
                    "timestamp": t,
                    "failure_type": row.get("failure_type", "unknown"),
                    "failure_level": int(row.get("failure_level", 1) or 1),
                })

        # Earliest failure timestamp per run represents the primary impending failure
        primary_failures: Dict[str, Dict[str, Any]] = {}
        for rid, f_list in failures_by_run.items():
            f_list.sort(key=lambda x: x["timestamp"])
            primary_failures[rid] = f_list[0]

        # 2. Gather model warnings per run
        warnings_by_run: Dict[str, List[float]] = defaultdict(list)
        for pred in predictions:
            # Handle predicted_label or prediction key
            yp = pred.get("predicted_label", pred.get("prediction", 0))
            if int(yp) == 1:
                rid = pred.get("run_id", "")
                t = float(pred.get("timestamp", 0.0))
                warnings_by_run[rid].append(t)

        for rid in warnings_by_run:
            warnings_by_run[rid].sort()

        # 3. Match warnings to failures under strict Multiple Warning Policy
        failure_records: List[FailureCentricRecord] = []
        lead_times: List[float] = []
        detected_count = 0

        for rid, f_info in sorted(primary_failures.items()):
            t_fail = f_info["timestamp"]
            f_type = f_info["failure_type"]
            f_level = f_info["failure_level"]

            run_warns = warnings_by_run.get(rid, [])
            # Policy: candidates are warnings with t_warn <= t_fail
            valid_candidate_warns = [tw for tw in run_warns if tw <= t_fail]

            if valid_candidate_warns:
                t_first_warn = valid_candidate_warns[0]
                lead = round(t_fail - t_first_warn, 4)
                detected = True
                detected_count += 1
                lead_times.append(lead)
            else:
                t_first_warn = None
                lead = None
                detected = False

            rec = FailureCentricRecord(
                failure_id=f"failure_{rid}_{round(t_fail, 4)}",
                run_id=rid,
                failure_timestamp=round(t_fail, 4),
                first_valid_warning_timestamp=round(t_first_warn, 4) if t_first_warn is not None else None,
                lead_time=lead,
                warning_count=len(valid_candidate_warns),
                detected=detected,
                failure_type=f_type,
                failure_level=f_level,
            )
            failure_records.append(rec)

        total_failures = len(primary_failures)
        coverage_pct = round((detected_count / total_failures * 100.0), 2) if total_failures > 0 else 0.0

        return {
            "total_failures_evaluated": total_failures,
            "detected_failures": detected_count,
            "detection_coverage_percent": coverage_pct,
            "mean_lead_time": round(float(np.mean(lead_times)), 4) if lead_times else 0.0,
            "median_lead_time": round(float(np.median(lead_times)), 4) if lead_times else 0.0,
            "min_lead_time": round(float(np.min(lead_times)), 4) if lead_times else 0.0,
            "max_lead_time": round(float(np.max(lead_times)), 4) if lead_times else 0.0,
            "failure_records": [r.to_dict() for r in failure_records],
        }
