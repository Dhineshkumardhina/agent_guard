"""Counterfactual and Perturbation Sensitivity Analysis - Phase 15.

Implements controlled sensitivity perturbations to evaluate model response:
1. Remove one interaction
2. Reduce contradiction rate by 50%
3. Reduce retry frequency by 50%
4. Modify confidence by +0.20
5. Remove top agent
6. Remove top interaction edge
7. Mask recent events

CRITICAL RESEARCH DISCLAIMER:
These perturbations constitute sensitivity analysis, NOT proof of physical causality.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import copy
import numpy as np

from ml.baselines.temporal_gnn.models import TemporalGraphFailurePredictor
from ml.baselines.temporal_gnn.dataset import (
    TemporalRunTrajectory,
    TemporalPredictionPoint,
    TemporalInteraction,
)
from ml.explainability.schema import (
    PerturbationResult,
    AgentAttribution,
    InteractionAttribution,
    CAUSALITY_DISCLAIMER,
)
from ml.explainability.temporal_gnn_explainer import TemporalGNNExplainer


class PerturbationAnalyzer:
    """Executes controlled sensitivity perturbations on Temporal GNN failure forecasts."""

    def __init__(self, explainer: TemporalGNNExplainer) -> None:
        self.explainer = explainer

    def run_all_perturbations(
        self,
        trajectory: TemporalRunTrajectory,
        prediction_point: TemporalPredictionPoint,
        top_agent: Optional[str] = None,
        top_edge: Optional[Tuple[str, str]] = None,
    ) -> List[PerturbationResult]:
        """Execute standard 7-perturbation sensitivity battery on a single prediction point.
        
        Args:
            trajectory: Temporal run trajectory.
            prediction_point: Target prediction point at t_pred.
            top_agent: Agent ID with highest attribution (or None to infer).
            top_edge: Directed pair (u, v) with highest attribution (or None to infer).
            
        Returns:
            List of PerturbationResult records quantifying sensitivity.
        """
        orig_prob = self.explainer._predict_point(trajectory, prediction_point)
        results: List[PerturbationResult] = []

        # 1. Perturbation: Reduce Contradiction Rate by 50%
        # contradiction_rate is node feature index 9
        prob_cont = self.explainer._predict_point(
            trajectory,
            prediction_point,
            feature_scale_node=(9, 0.5),
        )
        results.append(self._make_record(
            ptype="reduce_contradiction_50pct",
            target="contradiction_rate",
            orig=orig_prob,
            pert=prob_cont,
            desc="Reduced node contradiction_rate by 50% across all active agents.",
        ))

        # 2. Perturbation: Reduce Retry Frequency by 50%
        # retry_count is node feature index 4
        prob_retry = self.explainer._predict_point(
            trajectory,
            prediction_point,
            feature_scale_node=(4, 0.5),
        )
        results.append(self._make_record(
            ptype="reduce_retry_frequency_50pct",
            target="retry_count",
            orig=orig_prob,
            pert=prob_retry,
            desc="Reduced agent retry_count by 50% across all active agents.",
        ))

        # 3. Perturbation: Modify Confidence (+0.20)
        # average_confidence is node feature index 7
        prob_conf = self.explainer._predict_point(
            trajectory,
            prediction_point,
            feature_scale_node=(7, 1.2),  # Boost confidence
        )
        results.append(self._make_record(
            ptype="modify_confidence_plus20",
            target="average_confidence",
            orig=orig_prob,
            pert=prob_conf,
            desc="Boosted agent average_confidence by +20% across all active agents.",
        ))

        # 4. Perturbation: Remove One Interaction (Most recent event)
        recent_events = [inter for inter in trajectory.interactions if inter.timestamp <= prediction_point.timestamp]
        if recent_events:
            most_recent_step = recent_events[-1].step_idx
            prob_rm_event = self.explainer._predict_point(
                trajectory,
                prediction_point,
                excluded_interactions={most_recent_step},
            )
            results.append(self._make_record(
                ptype="remove_one_interaction",
                target=f"step_{most_recent_step}",
                orig=orig_prob,
                pert=prob_rm_event,
                desc="Removed the single most recent interaction event immediately prior to prediction time.",
            ))

        # 5. Perturbation: Mask Recent Events (last 2.0 seconds)
        cutoff_t = prediction_point.timestamp
        recent_window_steps = {
            inter.step_idx for inter in recent_events if (cutoff_t - inter.timestamp) <= 2.0
        }
        if recent_window_steps:
            prob_window = self.explainer._predict_point(
                trajectory,
                prediction_point,
                excluded_interactions=recent_window_steps,
            )
            results.append(self._make_record(
                ptype="mask_recent_events",
                target="recent_2s_window",
                orig=orig_prob,
                pert=prob_window,
                desc=f"Masked {len(recent_window_steps)} interaction events occurring within 2.0s prior to prediction.",
            ))

        # 6. Perturbation: Remove Top Agent
        target_agent = top_agent or (prediction_point.active_agents[0] if prediction_point.active_agents else "none")
        if target_agent != "none":
            prob_rm_agent = self.explainer._predict_point(
                trajectory,
                prediction_point,
                masked_agent=target_agent,
            )
            results.append(self._make_record(
                ptype="remove_top_agent",
                target=target_agent,
                orig=orig_prob,
                pert=prob_rm_agent,
                desc=f"Masked telemetry and memory states for primary agent '{target_agent}'.",
            ))

        # 7. Perturbation: Remove Top Edge
        if top_edge is not None:
            src, tgt = top_edge
            edge_steps = {
                inter.step_idx for inter in recent_events if inter.source_agent == src and inter.target_agent == tgt
            }
            if edge_steps:
                prob_rm_edge = self.explainer._predict_point(
                    trajectory,
                    prediction_point,
                    excluded_interactions=edge_steps,
                )
                results.append(self._make_record(
                    ptype="remove_top_edge",
                    target=f"{src}->{tgt}",
                    orig=orig_prob,
                    pert=prob_rm_edge,
                    desc=f"Masked all communications along directed channel '{src} -> {tgt}'.",
                ))

        return results

    def _make_record(
        self,
        ptype: str,
        target: str,
        orig: float,
        pert: float,
        desc: str,
    ) -> PerturbationResult:
        delta = pert - orig
        pct = (delta / max(1e-4, orig)) * 100.0

        if delta < -0.01:
            direction = "risk_decreased"
        elif delta > 0.01:
            direction = "risk_increased"
        else:
            direction = "neutral"

        return PerturbationResult(
            perturbation_type=ptype,
            target_entity=target,
            original_probability=round(orig, 4),
            perturbed_probability=round(pert, 4),
            delta_probability=round(delta, 4),
            percent_change=round(pct, 2),
            direction=direction,
            description=desc,
        )
