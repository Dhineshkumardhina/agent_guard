"""Explainability Visualizer - Phase 15.

Generates 8 publication-grade explainability visualization figures:
1. global_feature_importance.png: Global feature rankings with group color encoding
2. agent_importance_distribution.png: Multi-agent population attribution distribution
3. interaction_edge_importance.png: Directed interaction channel attribution
4. temporal_event_contributions.png: Event attribution as a function of recency
5. risk_trajectory_cases.png: Multi-case risk trajectory timelines with failure & warning markers
6. prediction_probability_timeline.png: Continuous risk timeline with annotated fault events
7. perturbation_sensitivity_analysis.png: Sensitivity delta plot across counterfactual perturbations
8. explanation_stability.png: Cross-seed and cross-horizon attribution consistency
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.explainability.schema import (
    FeatureAttribution,
    AgentAttribution,
    InteractionAttribution,
    TemporalEventAttribution,
    PerturbationResult,
    CaseStudyExplanation,
    GlobalImportanceReport,
    ExplanationStabilityReport,
)


class ExplainabilityVisualizer:
    """Generates publication figures for AgentGuard explainability and attribution."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def plot_global_feature_importance(
        self,
        features: List[FeatureAttribution],
        filename: str = "global_feature_importance.png",
        top_k: int = 15,
    ) -> Path:
        """Plot 1: Global feature importance rankings."""
        fig, ax = plt.subplots(figsize=(10, 6))

        top_feats = features[:top_k]
        names = [f.feature_name.replace("feat_", "").replace("_", " ").title() for f in reversed(top_feats)]
        scores = [f.importance_score for f in reversed(top_feats)]
        groups = [f.feature_group for f in reversed(top_feats)]

        group_colors = {
            "reliability": "#d95f02",
            "interaction": "#7570b3",
            "temporal": "#1b9e77",
            "agent_behavior": "#e7298a",
            "node": "#386cb0",
            "edge": "#f0027f",
        }
        colors = [group_colors.get(g, "#66a61e") for g in groups]

        bars = ax.barh(names, scores, color=colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("Relative Importance Score", fontsize=11, fontweight="bold")
        ax.set_title(f"Global Feature Importance Ranking (Top {len(top_feats)})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1.0)

        # Legend for groups
        handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in set(colors)]
        labels = [g.replace("_", " ").title() for g in set(groups)]
        ax.legend(handles, labels, title="Feature Category", loc="lower right", frameon=True)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_agent_importance_distribution(
        self,
        agent_reports: List[Dict[str, Any]],
        filename: str = "agent_importance_distribution.png",
    ) -> Path:
        """Plot 2: Multi-agent population attribution distribution."""
        fig, ax = plt.subplots(figsize=(8, 5))

        roles = [r["role"].capitalize() for r in agent_reports]
        importances = [r["mean_importance"] for r in agent_reports]
        counts = [r["frequency"] for r in agent_reports]

        x = np.arange(len(roles))
        width = 0.55

        bars = ax.bar(x, importances, width, color="#2b8cbe", edgecolor="black", alpha=0.85)
        ax.set_ylabel("Mean Attribution Score", fontsize=11, fontweight="bold")
        ax.set_title("Agent Role Failure-Risk Attribution Distribution", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(roles, fontsize=10)
        ax.set_ylim(0, max(importances) * 1.25 if importances else 1.0)

        for idx, bar in enumerate(bars):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}\n(n={counts[idx]})", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_interaction_edge_importance(
        self,
        interactions: List[Dict[str, Any]],
        filename: str = "interaction_edge_importance.png",
        top_k: int = 8,
    ) -> Path:
        """Plot 3: Directed interaction channel attribution."""
        fig, ax = plt.subplots(figsize=(9, 5))

        top_edges = interactions[:top_k]
        paths = [e["path"] for e in reversed(top_edges)]
        scores = [e["mean_importance"] for e in reversed(top_edges)]

        bars = ax.barh(paths, scores, color="#756bb1", edgecolor="black", alpha=0.85)
        ax.set_xlabel("Attribution Score", fontsize=11, fontweight="bold")
        ax.set_title(f"Critical Communication Channel Attribution (Top {len(top_edges)})", fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(scores) * 1.15 if scores else 1.0)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_temporal_event_contributions(
        self,
        events: List[TemporalEventAttribution],
        filename: str = "temporal_event_contributions.png",
    ) -> Path:
        """Plot 4: Temporal event recency vs attribution impact."""
        fig, ax = plt.subplots(figsize=(9, 5))

        times_before = [e.time_before_prediction for e in events]
        scores = [e.importance_score for e in events]
        types = [e.event_type for e in events]

        type_colors = {
            "error": "#e41a1c",
            "retry": "#ff7f00",
            "contradiction": "#984ea3",
            "message": "#377eb8",
        }
        colors = [type_colors.get(t, "#4daf4a") for t in types]

        scatter = ax.scatter(times_before, scores, c=colors, s=120, edgecolors="black", alpha=0.85, zorder=3)
        ax.set_xlabel("Time Elapsed Before Prediction Cutoff (Seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Event Attribution Score", fontsize=11, fontweight="bold")
        ax.set_title("Temporal Event Attribution vs Event Recency", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Invert x-axis so more recent events are on the right (closer to t_pred)
        ax.invert_xaxis()

        # Legend
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8) for col in set(colors)]
        labels = [t.capitalize() for t in set(types)]
        ax.legend(handles, labels, title="Event Type", loc="upper left", frameon=True)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_risk_trajectory_cases(
        self,
        case_trajectories: List[Dict[str, Any]],
        filename: str = "risk_trajectory_cases.png",
    ) -> Path:
        """Plot 5: Multi-case risk trajectory timelines."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
        axes = axes.flatten()

        for idx, case_data in enumerate(case_trajectories[:6]):
            ax = axes[idx]
            timeline = case_data.get("timeline", [])
            ctype = case_data.get("case_type", f"Case {idx+1}").replace("_", " ").title()
            thresh = case_data.get("threshold", 0.50)
            fail_t = case_data.get("failure_timestamp")

            if timeline:
                times = [pt["timestamp"] for pt in timeline]
                probs = [pt["predicted_probability"] for pt in timeline]
                ax.plot(times, probs, marker="o", linewidth=2.0, color="#2b8cbe", label="Predicted Risk P(F)")

            ax.axhline(thresh, color="red", linestyle="--", alpha=0.7, label=f"Threshold theta*={thresh:.2f}")
            if fail_t is not None:
                ax.axvline(fail_t, color="darkred", linestyle=":", linewidth=2, label=f"Failure t={fail_t:.1f}s")

            ax.set_title(ctype, fontsize=10, fontweight="bold")
            ax.set_xlabel("Simulation Time (s)", fontsize=9)
            ax.set_ylabel("Failure Probability", fontsize=9)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, linestyle="--", alpha=0.5)
            if idx == 0:
                ax.legend(fontsize=8, loc="upper left")

        plt.subplots_adjust(hspace=0.35, wspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.95)
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_prediction_probability_timeline(
        self,
        timeline: List[Dict[str, Any]],
        threshold: float = 0.50,
        failure_timestamp: Optional[float] = None,
        run_id: str = "",
        filename: str = "prediction_probability_timeline.png",
    ) -> Path:
        """Plot 6: Single high-resolution risk probability timeline."""
        fig, ax = plt.subplots(figsize=(10, 5))

        times = [pt["timestamp"] for pt in timeline]
        probs = [pt["predicted_probability"] for pt in timeline]

        ax.plot(times, probs, marker="o", color="#08519c", linewidth=2.5, label="Predicted Failure Probability P(t)")
        ax.fill_between(times, 0, probs, color="#bdd7e7", alpha=0.4)
        ax.axhline(threshold, color="#de2d26", linestyle="--", linewidth=1.8, label=f"Warning Threshold (theta*={threshold:.2f})")

        if failure_timestamp is not None:
            ax.axvline(failure_timestamp, color="#67000d", linestyle="-.", linewidth=2.0, label=f"Actual Failure ({failure_timestamp:.2f}s)")

        ax.set_xlabel("Chronological Time t (seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Risk P(Fail)", fontsize=11, fontweight="bold")
        ax.set_title(f"Dynamic Failure Risk Evolution Over Time ({run_id})", fontsize=12, fontweight="bold")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="upper left", frameon=True)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_perturbation_sensitivity(
        self,
        perturbations: List[PerturbationResult],
        filename: str = "perturbation_sensitivity_analysis.png",
    ) -> Path:
        """Plot 7: Controlled perturbation sensitivity deltas."""
        fig, ax = plt.subplots(figsize=(10, 5))

        labels = [p.perturbation_type.replace("_", " ").title() for p in reversed(perturbations)]
        deltas = [p.delta_probability for p in reversed(perturbations)]
        colors = ["#e41a1c" if d > 0 else "#377eb8" if d < 0 else "#999999" for d in deltas]

        bars = ax.barh(labels, deltas, color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_xlabel("Change in Failure Probability Delta P (Perturbed - Original)", fontsize=11, fontweight="bold")
        ax.set_title("Counterfactual Sensitivity Analysis (Controlled Perturbations)", fontsize=12, fontweight="bold")

        for bar in bars:
            w = bar.get_width()
            offset = 0.005 if w >= 0 else -0.025
            ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:+.3f}", va="center", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_explanation_stability(
        self,
        stability: ExplanationStabilityReport,
        filename: str = "explanation_stability.png",
    ) -> Path:
        """Plot 8: Explanation consistency across seeds and horizons."""
        fig, ax = plt.subplots(figsize=(8, 4.5))

        metrics = [
            "Feature Spearman (Seeds)",
            "Feature Jaccard (Seeds)",
            "Agent Spearman (Seeds)",
            "Agent Jaccard (Seeds)",
            "Feature Spearman (Horizons)",
            "Agent Spearman (Horizons)",
        ]
        values = [
            stability.feature_rank_correlation_seeds,
            stability.feature_top_k_jaccard_seeds,
            stability.agent_rank_correlation_seeds,
            stability.agent_top_k_jaccard_seeds,
            stability.feature_rank_correlation_horizons,
            stability.agent_rank_correlation_horizons,
        ]

        colors = ["#31a354" if v >= 0.70 else "#feb24c" if v >= 0.50 else "#f03b20" for v in values]
        bars = ax.barh(metrics, values, color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0.70, color="green", linestyle="--", alpha=0.6, label="Stability Benchmark (0.70)")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Correlation / Overlap Coefficient", fontsize=11, fontweight="bold")
        ax.set_title(f"Explanation Stability and Consistency Evaluation ({stability.model_name})", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=9)

        plt.tight_layout()
        out_path = self.output_dir / filename
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
