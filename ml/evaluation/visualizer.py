"""Evaluation Visualizer - Phase 12 (Section 28).

Generates publication-quality figures:
1. ROC curves (all models on comparable test set)
2. Precision-Recall curves
3. F1 Score vs Prediction Horizon (K in {1, 3, 5, 10})
4. AUPRC vs Prediction Horizon
5. Lead-time distributions (box/violin/histogram)
6. Calibration Curves / Reliability Diagrams
7. Subgroup Performance (Failure Levels, Topologies, Tasks)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class EvaluationVisualizer:
    """Generates publication-quality evaluation charts and figures."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Consistent style settings
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "grid.alpha": 0.3,
            "axes.grid": True,
        })
        self.color_palette = {
            "Rule-Based": "#7f7f7f",
            "Logistic Regression": "#aec7e8",
            "Random Forest": "#1f77b4",
            "XGBoost": "#2ca02c",
            "LSTM": "#ff7f0e",
            "GRU": "#d62728",
            "GCN": "#9467bd",
            "GAT": "#8c564b",
            "Temporal GNN": "#e377c2",
        }

    def _get_color(self, model_name: str) -> str:
        for k, col in self.color_palette.items():
            if k.lower() in model_name.lower():
                return col
        return "#17becf"

    def plot_roc_curves(
        self,
        curves_by_model: Dict[str, Dict[str, Any]],
        horizon: int = 1,
        filename: str = "roc_curves.png",
    ) -> str:
        """Plot comparative ROC curves for all models at a given horizon."""
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, curve_data in sorted(curves_by_model.items()):
            roc_data = curve_data.get("roc", {})
            fpr = roc_data.get("fpr", [])
            tpr = roc_data.get("tpr", [])
            auc = roc_data.get("auc", 0.5)

            if fpr and tpr:
                color = self._get_color(model_name)
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{model_name} (AUC = {auc:.3f})",
                    color=color,
                    lw=2,
                    alpha=0.85,
                )

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Chance (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (Recall / TPR)")
        ax.set_title(f"Receiver Operating Characteristic (ROC) — Horizon K={horizon}")
        ax.legend(loc="lower right", framealpha=0.9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_pr_curves(
        self,
        curves_by_model: Dict[str, Dict[str, Any]],
        horizon: int = 1,
        filename: str = "pr_curves.png",
    ) -> str:
        """Plot comparative Precision-Recall curves for all models at a given horizon."""
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, curve_data in sorted(curves_by_model.items()):
            pr_data = curve_data.get("pr", {})
            precision = pr_data.get("precision", [])
            recall = pr_data.get("recall", [])
            auc = pr_data.get("auc", 0.0)

            if precision and recall:
                color = self._get_color(model_name)
                ax.plot(
                    recall,
                    precision,
                    label=f"{model_name} (AUPRC = {auc:.3f})",
                    color=color,
                    lw=2,
                    alpha=0.85,
                )

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Recall (True Positive Rate)")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves — Horizon K={horizon}")
        ax.legend(loc="lower left", framealpha=0.9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_metric_vs_horizon(
        self,
        records: List[Dict[str, Any]],
        metric: str = "f1",
        ylabel: str = "F1 Score",
        filename: str = "f1_vs_horizon.png",
    ) -> str:
        """Plot metric evolution as prediction horizon K increases."""
        fig, ax = plt.subplots(figsize=(8, 5))

        models = sorted(list(set(r["model_name"] for r in records)))
        for model in models:
            sub = [r for r in records if r["model_name"] == model]
            sub.sort(key=lambda x: x["prediction_horizon"])
            horizons = [r["prediction_horizon"] for r in sub]
            vals = [r.get(metric, 0.0) for r in sub]

            if horizons:
                color = self._get_color(model)
                ax.plot(
                    horizons,
                    vals,
                    marker="o",
                    lw=2,
                    label=model,
                    color=color,
                    alpha=0.85,
                )

        ax.set_xlabel("Prediction Horizon K (Steps Ahead)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs Prediction Horizon K")
        ax.set_xticks([1, 3, 5, 10, 20])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(loc="best", framealpha=0.9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_calibration_curves(
        self,
        calibration_by_model: Dict[str, List[Dict[str, Any]]],
        filename: str = "calibration_curves.png",
    ) -> str:
        """Plot calibration curves / reliability diagrams."""
        fig, ax = plt.subplots(figsize=(7, 6))

        for model_name, bins in sorted(calibration_by_model.items()):
            if not bins:
                continue
            pred_means = [b["predicted_prob_mean"] for b in bins if b["sample_count"] > 0]
            obs_means = [b["observed_positive_ratio"] for b in bins if b["sample_count"] > 0]

            if pred_means:
                color = self._get_color(model_name)
                ax.plot(
                    pred_means,
                    obs_means,
                    marker="s",
                    lw=2,
                    label=model_name,
                    color=color,
                    alpha=0.85,
                )

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.5, label="Perfect Calibration")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of True Failures")
        ax.set_title("Calibration Reliability Diagram")
        ax.legend(loc="upper left", framealpha=0.9)
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_lead_time_distributions(
        self,
        lead_times_by_model: Dict[str, List[float]],
        filename: str = "lead_time_distributions.png",
    ) -> str:
        """Plot lead time boxplots across models."""
        fig, ax = plt.subplots(figsize=(9, 5))

        models = [m for m, lts in lead_times_by_model.items() if len(lts) > 0]
        data = [lead_times_by_model[m] for m in models]

        if data:
            bp = ax.boxplot(
                data,
                tick_labels=models,
                patch_artist=True,
                showmeans=True,
            )
            for patch, model in zip(bp["boxes"], models):
                patch.set_facecolor(self._get_color(model))
                patch.set_alpha(0.6)

        ax.set_ylabel("Early Warning Lead Time (Seconds)")
        ax.set_title("Distribution of Early Warning Lead Times by Model")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    def plot_subgroup_bar_chart(
        self,
        subgroup_records: List[Dict[str, Any]],
        category: str = "failure_level",
        metric: str = "f1",
        title: str = "Performance Across Failure Levels",
        filename: str = "subgroup_failure_level.png",
    ) -> str:
        """Plot grouped bar chart of performance across subgroups."""
        filtered = [r for r in subgroup_records if r.get("subgroup_category") == category]
        if not filtered:
            return ""

        groups = sorted(list(set(str(r["subgroup_value"]) for r in filtered)))
        models = sorted(list(set(r["model_name"] for r in filtered)))

        fig, ax = plt.subplots(figsize=(10, 5))
        n_models = len(models)
        width = 0.8 / max(1, n_models)
        x = np.arange(len(groups))

        for i, model in enumerate(models):
            vals = []
            for g in groups:
                m_matches = [
                    r.get(metric, 0.0)
                    for r in filtered
                    if r["model_name"] == model and str(r["subgroup_value"]) == g
                ]
                vals.append(m_matches[0] if m_matches else 0.0)

            offset = (i - n_models / 2.0 + 0.5) * width
            ax.bar(
                x + offset,
                vals,
                width,
                label=model,
                color=self._get_color(model),
                alpha=0.85,
            )

        ax.set_xlabel(category.replace("_", " ").capitalize())
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim([0.0, 1.05])
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path
