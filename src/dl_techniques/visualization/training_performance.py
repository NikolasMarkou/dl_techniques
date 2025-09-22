"""
Training and Performance Visualization Templates
=================================================

Ready-made templates for visualizing training dynamics, model performance,
and experiment comparisons.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .core import VisualizationPlugin, CompositeVisualization, PlotConfig, VisualizationContext


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TrainingHistory:
    """Container for training history data."""

    epochs: List[int]
    train_loss: List[float]
    val_loss: Optional[List[float]] = None
    train_metrics: Dict[str, List[float]] = None
    val_metrics: Dict[str, List[float]] = None
    grad_norms: Optional[Dict[str, List[float]]] = None

    def __post_init__(self):
        if self.train_metrics is None:
            self.train_metrics = {}
        if self.val_metrics is None:
            self.val_metrics = {}
        if self.grad_norms is None:
            self.grad_norms = {}


@dataclass
class ModelComparison:
    """Container for comparing multiple models."""

    model_names: List[str]
    metrics: Dict[str, Dict[str, float]]  # model_name -> metric_name -> value
    histories: Optional[Dict[str, TrainingHistory]] = None
    predictions: Optional[Dict[str, np.ndarray]] = None


# =============================================================================
# Training Visualization Templates
# =============================================================================

class TrainingCurvesVisualization(VisualizationPlugin):
    """Visualization for training curves with multiple metrics."""

    @property
    def name(self) -> str:
        return "training_curves"

    @property
    def description(self) -> str:
        return "Visualize training and validation curves for loss and metrics"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (TrainingHistory, dict))

    def create_visualization(
            self,
            data: Union[TrainingHistory, Dict[str, TrainingHistory]],
            metrics_to_plot: Optional[List[str]] = None,
            smooth_factor: float = 0.0,
            show_best_epoch: bool = True,
            **kwargs
    ) -> plt.Figure:
        """
        Create training curves visualization.

        Args:
            data: Training history data
            metrics_to_plot: Specific metrics to plot (None = all)
            smooth_factor: Smoothing factor for curves (0-1)
            show_best_epoch: Whether to mark best validation epoch
        """
        # Handle single or multiple histories
        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        # Determine metrics to plot
        if metrics_to_plot is None:
            all_metrics = set()
            for hist in histories.values():
                all_metrics.update(hist.train_metrics.keys())
                if hist.val_metrics:
                    all_metrics.update(hist.val_metrics.keys())
            metrics_to_plot = ["loss"] + sorted(list(all_metrics))

        # Create subplots
        n_metrics = len(metrics_to_plot)
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each metric
        for idx, metric in enumerate(metrics_to_plot[:n_metrics]):
            ax = axes[idx]

            for model_name, hist in histories.items():
                color = self.config.color_scheme.get_model_color(model_name, list(histories.keys()).index(model_name))

                # Plot training data
                if metric == "loss":
                    train_data = hist.train_loss
                    val_data = hist.val_loss
                else:
                    train_data = hist.train_metrics.get(metric, [])
                    val_data = hist.val_metrics.get(metric, []) if hist.val_metrics else []

                if train_data:
                    train_smooth = self._smooth_curve(train_data, smooth_factor)
                    ax.plot(hist.epochs[:len(train_smooth)], train_smooth,
                            label=f"{model_name} (train)", color=color, linewidth=2)

                if val_data:
                    val_smooth = self._smooth_curve(val_data, smooth_factor)
                    ax.plot(hist.epochs[:len(val_smooth)], val_smooth,
                            label=f"{model_name} (val)", color=color,
                            linewidth=2, linestyle='--')

                    # Mark best epoch
                    if show_best_epoch and metric == "loss":
                        best_idx = np.argmin(val_data)
                        ax.scatter(hist.epochs[best_idx], val_data[best_idx],
                                   color=color, s=100, zorder=5, marker='*',
                                   edgecolors='black', linewidth=1)

            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Log scale for loss
            if metric == "loss":
                ax.set_yscale('log')

        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Training History', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def _smooth_curve(self, values: List[float], factor: float) -> np.ndarray:
        """Apply exponential smoothing to curve."""
        if factor <= 0:
            return np.array(values)

        smoothed = []
        for i, val in enumerate(values):
            if i == 0:
                smoothed.append(val)
            else:
                smoothed.append(factor * smoothed[-1] + (1 - factor) * val)
        return np.array(smoothed)


class LearningRateScheduleVisualization(VisualizationPlugin):
    """Visualization for learning rate schedules."""

    @property
    def name(self) -> str:
        return "lr_schedule"

    @property
    def description(self) -> str:
        return "Visualize learning rate schedule over training"

    def can_handle(self, data: Any) -> bool:
        """
        Check if this plugin can handle the given data.

        Accepts:
        - A list of numeric values (learning rates)
        - A dict where values are lists of numeric values
        """
        # Check if data is a list of numbers
        if isinstance(data, list):
            return all(isinstance(x, (int, float, np.number)) for x in data)

        # Check if data is a dict with list values
        if isinstance(data, dict):
            for value in data.values():
                if not isinstance(value, list):
                    return False
                if not all(isinstance(x, (int, float, np.number)) for x in value):
                    return False
            return True

        return False

    def create_visualization(
            self,
            data: Union[List[float], Dict[str, List[float]]],
            show_phases: bool = True,
            phase_boundaries: Optional[List[int]] = None,
            **kwargs
    ) -> plt.Figure:
        """Create learning rate schedule visualization."""

        fig, ax = plt.subplots(figsize=self.config.fig_size)

        # Handle single or multiple schedules
        if isinstance(data, list):
            schedules = {"Learning Rate": data}
        else:
            schedules = data

        for idx, (name, lr_values) in enumerate(schedules.items()):
            color = self.config.color_scheme.get_model_color(name, idx)
            epochs = range(len(lr_values))
            ax.plot(epochs, lr_values, label=name, color=color, linewidth=2)

        # Add phase boundaries if provided
        if show_phases and phase_boundaries:
            for boundary in phase_boundaries:
                ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


# =============================================================================
# Model Comparison Templates
# =============================================================================

class ModelComparisonBarChart(VisualizationPlugin):
    """Bar chart comparison of model metrics."""

    @property
    def name(self) -> str:
        return "model_comparison_bars"

    @property
    def description(self) -> str:
        return "Compare models using grouped bar charts"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ModelComparison, dict))

    def create_visualization(
            self,
            data: Union[ModelComparison, Dict[str, Dict[str, float]]],
            metrics_to_show: Optional[List[str]] = None,
            sort_by: Optional[str] = None,
            show_values: bool = True,
            **kwargs
    ) -> plt.Figure:
        """Create model comparison bar chart."""

        # Convert to ModelComparison if needed
        if isinstance(data, dict):
            model_names = list(data.keys())
            metrics_dict = data
        else:
            model_names = data.model_names
            metrics_dict = data.metrics

        # Determine metrics to show
        if metrics_to_show is None:
            all_metrics = set()
            for model_metrics in metrics_dict.values():
                all_metrics.update(model_metrics.keys())
            metrics_to_show = sorted(list(all_metrics))

        # Sort models if requested
        if sort_by and sort_by in metrics_to_show:
            model_names = sorted(model_names,
                                 key=lambda m: metrics_dict[m].get(sort_by, 0),
                                 reverse=True)

        # Prepare data for plotting
        n_models = len(model_names)
        n_metrics = len(metrics_to_show)
        x = np.arange(n_models)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=self.config.fig_size)

        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_show):
            values = [metrics_dict[model].get(metric, 0) for model in model_names]
            offset = (i - n_metrics / 2) * width + width / 2
            bars = ax.bar(x + offset, values, width, label=metric,
                          alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add value labels on bars
            if show_values:
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{val:.3f}', ha='center', va='bottom',
                            fontsize=8)

        ax.set_xlabel('Models')
        ax.set_ylabel('Metric Value')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig


class PerformanceRadarChart(VisualizationPlugin):
    """Radar chart for multi-metric model comparison."""

    @property
    def name(self) -> str:
        return "performance_radar"

    @property
    def description(self) -> str:
        return "Compare models using radar charts"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (ModelComparison, dict))

    def create_visualization(
            self,
            data: Union[ModelComparison, Dict[str, Dict[str, float]]],
            metrics_to_show: Optional[List[str]] = None,
            normalize: bool = True,
            **kwargs
    ) -> plt.Figure:
        """Create radar chart comparison."""

        # Convert to ModelComparison if needed
        if isinstance(data, dict):
            model_names = list(data.keys())
            metrics_dict = data
        else:
            model_names = data.model_names
            metrics_dict = data.metrics

        # Determine metrics
        if metrics_to_show is None:
            all_metrics = set()
            for model_metrics in metrics_dict.values():
                all_metrics.update(model_metrics.keys())
            metrics_to_show = sorted(list(all_metrics))

        # Prepare data
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_show), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for idx, model in enumerate(model_names):
            values = [metrics_dict[model].get(metric, 0) for metric in metrics_to_show]

            # Normalize if requested
            if normalize:
                max_vals = [max(metrics_dict[m].get(metric, 0) for m in model_names)
                            for metric in metrics_to_show]
                values = [v / m if m > 0 else 0 for v, m in zip(values, max_vals)]

            values += values[:1]  # Complete the circle

            color = self.config.color_scheme.get_model_color(model, idx)
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_show)
        ax.set_ylim(0, 1 if normalize else None)
        ax.set_title('Model Performance Radar Chart', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        return fig


# =============================================================================
# Convergence Analysis Templates
# =============================================================================

class ConvergenceAnalysis(CompositeVisualization):
    """Comprehensive convergence analysis visualization."""

    @property
    def name(self) -> str:
        return "convergence_analysis"

    @property
    def description(self) -> str:
        return "Analyze training convergence patterns"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (TrainingHistory, dict))

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        super().__init__(config, context)
        self.add_subplot("Loss Convergence", self._plot_loss_convergence)
        self.add_subplot("Gradient Flow", self._plot_gradient_flow)
        self.add_subplot("Validation Gap", self._plot_validation_gap)
        self.add_subplot("Convergence Rate", self._plot_convergence_rate)

    def _plot_loss_convergence(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot loss convergence patterns."""
        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        for idx, (name, hist) in enumerate(histories.items()):
            color = self.config.color_scheme.get_model_color(name, idx)

            # Plot with log scale
            ax.semilogy(hist.epochs[:len(hist.train_loss)], hist.train_loss,
                        label=f"{name} (train)", color=color, alpha=0.7)
            if hist.val_loss:
                ax.semilogy(hist.epochs[:len(hist.val_loss)], hist.val_loss,
                            label=f"{name} (val)", color=color, linestyle='--')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_gradient_flow(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot gradient flow magnitude over time."""
        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        plotted_anything = False
        # The key we expect in the grad_norms dictionary.
        # This could be made a parameter in the future for more flexibility.
        grad_metric_name = "global_grad_norm"

        for idx, (name, hist) in enumerate(histories.items()):
            # Check if gradient data is available for this history
            if hist.grad_norms and grad_metric_name in hist.grad_norms:
                grad_data = hist.grad_norms[grad_metric_name]
                if grad_data:  # Ensure the list is not empty
                    color = self.config.color_scheme.get_model_color(name, idx)
                    ax.plot(hist.epochs[:len(grad_data)], grad_data,
                            label=name, color=color, linewidth=2)
                    plotted_anything = True

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Global Gradient Norm (L2)')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="--", alpha=0.3)

        if plotted_anything:
            ax.legend(fontsize=8)
        else:
            # Provide an informative message if no data was found
            ax.text(0.5, 0.5,
                    f'Gradient Flow\n(Requires "{grad_metric_name}" in TrainingHistory.grad_norms)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color='gray')

    def _plot_validation_gap(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot gap between training and validation loss."""
        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        for idx, (name, hist) in enumerate(histories.items()):
            if hist.val_loss:
                color = self.config.color_scheme.get_model_color(name, idx)
                gap = np.array(hist.val_loss) - np.array(hist.train_loss[:len(hist.val_loss)])
                ax.plot(hist.epochs[:len(gap)], gap, label=name, color=color, linewidth=2)
                ax.fill_between(hist.epochs[:len(gap)], 0, gap,
                                where=(gap > 0), alpha=0.3, color=color,
                                label='Overfitting' if idx == 0 else '')

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation - Training Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_convergence_rate(self, ax: plt.Axes, data: Any, **kwargs):
        """Plot convergence rate (loss reduction per epoch)."""
        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        for idx, (name, hist) in enumerate(histories.items()):
            color = self.config.color_scheme.get_model_color(name, idx)

            # Calculate rate of change
            loss_diff = np.diff(hist.train_loss)
            rate = -loss_diff  # Negative because loss should decrease

            ax.plot(hist.epochs[1:len(rate) + 1], rate,
                    label=name, color=color, alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Reduction Rate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)


# =============================================================================
# Overfitting Detection Template
# =============================================================================

class OverfittingAnalysis(VisualizationPlugin):
    """Visualization for detecting and analyzing overfitting."""

    @property
    def name(self) -> str:
        return "overfitting_analysis"

    @property
    def description(self) -> str:
        return "Detect and visualize overfitting patterns"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (TrainingHistory, dict))

    def create_visualization(
            self,
            data: Union[TrainingHistory, Dict[str, TrainingHistory]],
            patience: int = 10,
            **kwargs
    ) -> plt.Figure:
        """Create overfitting analysis visualization."""

        if isinstance(data, TrainingHistory):
            histories = {"Model": data}
        else:
            histories = data

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Loss curves with overfitting regions
        ax = axes[0, 0]
        for idx, (name, hist) in enumerate(histories.items()):
            color = self.config.color_scheme.get_model_color(name, idx)

            ax.plot(hist.epochs[:len(hist.train_loss)], hist.train_loss,
                    label=f"{name} (train)", color=color)
            if hist.val_loss:
                ax.plot(hist.epochs[:len(hist.val_loss)], hist.val_loss,
                        label=f"{name} (val)", color=color, linestyle='--')

                # Detect overfitting point
                overfit_epoch = self._detect_overfitting(hist.val_loss, patience)
                if overfit_epoch:
                    ax.axvline(x=overfit_epoch, color=color, alpha=0.5, linestyle=':')
                    ax.text(overfit_epoch, ax.get_ylim()[1] * 0.9,
                            f'Overfit @ {overfit_epoch}',
                            rotation=90, fontsize=8, color=color)

        ax.set_title('Loss Curves with Overfitting Detection')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Generalization gap over time
        ax = axes[0, 1]
        for idx, (name, hist) in enumerate(histories.items()):
            if hist.val_loss:
                color = self.config.color_scheme.get_model_color(name, idx)
                gap = np.array(hist.val_loss) - np.array(hist.train_loss[:len(hist.val_loss)])
                ax.plot(hist.epochs[:len(gap)], gap, label=name, color=color, linewidth=2)

        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Generalization Gap')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val Loss - Train Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. Early stopping points
        ax = axes[1, 0]
        for idx, (name, hist) in enumerate(histories.items()):
            if hist.val_loss:
                color = self.config.color_scheme.get_model_color(name, idx)
                best_epoch = np.argmin(hist.val_loss)
                ax.scatter(best_epoch, hist.val_loss[best_epoch],
                           color=color, s=100, label=f"{name} (best)")
                ax.plot(hist.epochs[:len(hist.val_loss)], hist.val_loss,
                        color=color, alpha=0.3)

        ax.set_title('Optimal Stopping Points')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Overfitting metrics summary
        ax = axes[1, 1]
        metrics_data = []
        for name, hist in histories.items():
            if hist.val_loss:
                overfit_epoch = self._detect_overfitting(hist.val_loss, patience)
                best_epoch = np.argmin(hist.val_loss)
                final_gap = hist.val_loss[-1] - hist.train_loss[-1]
                metrics_data.append([
                    name[:15],
                    best_epoch,
                    overfit_epoch if overfit_epoch else 'No',
                    f"{final_gap:.3f}"
                ])

        if metrics_data:
            table = ax.table(cellText=metrics_data,
                             colLabels=['Model', 'Best Epoch', 'Overfit Epoch', 'Final Gap'],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

        ax.axis('off')
        ax.set_title('Overfitting Metrics Summary')

        plt.suptitle('Overfitting Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig

    def _detect_overfitting(self, val_loss: List[float], patience: int) -> Optional[int]:
        """Detect epoch where overfitting begins."""
        best_loss = float('inf')
        patience_counter = 0

        for epoch, loss in enumerate(val_loss):
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                return epoch - patience

        return None


# =============================================================================
# Performance Summary Dashboard
# =============================================================================

class PerformanceDashboard(CompositeVisualization):
    """Comprehensive performance dashboard."""

    @property
    def name(self) -> str:
        return "performance_dashboard"

    @property
    def description(self) -> str:
        return "Complete performance analysis dashboard"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ModelComparison)

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        super().__init__(config, context)

    def create_visualization(
            self,
            data: ModelComparison,
            metric_to_display: Optional[str] = None,  # FIXED: Allow user to specify metric
            **kwargs
    ) -> plt.Figure:
        """
        Create comprehensive performance dashboard.

        Args:
            data: ModelComparison data
            metric_to_display: Specific metric to display in metric comparison (None = all metrics)
        """

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. Training curves (if available)
        if data.histories:
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_training_curves(ax1, data.histories)

        # 2. Metric comparison bars - FIXED
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_metric_bars(ax2, data, metric_to_display)

        # 3. Performance heatmap
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_performance_heatmap(ax3, data)

        # 4. Ranking table
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_ranking_table(ax4, data)

        # 5. Statistical comparison
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_statistical_comparison(ax5, data)

        plt.suptitle('Performance Dashboard', fontsize=18, fontweight='bold')
        return fig

    def _plot_training_curves(self, ax: plt.Axes, histories: Dict[str, TrainingHistory]):
        """Plot training curves."""
        for idx, (name, hist) in enumerate(histories.items()):
            color = self.config.color_scheme.get_model_color(name, idx)
            ax.plot(hist.epochs[:len(hist.val_loss)], hist.val_loss,
                    label=name, color=color, linewidth=2)

        ax.set_title('Validation Loss Comparison')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_metric_bars(self, ax: plt.Axes, data: ModelComparison,
                         metric_to_display: Optional[str] = None):
        """
        Plot metrics as bars.

        FIXED: Now allows specification of which metric to display,
        or displays all metrics as grouped bars if not specified.
        """
        # Get all available metrics
        all_metrics = set()
        for model_metrics in data.metrics.values():
            all_metrics.update(model_metrics.keys())

        if not all_metrics:
            ax.text(0.5, 0.5, 'No metrics available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            return

        # Determine which metrics to display
        if metric_to_display:
            # User specified a metric
            if metric_to_display not in all_metrics:
                # Metric not found - show available metrics
                ax.text(0.5, 0.5,
                       f'Metric "{metric_to_display}" not found.\nAvailable: {", ".join(sorted(all_metrics))}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                return
            metrics_to_show = [metric_to_display]
        else:
            # No metric specified - show all metrics
            metrics_to_show = sorted(list(all_metrics))

        n_models = len(data.model_names)
        n_metrics = len(metrics_to_show)

        if n_metrics == 1:
            # Single metric - simple bar chart
            metric = metrics_to_show[0]
            values = [data.metrics[model].get(metric, 0) for model in data.model_names]
            colors = [self.config.color_scheme.get_model_color(m, i)
                     for i, m in enumerate(data.model_names)]

            bars = ax.bar(range(n_models), values, color=colors, alpha=0.7)
            ax.set_xticks(range(n_models))
            ax.set_xticklabels(data.model_names, rotation=45, ha='right')
            ax.set_title(f'{metric.title()} Comparison')
            ax.set_ylabel(metric.title())

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            # Multiple metrics - grouped bar chart
            x = np.arange(n_models)
            width = 0.8 / n_metrics

            for i, metric in enumerate(metrics_to_show):
                values = [data.metrics[model].get(metric, 0) for model in data.model_names]
                offset = (i - n_metrics / 2) * width + width / 2
                bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)

                # Add value labels for smaller sets
                if n_models <= 3 and n_metrics <= 3:
                    for bar, val in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                               f'{val:.2f}', ha='center', va='bottom', fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(data.model_names, rotation=45, ha='right')
            ax.set_title('Metrics Comparison')
            ax.set_ylabel('Value')
            ax.legend(fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

    def _plot_performance_heatmap(self, ax: plt.Axes, data: ModelComparison):
        """Plot performance metrics heatmap."""
        # Create matrix
        metrics = sorted(set(m for model_metrics in data.metrics.values()
                             for m in model_metrics.keys()))
        matrix = np.zeros((len(data.model_names), len(metrics)))

        for i, model in enumerate(data.model_names):
            for j, metric in enumerate(metrics):
                matrix[i, j] = data.metrics[model].get(metric, 0)

        # Normalize columns
        matrix_norm = matrix / (matrix.max(axis=0) + 1e-10)

        im = ax.imshow(matrix_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.set_yticks(range(len(data.model_names)))
        ax.set_yticklabels(data.model_names)
        ax.set_title('Performance Heatmap (Normalized)')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_ranking_table(self, ax: plt.Axes, data: ModelComparison):
        """Plot model ranking table."""
        # Calculate average rank
        rankings = {}
        metrics = list(set(m for model_metrics in data.metrics.values()
                           for m in model_metrics.keys()))

        for metric in metrics:
            values = [(model, data.metrics[model].get(metric, 0))
                      for model in data.model_names]
            sorted_models = sorted(values, key=lambda x: x[1], reverse=True)

            for rank, (model, _) in enumerate(sorted_models, 1):
                if model not in rankings:
                    rankings[model] = []
                rankings[model].append(rank)

        # Create table data
        table_data = []
        for model in data.model_names:
            avg_rank = np.mean(rankings.get(model, []))
            table_data.append([model[:15], f"{avg_rank:.1f}"])

        table_data.sort(key=lambda x: float(x[1]))

        table = ax.table(cellText=table_data,
                         colLabels=['Model', 'Avg Rank'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)

        ax.axis('off')
        ax.set_title('Model Rankings')

    def _plot_statistical_comparison(self, ax: plt.Axes, data: ModelComparison):
        """Plot statistical comparison."""
        # Box plot of all metrics
        all_values = []
        labels = []

        for model in data.model_names:
            values = list(data.metrics[model].values())
            all_values.append(values)
            labels.append(model[:15])

        bp = ax.boxplot(all_values, labels=labels, patch_artist=True)

        # Color boxes
        for patch, model in zip(bp['boxes'], data.model_names):
            color = self.config.color_scheme.get_model_color(
                model, data.model_names.index(model))
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title('Statistical Distribution of All Metrics')
        ax.set_ylabel('Metric Values')
        ax.set_xlabel('Models')
        ax.grid(True, alpha=0.3, axis='y')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')