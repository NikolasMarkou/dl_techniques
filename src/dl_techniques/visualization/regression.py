"""
Regression Visualization Templates
==================================

Ready-made templates for analyzing regression models, including
residual analysis, prediction error plots, and normality checks.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .core import (
    VisualizationPlugin,
    CompositeVisualization,
    PlotConfig,
    VisualizationContext
)


# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------

@dataclass
class RegressionResults:
    """Container for regression evaluation results."""

    y_true: np.ndarray
    y_pred: np.ndarray
    model_name: Optional[str] = None
    feature_names: Optional[List[str]] = None

    def __post_init__(self):
        # Ensure arrays are flat
        self.y_true = np.array(self.y_true).flatten()
        self.y_pred = np.array(self.y_pred).flatten()


@dataclass
class MultiModelRegression:
    """Container for comparing multiple regression models."""

    results: Dict[str, RegressionResults]
    dataset_name: Optional[str] = None


# ---------------------------------------------------------------------
# Prediction Error Templates
# ---------------------------------------------------------------------

class PredictionErrorVisualization(VisualizationPlugin):
    """Visualize Predicted vs. Actual values."""

    @property
    def name(self) -> str:
        return "prediction_error"

    @property
    def description(self) -> str:
        return "Scatter plot of Predicted vs Actual values (Identity Plot)"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, RegressionResults)

    def create_visualization(
            self,
            data: RegressionResults,
            ax: Optional[plt.Axes] = None,
            show_identity: bool = True,
            **kwargs
    ) -> plt.Figure:
        """
        Create prediction error visualization.

        Args:
            data: Regression results
            ax: Optional matplotlib axes
            show_identity: Whether to show the perfect prediction line (y=x)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size)
        else:
            fig = ax.get_figure()

        # Calculate metrics for title/annotation
        r2 = r2_score(data.y_true, data.y_pred)

        # Plot data
        color = self.config.color_scheme.primary
        ax.scatter(data.y_true, data.y_pred, alpha=0.6,
                   color=color, edgecolor='white', linewidth=0.5, label='Predictions')

        # Determine limits
        min_val = min(data.y_true.min(), data.y_pred.min())
        max_val = max(data.y_true.max(), data.y_pred.max())
        padding = (max_val - min_val) * 0.05

        if show_identity:
            line_range = [min_val - padding, max_val + padding]
            ax.plot(line_range, line_range, 'k--', alpha=0.7, label='Perfect Fit')

        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        model_name = data.model_name if data.model_name else "Model"
        ax.set_title(f'Prediction Error - {model_name} ($R^2$={r2:.3f})')

        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


# ---------------------------------------------------------------------
# Residual Analysis Templates
# ---------------------------------------------------------------------

class ResidualsPlotVisualization(VisualizationPlugin):
    """Visualize residuals against predicted values."""

    @property
    def name(self) -> str:
        return "residuals_plot"

    @property
    def description(self) -> str:
        return "Plot residuals vs predicted values to check for heteroscedasticity"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, RegressionResults)

    def create_visualization(
            self,
            data: RegressionResults,
            ax: Optional[plt.Axes] = None,
            lowess: bool = False,
            **kwargs
    ) -> plt.Figure:
        """
        Create residuals plot.

        Args:
            data: Regression results
            ax: Optional matplotlib axes
            lowess: If True, uses seaborn to draw a locally weighted regression line
                    (useful for detecting non-linearity)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size)
        else:
            fig = ax.get_figure()

        residuals = data.y_true - data.y_pred
        color = self.config.color_scheme.secondary

        if lowess:
            sns.residplot(
                x=data.y_pred,
                y=residuals,
                lowess=True,
                color=color,
                scatter_kws={'alpha': 0.6, 'edgecolor': 'white', 'linewidth': 0.5},
                line_kws={'color': 'red', 'lw': 2},
                ax=ax
            )
        else:
            ax.scatter(data.y_pred, residuals, alpha=0.6,
                       color=color, edgecolor='white', linewidth=0.5)
            ax.axhline(0, color='black', linestyle='--', linewidth=1)

        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title(f'Residuals vs Predicted')
        ax.grid(True, alpha=0.3)

        return fig


class ResidualDistributionVisualization(VisualizationPlugin):
    """Visualize the distribution of residuals."""

    @property
    def name(self) -> str:
        return "residual_distribution"

    @property
    def description(self) -> str:
        return "Histogram and KDE of residuals to check normality"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, RegressionResults)

    def create_visualization(
            self,
            data: RegressionResults,
            ax: Optional[plt.Axes] = None,
            bins: int = 30,
            **kwargs
    ) -> plt.Figure:
        """Create residual distribution plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size)
        else:
            fig = ax.get_figure()

        residuals = data.y_true - data.y_pred

        # Plot Histogram
        sns.histplot(residuals, kde=True, bins=bins, ax=ax,
                     color=self.config.color_scheme.info, edgecolor='black', alpha=0.6)

        # Add mean line
        mean_res = np.mean(residuals)
        ax.axvline(mean_res, color='red', linestyle='--', label=f'Mean: {mean_res:.3f}')

        ax.set_xlabel('Residual Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig


class QQPlotVisualization(VisualizationPlugin):
    """Q-Q Plot to check for normality of residuals."""

    @property
    def name(self) -> str:
        return "qq_plot"

    @property
    def description(self) -> str:
        return "Quantile-Quantile plot of residuals"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, RegressionResults)

    def create_visualization(
            self,
            data: RegressionResults,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> plt.Figure:
        """Create Q-Q plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.fig_size)
        else:
            fig = ax.get_figure()

        residuals = data.y_true - data.y_pred

        # Calculate quantiles
        stats.probplot(residuals, dist="norm", plot=ax)

        # Customize appearance
        ax.get_lines()[0].set_markerfacecolor(self.config.color_scheme.primary)
        ax.get_lines()[0].set_markeredgecolor('white')
        ax.get_lines()[0].set_alpha(0.6)
        ax.get_lines()[1].set_color('red')  # The fit line

        ax.set_title('Q-Q Plot (Residuals)')
        ax.grid(True, alpha=0.3)

        return fig


# ---------------------------------------------------------------------
# Composite Dashboard
# ---------------------------------------------------------------------

class RegressionEvaluationDashboard(CompositeVisualization):
    """Comprehensive regression analysis dashboard."""

    @property
    def name(self) -> str:
        return "regression_dashboard"

    @property
    def description(self) -> str:
        return "4-plot dashboard for regression analysis"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, RegressionResults)

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        super().__init__(config, context)
        self.add_subplot("Prediction Error", self._plot_pred_error)
        self.add_subplot("Residuals vs Predicted", self._plot_residuals)
        self.add_subplot("Residual Distribution", self._plot_dist)
        self.add_subplot("Q-Q Plot", self._plot_qq)

    def _plot_pred_error(self, ax: plt.Axes, data: RegressionResults, **kwargs):
        """Wrapper for PredictionErrorVisualization."""
        viz = PredictionErrorVisualization(self.config, self.context)
        viz.create_visualization(data, ax=ax, **kwargs)

    def _plot_residuals(self, ax: plt.Axes, data: RegressionResults, **kwargs):
        """Wrapper for ResidualsPlotVisualization."""
        viz = ResidualsPlotVisualization(self.config, self.context)
        viz.create_visualization(data, ax=ax, lowess=True, **kwargs)

    def _plot_dist(self, ax: plt.Axes, data: RegressionResults, **kwargs):
        """Wrapper for ResidualDistributionVisualization."""
        viz = ResidualDistributionVisualization(self.config, self.context)
        viz.create_visualization(data, ax=ax, **kwargs)

    def _plot_qq(self, ax: plt.Axes, data: RegressionResults, **kwargs):
        """Wrapper for QQPlotVisualization."""
        viz = QQPlotVisualization(self.config, self.context)
        viz.create_visualization(data, ax=ax, **kwargs)

    def create_visualization(
            self,
            data: RegressionResults,
            **kwargs
    ) -> plt.Figure:
        """
        Overridden to add a metrics table to the dashboard.
        """
        # Create the standard composite layout (2x2)
        fig = super().create_visualization(data, layout=(2, 2), **kwargs)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(data.y_true, data.y_pred))
        mae = mean_absolute_error(data.y_true, data.y_pred)
        r2 = r2_score(data.y_true, data.y_pred)

        # Add a text box with metrics in the figure suptitle or bottom
        metrics_text = (f"Model: {data.model_name or 'Unknown'} | "
                        f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

        plt.suptitle(f"Regression Analysis Dashboard\n{metrics_text}",
                     fontsize=14, fontweight='bold')

        # Adjust layout to make room for the larger title
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        return fig