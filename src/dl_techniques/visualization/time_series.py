"""
Time Series Forecasting Visualization Templates
==============================================

Ready-made templates for visualizing time series forecasting results,
including point forecasts, prediction intervals, and model performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .core import VisualizationPlugin

# ---------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------


@dataclass
class TimeSeriesEvaluationResults:
    """
    Container for time series forecasting evaluation results.

    This structure holds all necessary data to visualize and compare
    model forecasts against ground truth.
    """
    # The complete test datasets
    all_inputs: np.ndarray  # Shape: (num_samples, input_length)
    all_true_forecasts: np.ndarray  # Shape: (num_samples, forecast_length)

    # Model predictions (can be optional)
    all_predicted_forecasts: Optional[np.ndarray] = None  # Point forecasts (e.g., median)
    all_predicted_quantiles: Optional[np.ndarray] = None  # Quantile forecasts

    # Metadata
    model_name: Optional[str] = "Model"
    quantile_levels: Optional[List[float]] = None


# ---------------------------------------------------------------------
# Time Series Visualization Templates
# ---------------------------------------------------------------------


class ForecastVisualization(VisualizationPlugin):
    """
    Comprehensive visualization for time series forecasts.

    This plugin can visualize point forecasts, probabilistic forecasts with
    quantile-based uncertainty bands, and compare them against the ground truth.
    """

    @property
    def name(self) -> str:
        return "forecast_visualization"

    @property
    def description(self) -> str:
        return "Visualize time series forecasts with point and quantile predictions"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, TimeSeriesEvaluationResults)

    def create_visualization(
            self,
            data: TimeSeriesEvaluationResults,
            ax: Optional[plt.Axes] = None,
            num_samples: int = 6,
            plot_type: str = 'auto',  # 'auto', 'point', 'quantile'
            **kwargs
    ) -> plt.Figure:
        """
        Create the forecast visualization.

        Args:
            data: TimeSeriesEvaluationResults data container.
            ax: Optional axes. Note: This plugin creates multiple subplots
                and will generate its own figure, ignoring this argument.
            num_samples: Number of random samples to plot.
            plot_type: Type of plot ('auto', 'point', 'quantile').
                       'auto' will plot quantiles if available, else point.
        """
        if data.all_inputs.shape[0] == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available for visualization.", ha='center')
            return fig

        # Determine plot type automatically
        if plot_type == 'auto':
            plot_type_to_use = 'quantile' if data.all_predicted_quantiles is not None else 'point'
        else:
            plot_type_to_use = plot_type

        # Create figure layout
        cols = min(3, num_samples)
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        axes = axes.flatten()

        # Select random samples
        total_samples = data.all_inputs.shape[0]
        sample_indices = np.random.choice(total_samples, size=min(num_samples, total_samples), replace=False)

        for i, sample_idx in enumerate(sample_indices):
            ax_sub = axes[i]
            input_seq = data.all_inputs[sample_idx]
            true_forecast = data.all_true_forecasts[sample_idx]
            input_len = len(input_seq)
            forecast_len = len(true_forecast)

            # Time axes
            input_x = np.arange(-input_len, 0)
            forecast_x = np.arange(0, forecast_len)

            # Plot input and true forecast
            ax_sub.plot(input_x, input_seq, color='blue', label='Input')
            ax_sub.plot(forecast_x, true_forecast, color='black', linewidth=2, label='True Forecast')
            ax_sub.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

            # Plot predictions
            if plot_type_to_use == 'quantile':
                self._plot_quantiles(ax_sub, data, sample_idx, forecast_x)
            elif plot_type_to_use == 'point' and data.all_predicted_forecasts is not None:
                predicted_forecast = data.all_predicted_forecasts[sample_idx]
                ax_sub.plot(forecast_x, predicted_forecast, color='red', linewidth=2, label='Point Forecast')

            ax_sub.set_title(f'Sample {sample_idx}')
            ax_sub.legend()
            ax_sub.grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(sample_indices), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f'Forecast Visualization - {data.model_name}', fontsize=16)
        plt.tight_layout()
        return fig

    def _plot_quantiles(self, ax: plt.Axes, data: TimeSeriesEvaluationResults, sample_idx: int, forecast_x: np.ndarray):
        """Helper function to plot quantile bands."""
        if data.all_predicted_quantiles is None or data.quantile_levels is None:
            return

        quantiles = np.array(data.quantile_levels)
        predictions = data.all_predicted_quantiles[sample_idx]  # Shape: [num_quantiles, forecast_len]
        n_quantiles = len(quantiles)

        # Find median and symmetric pairs for shading
        median_idx = np.argmin(np.abs(quantiles - 0.5))
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
        num_bands = median_idx

        for i in range(num_bands):
            lower_quantile = predictions[i, :]
            upper_quantile = predictions[-(i + 1), :]
            alpha = 0.4 - i * 0.08  # Decrease alpha for outer bands
            ax.fill_between(
                forecast_x, lower_quantile, upper_quantile,
                alpha=alpha, color=colors[i % len(colors)],
                label=f'{quantiles[i]:.1f}-{quantiles[-(i + 1)]:.1f} Quantile'
            )

        # Plot median
        if median_idx < n_quantiles:
            ax.plot(forecast_x, predictions[median_idx, :],
                      label='Median Forecast', color='red', linewidth=2)