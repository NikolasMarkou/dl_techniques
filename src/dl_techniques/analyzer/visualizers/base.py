"""
Base Visualizer Interface

Abstract base class for all visualizers with centralized legend management.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..data_types import AnalysisResults
from ..config import AnalysisConfig
from ..utils import lighten_color
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

# Legend styling constants
LEGEND_FONT_SIZE = 9
LEGEND_TITLE_FONT_SIZE = 11
LEGEND_MARKER_SIZE = 8
LEGEND_LINE_WIDTH = 2
LEGEND_ALPHA = 0.9
LEGEND_FRAME_ALPHA = 0.95
LEGEND_BORDER_WIDTH = 1

# Legend positioning constants - Reduced distance from plots
LEGEND_BBOX_X = 1.005  # Reduced from 1.02 to bring legend closer
LEGEND_BBOX_Y = 1.0
LEGEND_ANCHOR = 'upper left'

# ---------------------------------------------------------------------

class BaseVisualizer(ABC):
    """Abstract base class for all visualizers with centralized legend management."""

    def __init__(self, results: AnalysisResults, config: AnalysisConfig,
                 output_dir: Path, model_colors: Dict[str, str]):
        """
        Initialize the visualizer.

        Args:
            results: Analysis results to visualize
            config: Analysis configuration
            output_dir: Output directory for saving plots
            model_colors: Consistent color mapping for models
        """
        self.results = results
        self.config = config
        self.output_dir = output_dir
        self.model_colors = model_colors

        # Create consistent model ordering for all visualizers
        self.model_order = self._get_consistent_model_order()

    def _get_consistent_model_order(self) -> List[str]:
        """
        Get a consistent ordering of models across all visualizations.

        Returns:
            List of model names in consistent order (alphabetically sorted)
        """
        # Get all available models from different analysis results
        all_models = set()

        # Collect models from various analysis results
        if hasattr(self.results, 'model_metrics') and self.results.model_metrics:
            all_models.update(self.results.model_metrics.keys())
        if hasattr(self.results, 'weight_stats') and self.results.weight_stats:
            all_models.update(self.results.weight_stats.keys())
        if hasattr(self.results, 'calibration_metrics') and self.results.calibration_metrics:
            all_models.update(self.results.calibration_metrics.keys())
        if hasattr(self.results, 'confidence_metrics') and self.results.confidence_metrics:
            all_models.update(self.results.confidence_metrics.keys())
        if hasattr(self.results, 'information_flow') and self.results.information_flow:
            all_models.update(self.results.information_flow.keys())
        if hasattr(self.results, 'training_history') and self.results.training_history:
            all_models.update(self.results.training_history.keys())

        # Return alphabetically sorted list for consistency
        return sorted(list(all_models))

    def _create_figure_legend(self, fig: plt.Figure,
                            title: str = "Models",
                            include_all_models: bool = True,
                            specific_models: Optional[List[str]] = None) -> None:
        """
        Create a single legend for the entire figure with consistent model representation.

        Args:
            fig: The matplotlib figure to add the legend to
            title: Title for the legend
            include_all_models: Whether to include all available models or only those with data
            specific_models: Specific list of models to include in legend (overrides other options)
        """
        # Determine which models to include in legend
        if specific_models:
            models_to_include = [m for m in specific_models if m in self.model_order]
        elif include_all_models:
            models_to_include = self.model_order
        else:
            # Only include models that have data in current visualization
            models_to_include = self._get_models_with_data()

        if not models_to_include:
            logger.debug("No models available for legend creation")
            return

        # Create legend elements
        legend_elements = []
        for model_name in models_to_include:
            color = self.model_colors.get(model_name, '#333333')

            # Create a patch for the legend (circle marker with line color)
            legend_element = mpatches.Patch(
                facecolor=color,
                edgecolor='black',
                linewidth=LEGEND_BORDER_WIDTH,
                alpha=LEGEND_ALPHA,
                label=model_name
            )
            legend_elements.append(legend_element)

        # Add the legend to the figure
        legend = fig.legend(
            handles=legend_elements,
            title=title,
            bbox_to_anchor=(LEGEND_BBOX_X, LEGEND_BBOX_Y),
            loc=LEGEND_ANCHOR,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_TITLE_FONT_SIZE,
            framealpha=LEGEND_FRAME_ALPHA,
            edgecolor='black',
            fancybox=True,
            shadow=True
        )

        # Style the legend title
        legend.get_title().set_fontweight('bold')

        logger.debug(f"Created figure legend with {len(legend_elements)} models")

    def _get_models_with_data(self) -> List[str]:
        """
        Get list of models that have data in the current analysis results.

        This is a fallback method that subclasses can override to provide
        more specific logic for determining which models have relevant data.

        Returns:
            List of model names that have data for this visualization
        """
        # Default implementation: return all models in order
        return self.model_order

    def _get_model_color(self, model_name: str) -> str:
        """
        Get consistent color for a model.

        Args:
            model_name: Name of the model

        Returns:
            Hex color string for the model
        """
        return self.model_colors.get(model_name, '#333333')

    def _sort_models_consistently(self, models: List[str]) -> List[str]:
        """
        Sort a list of models according to the consistent ordering.

        Args:
            models: List of model names to sort

        Returns:
            Sorted list of model names
        """
        # Create ordering lookup
        order_lookup = {model: i for i, model in enumerate(self.model_order)}

        # Sort according to the consistent order
        return sorted(models, key=lambda x: order_lookup.get(x, float('inf')))

    @abstractmethod
    def create_visualizations(self) -> None:
        """Create all visualizations for this analyzer."""
        pass

    def _save_figure(self, fig: plt.Figure, name: str) -> None:
        """Save figure with configured settings."""
        try:
            filepath = self.output_dir / f"{name}.{self.config.save_format}"
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none', pad_inches=0.1)
            logger.info(f"Saved plot: {filepath}")
        except Exception as e:
            logger.error(f"Could not save figure {name}: {e}")

    def _lighten_color(self, color: str, factor: float) -> tuple:
        """Lighten a color by interpolating towards white."""
        return lighten_color(color, factor)