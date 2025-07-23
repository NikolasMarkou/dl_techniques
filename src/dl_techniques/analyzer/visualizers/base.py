"""
Base Visualizer Interface

Abstract base class for all visualizers to ensure consistent interface.
"""

from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..data_types import AnalysisResults
from ..config import AnalysisConfig
from ..utils import lighten_color
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

class BaseVisualizer(ABC):
    """Abstract base class for all visualizers."""

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