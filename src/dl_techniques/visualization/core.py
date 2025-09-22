"""
Visualization Framework Core Module
====================================

A modular, extensible visualization framework for machine learning experiments.
Provides a plugin-based architecture for creating various types of visualizations
with ready-made templates and full customization support.
"""

from __future__ import annotations

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Type
from enum import Enum
import json
import logging
import contextlib

# Configure matplotlib and seaborn defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Setup logging
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

class PlotStyle(Enum):
    """Available plot styles."""
    MINIMAL = "minimal"
    SCIENTIFIC = "scientific"
    DARK = "dark"
    PRESENTATION = "presentation"
    PUBLICATION = "publication"


@dataclass
class ColorScheme:
    """Color scheme configuration for visualizations."""

    primary: str = "#1f77b4"
    secondary: str = "#ff7f0e"
    success: str = "#2ca02c"
    warning: str = "#d62728"
    info: str = "#9467bd"
    background: str = "#ffffff"
    grid: str = "#e0e0e0"
    text: str = "#333333"

    # Model/experiment specific colors
    model_colors: Dict[str, str] = field(default_factory=dict)

    # Categorical palette for multiple items
    palette: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])

    def get_model_color(self, model_name: str, index: int = 0) -> str:
        """Get color for a specific model."""
        if model_name in self.model_colors:
            return self.model_colors[model_name]
        return self.palette[index % len(self.palette)]


@dataclass
class PlotConfig:
    """Configuration for plot appearance and behavior."""

    # Figure settings
    fig_size: Tuple[float, float] = (12, 8)
    dpi: int = 100
    save_dpi: int = 300

    # Style settings
    style: PlotStyle = PlotStyle.SCIENTIFIC
    color_scheme: ColorScheme = field(default_factory=ColorScheme)

    # Font settings
    font_family: str = "sans-serif"
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    annotation_fontsize: int = 9

    # Layout settings
    tight_layout: bool = True
    constrained_layout: bool = False
    subplot_spacing: float = 0.3

    # Grid settings
    show_grid: bool = True
    grid_alpha: float = 0.3
    grid_style: str = "--"

    # Legend settings
    legend_location: str = "best"
    legend_frameon: bool = True
    legend_shadow: bool = True
    legend_fancybox: bool = True

    # Save settings
    save_format: str = "png"
    transparent_background: bool = False
    bbox_inches: str = "tight"

    def get_style_params(self) -> Dict[str, Any]:
        """Get matplotlib rcParams for the configured style."""
        style_map = {
            PlotStyle.MINIMAL: {
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': False,
                'xtick.top': False,
                'ytick.right': False,
            },
            PlotStyle.SCIENTIFIC: {
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.axisbelow': True,
            },
            PlotStyle.DARK: {
                'figure.facecolor': '#2b2b2b',
                'axes.facecolor': '#2b2b2b',
                'axes.edgecolor': 'white',
                'axes.labelcolor': 'white',
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'grid.color': '#555555',
            },
            PlotStyle.PRESENTATION: {
                'font.size': 14,
                'axes.titlesize': 20,
                'axes.labelsize': 16,
                'xtick.labelsize': 14,
                'ytick.labelsize': 14,
                'legend.fontsize': 14,
                'lines.linewidth': 2.5,
            },
            PlotStyle.PUBLICATION: {
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'font.size': 10,
                'axes.linewidth': 0.5,
                'lines.linewidth': 1,
                'patch.linewidth': 0.5,
                'grid.linewidth': 0.5,
            }
        }

        params = {}
        if self.style in style_map:
            params.update(style_map[self.style])

        # Apply font settings
        params.update({
            'font.family': self.font_family,
            'axes.titlesize': self.title_fontsize,
            'axes.labelsize': self.label_fontsize,
            'xtick.labelsize': self.tick_fontsize,
            'ytick.labelsize': self.tick_fontsize,
            'legend.fontsize': self.legend_fontsize,
        })

        return params

    @contextlib.contextmanager
    def style_context(self):
        """Context manager for temporarily applying style settings."""
        # Store current rcParams
        old_params = {key: plt.rcParams[key] for key in self.get_style_params().keys()
                      if key in plt.rcParams}

        try:
            # Apply new params
            plt.rcParams.update(self.get_style_params())
            yield
        finally:
            # Restore old params
            plt.rcParams.update(old_params)


@dataclass
class VisualizationContext:
    """Context information for visualization generation."""

    experiment_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir: Path = field(default_factory=lambda: Path("visualizations"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_save_path(self, filename: str, subdir: Optional[str] = None) -> Path:
        """Get full save path for a file."""
        if subdir:
            save_dir = self.output_dir / self.experiment_name / self.timestamp / subdir
        else:
            save_dir = self.output_dir / self.experiment_name / self.timestamp

        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / filename


# =============================================================================
# Base Classes
# =============================================================================

class VisualizationPlugin(ABC):
    """Abstract base class for visualization plugins."""

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        """
        Initialize the visualization plugin.

        Args:
            config: Plot configuration
            context: Visualization context
        """
        self.config = config
        self.context = context

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this visualization plugin."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of this visualization plugin."""
        pass

    @abstractmethod
    def can_handle(self, data: Any) -> bool:
        """
        Check if this plugin can handle the given data.

        Args:
            data: Data to visualize

        Returns:
            True if this plugin can handle the data
        """
        pass

    @abstractmethod
    def create_visualization(
            self,
            data: Any,
            ax: Optional[plt.Axes] = None,
            **kwargs
    ) -> plt.Figure:
        """
        Create the visualization.

        Args:
            data: Data to visualize
            ax: Optional matplotlib axes to plot on.
            **kwargs: Additional arguments

        Returns:
            The created figure
        """
        pass

    def save_figure(self, fig: plt.Figure, name: str, subdir: Optional[str] = None) -> Path:
        """
        Save a figure with proper configuration.

        Args:
            fig: Figure to save
            name: Base name for the file
            subdir: Optional subdirectory

        Returns:
            Path where figure was saved
        """
        filename = f"{name}.{self.config.save_format}"
        save_path = self.context.get_save_path(filename, subdir)

        if self.config.tight_layout:
            fig.tight_layout()

        fig.savefig(
            save_path,
            dpi=self.config.save_dpi,
            format=self.config.save_format,
            transparent=self.config.transparent_background,
            bbox_inches=self.config.bbox_inches,
            pad_inches=0.1
        )

        logger.info(f"Saved figure to {save_path}")
        return save_path


class CompositeVisualization(VisualizationPlugin):
    """Base class for composite visualizations that combine multiple plots."""

    def __init__(self, config: PlotConfig, context: VisualizationContext):
        super().__init__(config, context)
        self.subplots: List[Tuple[str, Callable]] = []

    def add_subplot(self, name: str, plot_func: Callable) -> None:
        """Add a subplot to this composite visualization."""
        self.subplots.append((name, plot_func))

    def create_visualization(
            self,
            data: Any,
            ax: Optional[plt.Axes] = None,
            layout: Optional[Tuple[int, int]] = None,
            default_cols: int = 3,
            **kwargs
    ) -> plt.Figure:
        """Create a composite visualization with multiple subplots."""
        if ax is not None:
            logger.warning(
                f"{self.__class__.__name__} is a composite visualization and cannot be drawn "
                f"on a single provided axis. A new figure will be created."
            )

        if not self.subplots:
            raise ValueError("No subplots added to composite visualization")

        n_plots = len(self.subplots)
        if layout is None:
            # Auto-determine layout
            cols = min(default_cols, n_plots)
            rows = (n_plots + cols - 1) // cols
            layout = (rows, cols)

        fig, axes = plt.subplots(*layout, figsize=self.config.fig_size)
        if n_plots == 1:
            axes = [axes]
        elif layout[0] == 1 or layout[1] == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (name, plot_func) in enumerate(self.subplots[:n_plots]):
            ax_sub = axes[idx]
            plot_func(ax_sub, data, **kwargs)
            ax_sub.set_title(name, fontsize=self.config.title_fontsize)

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        return fig


# =============================================================================
# Visualization Manager
# =============================================================================

class VisualizationManager:
    """
    Central manager for all visualizations.

    This class provides a plugin-based architecture for managing different
    types of visualizations. It automatically discovers and loads visualization
    plugins, handles data routing, and manages the visualization pipeline.
    """

    def __init__(
            self,
            experiment_name: str,
            output_dir: Union[str, Path] = "visualizations",
            config: Optional[PlotConfig] = None,
            auto_discover: bool = True
    ):
        """
        Initialize the visualization manager.

        Args:
            experiment_name: Name of the experiment
            output_dir: Base directory for saving visualizations
            config: Plot configuration (uses defaults if None)
            auto_discover: Whether to auto-discover plugins
        """
        self.config = config or PlotConfig()
        self.context = VisualizationContext(
            experiment_name=experiment_name,
            output_dir=Path(output_dir)
        )

        self.plugins: Dict[str, VisualizationPlugin] = {}
        self.templates: Dict[str, Type[VisualizationPlugin]] = {}

        if auto_discover:
            self._discover_plugins()

    def register_plugin(self, plugin: VisualizationPlugin) -> None:
        """Register a visualization plugin."""
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name}")

    def register_template(self, name: str, template_class: Type[VisualizationPlugin]) -> None:
        """Register a visualization template class."""
        self.templates[name] = template_class
        logger.info(f"Registered template: {name}")

    def create_plugin_from_template(self, template_name: str) -> VisualizationPlugin:
        """Create a plugin instance from a template."""
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")

        template_class = self.templates[template_name]
        return template_class(self.config, self.context)

    @contextlib.contextmanager
    def style_context(self):
        """Context manager for applying this manager's style settings."""
        with self.config.style_context():
            yield

    def visualize(
            self,
            data: Any,
            plugin_name: Optional[str] = None,
            save: bool = True,
            show: bool = False,
            **kwargs
    ) -> Optional[plt.Figure]:
        """
        Create a visualization for the given data.

        Args:
            data: Data to visualize
            plugin_name: Specific plugin to use (auto-detect if None)
            save: Whether to save the figure
            show: Whether to show the figure
            **kwargs: Additional arguments for the plugin

        Returns:
            The created figure, or None if no suitable plugin found
        """
        # Find suitable plugin
        if plugin_name:
            if plugin_name not in self.plugins:
                # Try to create from template
                try:
                    plugin = self.create_plugin_from_template(plugin_name)
                    self.register_plugin(plugin)
                except ValueError:
                    logger.error(f"Plugin or template '{plugin_name}' not found")
                    return None
            plugin = self.plugins[plugin_name]
        else:
            # Auto-detect suitable plugin
            plugin = self._find_suitable_plugin(data)
            if not plugin:
                logger.warning("No suitable plugin found for data")
                return None

        # Create visualization with style context
        try:
            with self.style_context():
                fig = plugin.create_visualization(data, **kwargs)

                if save:
                    plugin.save_figure(fig, plugin.name)

                if show:
                    plt.show()
                else:
                    plt.close(fig)

                return fig

        except Exception as e:
            logger.error(f"Error creating visualization with {plugin.name}: {e}")
            return None

    def create_dashboard(
            self,
            data: Dict[str, Any],
            layout: Optional[Dict[str, Tuple[int, int]]] = None,
            save: bool = True,
            show: bool = False
    ) -> plt.Figure:
        """
        Create a dashboard with multiple visualizations.

        Args:
            data: Dictionary mapping plugin names to their data
            layout: Optional layout specification
            save: Whether to save the dashboard
            show: Whether to show the dashboard

        Returns:
            The created dashboard figure
        """
        if not layout:
            n_plots = len(data)
            cols = min(3, n_plots)
            rows = (n_plots + cols - 1) // cols
        else:
            rows = max(pos[0] for pos in layout.values()) + 1
            cols = max(pos[1] for pos in layout.values()) + 1

        with self.style_context():
            fig = plt.figure(figsize=(self.config.fig_size[0] * cols, self.config.fig_size[1] * rows))
            gs = fig.add_gridspec(rows, cols, hspace=0.3, wspace=0.3)

            for idx, (plugin_name, plugin_data) in enumerate(data.items()):
                if layout and plugin_name in layout:
                    row, col = layout[plugin_name]
                else:
                    row = idx // cols
                    col = idx % cols

                ax = fig.add_subplot(gs[row, col])

                # Get or create plugin
                if plugin_name in self.plugins:
                    plugin = self.plugins[plugin_name]
                else:
                    try:
                        plugin = self.create_plugin_from_template(plugin_name)
                    except ValueError:
                        logger.warning(f"Plugin '{plugin_name}' not found, skipping")
                        continue

                # Create visualization in subplot
                try:
                    plugin.create_visualization(plugin_data, ax=ax)
                except Exception as e:
                    logger.error(f"Error creating visualization for {plugin_name}: {e}")

            if save:
                save_path = self.context.get_save_path(f"dashboard.{self.config.save_format}")
                fig.savefig(
                    save_path,
                    dpi=self.config.save_dpi,
                    format=self.config.save_format,
                    bbox_inches=self.config.bbox_inches
                )

            if show:
                plt.show()
            else:
                plt.close(fig)

        return fig

    def _find_suitable_plugin(self, data: Any) -> Optional[VisualizationPlugin]:
        """
        Find a suitable plugin for the given data.

        Raises:
            ValueError: If multiple plugins can handle the data (ambiguous case)

        Returns:
            The suitable plugin, or None if no plugin can handle the data
        """
        suitable_plugins = []

        for plugin in self.plugins.values():
            if plugin.can_handle(data):
                suitable_plugins.append(plugin)

        if len(suitable_plugins) == 0:
            return None
        elif len(suitable_plugins) == 1:
            return suitable_plugins[0]
        else:
            # Multiple plugins can handle the data - force disambiguation
            plugin_names = [p.name for p in suitable_plugins]
            raise ValueError(
                f"Multiple plugins can handle this data type: {plugin_names}. "
                f"Please specify the plugin_name parameter to disambiguate."
            )

    def _discover_plugins(self) -> None:
        """Auto-discover and load available plugins."""
        # This would scan for plugin modules and auto-load them
        # For now, we'll manually register built-in templates
        pass

    def get_available_plugins(self) -> List[str]:
        """Get list of available plugin names."""
        return list(self.plugins.keys())

    def get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        return list(self.templates.keys())

    def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata about the visualizations."""
        self.context.metadata.update(metadata)
        metadata_path = self.context.get_save_path("metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.context.metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def create_color_palette(n_colors: int, palette_type: str = "husl") -> List[str]:
    """Create a color palette with n colors."""
    return sns.color_palette(palette_type, n_colors).as_hex()


def adjust_lightness(color: str, amount: float = 0.5) -> str:
    """
    Adjust the lightness of a color.

    Args:
        color: Hex color string
        amount: Amount to adjust (0=black, 1=white, 0.5=unchanged)

    Returns:
        Adjusted hex color string
    """
    import matplotlib.colors as mc
    import colorsys

    c = mc.hex2color(color)
    c = colorsys.rgb_to_hls(*c)
    c = colorsys.hls_to_rgb(c[0], amount, c[2])
    return mc.to_hex(c)