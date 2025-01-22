"""
Visualization Manager Module
==========================

A comprehensive module for managing and saving visualizations in machine learning projects.
Provides consistent styling, saving mechanisms, and organization for matplotlib-based visualizations.

Features:
    - Configurable visualization settings
    - Systematic file organization
    - High-quality output generation
    - Seaborn integration for better aesthetics
    - Type hints and comprehensive documentation
"""


import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.metrics import confusion_matrix
from typing import Union, Optional, Tuple, Dict, Any, List

# ------------------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------------------

from .logger import logger
from .datasets import MNISTData

# ------------------------------------------------------------------------------

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters.

    Args:
        fig_size: Default figure size for plots (width, height)
        dpi: DPI for saved images
        cmap: Default colormap for heatmaps
        save_format: Format for saving figures
        subplot_spacing: Default spacing between subplots
        title_fontsize: Default fontsize for titles
        label_fontsize: Default fontsize for labels
        tick_fontsize: Default fontsize for tick labels
        legend_fontsize: Default fontsize for legends
    """
    fig_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    cmap: str = 'viridis'
    save_format: str = 'png'
    subplot_spacing: float = 0.3
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

# ------------------------------------------------------------------------------


class VisualizationManager:
    """Manager for handling visualization creation and saving."""

    def __init__(
            self,
            output_dir: Union[str, Path],
            config: Optional[VisualizationConfig] = None,
            timestamp_dirs: bool = True
    ):
        """Initialize visualization manager.

        Args:
            output_dir: Base directory for saving visualizations
            config: Visualization configuration settings
            timestamp_dirs: Whether to create timestamped subdirectories
        """
        self.base_dir = Path(output_dir)
        if timestamp_dirs:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = self.base_dir / f"viz_{timestamp}"
        else:
            self.output_dir = self.base_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or VisualizationConfig()

        # Set up plotting style
        sns.set_style("whitegrid")
        plt.style.use(['seaborn-v0_8-whitegrid'])

        # Set default plot settings
        plt.rcParams.update({
            'figure.figsize': self.config.fig_size,
            'figure.dpi': self.config.dpi,
            'axes.titlesize': self.config.title_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.legend_fontsize
        })

    def get_save_path(self, name: str, subdir: Optional[str] = None) -> Path:
        """Get full save path for a visualization.

        Args:
            name: Base name for the file
            subdir: Optional subdirectory within output_dir

        Returns:
            Complete path where the file should be saved
        """
        if subdir:
            save_dir = self.output_dir / subdir
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = self.output_dir

        return save_dir / f"{name}.{self.config.save_format}"

    def save_figure(
            self,
            fig: plt.Figure,
            name: str,
            subdir: Optional[str] = None,
            close_fig: bool = True
    ) -> Path:
        """Save figure with proper configuration.

        Args:
            fig: Figure to save
            name: Base name for the file
            subdir: Optional subdirectory within output_dir
            close_fig: Whether to close the figure after saving

        Returns:
            Path where figure was saved
        """
        save_path = self.get_save_path(name, subdir)

        fig.tight_layout()
        fig.savefig(
            save_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            pad_inches=0.1,
            format=self.config.save_format
        )

        if close_fig:
            plt.close(fig)

        return save_path

    def create_figure(
            self,
            size: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """Create a new figure with proper settings.

        Args:
            size: Optional custom size for this figure

        Returns:
            Newly created figure
        """
        return plt.figure(figsize=size or self.config.fig_size)

    def plot_matrix(
            self,
            matrix: np.ndarray,
            title: str,
            xlabel: str,
            ylabel: str,
            name: str,
            subdir: Optional[str] = None,
            annot: bool = True,
            fmt: str = '.2f',
            cmap: Optional[str] = None
    ) -> Path:
        """Plot and save a matrix visualization (e.g., confusion matrix, correlation matrix).

        Args:
            matrix: 2D numpy array to visualize
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            name: Base name for saving
            subdir: Optional subdirectory for saving
            annot: Whether to annotate cells
            fmt: Format for annotations
            cmap: Optional custom colormap

        Returns:
            Path where figure was saved
        """
        fig = self.create_figure()
        ax = fig.add_subplot(111)

        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap or self.config.cmap,
            annot=annot,
            fmt=fmt
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return self.save_figure(fig, name, subdir)

    def plot_history(
            self,
            histories: Dict[str, Dict[str, List[float]]],
            metrics: List[str],
            name: str,
            subdir: Optional[str] = None,
            title: Optional[str] = None
    ) -> Path:
        """Plot training history metrics.

        Args:
            histories: Dictionary mapping model names to their metric histories
            metrics: List of metrics to plot
            name: Base name for saving
            subdir: Optional subdirectory for saving
            title: Optional title for the entire figure

        Returns:
            Path where figure was saved
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(
            1, n_metrics,
            figsize=(self.config.fig_size[0] * n_metrics, self.config.fig_size[1])
        )

        if n_metrics == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for model_name, history in histories.items():
                if metric in history:
                    ax.plot(
                        history[metric],
                        label=f'{model_name} (Training)',
                        linestyle='-'
                    )
                if f'val_{metric}' in history:
                    ax.plot(
                        history[f'val_{metric}'],
                        label=f'{model_name} (Validation)',
                        linestyle='--'
                    )

            ax.set_title(f'{metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)

        if title:
            fig.suptitle(title, fontsize=self.config.title_fontsize * 1.2)

        plt.tight_layout()
        return self.save_figure(fig, name, subdir)

    def compare_images(
            self,
            images: List[np.ndarray],
            titles: List[str],
            name: str,
            subdir: Optional[str] = None,
            cmap: Optional[str] = 'gray'
    ) -> Path:
        """Compare multiple images side by side.

        Args:
            images: List of images to compare
            titles: List of titles for each image
            name: Base name for saving
            subdir: Optional subdirectory for saving
            cmap: Colormap for images

        Returns:
            Path where figure was saved
        """
        n_images = len(images)
        fig, axes = plt.subplots(
            1, n_images,
            figsize=(self.config.fig_size[0] * n_images / 2, self.config.fig_size[1])
        )

        if n_images == 1:
            axes = [axes]

        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax)

        return self.save_figure(fig, name, subdir)

    def plot_confusion_matrices_comparison(
            self,
            y_true: np.ndarray,
            model_predictions: Dict[str, np.ndarray],
            name: str,
            subdir: Optional[str] = None,
            figsize: Optional[Tuple[int, int]] = None,
            normalize: bool = True,
            class_names: Optional[List[str]] = None
    ) -> Path:
        """Plot confusion matrices for multiple models side by side for comparison.

        Args:
            y_true: Ground truth labels (1D array of class indices)
            model_predictions: Dictionary mapping model names to their predictions (1D arrays of class indices)
            name: Base name for saving the figure
            subdir: Optional subdirectory for saving
            figsize: Optional custom figure size (width, height)
            normalize: Whether to normalize confusion matrices
            class_names: Optional list of class names for labels

        Returns:
            Path where the comparison figure was saved

        Raises:
            ValueError: If model_predictions is empty or predictions don't match ground truth shape
        """
        if not model_predictions:
            raise ValueError("No model predictions provided")

        # Ensure y_true is 1D array of class indices
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        n_models = len(model_predictions)
        n_classes = len(np.unique(y_true))

        if not class_names:
            class_names = [str(i) for i in range(n_classes)]

        # Calculate figure size based on number of models
        if figsize is None:
            width = min(20, self.config.fig_size[0] * n_models)
            height = min(15, self.config.fig_size[1] * (n_models + 1) // 2)
            figsize = (width, height)

        # Create subplot grid
        n_rows = (n_models + 2) // 3  # Adjust grid based on number of models
        n_cols = min(3, n_models)

        # Create figure with extra space for colorbar
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(n_rows, n_cols + 1, width_ratios=[*[1] * n_cols, 0.05])

        # Find global min/max values for consistent colormap scaling
        all_cms = []
        for model_name, y_pred in model_predictions.items():
            if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)

            if y_pred.shape != y_true.shape:
                print(f"Shape mismatch for {model_name}:")
                print(f"y_true shape: {y_true.shape}")
                print(f"y_pred shape: {y_pred.shape}")
                raise ValueError(f"Predictions shape mismatch for model {model_name}")

            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            all_cms.append(cm)

        vmin = min(cm.min() for cm in all_cms)
        vmax = max(cm.max() for cm in all_cms)

        # Plot confusion matrices
        axes = []
        for idx, (model_name, y_pred) in enumerate(model_predictions.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

            if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
                y_pred = np.argmax(y_pred, axis=1)

            cm = confusion_matrix(y_true, y_pred)
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot confusion matrix with consistent vmin/vmax
            sns_heatmap = sns.heatmap(
                cm,
                ax=ax,
                cmap=self.config.cmap,
                annot=True,
                fmt='.2f' if normalize else 'd',
                square=True,
                xticklabels=class_names,
                yticklabels=class_names,
                vmin=vmin,
                vmax=vmax,
                cbar=False  # Don't show individual colorbars
            )

            ax.set_title(f'{model_name}\nConfusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')

            # Store the mappable object from the first heatmap for the colorbar
            if idx == 0:
                mappable = sns_heatmap.collections[0]

        # Add a single colorbar for all plots using the mappable from the first heatmap
        cbar_ax = fig.add_subplot(gs[:, -1])
        plt.colorbar(mappable, cax=cbar_ax)

        # Remove empty subplots if any
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.add_subplot(gs[row, col]).remove()

        plt.tight_layout()
        return self.save_figure(fig, name, subdir)