import keras
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union


class ActivationDistributionAnalyzer:
    """Analyzer class for neural network activation distributions."""

    # Constants for analysis thresholds
    NEAR_ZERO_THRESHOLD: float = 1e-5  # Threshold for considering a value "near zero"
    EXACT_ZERO_THRESHOLD: float = 1e-7  # Threshold for considering a value "exactly zero"

    # Constants for histogram computation
    HIST_MIN_VAL: float = -10.0  # Minimum value for histogram binning
    HIST_MAX_VAL: float = 10.0  # Maximum value for histogram binning
    HIST_NUM_BINS: int = 100  # Number of bins for histogram

    # Constants for visualization
    DEFAULT_SAMPLE_SIZE: int = 1000  # Default number of samples to analyze
    DEFAULT_BINS: int = 50  # Default number of histogram bins
    MAX_COLS: int = 3  # Maximum number of columns in subplot grid
    FIGURE_SIZE_PER_PLOT: int = 5  # Size of each subplot in inches

    # Constants for file paths and names
    DISTRIBUTION_PLOT_FILENAME: str = 'activation_distributions.png'
    HEATMAP_PLOT_FILENAME: str = 'activation_heatmaps.png'
    STATS_FILENAME: str = 'activation_stats.txt'

    # Visualization settings
    PLOT_DPI: int = 300
    HISTOGRAM_ALPHA: float = 0.3
    GRID_ALPHA: float = 0.3

    # Color settings
    MEAN_LINE_COLOR: str = 'red'
    STD_LINE_COLOR: str = 'green'
    HEATMAP_COLORMAP: str = 'viridis'

    def __init__(
            self,
            model: keras.Model,
            layer_names: Optional[List[str]] = None
    ) -> None:
        """Initialize the activation analyzer.

        Args:
            model: Keras model to analyze
            layer_names: Optional list of layer names to analyze. If None, analyzes all layers
                       with activations
        """
        self.model = model
        self.layer_names = layer_names or self._get_activation_layer_names()
        self.activation_models = self._create_activation_models()

    def _get_activation_layer_names(self) -> List[str]:
        """Get names of layers that have activations.

        Returns:
            List of layer names that have activation functions
        """
        return [
            layer.name for layer in self.model.layers
            if isinstance(layer, (keras.layers.Conv2D, keras.layers.Dense))
        ]

    def _create_activation_models(self) -> Dict[str, keras.Model]:
        """Create models to extract activations from each layer.

        Returns:
            Dictionary mapping layer names to their respective activation models
        """
        activation_models = {}
        for layer_name in self.layer_names:
            layer = self.model.get_layer(layer_name)
            activation_models[layer_name] = keras.Model(
                inputs=self.model.input,
                outputs=layer.output
            )
        return activation_models

    def compute_activation_stats(
            self,
            input_data: np.ndarray,
            sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> Dict[str, Dict[str, Union[float, Dict[str, np.ndarray]]]]:
        """Compute statistical metrics and histograms for activations.

        Args:
            input_data: Input data to compute activations
            sample_size: Number of samples to use for computation

        Returns:
            Dictionary containing activation statistics and histograms for each layer
        """
        if len(input_data) > sample_size:
            indices = np.random.choice(len(input_data), sample_size, replace=False)
            input_data = input_data[indices]

        stats_dict = {}
        for layer_name, activation_model in self.activation_models.items():
            activations = activation_model.predict(input_data, verbose=0)
            flat_activations = activations.reshape(-1)

            # Compute basic statistics
            mean_val = float(np.mean(flat_activations))
            std_val = float(np.std(flat_activations))

            # Compute histogram with adaptive bounds
            hist_min = max(self.HIST_MIN_VAL, float(np.min(flat_activations)))
            hist_max = min(self.HIST_MAX_VAL, float(np.max(flat_activations)))

            # Compute histogram
            hist_counts, hist_edges = np.histogram(
                flat_activations,
                bins=self.HIST_NUM_BINS,
                range=(hist_min, hist_max)
            )

            # Calculate bin centers
            bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

            # Normalize histogram to get probability density
            hist_density = hist_counts / (np.sum(hist_counts) * (hist_edges[1] - hist_edges[0]))

            stats_dict[layer_name] = {
                # Basic statistics
                'mean': mean_val,
                'std': std_val,
                'min': float(np.min(flat_activations)),
                'max': float(np.max(flat_activations)),
                'sparsity': float(np.mean(np.abs(flat_activations) < self.NEAR_ZERO_THRESHOLD)),
                'positive_ratio': float(np.mean(flat_activations > 0)),
                'zero_ratio': float(np.mean(np.abs(flat_activations) < self.EXACT_ZERO_THRESHOLD)),

                # Distribution information
                'distribution': {
                    'bin_centers': bin_centers,
                    'density': hist_density,
                    'counts': hist_counts,
                    'bin_edges': hist_edges
                },

                # Additional distribution statistics
                'percentiles': {
                    'p1': float(np.percentile(flat_activations, 1)),
                    'p5': float(np.percentile(flat_activations, 5)),
                    'p25': float(np.percentile(flat_activations, 25)),
                    'p50': float(np.percentile(flat_activations, 50)),
                    'p75': float(np.percentile(flat_activations, 75)),
                    'p95': float(np.percentile(flat_activations, 95)),
                    'p99': float(np.percentile(flat_activations, 99))
                }
            }

        return stats_dict

    def plot_activation_distributions(
            self,
            input_data: np.ndarray,
            save_path: Path,
            sample_size: int = DEFAULT_SAMPLE_SIZE,
            n_bins: int = DEFAULT_BINS
    ) -> None:
        """Plot activation distributions for each layer.

        Args:
            input_data: Input data to compute activations
            save_path: Path to save the plots
            sample_size: Number of samples to use for visualization
            n_bins: Number of bins for the histogram
        """
        if len(input_data) > sample_size:
            indices = np.random.choice(len(input_data), sample_size, replace=False)
            input_data = input_data[indices]

        n_layers = len(self.layer_names)
        n_cols = min(self.MAX_COLS, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(self.FIGURE_SIZE_PER_PLOT * n_cols,
                     self.FIGURE_SIZE_PER_PLOT * n_rows)
        )
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (layer_name, activation_model) in enumerate(self.activation_models.items()):
            activations = activation_model.predict(input_data, verbose=0)
            flat_activations = activations.reshape(-1)

            # Calculate statistics
            mean_val = np.mean(flat_activations)
            std_val = np.std(flat_activations)
            sparsity = np.mean(np.abs(flat_activations) < self.NEAR_ZERO_THRESHOLD)
            positive_ratio = np.mean(flat_activations > 0)

            # Plot distribution
            sns.histplot(
                flat_activations,
                bins=n_bins,
                stat='density',
                element='step',
                fill=True,
                alpha=self.HISTOGRAM_ALPHA,
                ax=axes[idx]
            )

            # Add vertical lines for mean and ±std
            axes[idx].axvline(
                x=mean_val,
                color=self.MEAN_LINE_COLOR,
                linestyle='--',
                label='Mean'
            )
            axes[idx].axvline(
                x=mean_val + std_val,
                color=self.STD_LINE_COLOR,
                linestyle=':',
                label='+1 Std'
            )
            axes[idx].axvline(
                x=mean_val - std_val,
                color=self.STD_LINE_COLOR,
                linestyle=':',
                label='-1 Std'
            )

            axes[idx].set_title(
                f'Layer: {layer_name}\n'
                f'Mean: {mean_val:.3f}, Std: {std_val:.3f}\n'
                f'Sparsity: {sparsity:.1%}, Positive: {positive_ratio:.1%}'
            )
            axes[idx].set_xlabel('Activation Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=self.GRID_ALPHA)

        # Remove empty subplots
        for idx in range(len(self.activation_models), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(
            save_path / self.DISTRIBUTION_PLOT_FILENAME,
            dpi=self.PLOT_DPI,
            bbox_inches='tight'
        )
        plt.close()

    def plot_activation_heatmaps(
            self,
            input_data: np.ndarray,
            save_path: Path,
            sample_idx: int = 0
    ) -> None:
        """Plot activation heatmaps for convolutional layers.

        Args:
            input_data: Input data to compute activations
            save_path: Path to save the plots
            sample_idx: Index of the sample to visualize
        """
        conv_layers = [
            name for name in self.layer_names
            if isinstance(self.model.get_layer(name), keras.layers.Conv2D)
        ]

        if not conv_layers:
            return

        n_layers = len(conv_layers)
        fig, axes = plt.subplots(
            1,
            n_layers,
            figsize=(self.FIGURE_SIZE_PER_PLOT * n_layers,
                     self.FIGURE_SIZE_PER_PLOT)
        )
        if n_layers == 1:
            axes = [axes]

        sample = input_data[sample_idx:sample_idx + 1]

        for idx, layer_name in enumerate(conv_layers):
            activation_model = self.activation_models[layer_name]
            activations = activation_model.predict(sample, verbose=0)[0]

            # Average across channels
            avg_activation = np.mean(activations, axis=-1)

            im = axes[idx].imshow(avg_activation, cmap=self.HEATMAP_COLORMAP)
            axes[idx].set_title(f'Layer: {layer_name}')
            plt.colorbar(im, ax=axes[idx])
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(
            save_path / self.HEATMAP_PLOT_FILENAME,
            dpi=self.PLOT_DPI,
            bbox_inches='tight'
        )
        plt.close()


# Add to ActivationExperiment class:
def analyze_activation_distributions(self) -> None:
    """Analyze and visualize activation distributions for all models."""
    results_dir = Path('activation_analysis_results')
    results_dir.mkdir(exist_ok=True)

    for name, model in self.models.items():
        model_dir = results_dir / name
        model_dir.mkdir(exist_ok=True)

        analyzer = ActivationDistributionAnalyzer(model)

        # Compute and save activation statistics
        stats = analyzer.compute_activation_stats(self.x_test)

        # Plot distributions
        analyzer.plot_activation_distributions(
            self.x_test,
            save_path=model_dir
        )

        # Plot activation heatmaps
        analyzer.plot_activation_heatmaps(
            self.x_test,
            save_path=model_dir
        )

        # Save statistics to file
        with open(model_dir / ActivationDistributionAnalyzer.STATS_FILENAME, 'w') as f:
            f.write(f"Activation Statistics for {name} model:\n")
            f.write("=" * 50 + "\n")
            for layer_name, layer_stats in stats.items():
                f.write(f"\nLayer: {layer_name}\n")
                for metric, value in layer_stats.items():
                    f.write(f"{metric}: {value:.4f}\n")