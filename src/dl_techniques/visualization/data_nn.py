"""
Data Distribution and Neural Network Visualization Templates
=============================================================

Ready-made templates for data analysis, distribution visualization,
and neural network specific visualizations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import gaussian_kde
import keras

from .core import VisualizationPlugin, CompositeVisualization, PlotConfig, VisualizationContext


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DatasetInfo:
    """Container for dataset information."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    class_names: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ActivationData:
    """Container for neural network activation data."""

    layer_names: List[str]
    activations: Dict[str, np.ndarray]  # layer_name -> activations
    model_name: Optional[str] = None


@dataclass
class WeightData:
    """Container for neural network weight data."""

    layer_names: List[str]
    weights: Dict[str, List[np.ndarray]]  # layer_name -> [weights, biases]
    model_name: Optional[str] = None


# =============================================================================
# Data Distribution Templates
# =============================================================================

class DataDistributionAnalysis(VisualizationPlugin):
    """Comprehensive data distribution analysis."""

    @property
    def name(self) -> str:
        return "data_distribution"

    @property
    def description(self) -> str:
        return "Analyze and visualize data distributions"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (DatasetInfo, np.ndarray, pd.DataFrame))

    def create_visualization(
            self,
            data: Union[DatasetInfo, np.ndarray, pd.DataFrame],
            features_to_plot: Optional[List[int]] = None,
            plot_type: str = 'auto',  # 'hist', 'kde', 'box', 'violin', 'auto'
            **kwargs
    ) -> plt.Figure:
        """Create data distribution visualization."""

        # Convert to numpy array if needed
        if isinstance(data, DatasetInfo):
            x_data = data.x_train
            feature_names = data.feature_names
        elif isinstance(data, pd.DataFrame):
            x_data = data.values
            feature_names = list(data.columns)
        else:
            x_data = data
            feature_names = None

        # Flatten if needed
        if len(x_data.shape) > 2:
            x_data = x_data.reshape(x_data.shape[0], -1)

        # Select features to plot
        if features_to_plot is None:
            features_to_plot = list(range(min(12, x_data.shape[1])))

        n_features = len(features_to_plot)
        cols = min(4, n_features)
        rows = (n_features + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, feat_idx in enumerate(features_to_plot):
            ax = axes[idx]
            data_col = x_data[:, feat_idx]

            # Choose plot type
            if plot_type == 'auto':
                if len(np.unique(data_col)) < 20:
                    plot_type_use = 'hist'
                else:
                    plot_type_use = 'kde'
            else:
                plot_type_use = plot_type

            # Create plot
            if plot_type_use == 'hist':
                ax.hist(data_col, bins=30, edgecolor='black', alpha=0.7)
                ax.set_ylabel('Frequency')
            elif plot_type_use == 'kde':
                try:
                    kde = gaussian_kde(data_col)
                    x_range = np.linspace(data_col.min(), data_col.max(), 100)
                    ax.fill_between(x_range, kde(x_range), alpha=0.5)
                    ax.plot(x_range, kde(x_range), linewidth=2)
                    ax.set_ylabel('Density')
                except:
                    ax.hist(data_col, bins=30, edgecolor='black', alpha=0.7)
                    ax.set_ylabel('Frequency')
            elif plot_type_use == 'box':
                ax.boxplot(data_col, vert=True)
                ax.set_ylabel('Value')
            elif plot_type_use == 'violin':
                ax.violinplot([data_col], vert=True, showmeans=True)
                ax.set_ylabel('Value')

            # Labels
            if feature_names:
                ax.set_title(f'{feature_names[feat_idx]}')
            else:
                ax.set_title(f'Feature {feat_idx}')

            ax.set_xlabel('Value' if plot_type_use in ['hist', 'kde'] else '')

            # Add statistics
            mean_val = np.mean(data_col)
            std_val = np.std(data_col)
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'μ={mean_val:.2f}\nσ={std_val:.2f}',
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle('Data Distribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


class ClassBalanceVisualization(VisualizationPlugin):
    """Visualize class balance and imbalance in datasets."""

    @property
    def name(self) -> str:
        return "class_balance"

    @property
    def description(self) -> str:
        return "Visualize class distribution and balance"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (DatasetInfo, tuple))

    def create_visualization(
            self,
            data: Union[DatasetInfo, Tuple[np.ndarray, np.ndarray]],
            show_percentages: bool = True,
            **kwargs
    ) -> plt.Figure:
        """Create class balance visualization."""

        # Extract labels
        if isinstance(data, DatasetInfo):
            y_train = data.y_train
            y_test = data.y_test
            class_names = data.class_names
        else:
            y_train = data[0]
            y_test = data[1] if len(data) > 1 else None
            class_names = None

        # Handle one-hot encoded labels
        if len(y_train.shape) == 2:
            y_train = np.argmax(y_train, axis=1)
        if y_test is not None and len(y_test.shape) == 2:
            y_test = np.argmax(y_test, axis=1)

        # Count classes
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)

        if y_test is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes = list(axes) + [None]

        # 1. Training set distribution
        ax = axes[0]
        train_counts = np.bincount(y_train)
        colors = plt.cm.Set3(np.linspace(0, 1, len(train_counts)))

        wedges, texts, autotexts = ax.pie(
            train_counts,
            labels=class_names if class_names else unique_classes,
            autopct='%1.1f%%' if show_percentages else '',
            colors=colors,
            startangle=90
        )

        ax.set_title('Training Set Distribution')

        # 2. Bar chart comparison
        ax = axes[1]
        x = np.arange(n_classes)
        width = 0.35

        bars1 = ax.bar(x - width / 2 if y_test is not None else x,
                       train_counts, width,
                       label='Train', alpha=0.8, color='blue')

        if y_test is not None:
            test_counts = np.bincount(y_test)
            bars2 = ax.bar(x + width / 2, test_counts, width,
                           label='Test', alpha=0.8, color='orange')

        # Add value labels
        for bars in [bars1] + ([bars2] if y_test is not None else []):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names if class_names else unique_classes)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Imbalance ratio (if test set available)
        if y_test is not None:
            ax = axes[2]

            # Calculate imbalance metrics
            min_class_train = np.min(train_counts)
            max_class_train = np.max(train_counts)
            imbalance_ratio = max_class_train / (min_class_train + 1e-10)

            # Visualize as a gauge
            self._plot_imbalance_gauge(ax, imbalance_ratio)
            ax.set_title(f'Imbalance Ratio: {imbalance_ratio:.2f}')

        plt.suptitle('Class Balance Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def _plot_imbalance_gauge(self, ax: plt.Axes, ratio: float):
        """Plot imbalance ratio as a gauge."""

        # Define thresholds
        thresholds = [(1, 1.5, 'Balanced', 'green'),
                      (1.5, 3, 'Slight Imbalance', 'yellow'),
                      (3, 10, 'Moderate Imbalance', 'orange'),
                      (10, float('inf'), 'Severe Imbalance', 'red')]

        # Find current level
        for min_val, max_val, label, color in thresholds:
            if min_val <= ratio < max_val:
                current_level = label
                current_color = color
                break

        # Create semi-circular gauge
        theta = np.linspace(np.pi, 0, 100)
        r_outer = 1.0
        r_inner = 0.7

        # Background
        ax.fill_between(theta, r_inner, r_outer, color='lightgray', alpha=0.3)

        # Colored sections
        section_size = np.pi / len(thresholds)
        for i, (_, _, _, color) in enumerate(thresholds):
            theta_section = np.linspace(np.pi - i * section_size,
                                        np.pi - (i + 1) * section_size, 20)
            ax.fill_between(theta_section, r_inner, r_outer, color=color, alpha=0.5)

        # Indicator
        if ratio > 10:
            ratio = 10  # Cap for visualization
        angle = np.pi - (np.pi * np.log10(ratio) / np.log10(10))
        ax.plot([0, np.cos(angle)], [0, np.sin(angle)], 'k-', linewidth=3)
        ax.scatter([np.cos(angle)], [np.sin(angle)], s=100, color='black', zorder=5)

        ax.text(0, -0.3, current_level, ha='center', fontsize=12,
                fontweight='bold', color=current_color)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')


# =============================================================================
# Neural Network Architecture Visualization
# =============================================================================

class NetworkArchitectureVisualization(VisualizationPlugin):
    """Visualize neural network architecture."""

    @property
    def name(self) -> str:
        return "network_architecture"

    @property
    def description(self) -> str:
        return "Visualize neural network layer structure"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, keras.Model) or hasattr(data, 'layers')

    def create_visualization(
            self,
            data: keras.Model,
            show_params: bool = True,
            show_shapes: bool = True,
            orientation: str = 'vertical',  # 'vertical' or 'horizontal'
            **kwargs
    ) -> plt.Figure:
        """Create network architecture visualization."""

        fig, ax = plt.subplots(figsize=(14, 10))

        # Extract layer information
        layers_info = []
        for layer in data.layers:
            info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'params': layer.count_params(),
                'output_shape': layer.output_shape,
                'input_shape': layer.input_shape if hasattr(layer, 'input_shape') else None
            }
            layers_info.append(info)

        n_layers = len(layers_info)

        if orientation == 'vertical':
            self._plot_vertical_architecture(ax, layers_info, show_params, show_shapes)
        else:
            self._plot_horizontal_architecture(ax, layers_info, show_params, show_shapes)

        ax.set_title(f'Network Architecture - {data.name if hasattr(data, "name") else "Model"}',
                     fontsize=14, fontweight='bold')
        ax.axis('off')

        # Add summary statistics
        total_params = data.count_params()
        ax.text(0.02, 0.02, f'Total Parameters: {total_params:,}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        return fig

    def _plot_vertical_architecture(self, ax: plt.Axes, layers_info: List[Dict],
                                    show_params: bool, show_shapes: bool):
        """Plot architecture vertically."""

        n_layers = len(layers_info)
        y_positions = np.linspace(0.9, 0.1, n_layers)
        x_center = 0.5

        # Color map for layer types
        layer_colors = {
            'Dense': 'blue',
            'Conv2D': 'green',
            'MaxPooling2D': 'red',
            'Dropout': 'gray',
            'BatchNormalization': 'purple',
            'Flatten': 'orange',
            'Input': 'lightblue'
        }

        for i, (layer_info, y_pos) in enumerate(zip(layers_info, y_positions)):
            # Get color
            layer_type = layer_info['type']
            color = layer_colors.get(layer_type, 'lightgray')

            # Draw layer box
            width = 0.2
            height = 0.05
            rect = FancyBboxPatch(
                (x_center - width / 2, y_pos - height / 2),
                width, height,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor='black',
                alpha=0.7, linewidth=1
            )
            ax.add_patch(rect)

            # Layer name
            ax.text(x_center, y_pos, layer_info['name'][:20],
                    ha='center', va='center', fontsize=9, fontweight='bold')

            # Layer type
            ax.text(x_center - width / 2 - 0.02, y_pos, layer_type,
                    ha='right', va='center', fontsize=8, style='italic')

            # Parameters
            if show_params:
                ax.text(x_center + width / 2 + 0.02, y_pos,
                        f"{layer_info['params']:,} params",
                        ha='left', va='center', fontsize=8)

            # Shape
            if show_shapes and layer_info['output_shape']:
                shape_str = str(layer_info['output_shape'])
                ax.text(x_center, y_pos - height / 2 - 0.02, shape_str,
                        ha='center', va='top', fontsize=7, color='gray')

            # Connection line
            if i < n_layers - 1:
                ax.plot([x_center, x_center],
                        [y_pos - height / 2, y_positions[i + 1] + height / 2],
                        'k-', alpha=0.5, linewidth=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def _plot_horizontal_architecture(self, ax: plt.Axes, layers_info: List[Dict],
                                      show_params: bool, show_shapes: bool):
        """Plot architecture horizontally."""

        n_layers = len(layers_info)
        x_positions = np.linspace(0.1, 0.9, n_layers)
        y_center = 0.5

        # Similar to vertical but rotated
        # Implementation would be similar with x/y swapped
        pass  # Simplified for brevity


# =============================================================================
# Activation Visualization Templates
# =============================================================================

class ActivationVisualization(VisualizationPlugin):
    """Visualize neural network activations."""

    @property
    def name(self) -> str:
        return "activations"

    @property
    def description(self) -> str:
        return "Visualize layer activations"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ActivationData)

    def create_visualization(
            self,
            data: ActivationData,
            layers_to_show: Optional[List[str]] = None,
            plot_type: str = 'distribution',  # 'distribution', 'heatmap', 'stats'
            **kwargs
    ) -> plt.Figure:
        """Create activation visualization."""

        if layers_to_show is None:
            layers_to_show = list(data.activations.keys())[:6]

        n_layers = len(layers_to_show)

        if plot_type == 'distribution':
            fig, axes = plt.subplots(2, (n_layers + 1) // 2,
                                     figsize=(12, 8))
            axes = axes.flatten()

            for idx, layer_name in enumerate(layers_to_show):
                ax = axes[idx]
                activations = data.activations[layer_name].flatten()

                # Plot distribution
                ax.hist(activations, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', alpha=0.5)

                # Statistics
                mean_act = np.mean(activations)
                std_act = np.std(activations)
                ax.axvline(mean_act, color='blue', linestyle='--', alpha=0.5)

                ax.set_title(f'{layer_name}')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency')

                # Add text
                ax.text(0.02, 0.98,
                        f'μ={mean_act:.3f}\nσ={std_act:.3f}\n% dead={np.mean(activations == 0):.1%}',
                        transform=ax.transAxes, va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Hide unused
            for idx in range(n_layers, len(axes)):
                axes[idx].set_visible(False)

        elif plot_type == 'heatmap':
            fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 4))
            if n_layers == 1:
                axes = [axes]

            for idx, layer_name in enumerate(layers_to_show):
                ax = axes[idx]
                activations = data.activations[layer_name]

                # Reshape if needed for visualization
                if len(activations.shape) == 4:  # Conv layers
                    # Take mean over batch and channels for spatial view
                    act_mean = np.mean(activations, axis=(0, 3))
                    im = ax.imshow(act_mean, cmap='hot', aspect='auto')
                elif len(activations.shape) == 2:  # Dense layers
                    im = ax.imshow(activations[:min(100, len(activations))],
                                   cmap='hot', aspect='auto')
                    ax.set_xlabel('Neurons')
                    ax.set_ylabel('Samples')

                ax.set_title(layer_name)
                plt.colorbar(im, ax=ax, fraction=0.046)

        elif plot_type == 'stats':
            # Statistical summary
            fig, ax = plt.subplots(figsize=(12, 6))

            stats_data = []
            for layer_name in layers_to_show:
                acts = data.activations[layer_name].flatten()
                stats_data.append({
                    'Layer': layer_name[:15],
                    'Mean': np.mean(acts),
                    'Std': np.std(acts),
                    'Min': np.min(acts),
                    'Max': np.max(acts),
                    '% Zero': np.mean(acts == 0) * 100,
                    '% Positive': np.mean(acts > 0) * 100
                })

            df = pd.DataFrame(stats_data)

            # Plot as grouped bar chart
            df_plot = df.set_index('Layer')[['Mean', 'Std', '% Zero']]
            df_plot.plot(kind='bar', ax=ax, rot=45)

            ax.set_title('Activation Statistics Summary')
            ax.set_ylabel('Value')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

        model_name = data.model_name if data.model_name else "Model"
        plt.suptitle(f'Activation Analysis - {model_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


# =============================================================================
# Weight Visualization Templates
# =============================================================================

class WeightVisualization(VisualizationPlugin):
    """Visualize neural network weights."""

    @property
    def name(self) -> str:
        return "weights"

    @property
    def description(self) -> str:
        return "Visualize layer weights and biases"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, (WeightData, keras.Model))

    def create_visualization(
            self,
            data: Union[WeightData, keras.Model],
            layers_to_show: Optional[List[str]] = None,
            plot_type: str = 'distribution',  # 'distribution', 'matrix', 'filters'
            **kwargs
    ) -> plt.Figure:
        """Create weight visualization."""

        # Extract weight data if model
        if isinstance(data, keras.Model):
            weight_data = self._extract_weights(data)
        else:
            weight_data = data

        if layers_to_show is None:
            layers_to_show = list(weight_data.weights.keys())[:6]

        n_layers = len(layers_to_show)

        if plot_type == 'distribution':
            fig, axes = plt.subplots(2, (n_layers + 1) // 2, figsize=(12, 8))
            axes = axes.flatten()

            for idx, layer_name in enumerate(layers_to_show):
                ax = axes[idx]
                weights_list = weight_data.weights[layer_name]

                if len(weights_list) > 0:
                    weights = weights_list[0].flatten()

                    # Plot distribution
                    ax.hist(weights, bins=50, alpha=0.7, color='blue',
                            edgecolor='black', label='Weights')

                    # Plot bias if available
                    if len(weights_list) > 1:
                        biases = weights_list[1].flatten()
                        ax.hist(biases, bins=30, alpha=0.5, color='red',
                                edgecolor='black', label='Biases')

                    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
                    ax.set_title(f'{layer_name}')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend(fontsize=8)

                    # Statistics
                    ax.text(0.02, 0.98,
                            f'μ={np.mean(weights):.3f}\nσ={np.std(weights):.3f}',
                            transform=ax.transAxes, va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Hide unused
            for idx in range(n_layers, len(axes)):
                axes[idx].set_visible(False)

        elif plot_type == 'matrix':
            fig, axes = plt.subplots(1, n_layers, figsize=(3 * n_layers, 4))
            if n_layers == 1:
                axes = [axes]

            for idx, layer_name in enumerate(layers_to_show):
                ax = axes[idx]
                weights_list = weight_data.weights[layer_name]

                if len(weights_list) > 0:
                    weights = weights_list[0]

                    # Visualize weight matrix
                    if len(weights.shape) == 2:
                        im = ax.imshow(weights, cmap='RdBu_r', aspect='auto')
                    elif len(weights.shape) == 4:  # Conv weights
                        # Show first few filters
                        w_reshaped = weights[:, :, 0, :min(16, weights.shape[3])]
                        im = ax.imshow(self._arrange_filters(w_reshaped),
                                       cmap='RdBu_r', aspect='auto')
                    else:
                        continue

                    ax.set_title(layer_name[:20])
                    plt.colorbar(im, ax=ax, fraction=0.046)

        elif plot_type == 'filters':
            # For convolutional layers, show filters
            fig = plt.figure(figsize=(12, 8))
            n_cols = 4
            n_rows = (n_layers + n_cols - 1) // n_cols

            plot_idx = 1
            for layer_name in layers_to_show:
                weights_list = weight_data.weights[layer_name]

                if len(weights_list) > 0:
                    weights = weights_list[0]

                    if len(weights.shape) == 4:  # Conv2D weights
                        ax = plt.subplot(n_rows, n_cols, plot_idx)

                        # Show first 16 filters
                        n_filters = min(16, weights.shape[3])
                        filter_grid = self._create_filter_grid(weights, n_filters)

                        ax.imshow(filter_grid, cmap='gray')
                        ax.set_title(f'{layer_name[:15]}\n({n_filters} filters)')
                        ax.axis('off')

                        plot_idx += 1

        model_name = weight_data.model_name if hasattr(weight_data, 'model_name') else "Model"
        plt.suptitle(f'Weight Analysis - {model_name}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def _extract_weights(self, model: keras.Model) -> WeightData:
        """Extract weights from a Keras model."""

        layer_names = []
        weights_dict = {}

        for layer in model.layers:
            if layer.get_weights():
                layer_names.append(layer.name)
                weights_dict[layer.name] = layer.get_weights()

        return WeightData(
            layer_names=layer_names,
            weights=weights_dict,
            model_name=model.name if hasattr(model, 'name') else None
        )

    def _arrange_filters(self, filters: np.ndarray) -> np.ndarray:
        """Arrange filters in a grid for visualization."""

        n_filters = filters.shape[2]
        n_cols = int(np.ceil(np.sqrt(n_filters)))
        n_rows = int(np.ceil(n_filters / n_cols))

        h, w = filters.shape[:2]
        grid = np.zeros((h * n_rows, w * n_cols))

        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = filters[:, :, i]

        return grid

    def _create_filter_grid(self, weights: np.ndarray, n_filters: int) -> np.ndarray:
        """Create a grid of filters for visualization."""

        n_cols = int(np.ceil(np.sqrt(n_filters)))
        n_rows = int(np.ceil(n_filters / n_cols))

        h, w = weights.shape[:2]

        # Create grid with padding
        pad = 1
        grid = np.ones(((h + pad) * n_rows + pad,
                        (w + pad) * n_cols + pad)) * 0.5

        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols

            # Get filter (average over input channels)
            if weights.shape[2] > 1:
                filter_img = np.mean(weights[:, :, :, i], axis=2)
            else:
                filter_img = weights[:, :, 0, i]

            # Normalize
            filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)

            # Place in grid
            row_start = (h + pad) * row + pad
            col_start = (w + pad) * col + pad
            grid[row_start:row_start + h, col_start:col_start + w] = filter_img

        return grid


# =============================================================================
# Feature Map Visualization
# =============================================================================

class FeatureMapVisualization(VisualizationPlugin):
    """Visualize convolutional feature maps."""

    @property
    def name(self) -> str:
        return "feature_maps"

    @property
    def description(self) -> str:
        return "Visualize CNN feature maps"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, ActivationData)

    def create_visualization(
            self,
            data: ActivationData,
            sample_idx: int = 0,
            layers_to_show: Optional[List[str]] = None,
            max_features: int = 16,
            **kwargs
    ) -> plt.Figure:
        """Create feature map visualization."""

        if layers_to_show is None:
            # Find conv layers
            conv_layers = []
            for layer_name, acts in data.activations.items():
                if len(acts.shape) == 4:  # Batch, H, W, Channels
                    conv_layers.append(layer_name)
            layers_to_show = conv_layers[:4]

        if not layers_to_show:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'No convolutional layers found',
                    ha='center', va='center')
            ax.axis('off')
            return fig

        n_layers = len(layers_to_show)
        fig = plt.figure(figsize=(12, 3 * n_layers))

        for layer_idx, layer_name in enumerate(layers_to_show):
            activations = data.activations[layer_name]

            if len(activations.shape) != 4:
                continue

            # Get activations for specific sample
            sample_acts = activations[sample_idx]
            n_features = min(max_features, sample_acts.shape[-1])

            # Create grid
            n_cols = int(np.ceil(np.sqrt(n_features)))
            n_rows = int(np.ceil(n_features / n_cols))

            for i in range(n_features):
                ax = plt.subplot(n_layers, n_cols, layer_idx * n_cols + i + 1)

                feature_map = sample_acts[:, :, i]
                im = ax.imshow(feature_map, cmap='viridis', aspect='auto')
                ax.axis('off')

                if i == 0:
                    ax.set_title(f'{layer_name[:20]}', fontsize=10)

        plt.suptitle(f'Feature Maps (Sample {sample_idx})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig


# =============================================================================
# Gradient Visualization
# =============================================================================

class GradientVisualization(VisualizationPlugin):
    """Visualize gradients during training."""

    @property
    def name(self) -> str:
        return "gradients"

    @property
    def description(self) -> str:
        return "Visualize gradient flow and magnitudes"

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, dict) and 'gradients' in str(data)

    def create_visualization(
            self,
            data: Dict[str, np.ndarray],  # layer_name -> gradients
            plot_type: str = 'flow',  # 'flow', 'distribution', 'vanishing'
            **kwargs
    ) -> plt.Figure:
        """Create gradient visualization."""

        layer_names = list(data.keys())
        n_layers = len(layer_names)

        if plot_type == 'flow':
            fig, ax = plt.subplots(figsize=(12, 6))

            # Calculate gradient norms
            grad_norms = []
            for layer_name in layer_names:
                grad = data[layer_name]
                norm = np.linalg.norm(grad.flatten())
                grad_norms.append(norm)

            # Plot gradient flow
            x = range(n_layers)
            ax.plot(x, grad_norms, 'o-', linewidth=2, markersize=8)

            # Color by magnitude
            for i, (norm, layer) in enumerate(zip(grad_norms, layer_names)):
                if norm < 1e-6:
                    color = 'red'  # Vanishing
                elif norm > 100:
                    color = 'orange'  # Exploding
                else:
                    color = 'green'  # Healthy
                ax.scatter([i], [norm], color=color, s=100, zorder=5)

            ax.set_xlabel('Layer')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Flow Through Network')
            ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels([l[:10] for l in layer_names], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)

            # Add reference lines
            ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5,
                       label='Vanishing threshold')
            ax.axhline(y=100, color='orange', linestyle='--', alpha=0.5,
                       label='Exploding threshold')
            ax.legend()

        elif plot_type == 'distribution':
            fig, axes = plt.subplots(2, (n_layers + 1) // 2,
                                     figsize=(12, 8))
            axes = axes.flatten()

            for idx, layer_name in enumerate(layer_names[:len(axes)]):
                ax = axes[idx]
                grads = data[layer_name].flatten()

                ax.hist(grads, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(0, color='red', linestyle='--', alpha=0.5)

                ax.set_title(f'{layer_name[:20]}')
                ax.set_xlabel('Gradient Value')
                ax.set_ylabel('Frequency')

                # Add statistics
                ax.text(0.02, 0.98,
                        f'μ={np.mean(grads):.2e}\nσ={np.std(grads):.2e}',
                        transform=ax.transAxes, va='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Hide unused
            for idx in range(n_layers, len(axes)):
                axes[idx].set_visible(False)

        elif plot_type == 'vanishing':
            fig, ax = plt.subplots(figsize=(10, 6))

            # Analyze gradient vanishing
            vanishing_scores = []
            for layer_name in layer_names:
                grad = data[layer_name].flatten()
                # Score based on how many gradients are near zero
                vanishing = np.mean(np.abs(grad) < 1e-6)
                vanishing_scores.append(vanishing * 100)

            colors = ['red' if v > 50 else 'orange' if v > 20 else 'green'
                      for v in vanishing_scores]

            bars = ax.bar(range(n_layers), vanishing_scores, color=colors, alpha=0.7)

            ax.set_xlabel('Layer')
            ax.set_ylabel('% Vanishing Gradients')
            ax.set_title('Gradient Vanishing Analysis')
            ax.set_xticks(range(n_layers))
            ax.set_xticklabels([l[:10] for l in layer_names], rotation=45, ha='right')
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5)
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, vanishing_scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Gradient Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig