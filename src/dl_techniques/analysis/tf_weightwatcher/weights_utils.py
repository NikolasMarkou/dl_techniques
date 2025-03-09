"""
Utility functions for TensorFlow/Keras weight matrix analysis.

This module provides utility functions for analyzing weight matrices in TensorFlow/Keras models,
including layer type inference, weight extraction, and visualization.
"""

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# Import constants from constants module
from .constants import (
    LayerType, DEFAULT_BINS, DEFAULT_FIG_SIZE, DEFAULT_DPI
)


def infer_layer_type(layer: keras.layers.Layer) -> LayerType:
    """
    Determine the layer type for a given Keras layer.

    Args:
        layer: Keras layer to analyze.

    Returns:
        LayerType: The inferred type of the layer.
    """
    layer_class = layer.__class__.__name__.lower()

    if isinstance(layer, keras.layers.Dense) or 'dense' in layer_class:
        return LayerType.DENSE
    elif isinstance(layer, keras.layers.Conv1D) or 'conv1d' in layer_class:
        return LayerType.CONV1D
    elif isinstance(layer, keras.layers.Conv2D) or 'conv2d' in layer_class:
        return LayerType.CONV2D
    elif isinstance(layer, keras.layers.Conv3D) or 'conv3d' in layer_class:
        return LayerType.CONV3D
    elif isinstance(layer, keras.layers.Embedding) or 'embedding' in layer_class:
        return LayerType.EMBEDDING
    elif isinstance(layer, keras.layers.LSTM) or 'lstm' in layer_class:
        return LayerType.LSTM
    elif isinstance(layer, keras.layers.GRU) or 'gru' in layer_class:
        return LayerType.GRU
    elif isinstance(layer, keras.layers.LayerNormalization) or 'layernorm' in layer_class:
        return LayerType.NORM
    else:
        return LayerType.UNKNOWN


def get_layer_weights_and_bias(layer: keras.layers.Layer) -> (
        Tuple)[bool, Optional[np.ndarray], bool, Optional[np.ndarray]]:
    """
    Extract weights and biases from a Keras layer.

    Args:
        layer: Keras layer to extract weights and biases from.

    Returns:
        Tuple containing:
        - has_weights (bool): Whether layer has weights
        - weights (Optional[np.ndarray]): Weight matrix if available, else None
        - has_bias (bool): Whether layer has biases
        - bias (Optional[np.ndarray]): Bias vector if available, else None
    """
    has_weights, has_bias = False, False
    weights, bias = None, None

    layer_type = infer_layer_type(layer)

    # Get layer weights
    weights_list = layer.get_weights()

    if len(weights_list) > 0:
        if layer_type in [
            LayerType.DENSE,
            LayerType.CONV1D,
            LayerType.CONV2D,
            LayerType.CONV3D,
            LayerType.EMBEDDING
        ]:
            has_weights = True
            weights = weights_list[0]

            # Check for bias
            if hasattr(layer, 'use_bias') and layer.use_bias and len(weights_list) > 1:
                has_bias = True
                bias = weights_list[1]

    return has_weights, weights, has_bias, bias


def get_weight_matrices(
        weights: np.ndarray,
        layer_type: LayerType) -> Tuple[List[np.ndarray], int, int, float]:
    """
    Extract weight matrices from a layer's weights.

    Args:
        weights: Layer weights.
        layer_type: Type of layer.

    Returns:
        Tuple containing:
        - List of weight matrices.
        - N: Maximum dimension.
        - M: Minimum dimension.
        - rf: Receptive field size.
    """
    Wmats = []
    N, M, rf = 0, 0, 1.0

    if layer_type in [LayerType.DENSE, LayerType.EMBEDDING]:
        Wmats = [weights]
        N, M = max(weights.shape), min(weights.shape)

    elif layer_type == LayerType.CONV1D:
        # Conv1D weights shape: (kernel_size, input_dim, output_dim)
        # For Conv1D, we'll use the flattened input weight dimensions
        kernel_size, input_dim, output_dim = weights.shape
        rf = kernel_size

        # Reshape to 2D matrix for eigenvalue analysis
        weights_reshaped = weights.reshape(-1, output_dim)
        N, M = max(weights_reshaped.shape), min(weights_reshaped.shape)
        Wmats = [weights_reshaped]

    elif layer_type == LayerType.CONV2D:
        # For Conv2D, extract weight matrices for each filter position
        # Conv2D weights shape: (kernel_height, kernel_width, input_channels, output_channels)
        kh, kw, in_c, out_c = weights.shape
        rf = kh * kw

        # Extract individual filter matrices
        for i in range(kh):
            for j in range(kw):
                W = weights[i, j, :, :]

                # Make sure larger dimension is first for consistency
                if W.shape[0] < W.shape[1]:
                    W = W.T

                Wmats.append(W)

        # Set N and M based on the shapes of the extracted matrices
        if len(Wmats) > 0:
            N, M = max(Wmats[0].shape), min(Wmats[0].shape)

    return Wmats, N, M, rf


def plot_powerlaw_fit(
        evals: np.ndarray,
        alpha: float,
        xmin: float,
        D: float,
        sigma: float,
        layer_name: str,
        layer_id: int,
        savedir: str
) -> None:
    """
    Create and save power-law fit plots.

    Args:
        evals: Eigenvalues to plot.
        alpha: Power-law exponent.
        xmin: Minimum x value used for fit.
        D: Kolmogorov-Smirnov statistic.
        sigma: Standard error of alpha.
        layer_name: Name of the layer.
        layer_id: ID of the layer.
        savedir: Directory to save plots.
    """
    # Create save directory if it doesn't exist
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Log-log plot
    plt.figure(figsize=DEFAULT_FIG_SIZE)

    # Plot histogram with log-log scale
    hist, bin_edges = np.histogram(evals, bins=DEFAULT_BINS)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram to PDF
    hist = hist / (np.sum(hist) * (bin_edges[1] - bin_edges[0]))

    plt.loglog(bin_centers, hist, 'o', markersize=4, label='Eigenvalue distribution')

    # Add power-law fit line
    x_range = np.logspace(np.log10(xmin), np.log10(np.max(evals)), 100)
    y_fit = (alpha - 1) * (xmin ** (alpha - 1)) * x_range ** (-alpha)
    plt.loglog(x_range, y_fit, 'r-', label=f'Power-law fit: α={alpha:.3f}')

    plt.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3f}')

    plt.title(f"Log-Log ESD for {layer_name}\nα={alpha:.3f}, D={D:.3f}, σ={sigma:.3f}")
    plt.xlabel("Eigenvalue (λ)")
    plt.ylabel("Probability density")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{savedir}/layer_{layer_id}_powerlaw.png", dpi=DEFAULT_DPI)
    plt.close()

    # Additional plot: linear scale
    plt.figure(figsize=DEFAULT_FIG_SIZE)
    plt.hist(evals, bins=DEFAULT_BINS, density=True)
    plt.axvline(x=xmin, color='r', linestyle='--', label=f'xmin={xmin:.3f}')
    plt.title(f"Linear ESD for {layer_name}")
    plt.xlabel("Eigenvalue (λ)")
    plt.ylabel("Probability density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{savedir}/layer_{layer_id}_esd_linear.png", dpi=DEFAULT_DPI)
    plt.close()


def calculate_glorot_normalization_factor(
        N: int,
        M: int,
        rf: float = 1.0) -> float:
    """
    Calculate the Glorot normalization factor.

    This function computes the normalization factor based on the Glorot/Xavier
    initialization approach, which helps maintain variance across layers.

    Args:
        N: Maximum dimension of the weight matrix.
        M: Minimum dimension of the weight matrix.
        rf: Receptive field size (for convolutional layers).

    Returns:
        float: Glorot normalization factor.
    """
    # Fan-in and fan-out calculation
    fan_in = M * rf  # Input connections
    fan_out = N  # Output connections

    # Return the normalization factor using Glorot's formula
    return np.sqrt(2.0 / (fan_in + fan_out))


def create_weight_visualization(
        model: keras.Model,
        layer_index: int,
        figsize: Tuple[int, int] = (10, 6),
        cmap: str = 'viridis'
) -> Optional[plt.Figure]:
    """
    Creates a visualization of weight matrices for a specific layer.

    Args:
        model: Keras model containing the layer to visualize.
        layer_index: Index of the layer to visualize.
        figsize: Size of the figure to create.
        cmap: Colormap to use for visualization.

    Returns:
        Optional[plt.Figure]: Matplotlib figure object or None if visualization fails.
    """
    if layer_index < 0 or layer_index >= len(model.layers):
        raise ValueError(f"Layer index {layer_index} is out of range for model with {len(model.layers)} layers.")

    layer = model.layers[layer_index]
    layer_type = infer_layer_type(layer)

    # Get weights
    has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

    if not has_weights:
        return None

    fig = plt.figure(figsize=figsize)

    if layer_type == LayerType.DENSE:
        plt.imshow(weights, cmap=cmap, aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.title(f"Weight Matrix for {layer.name} (Dense Layer)")
        plt.xlabel("Output Units")
        plt.ylabel("Input Units")

    elif layer_type == LayerType.CONV2D:
        kh, kw, in_c, out_c = weights.shape

        # For Conv2D, we'll create a grid of filters
        grid_size = int(np.ceil(np.sqrt(out_c)))

        # Create a grid to display filters for the first input channel
        for i in range(min(out_c, grid_size * grid_size)):
            plt.subplot(grid_size, grid_size, i + 1)
            # Display the filter for the first input channel
            plt.imshow(weights[:, :, 0, i], cmap=cmap)
            plt.axis('off')

        plt.suptitle(f"Filters for {layer.name} (Conv2D Layer) - First Input Channel")

    plt.tight_layout()
    return fig


def compute_weight_statistics(
        model: keras.Model,
        include_layers: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Computes basic statistics for weights in model layers.

    Args:
        model: Keras model to analyze.
        include_layers: Optional list of layer indices to include.
                        If None, all layers with weights will be analyzed.

    Returns:
        Dictionary mapping layer indices to dictionaries of statistics.
    """
    stats = {}

    for layer_id, layer in enumerate(model.layers):
        if include_layers is not None and layer_id not in include_layers:
            continue

        # Get weights
        has_weights, weights, has_bias, bias = get_layer_weights_and_bias(layer)

        if not has_weights:
            continue

        layer_stats = {
            'name': layer.name,
            'type': infer_layer_type(layer).name,
            'weight_shape': weights.shape,
            'weight_mean': float(np.mean(weights)),
            'weight_std': float(np.std(weights)),
            'weight_min': float(np.min(weights)),
            'weight_max': float(np.max(weights)),
            'weight_median': float(np.median(weights)),
            'weight_sparsity': float(np.sum(weights == 0) / weights.size),
        }

        if has_bias:
            layer_stats.update({
                'bias_shape': bias.shape,
                'bias_mean': float(np.mean(bias)),
                'bias_std': float(np.std(bias)),
                'bias_min': float(np.min(bias)),
                'bias_max': float(np.max(bias)),
            })

        stats[layer_id] = layer_stats

    return stats


def extract_conv_filters(
        layer: keras.layers.Layer,
        filter_indices: Optional[List[int]] = None
) -> Optional[List[np.ndarray]]:
    """
    Extracts convolutional filters from a convolutional layer.

    Args:
        layer: Keras convolutional layer.
        filter_indices: Optional list of filter indices to extract.
                       If None, all filters will be extracted.

    Returns:
        Optional list of filter weight arrays or None if not a conv layer.
    """
    layer_type = infer_layer_type(layer)

    if layer_type not in [LayerType.CONV1D, LayerType.CONV2D, LayerType.CONV3D]:
        return None

    has_weights, weights, _, _ = get_layer_weights_and_bias(layer)

    if not has_weights:
        return None

    # For Conv2D: weights shape is (kh, kw, in_channels, out_channels)
    if layer_type == LayerType.CONV2D:
        filters = []
        _, _, _, num_filters = weights.shape

        indices = filter_indices if filter_indices is not None else range(num_filters)

        for i in indices:
            if i < num_filters:
                filters.append(weights[:, :, :, i])

        return filters

    # For Conv1D: weights shape is (kernel_size, in_channels, out_channels)
    elif layer_type == LayerType.CONV1D:
        filters = []
        _, _, num_filters = weights.shape

        indices = filter_indices if filter_indices is not None else range(num_filters)

        for i in indices:
            if i < num_filters:
                filters.append(weights[:, :, i])

        return filters

    return None