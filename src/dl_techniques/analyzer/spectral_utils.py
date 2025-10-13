"""
Utility functions for Keras weight matrix spectral analysis.

This module provides utility functions for analyzing weight matrices in Keras models,
including layer type inference, weight extraction, and visualization helpers.
"""

import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.analyzer.constants import (
    LayerType, SPECTRAL_DEFAULT_BINS, SPECTRAL_DEFAULT_FIG_SIZE, SPECTRAL_DEFAULT_DPI
)

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

def infer_layer_type(layer: keras.layers.Layer) -> LayerType:
    """
    Determine the layer type for a given Keras layer.

    Args:
        layer: Keras layer to analyze.

    Returns:
        LayerType: The inferred type of the layer.
    """
    layer_class = layer.__class__.__name__.lower()

    # Check by instance type first (more reliable)
    if isinstance(layer, keras.layers.Dense):
        return LayerType.DENSE
    elif isinstance(layer, keras.layers.Conv1D):
        return LayerType.CONV1D
    elif isinstance(layer, keras.layers.Conv2D):
        return LayerType.CONV2D
    elif isinstance(layer, keras.layers.Conv3D):
        return LayerType.CONV3D
    elif isinstance(layer, keras.layers.Embedding):
        return LayerType.EMBEDDING
    elif isinstance(layer, keras.layers.LSTM):
        return LayerType.LSTM
    elif isinstance(layer, keras.layers.GRU):
        return LayerType.GRU
    elif isinstance(layer, (keras.layers.LayerNormalization, keras.layers.BatchNormalization)):
        return LayerType.NORM
    # Fallback to string matching
    elif 'dense' in layer_class:
        return LayerType.DENSE
    elif 'conv1d' in layer_class:
        return LayerType.CONV1D
    elif 'conv2d' in layer_class:
        return LayerType.CONV2D
    elif 'conv3d' in layer_class:
        return LayerType.CONV3D
    elif 'embedding' in layer_class:
        return LayerType.EMBEDDING
    elif 'lstm' in layer_class:
        return LayerType.LSTM
    elif 'gru' in layer_class:
        return LayerType.GRU
    elif any(norm_type in layer_class for norm_type in ['layernorm', 'batchnorm', 'groupnorm']):
        return LayerType.NORM
    else:
        return LayerType.UNKNOWN

# ---------------------------------------------------------------------

def get_layer_weights_and_bias(layer: keras.layers.Layer) -> Tuple[bool, Optional[np.ndarray], bool, Optional[np.ndarray]]:
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
    try:
        weights_list = layer.get_weights()
    except:
        return has_weights, weights, has_bias, bias

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
        elif layer_type in [LayerType.LSTM, LayerType.GRU]:
            # For RNN layers, we typically have multiple weight matrices
            # For simplicity, we'll take the first one (input weights)
            if len(weights_list) >= 1:
                has_weights = True
                weights = weights_list[0]

    return has_weights, weights, has_bias, bias

# ---------------------------------------------------------------------

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
        kernel_size, input_dim, output_dim = weights.shape
        rf = kernel_size

        # Reshape to 2D matrix for eigenvalue analysis
        weights_reshaped = weights.reshape(-1, output_dim)
        N, M = max(weights_reshaped.shape), min(weights_reshaped.shape)
        Wmats = [weights_reshaped]

    elif layer_type == LayerType.CONV2D:
        # Conv2D weights shape: (kernel_height, kernel_width, input_channels, output_channels)
        kh, kw, in_c, out_c = weights.shape
        rf = kh * kw

        # For analysis, we can either:
        # 1. Reshape the entire tensor to 2D (simpler)
        # 2. Extract individual filter matrices (more detailed)

        # Option 1: Reshape entire tensor (more common in practice)
        weights_reshaped = weights.reshape(-1, out_c)
        N, M = max(weights_reshaped.shape), min(weights_reshaped.shape)
        Wmats = [weights_reshaped]

    elif layer_type == LayerType.CONV3D:
        # Conv3D weights shape: (kernel_d, kernel_h, kernel_w, input_channels, output_channels)
        kd, kh, kw, in_c, out_c = weights.shape
        rf = kd * kh * kw

        weights_reshaped = weights.reshape(-1, out_c)
        N, M = max(weights_reshaped.shape), min(weights_reshaped.shape)
        Wmats = [weights_reshaped]

    elif layer_type in [LayerType.LSTM, LayerType.GRU]:
        # RNN weights are typically 2D: (input_dim + hidden_dim, hidden_dim * gates)
        N, M = max(weights.shape), min(weights.shape)
        Wmats = [weights]

    return Wmats, N, M, rf

# ---------------------------------------------------------------------

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

    try:
        fig = plt.figure(figsize=figsize)

        if layer_type == LayerType.DENSE:
            plt.imshow(weights, cmap=cmap, aspect='auto')
            plt.colorbar(label='Weight Value')
            plt.title(f"Weight Matrix for {layer.name} (Dense Layer)")
            plt.xlabel("Output Units")
            plt.ylabel("Input Units")

        elif layer_type == LayerType.CONV2D:
            kh, kw, in_c, out_c = weights.shape
            grid_size = int(np.ceil(np.sqrt(min(out_c, 16))))  # Limit to 16 filters for clarity
            plt.suptitle(f"Filters for {layer.name} (Conv2D Layer) - Averaged over {in_c} Input Channels")

            for i in range(min(out_c, grid_size * grid_size)):
                ax = plt.subplot(grid_size, grid_size, i + 1)
                # Display the filter averaged over the input channels to get a
                # comprehensive view of the filter's spatial pattern.
                filter_visualization = np.mean(weights[:, :, :, i], axis=2)
                ax.imshow(filter_visualization, cmap=cmap)
                ax.axis('off')
                ax.set_title(f"Filter {i}", fontsize=8)


        elif layer_type == LayerType.CONV1D:
            kernel_size, in_c, out_c = weights.shape
            num_filters_to_show = min(out_c, 8)
            plt.suptitle(f"Filters for {layer.name} (Conv1D) - Averaged over {in_c} Input Channels")

            for i in range(num_filters_to_show):
                ax = plt.subplot(2, 4, i + 1)
                # Plot the filter averaged across all input channels.
                filter_visualization = np.mean(weights[:, :, i], axis=1)
                ax.plot(filter_visualization)
                ax.set_title(f"Filter {i}")
                ax.set_xlabel("Position")
                ax.set_ylabel("Weight")

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.warning(f"Error creating visualization for layer {layer_index}: {e}")
        return None

# ---------------------------------------------------------------------

def compute_weight_statistics(
        model: keras.Model,
        include_layers: Optional[List[int]] = None
) -> Dict[int, Dict[str, any]]:
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
            'type': infer_layer_type(layer).value,
            'weight_shape': weights.shape,
            'weight_mean': float(np.mean(weights)),
            'weight_std': float(np.std(weights)),
            'weight_min': float(np.min(weights)),
            'weight_max': float(np.max(weights)),
            'weight_median': float(np.median(weights)),
            'weight_sparsity': float(np.sum(weights == 0) / weights.size),
            'weight_l1_norm': float(np.sum(np.abs(weights))),
            'weight_l2_norm': float(np.sqrt(np.sum(weights ** 2))),
        }

        if has_bias:
            layer_stats.update({
                'bias_shape': bias.shape,
                'bias_mean': float(np.mean(bias)),
                'bias_std': float(np.std(bias)),
                'bias_min': float(np.min(bias)),
                'bias_max': float(np.max(bias)),
                'bias_l1_norm': float(np.sum(np.abs(bias))),
                'bias_l2_norm': float(np.sqrt(np.sum(bias ** 2))),
            })

        stats[layer_id] = layer_stats

    return stats

# ---------------------------------------------------------------------

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

    try:
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

        # For Conv3D: weights shape is (kd, kh, kw, in_channels, out_channels)
        elif layer_type == LayerType.CONV3D:
            filters = []
            _, _, _, _, num_filters = weights.shape

            indices = filter_indices if filter_indices is not None else range(num_filters)

            for i in indices:
                if i < num_filters:
                    filters.append(weights[:, :, :, :, i])

            return filters

    except Exception as e:
        logger.warning(f"Error extracting filters: {e}")

    return None