"""
Provide utilities to adapt Keras layer weights for spectral analysis.

This module serves as a crucial bridge between the high-level Keras API and
the low-level mathematical requirements of spectral analysis. The core
challenge in analyzing neural network weights is that different layer types
(Dense, Conv2D, Embedding, etc.) store their learnable parameters in
tensors of varying shapes and semantics. Spectral methods, however,
fundamentally operate on 2D matrices.

Architecture and Purpose
-----------------------
The primary architectural function of this module is to act as an adapter. It
abstracts away the heterogeneity of Keras layers, providing a standardized
interface for the `spectral_metrics` module. It inspects a given Keras
layer, identifies its type, and intelligently extracts and reshapes its
weight tensor into a 2D matrix representation upon which eigenvalue
decomposition can be performed. This allows the core analysis logic to remain
agnostic to the specifics of layer architecture.

Foundational Mathematics: Tensor Matricization
----------------------------------------------
The most critical transformation occurs in `get_weight_matrices` for
convolutional layers. A standard Conv2D layer's weights are stored in a 4D
tensor of shape `(kernel_height, kernel_width, in_channels, out_channels)`.
To analyze its spectral properties, this tensor must be "unfolded" or
"matricized" into a 2D matrix.

This module implements a standard matricization where the tensor is reshaped
into a matrix `W_matrix` of shape
`(kernel_height * kernel_width * in_channels, out_channels)`. This reshaping
is not arbitrary; the resulting matrix represents the linear transformation
applied by the convolutional filters. The spectrum of this matrix (i.e., its
singular values) captures the properties of this transformation, such as its
effective rank and the distribution of its principal components. The product
of the kernel dimensions (`kernel_height * kernel_width`) is also extracted
as the receptive field size (`rf`), which can be used for normalization in
more advanced spectral analyses.

References
----------
1.  Rahaman, N., et al. (2019). "On the Spectral Bias of Neural Networks."
    ICML. (Discusses the Fourier analysis of neural network layers, which
    relates to the matricization of convolutional kernels).
2.  Martin, C., & Mahoney, M. W. (2021). "Heavy-Tailed Universals in
    Deep Neural Networks." arXiv preprint arXiv:2106.07590.

"""

import keras
import numpy as np
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.analyzer.constants import LayerType

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
    except Exception:
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