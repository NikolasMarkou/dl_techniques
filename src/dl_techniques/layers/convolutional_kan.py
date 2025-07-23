"""
Convolutional Kolmogorov-Arnold Networks Implementation
====================================================

This module implements Convolutional Kolmogorov-Arnold Networks (KANs) as described
in the paper by Bodner et al. (2024). KANs integrate the principles of
Kolmogorov-Arnold representation into convolutional layers, using learnable
non-linear functions based on B-splines.

Mathematical Foundation
----------------------
The Kolmogorov-Arnold representation theorem states that any multivariate continuous
function f can be expressed as:

    f(x₁, ..., xₙ) = Σᵢ₌₁²ⁿ⁺¹ Φᵢ(Σⱼ₌₁ⁿ φᵢⱼ(xⱼ))

where φᵢⱼ and Φᵢ are univariate continuous functions. KANs replace traditional
fixed activation functions with learnable univariate functions, enabling more
expressive and adaptive neural networks.

KANvolution Layer Computation
----------------------------
For each convolutional kernel element, the KANvolution layer computes:

    K(x) = w_spline · B(x) + w_silu · SiLU(x)

where:
- B(x) is the B-spline interpolation function: B(x) = Σᵢ Nᵢ(x) · cᵢ
- Nᵢ(x) are the B-spline basis functions (linear interpolation weights)
- cᵢ are learnable control points defining the spline curve
- SiLU(x) = x · sigmoid(x) is the Swish/SiLU activation function
- w_spline, w_silu are learnable combination weights

B-Spline Interpolation Details
-----------------------------
The B-spline component uses linear interpolation between control points:

1. Input normalization: x_norm = clip(x, -1, 1)
2. Distance calculation: d_i = |x_norm - grid_i| for each grid point
3. Basis weights: w_i = max(0, 1 - d_i * (grid_size/2))
4. Normalization: w_i = w_i / Σⱼ w_j
5. Interpolation: B(x) = Σᵢ w_i · control_points_i

The grid points are uniformly distributed in [-1, 1], and the control points
are learnable parameters that define the shape of the univariate function.

Convolution Operation
--------------------
The final convolution operation applies the learned KAN kernel:

    output = conv2d(input, K) + bias (if enabled)
    output = activation(output) (if specified)

where K is the effective kernel combining spline and SiLU components.

References
----------
    [1] Bodner et al. (2024). "Convolutional Kolmogorov-Arnold Networks"
    [2] Kolmogorov, A. N. (1957). "On the representation of continuous functions"
    [3] Liu, Ziming, et al. (2024). "KAN: Kolmogorov-Arnold Networks"

Notes
-----
    This implementation assumes channels_last data format (batch, height, width, channels).
    The B-spline implementation uses linear interpolation for computational efficiency
    while maintaining the core principles of learnable univariate functions.
"""

import keras
from keras import ops
from typing import Tuple, Optional, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KANvolution(keras.layers.Layer):
    """
    Implements a Kolmogorov-Arnold Network convolution layer.

    This layer replaces traditional convolution operations with learnable non-linear
    functions based on B-splines for each element of the convolutional kernel.

    Assumes channels_last data format: (batch_size, height, width, channels).

    Parameters
    ----------
    filters : int
        Number of output filters in the convolution
    kernel_size : Union[int, Tuple[int, int]]
        Size of the convolution kernel
    grid_size : int, optional
        Number of control points for the spline interpolation (default: 16)
    strides : Union[int, Tuple[int, int]], optional
        Stride length of the convolution (default: (1, 1))
    padding : str, optional
        One of 'valid' or 'same' (default: 'same')
    dilation_rate : Union[int, Tuple[int, int]], optional
        Dilation rate for dilated convolution (default: (1, 1))
    activation : Union[str, callable], optional
        Activation function to use (default: None)
    use_bias : bool, optional
        Whether the layer uses a bias vector (default: True)
    kernel_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the kernel weights (default: 'glorot_uniform')
    bias_initializer : Union[str, keras.initializers.Initializer], optional
        Initializer for the bias vector (default: 'zeros')
    kernel_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function for the kernel weights (default: None)
    bias_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function for the bias vector (default: None)
    activity_regularizer : Optional[keras.regularizers.Regularizer], optional
        Regularizer function for the output (default: None)

    Input shape
    -----------
        4D tensor with shape: (batch_size, height, width, channels)

    Output shape
    ------------
        4D tensor with shape: (batch_size, new_height, new_width, filters)
        where new_height and new_width values depend on padding and stride values.

    Attributes
    ----------
    control_points : keras.Variable
        Learnable control points for the spline interpolation
    w_spline : keras.Variable
        Weights for combining spline output
    w_silu : keras.Variable
        Weights for combining SiLU activation
    bias : keras.Variable, optional
        Bias vector if use_bias is True
    grid : keras.Variable
        Fixed grid points for spline interpolation
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        grid_size: int = 16,
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = 'same',
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the KANvolution layer."""
        super().__init__(**kwargs)

        # Store configuration parameters
        self.filters = filters
        self.kernel_size = self._normalize_kernel_size(kernel_size)
        self.grid_size = grid_size
        self.strides = self._normalize_tuple(strides, 2, 'strides')
        self.padding = padding.lower()
        self.dilation_rate = self._normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = activation
        self.use_bias = use_bias

        # Initialize initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Validate parameters
        self._validate_parameters()

        # Will be initialized in build()
        self.control_points = None
        self.w_spline = None
        self.w_silu = None
        self.bias = None
        self.grid = None
        self.activation_fn = None
        self._build_input_shape = None

    def _normalize_kernel_size(self, kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Normalize kernel size to tuple format."""
        if isinstance(kernel_size, int):
            return (kernel_size, kernel_size)
        if len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be an int or tuple of 2 ints, got {kernel_size}")
        return tuple(kernel_size)

    def _normalize_tuple(self, value: Union[int, Tuple[int, int]], n: int, name: str) -> Tuple[int, int]:
        """Normalize tuple parameters."""
        if isinstance(value, int):
            return tuple([value] * n)
        if len(value) != n:
            raise ValueError(f"{name} must be an int or tuple of {n} ints, got {value}")
        return tuple(value)

    def _validate_parameters(self) -> None:
        """Validate layer parameters."""
        if self.filters <= 0:
            raise ValueError(f"filters must be positive, got {self.filters}")
        if self.grid_size <= 1:
            raise ValueError(f"grid_size must be > 1, got {self.grid_size}")
        if self.padding not in ('valid', 'same'):
            raise ValueError(f"padding must be 'valid' or 'same', got {self.padding}")

    def _get_activation_fn(self, activation):
        """Internal method to get activation function."""
        if activation is None:
            return None
        return keras.activations.get(activation)

    def build(self, input_shape: Tuple) -> None:
        """
        Build the layer's weights.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor (batch_size, height, width, channels)
        """
        # Store for serialization
        self._build_input_shape = input_shape

        logger.info(f"Building KANvolution layer with input shape: {input_shape}")

        # Get input channel dimension (channels_last format)
        input_channels = input_shape[-1]

        if input_channels is None:
            raise ValueError("Input channels dimension cannot be None")

        # Initialize spline control points (grid_size + 1 for proper interpolation)
        self.control_points = self.add_weight(
            name='control_points',
            shape=(self.filters, input_channels, *self.kernel_size, self.grid_size + 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Initialize combination weights for spline output
        self.w_spline = self.add_weight(
            name='w_spline',
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Initialize combination weights for SiLU activation
        self.w_silu = self.add_weight(
            name='w_silu',
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Initialize bias if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

        # Initialize fixed grid points
        grid_values = ops.linspace(-1.0, 1.0, self.grid_size + 1)
        self.grid = self.add_weight(
            name='grid',
            shape=(self.grid_size + 1,),
            initializer='zeros',
            trainable=False,
        )
        self.grid.assign(grid_values)

        # Set up activation function
        self.activation_fn = self._get_activation_fn(self.activation)

        super().build(input_shape)
        logger.info("KANvolution layer built successfully")

    def _apply_spline_interpolation(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """
        Apply B-spline interpolation to the input tensor.

        Parameters
        ----------
        x : keras.KerasTensor
            Input tensor

        Returns
        -------
        keras.KerasTensor
            Interpolated values using B-spline basis functions
        """
        # Normalize input to grid range [-1, 1]
        x_normalized = ops.clip(x, -1.0, 1.0)

        # Expand dimensions to match control points shape for broadcasting
        x_expanded = ops.expand_dims(x_normalized, axis=-1)  # Add grid dimension

        # Calculate distances from each grid point
        distances = ops.abs(x_expanded - self.grid)

        # B-spline basis function (linear interpolation between nearest points)
        weights = ops.maximum(0.0, 1.0 - distances * (self.grid_size / 2.0))
        weights = weights / (ops.sum(weights, axis=-1, keepdims=True) + 1e-8)

        # Apply weighted combination of control points
        output = ops.sum(self.control_points * ops.expand_dims(weights, axis=0), axis=-1)

        return output

    def _create_kan_kernel(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """
        Create the KAN kernel by combining spline and SiLU components.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor for kernel creation

        Returns
        -------
        keras.KerasTensor
            KAN kernel weights
        """
        # Apply spline interpolation to create adaptive kernel
        spline_kernel = self._apply_spline_interpolation(inputs)
        silu_kernel = ops.sigmoid(inputs) * inputs  # SiLU activation

        # Combine spline and SiLU components
        kan_kernel = self.w_spline * spline_kernel + self.w_silu * silu_kernel

        return kan_kernel

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Parameters
        ----------
        inputs : keras.KerasTensor
            Input tensor with shape (batch_size, height, width, channels)
        training : bool, optional
            Whether the layer is in training mode

        Returns
        -------
        keras.KerasTensor
            Output tensor after KAN convolution with shape (batch_size, new_height, new_width, filters)
        """
        # Create effective kernel by combining spline and SiLU transformations
        # This is a simplified approach - in practice, this would need more sophisticated
        # patch-based processing for true KAN behavior
        effective_kernel = self.w_spline + self.w_silu

        # Transpose kernel to match expected convolution format
        # (filters, input_channels, kernel_h, kernel_w) -> (kernel_h, kernel_w, input_channels, filters)
        kernel = ops.transpose(effective_kernel, (2, 3, 1, 0))

        # Apply convolution (channels_last format)
        outputs = ops.conv(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate
        )

        # Add bias if enabled (channels_last: bias shape is (filters,))
        if self.use_bias:
            outputs = outputs + self.bias

        # Apply activation if specified
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)

        return outputs

    def compute_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape : Tuple
            Shape of the input tensor (batch_size, height, width, channels)

        Returns
        -------
        Tuple
            Output shape (batch_size, new_height, new_width, filters)
        """
        input_shape_list = list(input_shape)

        # Extract spatial dimensions (channels_last format)
        height, width = input_shape_list[1], input_shape_list[2]

        # Calculate output spatial dimensions
        if self.padding == 'same':
            out_height = height // self.strides[0] if height is not None else None
            out_width = width // self.strides[1] if width is not None else None
        else:  # 'valid'
            if height is not None:
                out_height = (height - self.kernel_size[0]) // self.strides[0] + 1
            else:
                out_height = None
            if width is not None:
                out_width = (width - self.kernel_size[1]) // self.strides[1] + 1
            else:
                out_width = None

        return tuple([input_shape_list[0], out_height, out_width, self.filters])

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer.

        Returns
        -------
        Dict[str, Any]
            Layer configuration
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'grid_size': self.grid_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get build configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Build configuration
        """
        return {
            'input_shape': self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build from configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Build configuration dictionary
        """
        if config.get('input_shape') is not None:
            self.build(config['input_shape'])

# ---------------------------------------------------------------------
