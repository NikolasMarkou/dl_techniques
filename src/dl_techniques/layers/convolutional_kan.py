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

References
----------
    [1] Bodner et al. (2024). "Convolutional Kolmogorov-Arnold Networks"
    [2] Kolmogorov, A. N. (1957). "On the representation of continuous functions"
    [3] Liu, Ziming, et al. (2024). "KAN: Kolmogorov-Arnold Networks"
"""

import keras
import numpy as np
from keras import ops
from typing import Tuple, Optional, Union, Any, Dict, Callable

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class KANvolution(keras.layers.Layer):
    """
    Kolmogorov-Arnold Network convolution layer with learnable B-spline activations.

    This layer implements a convolutional operation where traditional fixed activation
    functions are replaced with learnable univariate functions based on B-splines.
    Each kernel element applies adaptive non-linear transformations that combine
    B-spline interpolation with SiLU activation for enhanced feature extraction.
    The KAN combination is K(x) = w_spline * B(x) + w_silu * SiLU(x), where B(x)
    uses linear interpolation over a grid of control points in [-1, 1] and input
    values are normalized using tanh for stable gradient flow.

    **Architecture Overview:**

    .. code-block:: text

        ┌─────────────────────────────────────────────────┐
        │  Input [batch, height, width, channels]         │
        └────────────────────┬────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  Patch Extraction: extract kernel-sized patches │
        └────────────────────┬────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  KAN Transformation                             │
        │  ┌──────────────────┬──────────────────────┐    │
        │  │  B-spline: Σ Nᵢ(x)·cᵢ │  SiLU: x·σ(x)   │    │
        │  └──────────┬───────┴──────────┬───────────┘    │
        │             ▼                  ▼                │
        │     w_spline·B(x)    +    w_silu·SiLU(x)        │
        └────────────────────┬────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  Convolution: apply transformed kernels         │
        └────────────────────┬────────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │  Bias Addition (optional) + Activation (optional)│
        └────────────────────┬─────────────────────────────┘
                             ▼
        ┌─────────────────────────────────────────────────┐
        │  Output [batch, new_height, new_width, filters] │
        └─────────────────────────────────────────────────┘

    :param filters: Number of output filters/channels. Must be positive.
    :type filters: int
    :param kernel_size: Spatial size of convolution kernel.
    :type kernel_size: Union[int, Tuple[int, int]]
    :param grid_size: Number of B-spline control points for learnable functions.
        Higher values allow more complex activation shapes. Must be > 1.
        Defaults to 16.
    :type grid_size: int
    :param strides: Stride length of convolution. Values > 1 reduce output size.
        Defaults to (1, 1).
    :type strides: Union[int, Tuple[int, int]]
    :param padding: Either 'valid' or 'same' (case-insensitive). Defaults to 'same'.
    :type padding: str
    :param dilation_rate: Dilation rate for convolution. Values > 1 create
        dilated/atrous convolution. Incompatible with strides > 1. Defaults to (1, 1).
    :type dilation_rate: Union[int, Tuple[int, int]]
    :param activation: Optional activation function applied after convolution.
        Can be string name or callable. None means linear. Defaults to None.
    :type activation: Optional[Union[str, Callable]]
    :param use_bias: Whether to add learnable bias vector to outputs. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weight matrices. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias vector. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer applied to all kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer applied to bias vector.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param activity_regularizer: Optional regularizer applied to layer output.
    :type activity_regularizer: Optional[keras.regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        grid_size: int = 16,
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = 'same',
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        activation: Optional[Union[str, Callable]] = None,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate required parameters
        if filters <= 0:
            raise ValueError(f"filters must be positive, got {filters}")
        if grid_size <= 1:
            raise ValueError(f"grid_size must be > 1, got {grid_size}")

        # Store ALL configuration for serialization
        self.filters = filters
        self.kernel_size = self._normalize_kernel_size(kernel_size)
        self.grid_size = grid_size
        self.strides = self._normalize_tuple(strides, 2, 'strides')
        self.padding = padding.lower()
        self.dilation_rate = self._normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias

        # Store serializable initializers and regularizers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Validate padding
        if self.padding not in ('valid', 'same'):
            raise ValueError(f"padding must be 'valid' or 'same', got {padding}")

        # Initialize weight attributes - created in build()
        self.control_points = None
        self.w_spline = None
        self.w_silu = None
        self.bias = None
        self.grid = None
        self._input_channels = None

    def _normalize_kernel_size(self, kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
        """Normalize kernel size to tuple format with validation.

        :param kernel_size: Kernel size as int or tuple.
        :type kernel_size: Union[int, Tuple[int, int]]
        :return: Normalized kernel size tuple.
        :rtype: Tuple[int, int]
        """
        if isinstance(kernel_size, int):
            if kernel_size <= 0:
                raise ValueError(f"kernel_size must be positive, got {kernel_size}")
            return (kernel_size, kernel_size)

        if not isinstance(kernel_size, (list, tuple)) or len(kernel_size) != 2:
            raise ValueError(f"kernel_size must be int or tuple of 2 ints, got {kernel_size}")

        if any(k <= 0 for k in kernel_size):
            raise ValueError(f"kernel_size values must be positive, got {kernel_size}")

        return tuple(kernel_size)

    def _normalize_tuple(
        self,
        value: Union[int, Tuple[int, int]],
        n: int,
        name: str
    ) -> Tuple[int, int]:
        """Normalize tuple parameters with validation.

        :param value: Value to normalize.
        :type value: Union[int, Tuple[int, int]]
        :param n: Expected tuple length.
        :type n: int
        :param name: Parameter name for error messages.
        :type name: str
        :return: Normalized tuple.
        :rtype: Tuple[int, int]
        """
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            return tuple([value] * n)

        if not isinstance(value, (list, tuple)) or len(value) != n:
            raise ValueError(f"{name} must be int or tuple of {n} ints, got {value}")

        if any(v <= 0 for v in value):
            raise ValueError(f"{name} values must be positive, got {value}")

        return tuple(value)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the layer's weights including B-spline control points and combination weights.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        logger.info(f"Building KANvolution layer with input shape: {input_shape}")

        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        input_channels = input_shape[-1]
        if input_channels is None:
            raise ValueError("Input channels dimension must be defined")
        self._input_channels = input_channels

        # Create B-spline control points for learnable univariate functions
        # Shape: (filters, input_channels, kernel_h, kernel_w, grid_size + 1)
        self.control_points = self.add_weight(
            name='control_points',
            shape=(self.filters, input_channels, *self.kernel_size, self.grid_size + 1),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Combination weights for B-spline component
        self.w_spline = self.add_weight(
            name='w_spline',
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Combination weights for SiLU component
        self.w_silu = self.add_weight(
            name='w_silu',
            shape=(self.filters, input_channels, *self.kernel_size),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        # Optional bias vector
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

        # Fixed grid points for B-spline interpolation in range [-1, 1].
        #
        # Materialized by an INITIALIZER, never by `add_weight(initializer='zeros')`
        # followed by `.assign()`. Keras 3 runs a symbolic build pass inside a
        # `StatelessScope` whenever this layer is first reached from a PARENT's
        # `call()` -- i.e. in every real model -- and that scope RECORDS the
        # `.assign()` and then DISCARDS it, leaving the knot vector all zeros: every
        # knot collapses onto a single point, so `_compute_bspline_basis` measures
        # every distance from the same location and the "learnable univariate
        # function" degenerates. Measured on CPU 2026-08-15: `grid[0]` is -1.0 on a
        # direct `.build(...)` and 0.0 through a parent layer's `call()`. See the
        # DECISION anchor in `layers/time_series/nbeats_blocks.py`
        # (`TrendBlock._create_polynomial_basis`) and decisions.md D-028.
        #
        # `np.linspace` (not `ops.linspace`) on purpose: a `keras.ops` tensor created
        # in `build()` belongs to the symbolic pass's scratch `FuncGraph` and raises
        # "out of scope" when the initializer later runs on the eager pass.
        grid_values = np.linspace(-1.0, 1.0, self.grid_size + 1, dtype='float32')
        self.grid = self.add_weight(
            name='grid',
            shape=(self.grid_size + 1,),
            initializer=lambda shape, dtype=None: ops.cast(
                ops.convert_to_tensor(grid_values), dtype or self.variable_dtype
            ),
            trainable=False,
        )

        logger.info("KANvolution layer built successfully")

        super().build(input_shape)

    def _compute_bspline_basis(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Compute linear B-spline basis functions for input values.

        Uses linear B-splines (degree 1) for computational efficiency while
        maintaining the learnable univariate function property of KANs.

        :param x: Input tensor values, should be normalized to [-1, 1] range.
        :type x: keras.KerasTensor
        :return: Basis function weights for each grid point.
        :rtype: keras.KerasTensor
        """
        # Clamp input to valid grid range for numerical stability
        x_clamped = ops.clip(x, -1.0, 1.0)

        # Expand dimensions for broadcasting with grid points
        x_expanded = ops.expand_dims(x_clamped, axis=-1)

        # Compute distances from each grid point
        grid_expanded = ops.expand_dims(self.grid, axis=0)
        distances = ops.abs(x_expanded - grid_expanded)

        # Linear B-spline basis: weight = max(0, 1 - distance / spacing)
        grid_spacing = 2.0 / self.grid_size  # Spacing in [-1, 1] range
        weights = ops.maximum(0.0, 1.0 - distances / grid_spacing)

        # Normalize weights to ensure they sum to 1
        weight_sum = ops.sum(weights, axis=-1, keepdims=True)
        normalized_weights = weights / (weight_sum + 1e-8)

        return normalized_weights

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass applying KAN transformation followed by convolution.

        Computes, per convolution tap, ``K(x) = w_spline * B(x) + w_silu * SiLU(x)``
        where ``B`` is the linear B-spline interpolation over ``control_points``
        and ``x`` is the tap value squashed into ``[-1, 1]`` by ``tanh``.

        :param inputs: Input tensor with shape (batch_size, height, width, channels).
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode (unused in this layer).
        :type training: Optional[bool]
        :return: Output tensor with shape (batch_size, new_height, new_width, filters).
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-08-22T035419-a11304c8/D-052
        # Do NOT "simplify" this back to `ops.conv(inputs, w_spline + w_silu)`.
        # That is what stood here until 2026-08-22 and it is not a KAN: it never
        # called `_compute_bspline_basis`, never read `control_points` and never
        # read `grid`, so the whole learnable-univariate-function apparatus this
        # module is named for was decorative. MEASURED at that revision, on CPU:
        # the layer's output equalled `ops.conv` with kernel `w_spline + w_silu`
        # to `max|delta| = 0.0` EXACTLY; `|f(2x) - 2f(x)| = 0.0` EXACTLY, i.e.
        # the layer was degree-1 homogeneous and therefore carried no
        # non-linearity at all, not even the advertised SiLU; and after one real
        # `SGD(lr=1.0)` step `control_points` moved `0.000000e+00` while
        # `w_spline` and `w_silu` moved by the IDENTICAL `1.221391e+00`,
        # confirming they were a redundant reparameterization of one kernel.
        # The cost of doing it properly is the `basis` tensor below, which is
        # `(batch, out_h, out_w, kh*kw*channels, grid_size + 1)` -- that is
        # inherent to a convolutional KAN, not an artifact of this code. Use a
        # small `grid_size` on large inputs. See decisions.md D-052.
        num_taps = self.kernel_size[0] * self.kernel_size[1] * self._input_channels

        # Patches in the SAME (kernel_h, kernel_w, channels) tap order that the
        # weight reshapes below assume.
        patches = ops.image.extract_patches(
            inputs,
            size=self.kernel_size,
            strides=self.strides,
            dilation_rate=self.dilation_rate,
            padding=self.padding,
        )

        # Normalize tap values into the spline's [-1, 1] grid domain. tanh is the
        # squashing the module docstring specifies, and it is what makes the
        # B-spline basis well defined for unbounded inputs.
        taps = ops.tanh(patches)

        # (batch, out_h, out_w, num_taps, grid_size + 1)
        basis = self._compute_bspline_basis(taps)

        # Weights are stored (filters, channels, kernel_h, kernel_w[, grid]);
        # transpose to (filters, kernel_h, kernel_w, channels[, grid]) so that
        # flattening matches `extract_patches`' tap order exactly.
        w_spline_flat = ops.reshape(
            ops.transpose(self.w_spline, (0, 2, 3, 1)), (self.filters, num_taps)
        )
        w_silu_flat = ops.reshape(
            ops.transpose(self.w_silu, (0, 2, 3, 1)), (self.filters, num_taps)
        )
        control_flat = ops.reshape(
            ops.transpose(self.control_points, (0, 2, 3, 1, 4)),
            (self.filters, num_taps, self.grid_size + 1),
        )

        # w_spline * B(x) folded into one contraction: the per-tap spline
        # coefficients are `w_spline[f, m] * control_points[f, m, :]`.
        spline_coeffs = ops.expand_dims(w_spline_flat, axis=-1) * control_flat
        spline_term = ops.einsum('bhwmg,fmg->bhwf', basis, spline_coeffs)

        # w_silu * SiLU(x) on the same normalized taps.
        silu_term = ops.einsum('bhwm,fm->bhwf', ops.silu(taps), w_silu_flat)

        outputs = spline_term + silu_term

        # Add bias if enabled
        if self.use_bias:
            outputs = ops.add(outputs, self.bias)

        # Apply post-convolution activation if specified
        if self.activation is not None:
            activation_fn = keras.activations.get(self.activation)
            outputs = activation_fn(outputs)

        return outputs

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape based on input shape and layer parameters.

        :param input_shape: Input tensor shape (batch_size, height, width, channels).
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (batch_size, new_height, new_width, filters).
        :rtype: Tuple[Optional[int], ...]
        """
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")

        batch_size, height, width, _ = input_shape

        # Compute spatial dimensions based on convolution parameters
        if self.padding == 'same':
            # Same padding: output_size = ceil(input_size / stride)
            if height is not None:
                out_height = (height + self.strides[0] - 1) // self.strides[0]
            else:
                out_height = None

            if width is not None:
                out_width = (width + self.strides[1] - 1) // self.strides[1]
            else:
                out_width = None
        else:  # 'valid' padding
            # Valid padding: output_size = ceil((input_size - kernel_size + 1) / stride)
            if height is not None:
                out_height = max(0, (height - self.kernel_size[0]) // self.strides[0] + 1)
            else:
                out_height = None

            if width is not None:
                out_width = max(0, (width - self.kernel_size[1]) // self.strides[1] + 1)
            else:
                out_width = None

        return (batch_size, out_height, out_width, self.filters)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all layer configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'grid_size': self.grid_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': serialize_activation(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
