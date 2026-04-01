"""
Zero-Centered Band RMS Normalization Layer for Enhanced Training Stability

This module implements Zero-Centered Band RMS Normalization, an advanced normalization
technique that combines three key innovations:
1. Zero-centering from ZeroCenteredRMSNorm for training stability
2. RMS normalization for computational efficiency and dimension independence
3. Band constraint from BandRMS for controlled representational flexibility

Mathematical Formulation:
    Given an input tensor x with shape (..., d), Zero-Centered Band RMS normalization:

    mu = mean(x) over specified axes
    x_centered = x - mu
    RMS(x_centered) = sqrt(mean(x_centered^2) + epsilon)
    x_norm = x_centered / RMS(x_centered)
    s = sigmoid(5.0 * band_param) * max_band_width + (1 - max_band_width)
    output = x_norm * s

    Where:
    - mu is the mean computed over specified axes (centering step)
    - RMS is computed from the centered input for stability
    - s is the learnable scaling factor constrained to [1-alpha, 1] band
    - alpha is the max_band_width parameter

Key Benefits:
    - Prevents mean drift and abnormal weight growth (zero-centering)
    - Maintains dimension-independent scaling (RMS normalization)
    - Provides controlled representational flexibility (band constraint)
    - Combines stability with expressiveness
    - Particularly effective for transformer architectures and large language models

References:
    - Builds upon ZeroCenteredRMSNorm concepts from Qwen3-Next
    - Extends BandRMS band constraint methodology
    - Combines benefits of LayerNorm stability with RMSNorm efficiency
"""

import keras
from keras import ops, initializers, regularizers
from typing import Any, Dict, Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ZeroCenteredBandRMSNorm(keras.layers.Layer):
    """
    Zero-Centered Root Mean Square Normalization with learnable band constraints.

    This layer implements a hybrid normalization approach that combines zero-centering
    for training stability with band-constrained RMS scaling for representational
    flexibility. It first centers inputs around zero, then normalizes by RMS, and
    finally applies learnable scaling within a constrained [1-alpha, 1] band.

    The normalization is computed as:

    1. Centering: mu = E[x], x_centered = x - mu
    2. RMS Computation: rms = sqrt(E[x_centered^2] + epsilon)
    3. Normalization: x_hat = x_centered / rms
    4. Band Scaling: s = sigmoid(5*beta) * alpha + (1-alpha), output = s * x_hat

    Where mu is computed per feature across normalization axes, rms is computed from
    centered inputs for enhanced stability, s is the learnable band scaling factor
    constrained to [1-alpha, 1], beta is the trainable band parameter, alpha is
    the max_band_width hyperparameter, and epsilon is a small constant for numerical
    stability.

    This creates a "thick shell" in the normalized space while maintaining zero-mean
    property, combining the benefits of LayerNorm stability, RMSNorm efficiency,
    and BandRMS representational flexibility.

    **Architecture Overview:**

    .. code-block:: text

        Input: x (batch, ..., features)
                │
                ▼
        ┌───────────────────────────────┐
        │  μ = mean(x) along axis       │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  x_centered = x - μ           │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  RMS = max(√(mean(x_c²)+ε),ε)│
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  normalized = x_centered / RMS│
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  σ = sigmoid(5 × band_param)  │
        │  scale = (1-α) + α × σ       │
        │  scale ∈ [1-α, 1]            │
        └────────────┬──────────────────┘
                     │
                     ▼
        ┌───────────────────────────────┐
        │  output = normalized × scale  │
        └────────────┬──────────────────┘
                     │
                     ▼
        Output: (batch, ..., features)

    :param max_band_width: Maximum allowed deviation from unit normalization
        (0 < alpha < 1). Controls the thickness of the representational band.
        When alpha=0.1, the output RMS will be constrained to [0.9, 1.0].
    :type max_band_width: float
    :param axis: Axis or axes along which to compute mean and RMS statistics.
        The default (-1) computes statistics over the last dimension. For multi-axis
        normalization, pass a tuple (e.g., (-2, -1) for normalizing over last two
        dimensions).
    :type axis: Union[int, Tuple[int, ...]]
    :param epsilon: Small constant added to denominator for numerical stability.
        Should be positive and typically in range [1e-8, 1e-5].
    :type epsilon: float
    :param band_initializer: Initializer for the band parameter. The band parameter
        controls the learned position within the [1-alpha, 1] constraint band.
        Common choices: "zeros" (start at lower bound), "ones" (start at upper bound).
    :type band_initializer: Union[str, initializers.Initializer]
    :param band_regularizer: Optional regularizer for the band parameter.
        Helps prevent the band parameter from becoming too extreme. If None,
        defaults to L2(1e-5) regularizer for stability.
    :type band_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments passed to the parent Layer class.

    :raises ValueError: If max_band_width is not between 0 and 1.
    :raises ValueError: If epsilon is not positive.
    :raises TypeError: If axis is not int or tuple of ints.
    """

    def __init__(
            self,
            max_band_width: float = 0.1,
            axis: Union[int, Tuple[int, ...]] = -1,
            epsilon: float = 1e-7,
            band_initializer: Union[str, initializers.Initializer] = "zeros",
            band_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the ZeroCenteredBandRMSNorm layer.

        :param max_band_width: Maximum deviation from unit normalization (0 < alpha < 1).
        :type max_band_width: float
        :param axis: Axis or axes along which to compute statistics.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float
        :param band_initializer: Initializer for the band parameter.
        :type band_initializer: Union[str, initializers.Initializer]
        :param band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
        :type band_regularizer: Optional[regularizers.Regularizer]
        :param kwargs: Additional layer arguments.
        :raises ValueError: If max_band_width is not between 0 and 1 or if epsilon
            is not positive.
        :raises TypeError: If axis is not int or tuple of ints.
        """
        super().__init__(**kwargs)

        # Validate inputs early
        self._validate_inputs(max_band_width, axis, epsilon)

        # Store ALL configuration parameters - required for get_config()
        self.max_band_width = max_band_width
        self.axis = axis
        self.epsilon = epsilon
        self.band_initializer = initializers.get(band_initializer)

        # Default regularizer if none provided
        self.band_regularizer = band_regularizer or regularizers.L2(1e-5)

        # Initialize weight attributes - created in build()
        self.band_param = None

        logger.debug(
            f"Initialized ZeroCenteredBandRMSNorm with max_band_width={max_band_width}, "
            f"axis={axis}, epsilon={epsilon}"
        )

    def _validate_inputs(
            self,
            max_band_width: float,
            axis: Union[int, Tuple[int, ...]],
            epsilon: float
    ) -> None:
        """
        Validate initialization parameters.

        :param max_band_width: Maximum allowed deviation from unit norm.
        :type max_band_width: float
        :param axis: Normalization axis/axes to validate.
        :type axis: Union[int, Tuple[int, ...]]
        :param epsilon: Small constant for numerical stability.
        :type epsilon: float
        :raises ValueError: If parameters are invalid.
        :raises TypeError: If axis type is invalid.
        """
        if not 0 < max_band_width < 1:
            raise ValueError(
                f"max_band_width must be between 0 and 1, got {max_band_width}"
            )

        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Validate axis type
        if isinstance(axis, (list, tuple)):
            if not all(isinstance(ax, int) for ax in axis):
                raise TypeError(f"All elements in axis must be integers, got {axis}")
        elif not isinstance(axis, int):
            raise TypeError(f"axis must be int or tuple of ints, got {type(axis)}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's own weights.

        This is called automatically when the layer first processes input.
        Following modern Keras 3 Pattern 1: Simple Layer (No Sub-layers).

        :param input_shape: Shape tuple indicating input tensor shape.
            First dimension (batch size) may be None.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Create a single scalar band parameter using add_weight()
        # This parameter controls the learned position within the [1-α, 1] band
        self.band_param = self.add_weight(
            name="band_param",
            shape=(),  # Scalar shape
            initializer=self.band_initializer,
            trainable=True,
            regularizer=self.band_regularizer
        )

        logger.debug("Created scalar band parameter for ZeroCenteredBandRMSNorm")

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply Zero-Centered Band RMS normalization to inputs.

        :param inputs: Input tensor of any shape. Normalization is applied along
            the axes specified during initialization.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating whether in training mode. Not used
            in this layer but kept for consistency with other normalization layers.
        :type training: Optional[bool]
        :return: Zero-centered band RMS normalized tensor with the same shape as
            inputs. The output RMS will be constrained to [1-max_band_width, 1]
            range.
        :rtype: keras.KerasTensor
        """
        # Store original dtype for casting back
        original_dtype = inputs.dtype

        # Cast to float32 for numerical stability in mixed precision training
        inputs_fp32 = ops.cast(inputs, "float32")

        # Step 1: Compute mean and center the input (zero-centering innovation)
        mean = ops.mean(
            inputs_fp32,
            axis=self.axis,
            keepdims=True
        )

        centered_inputs = inputs_fp32 - mean

        # Step 2: Compute RMS of centered inputs for dimension-independent scaling
        # Using mean(x²) instead of sum(x²) ensures normalization is independent
        # of vector dimension - critical for consistent behavior across layer widths
        mean_square = ops.mean(
            ops.square(centered_inputs),
            axis=self.axis,
            keepdims=True
        )

        # Clamp and compute RMS for stability
        rms = ops.maximum(
            ops.sqrt(mean_square + self.epsilon),
            self.epsilon
        )

        # Step 3: Normalize by RMS to achieve RMS=1 and L2_norm≈sqrt(D)
        normalized = centered_inputs / rms

        # Step 4: Apply learnable band scaling within [1-α, 1] range
        # Use sigmoid to map the band_param to [0, 1] with 5x multiplier for smoothness
        band_activation = ops.sigmoid(5.0 * self.band_param)

        # Scale the activation to be within [1-max_band_width, 1]
        # When band_activation = 0: scale = 1 - max_band_width
        # When band_activation = 1: scale = 1
        scale = (1.0 - self.max_band_width) + (self.max_band_width * band_activation)

        # Apply scaling to the normalized tensor
        # The single scalar scale is automatically broadcast to all elements
        output = normalized * scale

        # Cast back to original dtype
        return ops.cast(output, original_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple (same as input shape for normalization layers).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "max_band_width": self.max_band_width,
            "axis": self.axis,
            "epsilon": self.epsilon,
            "band_initializer": initializers.serialize(self.band_initializer),
            "band_regularizer": regularizers.serialize(self.band_regularizer),
        })
        return config

# ---------------------------------------------------------------------
