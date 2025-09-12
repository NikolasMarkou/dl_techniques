"""
Zero-Centered Band RMS Normalization Layer for Enhanced Training Stability

This module implements Zero-Centered Band RMS Normalization, an advanced normalization
technique that combines three key innovations:
1. Zero-centering from ZeroCenteredRMSNorm for training stability
2. RMS normalization for computational efficiency and dimension independence
3. Band constraint from BandRMS for controlled representational flexibility

Mathematical Formulation:
    Given an input tensor x with shape (..., d), Zero-Centered Band RMS normalization:

    μ = mean(x) over specified axes
    x_centered = x - μ
    RMS(x_centered) = sqrt(mean(x_centered²) + ε)
    x_norm = x_centered / RMS(x_centered)
    s = sigmoid(5.0 * band_param) * max_band_width + (1 - max_band_width)
    output = x_norm * s

    Where:
    - μ is the mean computed over specified axes (centering step)
    - RMS is computed from the centered input for stability
    - s is the learnable scaling factor constrained to [1-α, 1] band
    - α is the max_band_width parameter

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
    finally applies learnable scaling within a constrained [1-α, 1] band.

    **Intent**: Provide enhanced normalization that prevents mean drift and abnormal
    weight growth while maintaining computational efficiency and offering controlled
    representational flexibility through learnable band constraints.

    **Architecture**:
    ```
    Input(shape=[..., features])
           ↓
    Compute: μ = mean(x)
           ↓
    Center: x_centered = x - μ
           ↓
    Compute: rms = sqrt(mean(x_centered²) + ε)
           ↓
    Normalize: x_norm = x_centered / rms
           ↓
    Band Scale: s = sigmoid(5.0 * band_param) * α + (1 - α)
           ↓
    Apply: output = x_norm * s
           ↓
    Output(shape=[..., features])
    ```

    **Mathematical Operations**:
    1. **Centering**: μ = E[x], x_centered = x - μ
    2. **RMS Computation**: rms = √(E[x_centered²] + ε)
    3. **Normalization**: x̂ = x_centered / rms
    4. **Band Scaling**: s = sigmoid(5β) × α + (1-α), output = s × x̂

    Where:
    - μ is computed per feature across normalization axes
    - rms is computed from centered inputs for enhanced stability
    - s is the learnable band scaling factor constrained to [1-α, 1]
    - β is the trainable band parameter
    - α is the max_band_width hyperparameter
    - ε is a small constant for numerical stability

    This creates a "thick shell" in the normalized space while maintaining zero-mean
    property, combining the benefits of LayerNorm stability, RMSNorm efficiency,
    and BandRMS representational flexibility.

    Args:
        max_band_width: Maximum allowed deviation from unit normalization (0 < α < 1).
            Controls the thickness of the representational band. When α=0.1, the output
            RMS will be constrained to [0.9, 1.0]. Defaults to 0.1.
        axis: Axis or axes along which to compute mean and RMS statistics.
            The default (-1) computes statistics over the last dimension. For multi-axis
            normalization, pass a tuple (e.g., (-2, -1) for normalizing over last two
            dimensions). Defaults to -1.
        epsilon: Small constant added to denominator for numerical stability.
            Should be positive and typically in range [1e-8, 1e-5]. Defaults to 1e-7.
        band_initializer: Initializer for the band parameter. The band parameter
            controls the learned position within the [1-α, 1] constraint band.
            Common choices: "zeros" (start at lower bound), "ones" (start at upper bound).
            Defaults to "zeros".
        band_regularizer: Optional regularizer for the band parameter.
            Helps prevent the band parameter from becoming too extreme. If None,
            defaults to L2(1e-5) regularizer for stability. Defaults to None.
        **kwargs: Additional keyword arguments passed to the parent Layer class.

    Input shape:
        N-D tensor with shape: ``(batch_size, ..., features)``

        The layer can handle any dimensionality, with normalization applied along
        the specified axis/axes.

    Output shape:
        Same shape as input: ``(batch_size, ..., features)``

    Attributes:
        band_param: Learnable scalar parameter controlling position within the band.
            Created in build() method.

    Raises:
        ValueError: If max_band_width is not between 0 and 1.
        ValueError: If epsilon is not positive.
        TypeError: If axis is not int or tuple of ints.

    Example:
        Basic usage for transformer attention blocks:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.zero_centered_band_rms_norm import ZeroCenteredBandRMSNorm

            # Enhanced normalization for transformer layers
            inputs = keras.Input(shape=(512, 768))
            normalized = ZeroCenteredBandRMSNorm(
                max_band_width=0.1,
                axis=-1
            )(inputs)
            model = keras.Model(inputs, normalized)

        Stable transformer block with enhanced normalization:

        .. code-block:: python

            def enhanced_transformer_block(inputs, hidden_size=768):
                # Zero-centered band RMS norm before attention
                norm_inputs = ZeroCenteredBandRMSNorm(
                    max_band_width=0.15,
                    axis=-1,
                    epsilon=1e-6
                )(inputs)

                # Multi-head attention
                attention_out = keras.layers.MultiHeadAttention(
                    num_heads=12,
                    key_dim=64
                )(norm_inputs, norm_inputs)

                # Residual connection
                x = inputs + attention_out

                # Zero-centered band RMS norm before FFN
                norm_x = ZeroCenteredBandRMSNorm(
                    max_band_width=0.2
                )(x)

                # Feed-forward network
                ffn_out = keras.layers.Dense(3072, activation='gelu')(norm_x)
                ffn_out = keras.layers.Dense(hidden_size)(ffn_out)

                return x + ffn_out

        Large language model with custom regularization:

        .. code-block:: python

            class EnhancedLLMBlock(keras.layers.Layer):
                def __init__(self, hidden_size=4096, **kwargs):
                    super().__init__(**kwargs)

                    # Custom regularization for band parameters
                    custom_regularizer = keras.regularizers.L1L2(l1=1e-6, l2=1e-5)

                    self.attention_norm = ZeroCenteredBandRMSNorm(
                        max_band_width=0.12,
                        epsilon=1e-6,
                        band_regularizer=custom_regularizer
                    )
                    self.ffn_norm = ZeroCenteredBandRMSNorm(
                        max_band_width=0.08,
                        epsilon=1e-6,
                        band_initializer='random_uniform'
                    )

        Multi-axis normalization for convolutional applications:

        .. code-block:: python

            # Normalize over spatial and channel dimensions
            inputs = keras.Input(shape=(32, 32, 256))

            # Spatial normalization with band constraints
            spatial_norm = ZeroCenteredBandRMSNorm(
                axis=(1, 2),  # Height and width dimensions
                max_band_width=0.2
            )(inputs)

        Mixed precision training optimization:

        .. code-block:: python

            # Enhanced stability for mixed precision
            keras.mixed_precision.set_global_policy('mixed_float16')

            inputs = keras.Input(shape=(1024,), dtype='float16')
            normalized = ZeroCenteredBandRMSNorm(
                max_band_width=0.15,
                epsilon=1e-5,  # Slightly larger for fp16 stability
                band_regularizer=keras.regularizers.L2(1e-4)
            )(inputs)

    Note:
        - Combines zero-centering stability with band constraint flexibility
        - Prevents abnormal weight growth while allowing representational adaptation
        - Particularly effective in transformer architectures and large language models
        - The band parameter learns the optimal position within the constraint range
        - Implementation handles mixed precision training with appropriate casting
        - Follows modern Keras 3 patterns for robust serialization
        - The single scalar band parameter is broadcast across all features
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

        Args:
            max_band_width: Maximum deviation from unit normalization (0 < α < 1).
            axis: Axis or axes along which to compute statistics.
            epsilon: Small constant for numerical stability.
            band_initializer: Initializer for the band parameter.
            band_regularizer: Regularizer for the band parameter. Default is L2(1e-5).
            **kwargs: Additional layer arguments.

        Raises:
            ValueError: If max_band_width is not between 0 and 1 or if epsilon is not positive.
            TypeError: If axis is not int or tuple of ints.
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

        Args:
            max_band_width: Maximum allowed deviation from unit norm.
            axis: Normalization axis/axes to validate.
            epsilon: Small constant for numerical stability.

        Raises:
            ValueError: If parameters are invalid.
            TypeError: If axis type is invalid.
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

        Args:
            input_shape: Shape tuple indicating input tensor shape.
                First dimension (batch size) may be None.
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

        Args:
            inputs: Input tensor of any shape. Normalization is applied along
                the axes specified during initialization.
            training: Boolean indicating whether in training mode. Not used
                in this layer but kept for consistency with other normalization layers.

        Returns:
            Zero-centered band RMS normalized tensor with the same shape as inputs.
            The output RMS will be constrained to [1-max_band_width, 1] range.

        Note:
            The computation is performed in float32 for numerical stability in mixed
            precision training, then cast back to the original input dtype.
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

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape for normalization layers).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for serialization.

        Following modern Keras 3 patterns, this method returns ALL constructor
        arguments needed to recreate this layer instance.

        Returns:
            Dictionary containing all constructor arguments.
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
