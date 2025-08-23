"""
Global Response Normalization (GRN) Layer Implementation
=======================================================

Theory and Background:
---------------------
Global Response Normalization (GRN) is a normalization technique introduced in ConvNeXt V2
that enhances inter-channel feature competition by normalizing features across spatial
dimensions and applying learnable scale and bias parameters.

## Key Concepts:
1. **Inter-channel Competition**: GRN promotes competition between channels by normalizing
   based on the global response across all channels
2. **Spatial Normalization**: Computes L2 norms across spatial dimensions for each channel
3. **Global Scaling**: Normalizes by the mean L2 norm across all channels
4. **Learnable Parameters**: Applies trainable gamma (scale) and beta (bias) parameters
5. **Residual Connection**: Adds the normalized response to the original input

## Mathematical Formulation:
For input X with shape (B, H, W, C):
1. Compute spatial L2 norm: N_c = ||X_c||_2 for each channel c
2. Compute global mean: μ = mean(N_c) across all channels
3. Normalize: N'_c = N_c / (μ + ε)
4. Apply learnable parameters: Y = X + γ * (X ⊙ N') + β
where ⊙ denotes element-wise multiplication with broadcasting

## Benefits:
- **Improved Feature Selectivity**: Enhances the most informative channels
- **Better Gradient Flow**: Maintains residual connections for stable training
- **Computational Efficiency**: Lightweight normalization with minimal overhead
- **Channel Competition**: Promotes specialization among different channels

## References:
[1] Woo, S., et al. (2023). "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
    arXiv:2301.00808
[2] Liu, Z., et al. (2022). "A ConvNet for the 2020s"
    arXiv:2201.03545
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Union, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class GlobalResponseNormalization(keras.layers.Layer):
    """
    Global Response Normalization (GRN) layer using backend-agnostic Keras operations.

    This layer implements the GRN operation from the ConvNeXt V2 paper, which enhances
    inter-channel feature competition. It normalizes features across spatial dimensions
    and applies learnable scale and bias parameters with a residual connection.

    The operation flow is:
    1. Compute L2 norm across spatial dimensions for each channel
    2. Normalize by the mean of the L2 norm across channels
    3. Apply learnable scaling (gamma) and bias (beta)
    4. Add the result to the input (residual connection)

    Args:
        eps: Small constant for numerical stability. Must be positive. Defaults to 1e-6.
        gamma_initializer: Initializer for gamma (scale) weights. Defaults to 'ones'.
        beta_initializer: Initializer for beta (bias) weights. Defaults to 'zeros'.
        gamma_regularizer: Optional regularizer for gamma weights. Defaults to None.
        beta_regularizer: Optional regularizer for beta weights. Defaults to None.
        activity_regularizer: Optional regularizer for the layer output. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``

    Output shape:
        Same shape as input: ``(batch_size, height, width, channels)``

    Raises:
        ValueError: If eps <= 0.
        ValueError: If input shape is not 4D.
        ValueError: If channel dimension is not defined.

    Example:
        Basic usage in ConvNeXt-style block:

        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization

            # Basic usage
            inputs = keras.Input(shape=(32, 32, 64))
            normalized = GlobalResponseNormalization()(inputs)
            model = keras.Model(inputs, normalized)

        Integration in ConvNeXt V2 block:

        .. code-block:: python

            def convnext_v2_block(inputs, filters=96):
                # Depthwise convolution
                x = keras.layers.DepthwiseConv2D(7, padding='same')(inputs)
                x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

                # Pointwise convolution (1x1)
                x = keras.layers.Conv2D(filters * 4, 1)(x)
                x = keras.layers.Activation('gelu')(x)

                # Global Response Normalization
                x = GlobalResponseNormalization(eps=1e-6)(x)

                # Output projection
                x = keras.layers.Conv2D(filters, 1)(x)

                # Residual connection
                return inputs + x

        Custom configuration:

        .. code-block:: python

            # With custom parameters and regularization
            grn = GlobalResponseNormalization(
                eps=1e-5,
                gamma_initializer='random_uniform',
                beta_initializer='zeros',
                gamma_regularizer=keras.regularizers.L2(1e-4)
            )
            outputs = grn(inputs)

        Mixed precision training:

        .. code-block:: python

            # GRN works well with mixed precision
            keras.mixed_precision.set_global_policy('mixed_float16')

            inputs = keras.Input(shape=(224, 224, 96), dtype='float16')
            grn_out = GlobalResponseNormalization(eps=1e-5)(inputs)
            model = keras.Model(inputs, grn_out)

    Note:
        - This implementation follows modern Keras 3 patterns for robust serialization
        - Designed specifically for 4D tensors (2D spatial + channels)
        - Maintains residual connections for stable gradient flow
        - Computationally efficient with minimal overhead

    References:
        ConvNeXt V2 paper: https://arxiv.org/abs/2301.00808
    """

    def __init__(
        self,
        eps: float = 1e-6,
        gamma_initializer: Union[str, keras.initializers.Initializer] = 'ones',
        beta_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the GlobalResponseNormalization layer.

        Args:
            eps: Small constant for numerical stability. Must be positive.
            gamma_initializer: Initializer for gamma (scale) weights.
            beta_initializer: Initializer for beta (bias) weights.
            gamma_regularizer: Regularizer for gamma weights.
            beta_regularizer: Regularizer for beta weights.
            activity_regularizer: Regularizer for the layer output.
            **kwargs: Additional keyword arguments for the Layer parent class.

        Raises:
            ValueError: If eps <= 0.
        """
        super().__init__(**kwargs)

        # Validate parameters
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # Store ALL configuration parameters - required for get_config()
        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Initialize weight attributes - created in build()
        self.gamma = None
        self.beta = None

        logger.debug(f"Initialized GlobalResponseNormalization with eps={eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's own weights.

        This is called automatically when the layer first processes input.
        Following modern Keras 3 Pattern 1: Simple Layer (No Sub-layers).

        Args:
            input_shape: Tuple of integers defining the input shape.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Input validation
        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be 4D (batch, height, width, channels), "
                             f"got {len(input_shape)}D")

        # Get number of channels
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined")

        logger.debug(f"Building GlobalResponseNormalization with {channels} channels")

        # Create layer's own weights using add_weight()
        self.gamma = self.add_weight(
            name="gamma",
            shape=(1, 1, 1, channels),
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=True
        )

        self.beta = self.add_weight(
            name="beta",
            shape=(1, 1, 1, channels),
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=True
        )

        # Set input spec for automatic shape validation
        self.input_spec = keras.layers.InputSpec(
            ndim=4, axes={-1: channels}
        )

        # Always call parent build at the end
        super().build(input_shape)
        logger.debug("GlobalResponseNormalization build completed")

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply global response normalization to the input tensor.

        This method implements the core GRN algorithm:
        1. Computes L2 norms across spatial dimensions for each channel
        2. Normalizes by the global mean norm across all channels
        3. Applies learnable scale (gamma) and bias (beta) parameters
        4. Adds the result to the original input (residual connection)

        The mathematical operation is:
        Y = X + γ * (X ⊙ (||X||_spatial / mean(||X||_spatial))) + β

        Args:
            inputs: Input tensor of shape (batch_size, height, width, channels).
            training: Whether in training mode (unused, included for API compatibility).

        Returns:
            Normalized tensor of the same shape as input.
        """
        # Get input shape for reshaping operations
        batch_size = ops.shape(inputs)[0]
        height = ops.shape(inputs)[1]
        width = ops.shape(inputs)[2]
        channels = ops.shape(inputs)[3]

        # Step 1: Reshape to (batch_size, pixels, channels) for efficient norm calculation
        # This flattens spatial dimensions while keeping channels separate
        reshaped = ops.reshape(inputs, (batch_size, height * width, channels))

        # Step 2: Compute L2 norm across spatial dimensions (axis=1)
        # Shape: (batch_size, 1, channels)
        norm_squared = ops.sum(ops.square(reshaped), axis=1, keepdims=True)
        norm = ops.sqrt(norm_squared + self.eps)  # Add epsilon for numerical stability

        # Step 3: Normalize by mean norm across channels
        # Compute global mean across channels: (batch_size, 1, 1)
        mean_norm = ops.mean(norm, axis=-1, keepdims=True)

        # Normalize each channel's norm by the global mean
        # Shape: (batch_size, 1, channels)
        norm_channels = norm / (mean_norm + self.eps)

        # Step 4: Reshape norm back to spatial dimensions for broadcasting
        # Shape: (batch_size, height, width, channels)
        norm_spatial = ops.reshape(norm_channels, (batch_size, 1, 1, channels))

        # Step 5: Apply GRN transformation with residual connection
        # Original ConvNeXt V2 formula: output = x + gamma * (x * normalized) + beta
        scaled = inputs * norm_spatial * self.gamma
        transformed = scaled + self.beta
        output = inputs + transformed

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple (same as input shape).
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
            "eps": float(self.eps),
            "gamma_initializer": keras.initializers.serialize(self.gamma_initializer),
            "beta_initializer": keras.initializers.serialize(self.beta_initializer),
            "gamma_regularizer": keras.regularizers.serialize(self.gamma_regularizer),
            "beta_regularizer": keras.regularizers.serialize(self.beta_regularizer),
            "activity_regularizer": keras.regularizers.serialize(self.activity_regularizer),
        })
        return config

# ---------------------------------------------------------------------
