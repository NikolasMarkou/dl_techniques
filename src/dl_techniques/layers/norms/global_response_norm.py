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
        eps: Float, small constant for numerical stability. Must be positive.
            Defaults to 1e-6.
        gamma_initializer: Initializer for gamma (scale) weights. Can be string name
            or keras.initializers.Initializer instance. Defaults to 'ones'.
        beta_initializer: Initializer for beta (bias) weights. Can be string name
            or keras.initializers.Initializer instance. Defaults to 'zeros'.
        gamma_regularizer: Optional regularizer for gamma weights. Can be string name
            or keras.regularizers.Regularizer instance. Defaults to None.
        beta_regularizer: Optional regularizer for beta weights. Can be string name
            or keras.regularizers.Regularizer instance. Defaults to None.
        activity_regularizer: Optional regularizer for the layer output. Can be string
            name or keras.regularizers.Regularizer instance. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        The layer expects data in channels-last format.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, channels)`.
        Same shape as input.

    Attributes:
        gamma: Scale parameter weights of shape (1, 1, 1, channels).
        beta: Bias parameter weights of shape (1, 1, 1, channels).

    Example:
        ```python
        # Basic usage
        layer = GlobalResponseNormalization()

        # With custom parameters
        layer = GlobalResponseNormalization(
            eps=1e-8,
            gamma_initializer='glorot_uniform',
            gamma_regularizer=keras.regularizers.L2(1e-4)
        )

        # In a model
        inputs = keras.Input(shape=(224, 224, 3))
        x = keras.layers.Conv2D(64, 3, activation='relu')(inputs)
        x = GlobalResponseNormalization()(x)
        outputs = keras.layers.Conv2D(10, 1)(x)
        model = keras.Model(inputs, outputs)
        ```

    Raises:
        ValueError: If eps <= 0.
        ValueError: If input is not 4D during forward pass.
        ValueError: If channel dimension is None during build.

    References:
        ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
        (https://arxiv.org/abs/2301.00808)

    Note:
        This implementation follows the modern Keras 3 pattern where weights
        are created in build() and all configuration is stored in __init__.
        This ensures proper serialization and avoids common build errors.
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
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        # Validate eps parameter
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # 1. Store ALL configuration arguments as instance attributes
        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)

        # 2. No sub-layers to create for this layer

        # 3. Initialize weight attributes to None - they'll be created in build()
        self.gamma = None
        self.beta = None

        logger.debug(f"Initialized GlobalResponseNormalization with eps={eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights.

        This method is called automatically by Keras when the layer is first used.
        It creates the layer's gamma and beta weights using add_weight().

        Args:
            input_shape: Shape tuple of the input tensor, including batch dimension.
                Expected shape: (batch_size, height, width, channels).

        Raises:
            ValueError: If input shape is not 4D.
            ValueError: If channel dimension is not defined.
        """
        # Validate input shape
        if len(input_shape) != 4:
            raise ValueError(
                f"GlobalResponseNormalization expects 4D input "
                f"(batch_size, height, width, channels), got {len(input_shape)}D: {input_shape}"
            )

        # Get number of channels
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("The channel dimension (last dimension) must be defined")

        logger.debug(f"Building GlobalResponseNormalization with {channels} channels")

        # CREATE the layer's weights using add_weight()
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1, 1, 1, channels),
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=True,
        )

        self.beta = self.add_weight(
            name='beta',
            shape=(1, 1, 1, channels),
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=True,
        )

        # Let Keras know the build is complete
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
            training: Boolean indicating whether the layer should behave in training
                mode or inference mode. Not used in this layer but included for
                consistency with Keras API.

        Returns:
            Normalized tensor of the same shape as input with enhanced inter-channel
            feature competition.

        Raises:
            ValueError: If input tensor is not 4D.
        """
        # Validate input shape at runtime
        input_shape = ops.shape(inputs)
        if len(input_shape) != 4:
            raise ValueError(
                f"GlobalResponseNormalization expects 4D input, got {len(input_shape)}D"
            )

        # Get input shape for reshaping operations
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]

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
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple (same as input shape for GRN).
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        This method must return ALL arguments needed to recreate the layer
        via __init__. Uses keras serializers for complex objects.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            'eps': self.eps,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
        })
        return config

    # Modern Keras 3: NO get_build_config or build_from_config needed.
    # Keras handles the build lifecycle automatically.