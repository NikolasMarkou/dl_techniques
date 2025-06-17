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
        eps: Small constant for numerical stability. Must be positive.
        gamma_initializer: Initializer for gamma (scale) weights.
        beta_initializer: Initializer for beta (bias) weights.
        gamma_regularizer: Regularizer for gamma weights.
        beta_regularizer: Regularizer for beta weights.
        activity_regularizer: Regularizer for the layer output.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Call arguments:
        inputs: Input tensor of shape `(batch_size, height, width, channels)`.
        training: Boolean indicating whether the layer should behave in training mode.

    Returns:
        output: Normalized tensor of the same shape as input with enhanced inter-channel
               feature competition.

    Raises:
        ValueError: If eps <= 0.
        ValueError: If input shape is not 4D.
        ValueError: If channel dimension is not defined.

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
        super().__init__(**kwargs)

        # Validate parameters
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        # Store configuration parameters
        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Initialize weight attributes to None - will be created in build()
        self.gamma = None
        self.beta = None

        # Store build input shape for serialization
        self._build_input_shape = None
        self._channels = None

        logger.info(f"Initialized GlobalResponseNormalization with eps={eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer by creating weights and setting up input specifications.

        Args:
            input_shape: Tuple of integers defining the input shape.

        Raises:
            ValueError: If input shape is invalid.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Input validation
        if len(input_shape) != 4:
            raise ValueError(f"Input shape must be 4D (batch, height, width, channels), "
                             f"got {len(input_shape)}D")

        # Get number of channels
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined")

        self._channels = channels
        logger.debug(f"Building GlobalResponseNormalization with {channels} channels")

        # Create trainable parameters
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
            input_shape: Shape of the input tensor.

        Returns:
            Output shape (same as input shape).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration.
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

    def get_build_config(self) -> Dict[str, Any]:
        """
        Get the build configuration for serialization.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """
        Build the layer from a configuration dictionary.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])