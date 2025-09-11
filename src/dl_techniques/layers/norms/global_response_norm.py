"""
Global Response Normalization (GRN) Layer Implementation
=======================================================

Theory and Background:
---------------------
Global Response Normalization (GRN) is a normalization technique introduced in ConvNeXt V2
that enhances inter-channel feature competition by normalizing features across spatial
dimensions and applying learnable scale and bias parameters.

This implementation is generalized to handle 2D, 3D, and 4D tensors.

## Key Concepts:
1. **Inter-channel Competition**: GRN promotes competition between channels by normalizing
   based on the global response across all other dimensions for each channel.
2. **Feature Normalization**: Computes L2 norms across spatial/sequence dimensions (for 3D/4D)
   or uses the feature's own value (for 2D) for each channel.
3. **Global Scaling**: Normalizes by the mean norm across all channels.
4. **Learnable Parameters**: Applies trainable gamma (scale) and beta (bias) parameters.
5. **Residual Connection**: Adds the normalized response to the original input.

## Mathematical Formulation:
For an input X with rank N:
1. Compute feature norm: N_c = ||X_c||_2 over axes (1, ..., N-2) for each channel c.
2. Compute global mean: μ = mean(N_c) across all channels.
3. Normalize: N'_c = N_c / (μ + ε).
4. Apply learnable parameters: Y = X + γ * (X ⊙ N') + β,
   where ⊙ denotes element-wise multiplication with broadcasting.

## Benefits:
- **Improved Feature Selectivity**: Enhances the most informative channels.
- **Better Gradient Flow**: Maintains residual connections for stable training.
- **Computational Efficiency**: Lightweight normalization with minimal overhead.
- **Versatility**: Handles 2D, 3D, and 4D data seamlessly.

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
    Global Response Normalization (GRN) layer supporting 2D, 3D, and 4D inputs.

    This layer implements the GRN operation from the ConvNeXt V2 paper, generalized
    to handle 2D, 3D, and 4D tensors. It enhances inter-channel feature competition by
    normalizing features and applying learnable scale and bias parameters with a
    residual connection.

    The operation flow is:
    1. Compute L2 norm across spatial/sequence dimensions for each channel. For 2D data,
       this simplifies to the absolute value.
    2. Normalize by the mean of the L2 norm across channels.
    3. Apply learnable scaling (gamma) and bias (beta).
    4. Add the result to the input (residual connection).

    Args:
        eps: Small constant for numerical stability. Must be positive. Defaults to 1e-6.
        gamma_initializer: Initializer for gamma (scale) weights. Defaults to 'ones'.
        beta_initializer: Initializer for beta (bias) weights. Defaults to 'zeros'.
        gamma_regularizer: Optional regularizer for gamma weights. Defaults to None.
        beta_regularizer: Optional regularizer for beta weights. Defaults to None.
        activity_regularizer: Optional regularizer for the layer output. Defaults to None.
        **kwargs: Additional keyword arguments for the Layer parent class.

    Input shape:
        - 2D tensor: ``(batch_size, features)``
        - 3D tensor: ``(batch_size, sequence_length, features)``
        - 4D tensor: ``(batch_size, height, width, channels)``

    Output shape:
        Same shape as input.

    Raises:
        ValueError: If eps <= 0.
        ValueError: If input rank is not 2, 3, or 4.
        ValueError: If the feature/channel dimension is not defined.

    Example:
        `GlobalResponseNormalization` can be used with various data formats.

        **4D Data (e.g., in a CNN)**
        .. code-block:: python

            import keras
            from dl_techniques.layers.norms.global_response_norm import GlobalResponseNormalization

            inputs_4d = keras.Input(shape=(32, 32, 64))
            normalized_4d = GlobalResponseNormalization()(inputs_4d)
            model_4d = keras.Model(inputs_4d, normalized_4d)
            print(model_4d.output_shape)
            # Output: (None, 32, 32, 64)

        **3D Data (e.g., in a Transformer)**
        .. code-block:: python

            inputs_3d = keras.Input(shape=(50, 128))  # (batch, seq_len, features)
            normalized_3d = GlobalResponseNormalization()(inputs_3d)
            model_3d = keras.Model(inputs_3d, normalized_3d)
            print(model_3d.output_shape)
            # Output: (None, 50, 128)

        **2D Data (e.g., in an MLP)**
        .. code-block:: python

            inputs_2d = keras.Input(shape=(256,))  # (batch, features)
            normalized_2d = GlobalResponseNormalization()(inputs_2d)
            model_2d = keras.Model(inputs_2d, normalized_2d)
            print(model_2d.output_shape)
            # Output: (None, 256)

    Note:
        - This implementation follows modern Keras 3 patterns for robust serialization.
        - Designed for inputs where the last dimension is the channel/feature dimension.
        - Maintains residual connections for stable gradient flow.

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

        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.eps = eps
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        self.gamma = None
        self.beta = None

        logger.debug(f"Initialized GlobalResponseNormalization with eps={eps}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the layer's weights, adapted for 2D, 3D, or 4D inputs.

        Args:
            input_shape: Tuple of integers defining the input shape.

        Raises:
            ValueError: If input rank is not 2, 3, or 4, or if the channel
                        dimension is not defined.
        """
        rank = len(input_shape)
        if rank not in [2, 3, 4]:
            raise ValueError(
                f"Input rank must be 2, 3, or 4 (batch, [dims...], channels), "
                f"but got rank {rank}"
            )

        channels = input_shape[-1]
        if channels is None:
            raise ValueError("The channel/feature dimension (last axis) must be defined.")

        logger.debug(f"Building GlobalResponseNormalization for rank {rank} with {channels} channels")

        # Define shape for gamma and beta to allow broadcasting
        param_shape = (1,) * (rank - 1) + (channels,)

        self.gamma = self.add_weight(
            name="gamma",
            shape=param_shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=param_shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            trainable=True,
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

        This method implements the core GRN algorithm for 2D, 3D, or 4D inputs.

        Args:
            inputs: Input tensor of shape (batch, ..., channels).
            training: Whether in training mode (unused, for API compatibility).

        Returns:
            Normalized tensor of the same shape as input.
        """
        rank = ops.ndim(inputs)

        # Step 1: Compute the L2 norm over spatial/sequence dimensions.
        # For rank 2 (batch, features), axes is empty, so sum is identity,
        # and norm becomes the absolute value.
        # For rank 3 (batch, seq, features), axis is (1,).
        # For rank 4 (batch, h, w, channels), axis is (1, 2).
        axes_to_reduce = tuple(range(1, rank - 1))
        norm = ops.sqrt(
            ops.sum(ops.square(inputs), axis=axes_to_reduce, keepdims=True) + self.eps
        )

        # Step 2: Normalize by the mean norm across channels.
        mean_norm = ops.mean(norm, axis=-1, keepdims=True)
        normalized_norm = norm / (mean_norm + self.eps)

        # Step 3: Apply GRN transformation with residual connection.
        # The shapes of norm, gamma, and beta are broadcastable to the input shape.
        transformed = self.gamma * (inputs * normalized_norm) + self.beta
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