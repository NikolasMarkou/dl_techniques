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
    """Global Response Normalization (GRN) layer for 2D, 3D, and 4D inputs.

    Implements the GRN operation from ConvNeXt V2, enhancing inter-channel feature
    competition. For each channel, computes the L2 norm across spatial/sequence
    dimensions, normalizes by the mean norm across channels, then applies learnable
    gamma and beta with a residual connection:
    ``Y = X + γ * (X ⊙ (norm_c / (mean(norm) + ε))) + β``.
    Generalizes to 2D (MLP), 3D (sequence), and 4D (image) inputs.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │         Input (X)                │
        │  shape: (B, [spatial...], C)     │
        └──────────────┬───────────────────┘
                       │
                       ├──────────────────────┐
                       │                      │
                       ▼                      │
        ┌──────────────────────────────┐      │
        │  L2 Norm per channel over    │      │
        │  spatial dims: norm_c        │      │
        └──────────────┬───────────────┘      │
                       │                      │
                       ▼                      │
        ┌──────────────────────────────┐      │
        │  Mean norm across channels:  │      │
        │  μ = mean(norm_c)            │      │
        └──────────────┬───────────────┘      │
                       │                      │
                       ▼                      │
        ┌──────────────────────────────┐      │
        │  Normalize: norm_c / (μ + ε) │      │
        └──────────────┬───────────────┘      │
                       │                      │
                       ▼                      │
        ┌──────────────────────────────┐      │
        │  γ × (X ⊙ norm') + β         │      │
        └──────────────┬───────────────┘      │
                       │                      │
                       ▼                      ▼
        ┌──────────────────────────────────────┐
        │  Residual: X + transformed           │
        └──────────────┬───────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │         Output (Y)               │
        │  shape: (B, [spatial...], C)     │
        └──────────────────────────────────┘

    :param eps: Small constant for numerical stability. Must be positive.
        Defaults to 1e-6.
    :type eps: float
    :param gamma_initializer: Initializer for gamma (scale) weights.
        Defaults to ``'ones'``.
    :type gamma_initializer: Union[str, keras.initializers.Initializer]
    :param beta_initializer: Initializer for beta (bias) weights.
        Defaults to ``'zeros'``.
    :type beta_initializer: Union[str, keras.initializers.Initializer]
    :param gamma_regularizer: Optional regularizer for gamma weights.
    :type gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param beta_regularizer: Optional regularizer for beta weights.
    :type beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param activity_regularizer: Optional regularizer for the layer output.
    :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]

    :raises ValueError: If eps <= 0.
    :raises ValueError: If input rank is not 2, 3, or 4.
    :raises ValueError: If the feature/channel dimension is not defined.
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
        """Initialize the GlobalResponseNormalization layer.

        :param eps: Small constant for numerical stability. Must be positive.
        :type eps: float
        :param gamma_initializer: Initializer for gamma (scale) weights.
        :type gamma_initializer: Union[str, keras.initializers.Initializer]
        :param beta_initializer: Initializer for beta (bias) weights.
        :type beta_initializer: Union[str, keras.initializers.Initializer]
        :param gamma_regularizer: Regularizer for gamma weights.
        :type gamma_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param beta_regularizer: Regularizer for beta weights.
        :type beta_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
        :param activity_regularizer: Regularizer for the layer output.
        :type activity_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]

        :raises ValueError: If eps <= 0.
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
        """Create the layer's weights, adapted for 2D, 3D, or 4D inputs.

        :param input_shape: Tuple of integers defining the input shape.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If input rank is not 2, 3, or 4, or if the
            channel dimension is not defined.
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
        """Apply global response normalization to the input tensor.

        Implements the core GRN algorithm for 2D, 3D, or 4D inputs.

        :param inputs: Input tensor of shape ``(batch, ..., channels)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode (unused, for API compatibility).
        :type training: Optional[bool]

        :return: Normalized tensor of the same shape as input.
        :rtype: keras.KerasTensor
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
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Output shape tuple (same as input shape).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        :rtype: Dict[str, Any]
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
