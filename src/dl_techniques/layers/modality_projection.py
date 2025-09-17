"""Projects visual features into a language model's embedding space.

This layer serves as a critical bridge in Vision-Language Models (VLMs),
transforming visual representations from an image encoder into a format
compatible with a language model's textual embedding space. Its design
addresses two fundamental challenges: aligning the distinct statistical
properties of visual and textual modalities, and managing the computational
burden imposed by long sequences of visual tokens.

Architectural and Mathematical Underpinnings:

The layer employs a two-stage process that prioritizes both information
preservation and computational efficiency.

1.  **Information-Preserving Token Reduction**: The primary computational
    bottleneck in processing visual features with a Transformer-based
    language model is the quadratic complexity of the self-attention
    mechanism with respect to sequence length. High-resolution images
    produce a large number of visual tokens, making direct processing
    infeasible.

    To mitigate this, the layer first reduces the number of tokens using a
    space-to-depth transformation, commonly known as `PixelShuffle`.
    Unlike pooling or strided convolutions which discard spatial
    information, this operation is a lossless rearrangement. For a given
    `scale_factor` `s`, it reshapes `s x s` blocks of spatial tokens into a
    single, more descriptive token by folding the spatial dimensions into
    the channel dimension. This reduces the sequence length by a factor of
    `s²` while preserving all the original information, concentrating it
    into a richer, more compact sequence of tokens.

2.  **Cross-Modal Linear Projection**: After the sequence is shortened, the
    resulting tokens are passed through a `Dense` layer. This is the core
    projector that learns an affine transformation to map the (reshaped)
    visual feature space into the target language embedding space. The
    weights of this projection, `y = xW + b`, are learned end-to-end,
    allowing the model to discover the optimal alignment between visual
    concepts and their corresponding representations in the language model's
    semantic space. Optional GELU activation and Layer Normalization are
    applied to stabilize training and match the architectural conventions of
    modern Transformers.

This design provides an efficient and effective mechanism for cross-modal
grounding, enabling language models to seamlessly integrate and reason about
visual information.

References:
    - Shi, W., et al. (2016). Real-Time Single Image and Video
      Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural
      Network. *CVPR*. (Introduced the concept of Pixel Shuffle).
    - Radford, A., et al. (2021). Learning Transferable Visual Models From
      Natural Language Supervision. *ICML*. (Established the pattern for
      projecting modalities in CLIP).
    - Alayrac, J., et al. (2022). Flamingo: a Visual Language Model for
      Few-Shot Learning. *NeurIPS*. (Demonstrates advanced projection
      architectures in VLMs).
"""

import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..utils.logger import logger
from .pixel_shuffle import PixelShuffle

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class ModalityProjection(keras.layers.Layer):
    """Modality projection layer for nanoVLM.

    Projects visual features to language embedding space using pixel shuffle
    for token reduction followed by linear projection.

    This layer combines token reduction through pixel shuffling with learnable
    projection to align visual and textual representations in a shared embedding space.

    Args:
        input_dim: Input feature dimension. Must be positive integer.
        output_dim: Output embedding dimension. Must be positive integer.
        scale_factor: Pixel shuffle scale factor for token reduction. Must be positive integer.
            Defaults to 2. Higher values reduce more tokens but may lose spatial information.
        use_gelu: Whether to use GELU activation after projection. Defaults to True.
        use_layer_norm: Whether to apply layer normalization after projection. Defaults to True.
        projection_kernel_initializer: Initializer for the projection layer weights.
            Defaults to 'glorot_uniform'.
        projection_bias_initializer: Initializer for the projection layer bias.
            Defaults to 'zeros'.
        projection_kernel_regularizer: Regularizer for the projection layer weights.
            Defaults to None.
        projection_bias_regularizer: Regularizer for the projection layer bias.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the base Layer class.

    Raises:
        ValueError: If input_dim, output_dim, or scale_factor are not positive integers.

    Example:
        >>> # Create projection layer
        >>> projection = ModalityProjection(
        ...     input_dim=768,
        ...     output_dim=512,
        ...     scale_factor=2,
        ...     use_gelu=True,
        ...     use_layer_norm=True
        ... )
        >>>
        >>> # Input: visual features [batch, 196, 768]
        >>> visual_features = keras.random.normal((32, 196, 768))
        >>> projected = projection(visual_features)
        >>> print(projected.shape)  # (32, 49, 512) - tokens reduced by 4x
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            scale_factor: int = 2,
            use_gelu: bool = True,
            use_layer_norm: bool = True,
            projection_kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            projection_bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            projection_kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            projection_bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive integer, got {scale_factor}")

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale_factor = scale_factor
        self.use_gelu = use_gelu
        self.use_layer_norm = use_layer_norm

        # Store initializers and regularizers
        self.projection_kernel_initializer = keras.initializers.get(projection_kernel_initializer)
        self.projection_bias_initializer = keras.initializers.get(projection_bias_initializer)
        self.projection_kernel_regularizer = keras.regularizers.get(projection_kernel_regularizer)
        self.projection_bias_regularizer = keras.regularizers.get(projection_bias_regularizer)

        # FIX: Instantiate sublayers in __init__ for proper serialization
        self.pixel_shuffle = PixelShuffle(
            scale_factor=self.scale_factor,
            name='pixel_shuffle'
        )
        self.projection_dense = keras.layers.Dense(
            units=self.output_dim,
            kernel_initializer=self.projection_kernel_initializer,
            bias_initializer=self.projection_bias_initializer,
            kernel_regularizer=self.projection_kernel_regularizer,
            bias_regularizer=self.projection_bias_regularizer,
            name='projection_dense'
        )
        self.projection_activation = None
        if self.use_gelu:
            self.projection_activation = keras.layers.Activation('gelu', name='projection_gelu')

        self.projection_norm = None
        if self.use_layer_norm:
            self.projection_norm = keras.layers.LayerNormalization(
                axis=-1,
                epsilon=1e-6,
                name='projection_norm'
            )

        # Store build information for serialization
        self._build_input_shape = None

        logger.info(
            f"Initialized ModalityProjection: {input_dim} -> {output_dim}, "
            f"scale_factor={scale_factor}, gelu={use_gelu}, norm={use_layer_norm}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the modality projection components.

        Args:
            input_shape: Shape tuple indicating input shape. Expected format is
                (batch_size, sequence_length, input_dim).

        Raises:
            ValueError: If input shape is invalid or incompatible.
        """
        # Store input shape for serialization
        self._build_input_shape = input_shape

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected 3D input shape (batch, sequence, features), got {input_shape}"
            )

        input_features = input_shape[-1]
        if input_features != self.input_dim:
            raise ValueError(
                f"Input shape last dimension ({input_features}) doesn't match "
                f"specified input_dim ({self.input_dim})"
            )

        # The layers are now created in __init__, so we just call super().build()
        # Keras will handle building the sub-layers.
        super().build(input_shape)

        pixel_shuffle_output_shape = self.pixel_shuffle.compute_output_shape(input_shape)
        logger.debug(
            f"Built ModalityProjection with input_shape={input_shape}, "
            f"pixel_shuffle_output_shape={pixel_shuffle_output_shape}"
        )

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply modality projection.

        Args:
            inputs: Visual features of shape [batch_size, num_tokens, input_dim].
            training: Boolean indicating whether the layer should behave in training
                mode or inference mode. Passed to sublayers that behave differently
                during training and inference.

        Returns:
            Projected features of shape [batch_size, reduced_tokens, output_dim]
            where reduced_tokens = 1 + (num_tokens-1) / (scale_factor ** 2).
        """
        # Apply pixel shuffle to reduce number of tokens
        x = self.pixel_shuffle(inputs)

        # Apply dense projection
        x = self.projection_dense(x)

        # Apply optional GELU activation
        if self.projection_activation is not None:
            x = self.projection_activation(x)

        # Apply optional layer normalization
        if self.projection_norm is not None:
            x = self.projection_norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input.

        Returns:
            Output shape tuple.
        """
        # Get pixel shuffle output shape
        shuffled_shape = self.pixel_shuffle.compute_output_shape(input_shape)
        shuffled_shape_list = list(shuffled_shape)

        # Update last dimension with output_dim
        output_shape_list = shuffled_shape_list[:-1] + [self.output_dim]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Returns the layer configuration for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "scale_factor": self.scale_factor,
            "use_gelu": self.use_gelu,
            "use_layer_norm": self.use_layer_norm,
            "projection_kernel_initializer": keras.initializers.serialize(self.projection_kernel_initializer),
            "projection_bias_initializer": keras.initializers.serialize(self.projection_bias_initializer),
            "projection_kernel_regularizer": keras.regularizers.serialize(self.projection_kernel_regularizer),
            "projection_bias_regularizer": keras.regularizers.serialize(self.projection_bias_regularizer),
        })
        return config

    def get_build_config(self) -> Dict[str, Any]:
        """Get the config needed to build the layer from a config.

        This method is needed for proper model saving and loading.

        Returns:
            Dictionary containing the build configuration.
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Build the layer from a config created with get_build_config.

        Args:
            config: Dictionary containing the build configuration.
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
