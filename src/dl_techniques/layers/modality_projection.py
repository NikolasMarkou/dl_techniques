"""
Modality Projection Layer for nanoVLM.

This module implements a modality projection layer that projects visual features
to language embedding space using pixel shuffle for token reduction followed by
linear projection.
"""


import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# lolca imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

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

        # Initialize sublayers to None - will be created in build()
        self.pixel_shuffle = None
        self.projection_dense = None
        self.projection_activation = None
        self.projection_norm = None

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

        # Create pixel shuffle layer for token reduction
        # Assuming PixelShuffle layer exists in the project
        from dl_techniques.layers import PixelShuffle
        self.pixel_shuffle = PixelShuffle(scale_factor=self.scale_factor)

        # Calculate expected input dimension after pixel shuffle
        shuffled_dim = self.input_dim * (self.scale_factor ** 2)

        # Build projection dense layer
        self.projection_dense = keras.layers.Dense(
            units=self.output_dim,
            kernel_initializer=self.projection_kernel_initializer,
            bias_initializer=self.projection_bias_initializer,
            kernel_regularizer=self.projection_kernel_regularizer,
            bias_regularizer=self.projection_bias_regularizer,
            name='projection_dense'
        )

        # Build optional activation layer
        if self.use_gelu:
            self.projection_activation = keras.layers.Activation('gelu', name='projection_gelu')

        # Build optional normalization layer
        if self.use_layer_norm:
            self.projection_norm = keras.layers.LayerNormalization(
                axis=-1,
                epsilon=1e-6,
                name='projection_norm'
            )

        # Build sublayers with appropriate shapes
        pixel_shuffle_output_shape = self.pixel_shuffle.compute_output_shape(input_shape)
        self.projection_dense.build(pixel_shuffle_output_shape)

        if self.projection_activation is not None:
            dense_output_shape = self.projection_dense.compute_output_shape(pixel_shuffle_output_shape)
            self.projection_activation.build(dense_output_shape)

        if self.projection_norm is not None:
            if self.projection_activation is not None:
                activation_output_shape = dense_output_shape
            else:
                activation_output_shape = self.projection_dense.compute_output_shape(pixel_shuffle_output_shape)
            self.projection_norm.build(activation_output_shape)

        super().build(input_shape)

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
            where reduced_tokens = num_tokens / (scale_factor ** 2).
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
        # Convert to list for consistent manipulation
        input_shape_list = list(input_shape)

        # Get pixel shuffle output shape
        shuffled_shape = self.pixel_shuffle.compute_output_shape(tuple(input_shape_list))
        shuffled_shape_list = list(shuffled_shape)

        # Update last dimension with output_dim
        output_shape_list = shuffled_shape_list[:-1] + [self.output_dim]

        # Return as tuple for consistency
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
