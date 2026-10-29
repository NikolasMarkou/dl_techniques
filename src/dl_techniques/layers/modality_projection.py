"""Project visual features into a language embedding space.

``ModalityProjection`` shortens a visual token sequence before a text
Transformer attends over it, since self-attention cost grows quadratically with
sequence length. A ``PixelShuffle`` space-to-depth transform folds ``s x s``
blocks of spatial tokens into one wider token, cutting the sequence by ``s^2``
while keeping every value, so nothing is pooled away. A ``Dense`` layer then
learns the affine map ``y = xW + b`` from the reshaped visual space into the
language embedding space, followed by optional GELU and layer normalization,
both on by default.

The input must be rank 3 with its last axis equal to ``input_dim``, and its
token count must divide by ``scale_factor ** 2`` for the fold to come out even.
The activation is ``exact_gelu`` rather than the stock approximation.

References:
    - Shi et al., 2016. Real-Time Single Image and Video Super-Resolution
      Using an Efficient Sub-Pixel Convolutional Neural Network.
    - Radford et al., 2021. Learning Transferable Visual Models From
      Natural Language Supervision.
    - Alayrac et al., 2022. Flamingo: a Visual Language Model for Few-Shot
      Learning.
"""

import keras
from typing import Optional, Tuple, Union, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.activations.exact_activations import exact_gelu
from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique

from .pooling.pixel_shuffle import PixelShuffle

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.modality_projection")
class ModalityProjection(keras.layers.Layer):
    """Fold visual tokens together and project them to the language space.

    ``PixelShuffle`` rearranges ``s x s`` blocks of spatial tokens into one
    token by moving the spatial extent into the channel axis, so the sequence
    shortens by ``s^2`` and no value is discarded. The ``Dense`` layer that
    follows learns an affine map aligning the widened visual features with the
    target language embedding space.

    Architecture:

    .. code-block:: text

                input [B, N, input_dim]
                                ▼
                ┌───────────────────────────────┐
                │ PixelShuffle(scale_factor)    │
                └───────────────┬───────────────┘
                 [B, N/s^2, input_dim*s^2]
                                ▼
                ┌───────────────────────────────┐
                │ Dense(output_dim)             │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ exact GELU                    │
                │  (optional)                   │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │ LayerNormalization eps 1e-6   │
                │  (optional)                   │
                └───────────────┬───────────────┘
                                ▼
                 [B, N/s^2, output_dim]

    Shape bookkeeping:

    .. code-block:: text

        stage                 tokens        channels
        ───────────────────   ───────────   ───────────────
        input                 N             input_dim
        after PixelShuffle    N / s^2       input_dim * s^2
        after Dense           N / s^2       output_dim

    :param input_dim: Input feature dimension. Must be a positive integer, and
        the input's last axis must equal it.
    :type input_dim: int
    :param output_dim: Output embedding dimension. Must be a positive integer.
    :type output_dim: int
    :param scale_factor: Pixel shuffle scale factor ``s``; the sequence
        shortens by ``s^2``. Must be a positive integer. Defaults to 2.
    :type scale_factor: int
    :param use_gelu: Apply ``exact_gelu`` after the projection. Defaults to
        True.
    :type use_gelu: bool
    :param use_layer_norm: Apply layer normalization last. Defaults to True.
    :type use_layer_norm: bool
    :param projection_kernel_initializer: Initializer for projection weights.
        Defaults to ``'glorot_uniform'``.
    :type projection_kernel_initializer: Union[str, keras.initializers.Initializer]
    :param projection_bias_initializer: Initializer for projection bias.
        Defaults to ``'zeros'``.
    :type projection_bias_initializer: Union[str, keras.initializers.Initializer]
    :param projection_kernel_regularizer: Regularizer for projection weights.
    :type projection_kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param projection_bias_regularizer: Regularizer for projection bias.
    :type projection_bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the base Layer class.

    Input shape:
        3D tensor ``(batch_size, num_tokens, input_dim)``.

    Output shape:
        3D tensor ``(batch_size, num_tokens / scale_factor**2, output_dim)``.

    :raises ValueError: From ``__init__``, if ``input_dim``, ``output_dim`` or
        ``scale_factor`` is not a positive integer. From ``build``, if the
        input is not rank 3 or its last axis does not equal ``input_dim``.
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

        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(f"output_dim must be a positive integer, got {output_dim}")
        if not isinstance(scale_factor, int) or scale_factor <= 0:
            raise ValueError(f"scale_factor must be a positive integer, got {scale_factor}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale_factor = scale_factor
        self.use_gelu = use_gelu
        self.use_layer_norm = use_layer_norm

        self.projection_kernel_initializer = keras.initializers.get(projection_kernel_initializer)
        self.projection_bias_initializer = keras.initializers.get(projection_bias_initializer)
        self.projection_kernel_regularizer = keras.regularizers.get(projection_kernel_regularizer)
        self.projection_bias_regularizer = keras.regularizers.get(projection_bias_regularizer)

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
            # DECISION plan-2026-10-14T062616-81938e81/D-277: exact_gelu is
            # float64-exact and bit-identical to the stock GELU at float32.
            self.projection_activation = keras.layers.Activation(
                exact_gelu, name='projection_gelu')

        self.projection_norm = None
        if self.use_layer_norm:
            self.projection_norm = keras.layers.LayerNormalization(
                axis=-1,
                epsilon=1e-6,
                name='projection_norm'
            )

        self._build_input_shape = None

        logger.info(
            f"Initialized ModalityProjection: {input_dim} -> {output_dim}, "
            f"scale_factor={scale_factor}, gelu={use_gelu}, norm={use_layer_norm}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build each sub-layer in order, threading the shape through.

        Building them here rather than lazily means every weight variable
        exists before a checkpoint is restored.

        :param input_shape: Shape tuple ``(batch_size, sequence_length, input_dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 3, or its last axis does
            not equal ``input_dim``.
        """
        # A second build, from a parent reaching an already-built projection,
        # must return early: Keras refuses to add state to a built layer.
        if self.built:
            return

        self._build_input_shape = input_shape

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

        self.pixel_shuffle.build(input_shape)
        pixel_shuffle_output_shape = self.pixel_shuffle.compute_output_shape(input_shape)

        self.projection_dense.build(pixel_shuffle_output_shape)
        dense_output_shape = self.projection_dense.compute_output_shape(
            pixel_shuffle_output_shape
        )

        if self.projection_activation is not None:
            self.projection_activation.build(dense_output_shape)
        if self.projection_norm is not None:
            self.projection_norm.build(dense_output_shape)

        logger.debug(
            f"Built ModalityProjection with input_shape={input_shape}, "
            f"pixel_shuffle_output_shape={pixel_shuffle_output_shape}"
        )

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Fold the tokens, project them, and normalize.

        :param inputs: Visual features of shape ``[batch_size, num_tokens, input_dim]``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Only
            the normalization reads it; nothing else here is mode-dependent.
        :type training: Optional[bool]
        :return: Projected features of shape ``[batch_size, reduced_tokens, output_dim]``.
        :rtype: keras.KerasTensor
        """
        x = self.pixel_shuffle(inputs)

        x = self.projection_dense(x)

        if self.projection_activation is not None:
            x = self.projection_activation(x)

        if self.projection_norm is not None:
            x = self.projection_norm(x, training=training)

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute the output shape, taking the token count from PixelShuffle.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        shuffled_shape = self.pixel_shuffle.compute_output_shape(input_shape)
        shuffled_shape_list = list(shuffled_shape)

        output_shape_list = shuffled_shape_list[:-1] + [self.output_dim]

        return tuple(output_shape_list)

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
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
        """Return the shape needed to rebuild this layer on deserialization.

        :return: Dictionary containing the build configuration.
        :rtype: Dict[str, Any]
        """
        return {
            "input_shape": self._build_input_shape,
        }

    def build_from_config(self, config: Dict[str, Any]) -> None:
        """Rebuild the layer from a config made by :meth:`get_build_config`.

        :param config: Dictionary containing the build configuration.
        :type config: Dict[str, Any]
        """
        if config.get("input_shape") is not None:
            self.build(config["input_shape"])

# ---------------------------------------------------------------------
