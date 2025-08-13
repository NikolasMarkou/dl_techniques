"""
This module provides the `ModalityProjection` layer, a critical component for
multimodal architectures like Vision-Language Models (VLMs).

Its primary function is to act as a "bridge" between the visual and textual
modalities. It takes high-dimensional feature vectors from a vision encoder (e.g.,
the output of a Vision Transformer) and transforms them into a format that is
compatible with a language model's embedding space. This alignment is essential for
the language model to be able to "understand" and reason about the visual input.

This layer achieves this alignment through a two-stage process that prioritizes
computational efficiency:

1.  **Efficient Token Reduction via Pixel Shuffle:**
    -   Before any learned projection, the layer first reduces the number of visual
        tokens. This is a crucial optimization for handling high-resolution visual
        features, as the computational cost of attention in the subsequent language
        model is quadratic with respect to the sequence length.
    -   Instead of using pooling or a strided convolution (which discard information),
        it employs `PixelShuffle` (a space-to-depth operation). This operation
        rearranges spatial information into the channel dimension, effectively merging
        small blocks of tokens (e.g., a 2x2 block) into a single, richer token. This
        reduces the token count significantly while preserving all the original
        information.

2.  **Cross-Modality Linear Projection:**
    -   After the token sequence has been shortened, the resulting feature vectors
        are passed through a standard linear (`Dense`) layer.
    -   This is the main projection step, which learns a mapping from the (rearranged)
        visual feature space to the target `output_dim`. This `output_dim` is typically
        chosen to match the embedding dimension of the language model that will
        receive these features.

3.  **Optional Post-Processing:**
    -   To stabilize the output and introduce additional non-linearity, the layer can
        optionally apply a `GELU` activation and a `LayerNormalization` layer. Layer
        normalization is standard practice in Transformer-based architectures to
        ensure the outputs are well-conditioned for the subsequent model layers.

In essence, this layer provides a computationally efficient and effective method to
not only align visual and textual representations but also to reduce the sequence
length of visual tokens, making the overall VLM more performant.
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

        self.input_dim, self.output_dim, self.scale_factor = input_dim, output_dim, scale_factor
        self.use_gelu, self.use_layer_norm = use_gelu, use_layer_norm
        self.projection_kernel_initializer = keras.initializers.get(projection_kernel_initializer)
        self.projection_bias_initializer = keras.initializers.get(projection_bias_initializer)
        self.projection_kernel_regularizer = keras.regularizers.get(projection_kernel_regularizer)
        self.projection_bias_regularizer = keras.regularizers.get(projection_bias_regularizer)

        self.pixel_shuffle = PixelShuffle(scale_factor=self.scale_factor, name='pixel_shuffle')
        self.projection_dense = keras.layers.Dense(
            units=self.output_dim, kernel_initializer=self.projection_kernel_initializer,
            bias_initializer=self.projection_bias_initializer, kernel_regularizer=self.projection_kernel_regularizer,
            bias_regularizer=self.projection_bias_regularizer, name='projection_dense'
        )
        self.projection_activation = keras.layers.Activation('gelu', name='projection_gelu') if self.use_gelu else None
        self.projection_norm = keras.layers.LayerNormalization(epsilon=1e-6, name='projection_norm') if self.use_layer_norm else None
        logger.info(f"Initialized ModalityProjection: {input_dim} -> {output_dim}, scale_factor={scale_factor}, gelu={use_gelu}, norm={use_layer_norm}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")
        if input_shape[-1] != self.input_dim:
            raise ValueError(f"Input shape last dim ({input_shape[-1]}) doesn't match input_dim ({self.input_dim})")
        super().build(input_shape)

    def call(self, inputs: keras.KerasTensor, training: Optional[bool] = None) -> keras.KerasTensor:
        x = self.pixel_shuffle(inputs)
        x = self.projection_dense(x)
        if self.projection_activation: x = self.projection_activation(x)
        if self.projection_norm: x = self.projection_norm(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        shuffled_shape = list(self.pixel_shuffle.compute_output_shape(input_shape))
        return tuple(shuffled_shape[:-1] + [self.output_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim, "output_dim": self.output_dim, "scale_factor": self.scale_factor,
            "use_gelu": self.use_gelu, "use_layer_norm": self.use_layer_norm,
            "projection_kernel_initializer": keras.initializers.serialize(self.projection_kernel_initializer),
            "projection_bias_initializer": keras.initializers.serialize(self.projection_bias_initializer),
            "projection_kernel_regularizer": keras.regularizers.serialize(self.projection_kernel_regularizer),
            "projection_bias_regularizer": keras.regularizers.serialize(self.projection_bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
