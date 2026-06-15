"""
Lossless 4D spatial-to-depth downsampling (pixel-unshuffle) with optional projection.

Implements the inverse of the standard pixel-shuffle (depth-to-space) on
``(B, H, W, C)`` feature maps: a stride-``s`` non-overlapping rearrangement
that reduces each spatial dimension by ``s`` while multiplying the channel
dimension by ``s**2``. The rearrangement is parameter-free and information-
preserving (no aliasing, no learnable kernel). When ``out_channels`` is set,
a 1x1 ``Conv2D`` projects the ``s**2 * C`` output back to a target channel
count, producing the canonical "pixel-unshuffle + 1x1" anti-alias-free
downsample used as the gradient-highway skip path in axis B of the
Clifford-algebra-compliant downsampling design space
(``analyses/analysis_2026-04-30_41b5e415/summary.md``).

The pre-existing ``PixelShuffle`` layer in this library operates on 3-D ViT
token tensors ``(B, 1+H*W, C)`` and is therefore not interchangeable with
this layer.

References:
    - Shi et al. (2016). "Real-Time Single Image and Video Super-Resolution
      Using an Efficient Sub-Pixel Convolutional Neural Network".
      https://arxiv.org/abs/1609.05158 (introduces depth-to-space; this is
      the inverse transform).
"""

from typing import Any, Dict, Optional

import keras
from keras import initializers, layers, ops, regularizers

from ..utils.logger import logger

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class PixelUnshuffle2D(keras.layers.Layer):
    """Lossless space-to-depth downsampling with optional 1x1 projection.

    Rearranges non-overlapping ``scale x scale`` spatial blocks into the
    channel dimension, mapping ``(B, H, W, C) -> (B, H/scale, W/scale,
    scale**2 * C)``. The rearrangement carries no parameters and preserves
    every input value. When ``out_channels`` is provided, a learnable 1x1
    ``Conv2D`` projects the rearranged tensor to ``out_channels``.

    :param scale: Downsampling factor for both spatial axes. Must divide
        the input height and width evenly. Default ``2``.
    :type scale: int
    :param out_channels: If provided, append a 1x1 ``Conv2D`` that projects
        ``scale**2 * C`` -> ``out_channels``. Default ``None`` (no projection).
    :type out_channels: Optional[int]
    :param use_bias: Whether the optional 1x1 projection has bias. Default
        ``True``.
    :type use_bias: bool
    :param kernel_initializer: Initialiser for the optional projection.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Bias initialiser for the optional projection.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional projection-kernel regulariser.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional projection-bias regulariser.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    Example:

    .. code-block:: python

        x = keras.layers.Input(shape=(32, 32, 96))
        # Lossless: -> (None, 16, 16, 384)
        y = PixelUnshuffle2D(scale=2)(x)
        # With projection: -> (None, 16, 16, 96)
        z = PixelUnshuffle2D(scale=2, out_channels=96)(x)
    """

    def __init__(
        self,
        scale: int = 2,
        out_channels: Optional[int] = None,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(scale, int) or scale < 1:
            raise ValueError(
                f"scale must be a positive integer, got {scale!r}"
            )
        if out_channels is not None and (
            not isinstance(out_channels, int) or out_channels < 1
        ):
            raise ValueError(
                f"out_channels must be a positive int or None, got "
                f"{out_channels!r}"
            )

        self.scale = scale
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.proj: Optional[layers.Conv2D] = None
        if out_channels is not None:
            self.proj = layers.Conv2D(
                filters=out_channels,
                kernel_size=1,
                strides=1,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="proj",
            )

    def build(self, input_shape: Any) -> None:
        if self.built:
            return
        c = input_shape[-1]
        if c is None:
            raise ValueError(
                "PixelUnshuffle2D requires a static channel dimension."
            )
        # H and W static-divisibility is checked at call time, not build time
        # (build receives the symbolic Input shape and must accept None HW).
        if self.proj is not None:
            self.proj.build((input_shape[0], None, None, c * self.scale ** 2))
        super().build(input_shape)
        logger.debug(
            f"PixelUnshuffle2D built: scale={self.scale}, "
            f"out_channels={self.out_channels}, in_channels={c}"
        )

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        s = self.scale
        # Fully dynamic shape so symbolic build with None H/W is fine.
        shape = ops.shape(inputs)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        # (B, H/s, s, W/s, s, C)
        x = ops.reshape(inputs, (b, h // s, s, w // s, s, c))
        # (B, H/s, W/s, s, s, C)
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        # (B, H/s, W/s, s*s*C)
        x = ops.reshape(x, (b, h // s, w // s, s * s * c))

        if self.proj is not None:
            x = self.proj(x, training=training)
        return x

    def compute_output_shape(self, input_shape: Any) -> Any:
        b, h, w, c = input_shape
        new_h = None if h is None else h // self.scale
        new_w = None if w is None else w // self.scale
        new_c = self.out_channels if self.out_channels is not None else (
            None if c is None else c * self.scale ** 2
        )
        return (b, new_h, new_w, new_c)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "scale": self.scale,
                "out_channels": self.out_channels,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------


# DECISION plan_2026-06-15_00924f53/D-002: keras.layers.DepthToSpace and
# keras.ops(.nn).depth_to_space do NOT exist in Keras 3.8; this is the NHWC
# pixel-shuffle (depth->space), the exact inverse of PixelUnshuffle2D. Do NOT
# replace this with keras.layers.DepthToSpace / keras.ops.nn.depth_to_space
# (neither symbol exists in this build) nor with a Lambda (breaks .keras
# round-trip). The reshape->transpose(0,1,3,2,4,5)->reshape order is the
# inverse of PixelUnshuffle2D.call and is pinned by the round-trip test in
# tests/test_layers/test_pixel_shuffle_2d.py. See decisions.md D-002.
@keras.saving.register_keras_serializable()
class PixelShuffle2D(keras.layers.Layer):
    """Lossless depth-to-space upsampling (pixel-shuffle), NHWC.

    Rearranges the channel dimension into non-overlapping
    ``block_size x block_size`` spatial blocks, mapping
    ``(B, H, W, C) -> (B, H*block_size, W*block_size, C/block_size**2)``. The
    rearrangement carries no parameters and preserves every input value. This
    is the exact inverse of :class:`PixelUnshuffle2D` (space-to-depth): for
    ``r = block_size``, ``PixelUnshuffle2D(scale=r)(PixelShuffle2D(r)(x)) == x``.

    :param block_size: Upsampling factor for both spatial axes. The channel
        dimension must be divisible by ``block_size**2``. Default ``2``.
    :type block_size: int
    :param kwargs: Additional keyword arguments for the Layer base class.

    Example:

    .. code-block:: python

        x = keras.layers.Input(shape=(16, 16, 12))
        # -> (None, 32, 32, 3)
        y = PixelShuffle2D(block_size=2)(x)
    """

    def __init__(self, block_size: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError(
                f"block_size must be a positive integer, got {block_size!r}"
            )
        self.block_size = block_size

    def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
        r = self.block_size
        # Fully dynamic shape so symbolic build with None H/W is fine.
        shape = ops.shape(inputs)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        # Inverse of PixelUnshuffle2D.call:
        # (B, H, W, r*r*C') -> (B, H, r, W, r, C')
        x = ops.reshape(inputs, (b, h, w, r, r, c // (r * r)))
        # (B, H, r, W, r, C')  -- (0,1,3,2,4,5) is its own inverse
        x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
        # (B, H*r, W*r, C')
        x = ops.reshape(x, (b, h * r, w * r, c // (r * r)))
        return x

    def compute_output_shape(self, input_shape: Any) -> Any:
        b, h, w, c = input_shape
        r = self.block_size
        new_h = None if h is None else h * r
        new_w = None if w is None else w * r
        new_c = None if c is None else c // (r * r)
        return (b, new_h, new_w, new_c)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config


# ---------------------------------------------------------------------
