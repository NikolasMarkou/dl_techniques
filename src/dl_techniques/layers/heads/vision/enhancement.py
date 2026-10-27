"""Enhancement head: a restored or upscaled image."""

from typing import Dict, Optional, Sequence, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .utils import (
    _refuse_common_layers_call_never_runs,
    _validated_block_dims,
)


@register_dl_technique("dl_techniques.layers.heads.vision.enhancement")
class EnhancementHead(BaseVisionHead):
    """
    A restored or upscaled image, built from a feature map.

    Use this for denoising, super-resolution and the other image-to-image
    tasks. ``ConvBlock`` stages refine the features, one per ``block_dims``
    entry (three at ``hidden_dim`` by default). Then one conv produces the
    image: a transposed conv when ``scale_factor > 1``, and a 3x3
    same-resolution conv otherwise.

    ``block_dims=()`` builds no refinement block, so the head is exactly one
    3x3 conv to ``output_channels``: the tail conv of a restoration network
    such as SCUNet's ``m_tail`` (``output_use_bias=False``) or DarkIR's
    ``ending`` (bias kept). Any long skip or global residual around that
    conv is the model's to apply; the head adds none.

    This head applies none of the common stages and no dropout, so it builds
    none: ``use_attention`` and ``use_ffn`` default to ``False``,
    ``dropout_rate`` to ``0.0``, and any other value raises ``ValueError``
    (its ``ConvBlock`` stages take no dropout). ``build`` does not call
    ``_common_processed_shape`` either. The head does read ``hidden_dim``,
    ``normalization_type`` and ``activation_type``, which it hands to its own
    ``ConvBlock`` stages. Its output is unactivated: a restored image, not a
    score.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                          │
                          ▼
        ┌─────────────────────────────────┐
        │ enhance_blocks: one ConvBlock   │ default: 3 x hidden_dim
        │ per block_dims entry (0 or more)│ () : none
        └────────────────┬────────────────┘
                          ▼
                 ┌────────┴────────┐
                 ▼                 ▼
          scale_factor > 1     otherwise
        ┌─────────────────┐ ┌─────────────────┐
        │ upsample        │ │ output_conv     │
        │ Conv2DTranspose │ │ Conv2D 3x3      │
        └────────┬────────┘ └────────┬────────┘
                 └─────────┬─────────┘
                           ▼
                       'enhanced'

        enhanced: (B, H*s, W*s, output_channels) when s > 1,
                  else (B, H, W, output_channels). s = scale_factor.

    The class name is frozen. This class once lived inside
    ``create_enhancement_head`` as a closure-local registered class, and was
    lifted to module scope. Do NOT re-nest it there. Its decorator is now
    ``@register_dl_technique("dl_techniques.layers.heads.vision.enhancement")``,
    so ``get_registered_name`` resolves
    ``dl_techniques.layers.heads.vision.enhancement>EnhancementHead`` (it was
    ``...vision.factory>EnhancementHead`` until 2026-10-27, when the key moved
    to the defining module). The helper also mints a ``Custom>EnhancementHead``
    alias keyed on the bare class name, so a rename would drop it.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'enhanced': (batch, out_height, out_width, output_channels)}``.

    :param output_channels: Channel count of the output image.
    :type output_channels: int
    :param scale_factor: Spatial upscaling factor. Above 1 selects the
        transposed-conv path; 1 selects the same-resolution conv.
    :type scale_factor: int
    :param block_dims: Width of each refinement ``ConvBlock``, in order.
        ``None`` means ``(hidden_dim, hidden_dim, hidden_dim)``. ``()``
        builds no block, so the output conv reads the input directly.
    :type block_dims: Optional[Sequence[int]]
    :param output_use_bias: Whether the output conv (``upsample`` or
        ``output_conv``) has a bias. Defaults to ``True``.
    :type output_use_bias: bool
    :param kwargs: Forwarded to :class:`BaseVisionHead`.

    :raises ValueError: If ``use_attention`` or ``use_ffn`` is ``True``,
        ``dropout_rate`` is above 0, or ``block_dims`` holds a width that is
        not a positive integer, or ``output_channels`` or ``scale_factor`` is
        not a positive integer.

    :ivar enhance_blocks: One refinement ``ConvBlock`` per resolved
        ``block_dims`` entry, named ``enhance_block_{i}``.
    :vartype enhance_blocks: List[ConvBlock]
    :ivar upsample: Transposed conv, built only when ``scale_factor > 1``.
    :vartype upsample: keras.layers.Conv2DTranspose
    :ivar output_conv: Output conv, built only when ``scale_factor <= 1``.
    :vartype output_conv: keras.layers.Conv2D
    """

    def __init__(
            self,
            output_channels: int = 3,
            scale_factor: int = 1,
            block_dims: Optional[Sequence[int]] = None,
            output_use_bias: bool = True,
            **kwargs
    ):
        """
        Store the configuration and build the enhancement layers.

        No common layer is built: this head's ``call`` applies none. Only one
        of ``upsample`` and ``output_conv`` is built, chosen by
        ``scale_factor``.

        :param output_channels: Channel count of the output image.
        :type output_channels: int
        :param scale_factor: Spatial upscaling factor.
        :type scale_factor: int
        :param block_dims: Refinement block widths; ``None`` is three blocks
            at ``hidden_dim``, ``()`` is none.
        :type block_dims: Optional[Sequence[int]]
        :param output_use_bias: Whether the output conv has a bias.
        :type output_use_bias: bool
        :param kwargs: Forwarded to :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``use_attention`` or ``use_ffn`` is ``True``,
            ``dropout_rate`` is above 0, a ``block_dims`` entry is not a
            positive integer, or ``output_channels`` or ``scale_factor`` is
            not a positive integer.
        """
        _refuse_common_layers_call_never_runs(
            kwargs, type(self).__name__,
            ('use_attention', 'use_ffn', 'dropout_rate'))
        super().__init__(**kwargs)
        owner = type(self).__name__
        self.output_channels = positive_int(
            output_channels, 'output_channels', owner)
        self.scale_factor = positive_int(scale_factor, 'scale_factor', owner)
        self.block_dims = _validated_block_dims(block_dims)
        self.output_use_bias = bool(output_use_bias)
        resolved_dims = (
            (self.hidden_dim,) * 3 if self.block_dims is None
            else self.block_dims
        )

        # Enhancement-specific layers. Named explicitly: unnamed, each took
        # Keras' process-global auto suffix (`conv_block_102`, `conv2d_20`),
        # so two instances of one model disagreed by weight path and a
        # `.keras` reload drifted every path under the head.
        self.enhance_blocks = [
            ConvBlock(
                filters=filters,
                kernel_size=3,
                normalization_type=self.normalization_type,
                activation_type=self.activation_type,
                name=f'enhance_block_{i}'
            )
            for i, filters in enumerate(resolved_dims)
        ]

        if self.scale_factor > 1:
            # For super-resolution
            self.upsample = keras.layers.Conv2DTranspose(
                filters=self.output_channels,
                kernel_size=3,
                strides=self.scale_factor,
                padding='same',
                use_bias=self.output_use_bias,
                name='upsample'
            )
        else:
            # For denoising and other tasks
            self.output_conv = keras.layers.Conv2D(
                filters=self.output_channels,
                kernel_size=3,
                padding='same',
                use_bias=self.output_use_bias,
                name='output_conv'
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the refinement stack and the output conv.

        Each ``ConvBlock`` is built on the shape the previous one returns.
        The head owns no common layer, so the base ``build`` only marks it
        built.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = tuple(input_shape)
        for block in self.enhance_blocks:
            block.build(shape)
            shape = tuple(block.compute_output_shape(shape))

        if self.scale_factor > 1:
            self.upsample.build(shape)
        else:
            self.output_conv.build(shape)

        # This head owns no common layer, so this only marks it built.
        super().build(input_shape)

    def call(self, inputs, training=None):
        """
        Refine the features, then produce the output image.

        No common stage runs here, and none is built. See the class
        docstring.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to each ``ConvBlock``.
        :type training: Optional[bool]
        :return: Dict with a single ``'enhanced'`` key.
        :rtype: Dict[str, keras.KerasTensor]
        """
        x = inputs

        for block in self.enhance_blocks:
            x = block(x, training=training)

        if self.scale_factor > 1:
            x = self.upsample(x)
        else:
            x = self.output_conv(x)

        return {'enhanced': x}

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shape.

        With ``scale_factor > 1`` the transposed conv scales height and width
        by ``scale_factor``. Otherwise the output conv keeps them. Channels
        become ``output_channels`` either way.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with the ``'enhanced'`` output shape.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        scale = self.scale_factor if self.scale_factor > 1 else 1
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        return {'enhanced': (batch, out_height, out_width, self.output_channels)}

    def get_config(self):
        """
        Return the constructor arguments for serialization.

        Adds this head's four arguments to the base configuration.

        :return: Config dict carrying ``output_channels``, ``scale_factor``,
            ``block_dims`` and ``output_use_bias``.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'scale_factor': self.scale_factor,
            'block_dims': (
                None if self.block_dims is None else list(self.block_dims)
            ),
            'output_use_bias': self.output_use_bias,
        })
        return config
