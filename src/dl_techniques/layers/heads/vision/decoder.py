"""Progressive decoder shared by the dense-regression heads.

Three stride-2 transposed-conv stages (each preceded by a ``ConvBlock``)
double the resolution and halve the width, so the output is 8x the input.
Used by :class:`DepthEstimationHead` (prefix ``'depth'``) and by
:class:`SurfaceNormalsHead` / :class:`OpticalFlowHead` (prefix = their
output key).

The two flat lists of layers (``conv_blocks`` and ``upsamples``) are kept
flat and paired by index — Keras does not save layers held in nested lists
(D-027).
"""

from typing import List, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from ..validation import positive_int
from .utils import _require_hidden_dim_at_least

#: Doublings of the progressive decoder: three stride-2 transposed convs, so
#: the output is 8x the input, and the width halves three times from
#: ``hidden_dim`` (``hidden_dim // 8`` channels at the end).
_PROGRESSIVE_DOUBLINGS = 3


def _create_progressive_decoder(
        head,
        prefix: str
) -> Tuple[List[ConvBlock], List[keras.layers.Conv2DTranspose]]:
    """
    Create the progressive decoder: three ``ConvBlock`` then stride-2
    ``Conv2DTranspose`` stages, each halving the width.

    Interface contract, shared by :class:`DepthEstimationHead` (progressive
    mode, ``prefix='depth'``) and :class:`SurfaceNormalsHead` /
    :class:`OpticalFlowHead` (``prefix`` = their output key):

    * Stage ``i`` is ``ConvBlock(hidden_dim // 2**i)`` named
      ``f'{prefix}_conv_{i}'``, then ``Conv2DTranspose(hidden_dim //
      2**(i+1), 3, strides=2)`` named ``f'{prefix}_upsample_{i}'``. The
      ``ConvBlock`` s take the head's ``normalization_type`` and
      ``activation_type`` and no dropout (anchor D-050).
    * Returns two FLAT lists paired by index (anchor D-027: a nested list is
      not saved by Keras).
    * Raises ``ValueError`` when ``hidden_dim < 8``, the width the last
      transposed conv needs.

    :param head: The head that owns the decoder; its ``hidden_dim``,
        ``normalization_type`` and ``activation_type`` are read.
    :type head: BaseVisionHead
    :param prefix: Layer-name prefix.
    :type prefix: str
    :return: ``(conv_blocks, upsamples)``.
    :rtype: Tuple[List[ConvBlock], List[keras.layers.Conv2DTranspose]]
    :raises ValueError: If ``hidden_dim`` is below 8.
    """
    _require_hidden_dim_at_least(
        head.hidden_dim, 2 ** _PROGRESSIVE_DOUBLINGS, type(head).__name__,
        "the progressive decoder halves it three times, and its last "
        "transposed conv takes hidden_dim // 8 channels")

    # DECISION plan-2026-10-14T062616-81938e81/D-027
    # Two FLAT lists, paired by index. Do NOT nest them back into a list of
    # [conv, upsample] pairs: Keras does not track a layer inside a nested
    # list for saving, so a .keras round trip silently restored none of these
    # six kernels (measured: the reloaded depth differed by up to 17.3 with no
    # warning). See decisions.md D-027.
    conv_blocks: List[ConvBlock] = []
    upsamples: List[keras.layers.Conv2DTranspose] = []
    channels = head.hidden_dim
    for i in range(_PROGRESSIVE_DOUBLINGS):
        conv_blocks.append(
            ConvBlock(
                filters=channels,
                kernel_size=3,
                normalization_type=head.normalization_type,
                activation_type=head.activation_type,
                name=f'{prefix}_conv_{i}'
            )
        )
        upsamples.append(
            keras.layers.Conv2DTranspose(
                filters=channels // 2,
                kernel_size=3,
                strides=2,
                padding='same',
                name=f'{prefix}_upsample_{i}'
            )
        )
        channels = channels // 2
    return conv_blocks, upsamples


def _build_progressive_decoder(
        conv_blocks: List[ConvBlock],
        upsamples: List[keras.layers.Conv2DTranspose],
        shape: Tuple[Optional[int], ...]
) -> Tuple[Optional[int], ...]:
    """
    Build the decoder from :func:`_create_progressive_decoder` stage by stage.

    :param conv_blocks: The decoder's ``ConvBlock`` s.
    :type conv_blocks: List[ConvBlock]
    :param upsamples: The decoder's transposed convs, paired by index.
    :type upsamples: List[keras.layers.Conv2DTranspose]
    :param shape: The decoder's input shape.
    :type shape: Tuple[Optional[int], ...]
    :return: The decoder's output shape.
    :rtype: Tuple[Optional[int], ...]
    """
    for conv_block, upsample in zip(conv_blocks, upsamples):
        conv_block.build(shape)
        shape = tuple(conv_block.compute_output_shape(shape))
        upsample.build(shape)
        shape = tuple(upsample.compute_output_shape(shape))
    return shape


def _run_progressive_decoder(
        conv_blocks: List[ConvBlock],
        upsamples: List[keras.layers.Conv2DTranspose],
        x: keras.KerasTensor,
        training: Optional[bool]
) -> keras.KerasTensor:
    """
    Apply the decoder from :func:`_create_progressive_decoder`.

    :param conv_blocks: The decoder's ``ConvBlock`` s.
    :type conv_blocks: List[ConvBlock]
    :param upsamples: The decoder's transposed convs, paired by index.
    :type upsamples: List[keras.layers.Conv2DTranspose]
    :param x: The decoder's input.
    :type x: keras.KerasTensor
    :param training: Keras training flag, forwarded to each ``ConvBlock``.
    :type training: Optional[bool]
    :return: The decoder's output, 8x the input resolution.
    :rtype: keras.KerasTensor
    """
    for conv_block, upsample in zip(conv_blocks, upsamples):
        x = conv_block(x, training=training)
        x = upsample(x)
    return x
