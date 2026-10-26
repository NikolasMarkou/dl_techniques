"""Dense vector heads: surface normals and optical flow."""

from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ...conv_blocks.conv_block import ConvBlock
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .decoder import (
    _create_progressive_decoder,
    _build_progressive_decoder,
    _run_progressive_decoder,
    _PROGRESSIVE_DOUBLINGS,
)
from .utils import _refuse_dropout_no_layer_applies


def _unit_normalize(x: keras.KerasTensor) -> keras.KerasTensor:
    """
    L2-normalize the last axis, in float32 or wider, then return the input
    dtype.

    A float16 sum of squares overflows above about 256 per component (65504
    is float16's ceiling), and ``rsqrt`` of ``inf`` is 0, which would turn a
    large normal into the zero vector; so a narrower float is normalized in
    float32 and cast back. ``epsilon=1e-12`` is ``torch.nn.functional
    .normalize``'s, the call DSINE ends its normals head with.

    :param x: Tensor whose last axis holds the vectors.
    :type x: keras.KerasTensor
    :return: ``x / max(||x||, 1e-12)`` along the last axis, in ``x``'s dtype.
    :rtype: keras.KerasTensor
    """
    dtype = getattr(x.dtype, "name", None) or str(x.dtype)
    if dtype in ('float16', 'bfloat16'):
        normalized = keras.ops.normalize(
            keras.ops.cast(x, 'float32'), axis=-1, order=2, epsilon=1e-12)
        return keras.ops.cast(normalized, dtype)
    return keras.ops.normalize(x, axis=-1, order=2, epsilon=1e-12)


class _DenseVectorHeadBase(BaseVisionHead):
    """
    A per-pixel vector map at eight times the input resolution.

    The shared body of :class:`SurfaceNormalsHead` and
    :class:`OpticalFlowHead`: the optional attention and FFN, the progressive
    decoder :class:`DepthEstimationHead` uses (three ``ConvBlock`` then
    stride-2 ``Conv2DTranspose`` stages), then ``output_conv``, a linear 3x3
    conv to ``_CHANNELS`` channels, optionally L2-normalized per pixel.

    A subclass sets three class attributes: ``_OUTPUT_KEY`` (the one output
    key, also the layer-name prefix), ``_CHANNELS`` (fixed by the geometry,
    so there is no ``output_channels`` argument) and ``_UNIT_NORM``.

    The decoder applies no dropout, so ``dropout_rate`` defaults to 0.0 and
    a positive rate is refused unless ``use_attention`` or ``use_ffn`` is
    on. ``use_attention`` and ``use_ffn`` default to ``False``.

    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar decoder_conv_blocks: The three decoder ``ConvBlock`` s.
    :vartype decoder_conv_blocks: List[ConvBlock]
    :ivar decoder_upsamples: The three stride-2 ``Conv2DTranspose`` layers.
    :vartype decoder_upsamples: List[keras.layers.Conv2DTranspose]
    :ivar output_conv: Linear ``Conv2D(_CHANNELS, 3)``.
    :vartype output_conv: keras.layers.Conv2D
    """

    _OUTPUT_KEY: str = ''
    _CHANNELS: int = 0
    _UNIT_NORM: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """
        Build the decoder and the output conv.

        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If ``output_channels`` is passed, a positive
            ``dropout_rate`` would reach no layer, or ``hidden_dim < 8``.
        """
        owner = type(self).__name__
        if 'output_channels' in kwargs:
            raise ValueError(
                f"{owner} emits {self._CHANNELS} channels by definition and "
                f"takes no output_channels (got "
                f"{kwargs['output_channels']!r})."
            )
        kwargs.setdefault('use_attention', False)
        kwargs.setdefault('use_ffn', False)
        _refuse_dropout_no_layer_applies(kwargs, owner)
        super().__init__(**kwargs)

        self.decoder_conv_blocks, self.decoder_upsamples = (
            _create_progressive_decoder(self, self._OUTPUT_KEY))
        # Linear: RAFT's FlowHead and PWC-Net's predict_flow end in a 3x3 conv
        # to 2 channels with no activation; the normals heads of Bae et al.
        # 2021 and DSINE end in a linear conv that is then L2-normalized.
        self.output_conv = keras.layers.Conv2D(
            filters=self._CHANNELS,
            kernel_size=3,
            padding='same',
            name=f'{self._OUTPUT_KEY}_head'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the decoder and the output conv on the common-processed shape.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = _build_progressive_decoder(
            self.decoder_conv_blocks, self.decoder_upsamples,
            self._common_processed_shape(input_shape))
        self.output_conv.build(shape)
        # The common attention and ffn, on the raw input shape.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Run the common stages, the decoder and the output conv.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: ``{_OUTPUT_KEY: (batch, 8 * height, 8 * width,
            _CHANNELS)}``.
        :rtype: Dict[str, keras.KerasTensor]
        """
        x = inputs
        if self.use_attention:
            x = self._apply_attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)
        x = _run_progressive_decoder(
            self.decoder_conv_blocks, self.decoder_upsamples, x, training)
        x = self.output_conv(x)
        if self._UNIT_NORM:
            x = _unit_normalize(x)
        return {self._OUTPUT_KEY: x}

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the output shape: 8x the input height and width.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: ``{_OUTPUT_KEY: (batch, 8 * height, 8 * width, _CHANNELS)}``.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        scale = 2 ** _PROGRESSIVE_DOUBLINGS
        return {self._OUTPUT_KEY: (
            batch,
            height * scale if height is not None else None,
            width * scale if width is not None else None,
            self._CHANNELS,
        )}


@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class SurfaceNormalsHead(_DenseVectorHeadBase):
    """
    A unit surface normal per pixel, at eight times the input resolution.

    The progressive decoder, then a linear 3x3 conv to 3 channels, then a
    per-pixel L2 normalization, so every output vector has unit length and
    any sign. Reference: Bae et al. 2021 (``surface_normal_uncertainty``,
    ``norm_normalize``: the xyz channels over their L2 norm) and DSINE (Bae
    and Davison 2024: ``F.normalize(normal, dim=1)`` after a linear
    prediction head). The norm is taken in float32 under a float16 policy.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention, ffn       (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 3 x (normals_conv_i ConvBlock,  │
        │ normals_upsample_i stride 2)    │
        └────────────────┬────────────────┘
                         ▼ 8x spatial, hidden_dim // 8 channels
        ┌─────────────────────────────────┐
        │ normals_head: Conv2D 3x3, 3 ch  │
        │ linear                          │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ x / max(||x||_2, 1e-12)         │
        └────────────────┬────────────────┘
                         ▼
                     'normals'  (B, 8H, 8W, 3), unit length

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'normals': (batch, 8 * height, 8 * width, 3)}``.

    :param kwargs: Arguments for :class:`BaseVisionHead`. There is no
        ``output_channels``: a normal has 3 components.
    """

    _OUTPUT_KEY = 'normals'
    _CHANNELS = 3
    _UNIT_NORM = True


@register_dl_technique("dl_techniques.layers.heads.vision.factory")
class OpticalFlowHead(_DenseVectorHeadBase):
    """
    A signed 2D displacement per pixel, at eight times the input resolution.

    The progressive decoder, then a linear 3x3 conv to 2 channels: no
    activation and no range map, so the flow takes any sign and magnitude.
    Reference: RAFT's ``FlowHead`` (``conv2(relu(conv1(x)))``, ``conv2 =
    nn.Conv2d(hidden_dim, 2, 3, padding=1)``) and PWC-Net's ``predict_flow``
    (``nn.Conv2d(in_planes, 2, 3, 1, 1, bias=True)``).

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention, ffn       (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 3 x (flow_conv_i ConvBlock,     │
        │ flow_upsample_i stride 2)       │
        └────────────────┬────────────────┘
                         ▼ 8x spatial, hidden_dim // 8 channels
        ┌─────────────────────────────────┐
        │ flow_head: Conv2D 3x3, 2 ch     │
        │ linear                          │
        └────────────────┬────────────────┘
                         ▼
                       'flow'  (B, 8H, 8W, 2), signed

    The flow is in whatever units the training target uses; the head does
    not rescale it to the 8x output resolution.

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'flow': (batch, 8 * height, 8 * width, 2)}``.

    :param kwargs: Arguments for :class:`BaseVisionHead`. There is no
        ``output_channels``: a flow vector has 2 components.
    """

    _OUTPUT_KEY = 'flow'
    _CHANNELS = 2
    _UNIT_NORM = False
