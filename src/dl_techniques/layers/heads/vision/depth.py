"""Depth estimation head: one depth map at 8x the input resolution."""

import math
import numbers
from typing import Any, Dict, Optional, Tuple

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..validation import positive_int
from dl_techniques.utils.keras_registration import register_dl_technique
from .base import BaseVisionHead
from .decoder import (
    _create_progressive_decoder,
    _build_progressive_decoder,
    _run_progressive_decoder,
)
from .utils import _refuse_dropout_no_layer_applies

#: Largest finite float16: a range above it scales to inf under ``mixed_float16``.
_FLOAT16_MAX = 65504.0


@register_dl_technique("dl_techniques.layers.heads.vision.depth")
class DepthEstimationHead(BaseVisionHead):
    """
    A depth map, at eight times the input resolution.

    Three stages each refine the features and then double height and width,
    so the output is ``8H`` by ``8W``. A sigmoid conv produces a normalized
    map, which is then scaled into ``[min_depth, max_depth]``.

    **Architecture Overview:**

    .. code-block:: text

        inputs (B, H, W, C)
                         │
                         ▼
        ┌─────────────────────────────────┐
        │ attention            (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ ffn                  (optional) │
        └────────────────┬────────────────┘
                         ▼
        ┌─────────────────────────────────┐
        │ 3 x (depth_conv_blocks[i], then │
        │ depth_upsamples[i] stride 2)    │
        └────────────────┬────────────────┘
                         ▼ 8x spatial, channels halved 3x
        ┌─────────────────────────────────┐
        │ depth_head: Conv2D 3x3  sigmoid │
        └────────────────┬────────────────┘
                         ▼ depth_normalized, in [0, 1]
                 ┌───────┴────────────┐
                 ▼                    ▼
        ┌──────────────────┐ ┌──────────────────┐
        │ use_log_depth:   │ │ passed through   │
        │ exp(...) or a    │ │ with no change   │
        │ linear rescale   │ │                  │
        └────────┬─────────┘ └────────┬─────────┘
                 ▼                    ▼
              'depth'            'confidence'

        depth, confidence: (B, 8H, 8W, output_channels)

    ``'confidence'`` is ``depth_normalized`` itself, returned unchanged. It
    is the sigmoid output the depth was scaled from, not a second estimate
    and not an uncertainty. Treat it as a confidence only if you have checked
    that reading against your data.

    The scaling has two forms. With ``use_log_depth`` the map is read as a
    position in log space: ``exp(x * (log(max_depth) - log(min_depth)) +
    log(min_depth))``. Otherwise it is read linearly: ``x * (max_depth -
    min_depth) + min_depth``.

    Surface normals and optical flow have heads of their own,
    :class:`SurfaceNormalsHead` and :class:`OpticalFlowHead`: they share
    this progressive decoder, but neither output is a depth range.

    The decoder applies no dropout (the DPT/MiDaS depth head has none), so
    ``dropout_rate`` defaults to 0.0 here and a positive rate is refused
    unless ``use_attention`` or ``use_ffn`` is on, where it is that stage's
    dropout.

    A ``decoder_style='dpt'`` mode (Conv-BatchNorm-ReLU stages over one
    feature map) existed until 2026-10-20; it was deleted with every
    ``dpt_*`` argument when its last consumer, DepthAnything, moved to the
    reference DPT head in ``models/vision/depth_anything`` (D-222 of
    plan-2026-10-14T062616-81938e81). Keras refuses either keyword now
    (``ValueError``, an unrecognized keyword argument).

    Input shape:
        ``(batch, height, width, channels)``.

    Output shape:
        ``{'depth': (batch, 8 * height, 8 * width, output_channels),
        'confidence': the same shape}``.

    :param output_channels: Channel count of the output map. 1 for depth.
    :type output_channels: int
    :param min_depth: Lower end of the output range. Finite and below
        ``max_depth``; also above 0 with ``use_log_depth``.
    :type min_depth: float
    :param max_depth: Upper end of the output range. Finite, and at most
        65504 when the head computes in float16.
    :type max_depth: float
    :param use_log_depth: Scale in log space instead of linearly.
    :type use_log_depth: bool
    :param kwargs: Arguments for :class:`BaseVisionHead`.

    :ivar depth_conv_blocks: The three refinement ``ConvBlock`` s.
    :vartype depth_conv_blocks: List[ConvBlock]
    :ivar depth_upsamples: The three stride-2 ``Conv2DTranspose`` layers,
        each applied after the ``ConvBlock`` of the same index.
    :vartype depth_upsamples: List[keras.layers.Conv2DTranspose]
    :ivar depth_head: Output ``Conv2D(output_channels)`` with sigmoid.
    :vartype depth_head: keras.layers.Conv2D
    """

    def __init__(
            self,
            output_channels: int = 1,
            min_depth: float = 0.1,
            max_depth: float = 100.0,
            use_log_depth: bool = True,
            **kwargs: Any
    ) -> None:
        """
        Store the configuration, then build the depth layers.

        :param output_channels: Channel count of the output map.
        :type output_channels: int
        :param min_depth: Lower end of the output range.
        :type min_depth: float
        :param max_depth: Upper end of the output range.
        :type max_depth: float
        :param use_log_depth: Scale in log space instead of linearly.
        :type use_log_depth: bool
        :param kwargs: Arguments for :class:`BaseVisionHead`.
        :return: None.
        :rtype: None
        :raises ValueError: If the depth range cannot be scaled into.
        """
        # Set defaults for DepthEstimationHead before calling parent constructor
        kwargs.setdefault('use_attention', False)
        kwargs.setdefault('use_ffn', False)
        _refuse_dropout_no_layer_applies(kwargs, type(self).__name__)
        super().__init__(**kwargs)
        self.output_channels = positive_int(
            output_channels, 'output_channels', type(self).__name__)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_log_depth = use_log_depth
        self._check_depth_range()

        # Progressive upsampling with refinement: the shared decoder, named
        # depth_conv_{i} / depth_upsample_{i} (two flat lists).
        self.depth_conv_blocks, self.depth_upsamples = (
            _create_progressive_decoder(self, 'depth'))

        # Depth prediction layer
        self.depth_head = keras.layers.Conv2D(
            filters=self.output_channels,
            kernel_size=3,
            padding='same',
            # Sigmoid keeps the output in [0, 1]. call() then scales it into
            # the [min_depth, max_depth] range.
            activation='sigmoid',
            name='depth_head'
        )

    def _check_depth_range(self) -> None:
        """
        Refuse a depth range the scaling in ``call`` cannot produce.

        A range that is not finite and increasing gives NaN or an inverted
        map, ``log(min_depth)`` needs ``min_depth > 0``, and a ``max_depth``
        above the float16 maximum scales to inf in that dtype.

        :return: None.
        :rtype: None
        :raises ValueError: Naming the head and the range.
        """
        owner = type(self).__name__
        for name, value in (('min_depth', self.min_depth), ('max_depth', self.max_depth)):
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value)):
                raise ValueError(
                    f"{owner}: {name} must be a finite number, got {value!r}"
                )
        if self.min_depth >= self.max_depth:
            raise ValueError(
                f"{owner}: min_depth must be below max_depth, got "
                f"min_depth={self.min_depth!r}, max_depth={self.max_depth!r}"
            )
        if self.use_log_depth and self.min_depth <= 0:
            raise ValueError(
                f"{owner}: use_log_depth=True takes log(min_depth), so min_depth "
                f"must be above 0, got {self.min_depth!r}. Pass "
                f"use_log_depth=False for a range that starts at 0."
            )
        if self.compute_dtype == 'float16' and self.max_depth > _FLOAT16_MAX:
            raise ValueError(
                f"{owner}: max_depth={self.max_depth!r} is above the float16 "
                f"maximum {_FLOAT16_MAX} and would scale to inf; lower it or "
                f"compute this head in float32."
            )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the depth stack on the common-processed shape.

        :param input_shape: Input feature-map shape.
        :type input_shape: Tuple[Optional[int], ...]
        :return: None.
        :rtype: None
        """
        shape = _build_progressive_decoder(
            self.depth_conv_blocks, self.depth_upsamples,
            self._common_processed_shape(input_shape))
        self.depth_head.build(shape)

        # The common attention and ffn, on the raw input shape.
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> Dict[str, keras.KerasTensor]:
        """
        Refine, upsample, predict, then scale into the depth range.

        :param inputs: Feature map of shape ``(batch, height, width,
            channels)``.
        :type inputs: keras.KerasTensor
        :param training: Keras training flag, forwarded to every sub-layer.
        :type training: Optional[bool]
        :return: Dict with ``'depth'`` and ``'confidence'``, where
            ``'confidence'`` is the unscaled sigmoid output.
        :rtype: Dict[str, keras.KerasTensor]
        """

        x = inputs

        # Apply common processing
        if self.use_attention:
            x = self._apply_attention(x, training=training)
        if self.use_ffn:
            x = self.ffn(x, training=training)

        # Progressive refinement and upsampling
        x = _run_progressive_decoder(
            self.depth_conv_blocks, self.depth_upsamples, x, training)

        # Predict normalized depth
        depth_normalized = self.depth_head(x)

        # Scale to actual depth range
        if self.use_log_depth:
            # Convert from log space. The logs are taken in the map's dtype:
            # keras.ops.log of a Python float is float32, which TensorFlow
            # refuses to multiply with a float16 map.
            dtype = depth_normalized.dtype
            log_min = keras.ops.log(keras.ops.cast(self.min_depth, dtype))
            log_max = keras.ops.log(keras.ops.cast(self.max_depth, dtype))
            depth = keras.ops.exp(depth_normalized * (log_max - log_min) + log_min)
        else:
            # Linear scaling
            depth = depth_normalized * (self.max_depth - self.min_depth) + self.min_depth

        return {
            'depth': depth,
            # This is the sigmoid output the depth was scaled from, not a
            # separate estimate. Read the class docstring before using it.
            'confidence': depth_normalized
        }

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """
        Report the two output shapes.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Dict with ``'depth'`` and ``'confidence'`` shapes.
        :rtype: Dict[str, Tuple[Optional[int], ...]]
        """
        batch, height, width = input_shape[0], input_shape[1], input_shape[2]
        # Three transposed-conv upsamples, stride 2 each.
        scale = 8
        out_height = height * scale if height is not None else None
        out_width = width * scale if width is not None else None
        shape = (batch, out_height, out_width, self.output_channels)
        return {'depth': shape, 'confidence': shape}

    def get_config(self) -> Dict[str, Any]:
        """
        Return the constructor arguments for serialization.

        :return: Config dict carrying ``output_channels``, ``min_depth``,
            ``max_depth`` and ``use_log_depth`` on top of the base
            configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'output_channels': self.output_channels,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'use_log_depth': self.use_log_depth,
        })
        return config
