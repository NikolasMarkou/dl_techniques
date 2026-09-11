"""
Keypoint detection and description in one forward pass.

Defines :class:`SuperPoint`, which returns a keypoint logit grid and a
descriptor field from one shared encoder. Detection is a 65-way choice per
8x8 pixel cell on an H/8 x W/8 grid: 64 classes for the pixels of the cell
and one dustbin class for "no keypoint", so pixel-level detection needs no
decoder and the softmax keeps at most one keypoint per cell. The descriptor
head predicts a coarse field on the same grid, resizes it bicubically to the
input size the model was built with, and L2-normalizes each pixel, so
comparing two descriptors is one dot product. The encoder is a three-stage
ConvNeXt V2 run at strides=2, which puts the last stage at H/8. A caller
gets raw logits; the softmax, the dropped dustbin channel and the reshape
from (H/8, W/8, 64) to (H, W, 1) happen outside this model. H and W should
be divisible by 8, and the descriptor output size is fixed at construction.

References:
    - DeTone et al., 2018. SuperPoint: Self-Supervised Interest Point Detection
      and Description. CVPR 2018 Workshops.
      (https://arxiv.org/abs/1712.07629)
    - Woo et al., 2023. ConvNeXt V2: Co-designing and Scaling ConvNets with
      Masked Autoencoders. CVPR 2023. (https://arxiv.org/abs/2301.00808)
    - Sarlin et al., 2020. SuperGlue: Learning Feature Matching with Graph
      Neural Networks. CVPR 2020. (https://arxiv.org/abs/1911.11763)
    - Lowe, 2004. Distinctive Image Features from Scale-Invariant Keypoints.
      IJCV 60(2).

"""

import keras
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any, Sequence

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision.convnext.convnext_v2 import ConvNeXtV2
from dl_techniques.utils.activation_serialization import (
    serialize_activation,
    deserialize_activation,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.superpoint.model")
class SuperPoint(keras.Model):
    """Detect interest points and describe them in a single forward pass.

    Runs a ConvNeXt V2 encoder and a shared 1x1 neck, then splits into a
    detector head that emits 65-class logits on an H/8 x W/8 grid and a
    descriptor head whose coarse field is resized to full resolution and
    L2-normalized per pixel. The detector output is raw logits; the softmax
    is applied by the loss.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
                │
                ▼
        ┌───────────────────────────────────┐
        │ ConvNeXtV2  strides=2, no top     │
        │ stem /2, stage /4, stage /8       │
        └───────────────────────────────────┘
                │ [B, H/8, W/8, dims[-1]]
                ▼
        ┌───────────────────────────────────┐
        │ proj  conv 1x1  (shared neck)     │
        └───────────────────────────────────┘
                │ [B, H/8, W/8, descriptor_dim]
                ├──────────────────────┐
                ▼                      ▼
        ┌────────────────┐  ┌──────────────────────┐
        │ detector_head  │  │ descriptor_head      │
        │ conv 1x1       │  │ conv 1x1             │
        └────────────────┘  └──────────────────────┘
                │                      │
                │                      ▼
                │           ┌──────────────────────┐
                │           │ resize bicubic       │
                │           │ float32, to (H, W)   │
                │           └──────────────────────┘
                │                      │
                │                      ▼
                │           ┌──────────────────────┐
                │           │ l2 normalize, axis-1 │
                │           └──────────────────────┘
                ▼                      ▼
           "keypoints"            "descriptors"
           [B, H/8, W/8, 65]      [B, H, W, descriptor_dim]

    The resize target is the construction-time (H, W), not the runtime one.

    Detector channels:

    .. code-block:: text

        ┌──────────────────────────────────────────┐
        │ channels 0..63   one per pixel of the    │
        │                  8x8 cell                │
        ├──────────────────────────────────────────┤
        │ channel 64       dustbin: no keypoint    │
        │                  in this cell            │
        └──────────────────────────────────────────┘

    A softmax over the 65 channels makes the cell classes exclusive.

    Variants:

    .. code-block:: text

        variant   depths        dims
        tiny      [3, 3, 9]     [96, 192, 384]
        base      [3, 3, 27]    [128, 256, 512]
        large     [3, 3, 27]    [192, 384, 768]

    :param depths: Sequence[int], ConvNeXt V2 blocks per stage (3 stages).
        Default `(3, 3, 9)`. Length must equal `len(dims)`.
    :param dims: Sequence[int], channel width per stage (3 stages). Default
        `(96, 192, 384)`.
    :param input_shape: Tuple[int, int, int], `(height, width, channels)`.
        Default `(256, 256, 1)` for grayscale. H and W should be divisible by 8
        so the coarse maps are exactly `H/8 x W/8`.
    :param descriptor_dim: int, descriptor channel count and neck width. Default `256`.
    :param drop_path_rate: float, stochastic-depth rate for the encoder. Default `0.0`.
    :param kernel_size: int or tuple, ConvNeXt V2 block kernel size. Default `7`.
    :param activation: str or callable, ConvNeXt V2 block activation. Default `"gelu"`.
    :param use_bias: bool, whether encoder and head convolutions use bias. Default `True`.
    :param kernel_regularizer: Optional regularizer for encoder and head kernels.
    :param **kwargs: forwarded to `keras.Model`.

    Input shape:
        4D tensor `(batch, height, width, channels)`.

    Output shape:
        A dict:
            - `"keypoints"`: `(batch, height // 8, width // 8, 65)` raw logits.
            - `"descriptors"`: `(batch, height, width, descriptor_dim)`, unit-L2 along axis -1.

    Example:
        >>> model = SuperPoint.from_variant("tiny", input_shape=(256, 256, 1))
        >>> out = model(keras.ops.zeros((1, 256, 256, 1)))
        >>> out["keypoints"].shape, out["descriptors"].shape
        ((1, 32, 32, 65), (1, 256, 256, 256))
    """

    # 3-stage slices of ConvNeXt V2 tiny / base / large.
    MODEL_VARIANTS = {
        "tiny": {"depths": [3, 3, 9], "dims": [96, 192, 384]},
        "base": {"depths": [3, 3, 27], "dims": [128, 256, 512]},
        "large": {"depths": [3, 3, 27], "dims": [192, 384, 768]},
    }

    # 64 pixels of an 8x8 cell plus one dustbin class.
    DETECTOR_CHANNELS = 65
    ENCODER_STRIDES = 2

    def __init__(
            self,
            depths: Sequence[int] = (3, 3, 9),
            dims: Sequence[int] = (96, 192, 384),
            input_shape: Tuple[int, int, int] = (256, 256, 1),
            descriptor_dim: int = 256,
            drop_path_rate: float = 0.0,
            kernel_size: Union[int, Tuple[int, int]] = 7,
            activation: str = "gelu",
            use_bias: bool = True,
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (H, W, C), got {input_shape}")
        if descriptor_dim <= 0:
            raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}")

        self.depths = list(depths)
        self.dims = list(dims)
        self._input_shape = tuple(input_shape)
        self.descriptor_dim = descriptor_dim
        self.drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

        # Static spatial dims give the descriptor resize a graph-safe target.
        self.input_height, self.input_width, self.input_channels = self._input_shape

        # DECISION plan_2026-06-18_e1411ebf/D-001: encoder runs at strides=2; the default
        # 4 gives /4, /16, /64 and never reaches H/8. See decisions.md.
        self.encoder = ConvNeXtV2(
            depths=self.depths,
            dims=self.dims,
            strides=self.ENCODER_STRIDES,
            include_top=False,
            drop_path_rate=self.drop_path_rate,
            kernel_size=self.kernel_size,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            input_shape=self._input_shape,
            name="encoder",
        )

        # Both heads read this neck, so descriptor_dim sets its width.
        self.proj = keras.layers.Conv2D(
            filters=self.descriptor_dim,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            name="proj",
        )

        # Emits raw logits; the softmax belongs to the loss.
        self.detector_head = keras.layers.Conv2D(
            filters=self.DETECTOR_CHANNELS,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            name="detector_head",
        )

        self.descriptor_head = keras.layers.Conv2D(
            filters=self.descriptor_dim,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            name="descriptor_head",
        )

        logger.info(
            f"Created SuperPoint (depths={self.depths}, dims={self.dims}, "
            f"descriptor_dim={self.descriptor_dim}) for input {self._input_shape}"
        )

    def build(self, input_shape):
        """Build the encoder, the neck and both heads in forward order.

        Building every sublayer here instead of on the first call means all
        weights exist before a `.keras` weight restore, which otherwise skips
        weights that do not yet exist.

        :param input_shape: input shape tuple `(batch, H, W, C)`.
        """
        self.encoder.build(input_shape)
        encoder_out_shape = self.encoder.compute_output_shape(input_shape)

        self.proj.build(encoder_out_shape)
        neck_shape = self.proj.compute_output_shape(encoder_out_shape)

        self.detector_head.build(neck_shape)
        self.descriptor_head.build(neck_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Run the encoder, the neck and both heads.

        :param inputs: 4D tensor `(batch, height, width, channels)`.
        :param training: bool or None, training-mode flag forwarded to sublayers.

        :return: Dict with `"keypoints"` (raw logits, `(B, H/8, W/8, 65)`) and `"descriptors"`
            (unit-L2 along channels, `(B, H, W, descriptor_dim)`).
        """
        feat = self.encoder(inputs, training=training)
        neck = self.proj(feat, training=training)

        keypoints = self.detector_head(neck, training=training)
        desc_coarse = self.descriptor_head(neck, training=training)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-060: resize in float32 and cast back;
        # at float16 TensorFlow's ResizeBicubic returns a None gradient. See decisions.md.
        desc = keras.ops.image.resize(
            keras.ops.cast(desc_coarse, "float32"),
            size=(self.input_height, self.input_width),
            interpolation="bicubic",
        )
        desc = keras.ops.cast(desc, self.compute_dtype)

        # DECISION plan-2026-08-19T163559-499b6f0e/D-050: the floor is finfo(dtype).tiny;
        # np.float16(1e-12) is 0.0, so a zero descriptor gave NaN. See decisions.md.
        norm_eps = max(1e-12, float(np.finfo(self.compute_dtype).tiny))
        desc = desc / (keras.ops.norm(desc, axis=-1, keepdims=True) + norm_eps)

        return {"keypoints": keypoints, "descriptors": desc}

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple]:
        """Compute the output shapes of both heads.

        :param input_shape: input shape tuple `(batch, H, W, C)`. Anything that is
            not 4D falls back to the construction-time height and width.

        :return: Dict mapping `"keypoints"` and `"descriptors"` to their output shapes.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-119: only the detector grid reads
        # input_shape; descriptors keep the construction-time size. See decisions.md.
        batch = input_shape[0] if len(input_shape) == 4 else None
        stride = self.ENCODER_STRIDES ** len(self.depths)
        height = input_shape[-3] if len(input_shape) == 4 else self.input_height
        width = input_shape[-2] if len(input_shape) == 4 else self.input_width
        grid_h = height // stride if height is not None else None
        grid_w = width // stride if width is not None else None
        return {
            "keypoints": (batch, grid_h, grid_w, self.DETECTOR_CHANNELS),
            "descriptors": (batch, self.input_height, self.input_width, self.descriptor_dim),
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments needed to rebuild this model.

        :return: Dict of serialized config values.
        """
        config = super().get_config()
        config.update({
            "depths": self.depths,
            "dims": self.dims,
            "input_shape": self._input_shape,
            "descriptor_dim": self.descriptor_dim,
            "drop_path_rate": self.drop_path_rate,
            "kernel_size": self.kernel_size,
            "activation": serialize_activation(self.activation),
            "use_bias": self.use_bias,
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SuperPoint":
        """Rebuild a SuperPoint instance from a config dict.

        :param config: Dict as returned by :meth:`get_config`.

        :return: A `SuperPoint` instance.
        """
        if config.get("kernel_regularizer") is not None:
            config["kernel_regularizer"] = keras.regularizers.deserialize(
                config["kernel_regularizer"]
            )
        return cls(**config)

    @classmethod
    def from_variant(cls, variant: str, **kwargs) -> "SuperPoint":
        """Create a SuperPoint model from a named variant.

        :param variant: one of `"tiny"`, `"base"`, `"large"`.
        :param **kwargs: forwarded to the constructor (e.g. `input_shape`, `descriptor_dim`).

        :return: A `SuperPoint` instance.

        :raises ValueError: if `variant` is not a known variant.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )
        cfg = cls.MODEL_VARIANTS[variant]
        logger.info(f"Creating SuperPoint-{variant.upper()}")
        return cls(depths=cfg["depths"], dims=cfg["dims"], **kwargs)


# ---------------------------------------------------------------------


def create_superpoint(
        variant: str = "base",
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        **kwargs
) -> SuperPoint:
    """Build a SuperPoint model from a variant name and an input shape.

    :param variant: one of `"tiny"`, `"base"`, `"large"`. Default `"base"`.
    :param input_shape: `(height, width, channels)`. Default `(256, 256, 1)`.
    :param **kwargs: forwarded to `SuperPoint.from_variant`.

    :return: A `SuperPoint` instance.
    """
    return SuperPoint.from_variant(variant, input_shape=input_shape, **kwargs)

# ---------------------------------------------------------------------
