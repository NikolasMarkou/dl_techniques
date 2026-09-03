"""Joint interest-point detection and description in a single forward pass.

Defines :class:`SuperPoint`, a shared-encoder model that predicts a keypoint
heatmap and a descriptor field from one representation, so the two tasks
share their evidence and train jointly under a self-supervised objective
derived from homographic warps rather than annotated keypoints.

Detection is classification, not regression: each 8x8 pixel cell at
resolution ``H/8 x W/8`` predicts which of its 64 pixels holds a keypoint,
or, via a 65th dustbin class, that it holds none. This resolves full pixel
detection with no decoder, gives implicit within-cell non-maximum
suppression through the softmax, and gives the "no keypoint" case an
explicit class instead of a low score. The descriptor head predicts a
coarse ``descriptor_dim``-channel field at the same resolution, resizes it
bicubically to full resolution, and L2-normalizes it per pixel, so matching
reduces to a single dot product. The encoder replaces the original VGG-style
network with a nested three-stage ConvNeXt V2 backbone run at
``strides=2`` rather than its default 4, which is what makes three stages
land on exactly ``H/8``.

Decoding the 65-channel output into pixel coordinates is not part of this
model: the forward pass returns raw logits (softmax lives in the loss), and
recovering a heatmap is a softmax, dropping the dustbin channel, then a
depth-to-space reshape from ``(H/8, W/8, 64)`` to ``(H, W, 1)``. The model
is tied to the ``input_shape`` it was built with; ``H`` and ``W`` should be
divisible by 8.

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

import numpy as np
import keras
from typing import List, Optional, Union, Tuple, Dict, Any, Sequence

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
    """SuperPoint interest-point detector + descriptor with a ConvNeXt V2 encoder.

    Produces, in a single forward pass, a keypoint-detection heatmap (raw
    logits over a 65-class 8x8-cell-plus-dustbin grid) and a full-resolution,
    unit-L2 descriptor field, sharing one ConvNeXt V2 encoder and a 1x1
    projection neck. The detector head emits logits; softmax lives in the
    loss, per repo convention. The descriptor field is L2-normalized along
    the channel axis at every pixel.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
            |
        ConvNeXtV2(strides=2, include_top=False, depths[:3], dims[:3])
            |  stem /2, stage 1 down /4, stage 2 down /8
        feat [B, H/8, W/8, dims[2]]
            |
        proj  Conv2D 1x1 -> [B, H/8, W/8, descriptor_dim]  (shared neck)
            +-----------------------------+
            |                             |
        detector_head Conv2D 1x1   descriptor_head Conv2D 1x1
        -> [B, H/8, W/8, 65]        -> [B, H/8, W/8, descriptor_dim]
           (logits)                        |
                                        resize bicubic -> [H, W]
                                        L2-normalize (axis=-1)
                                            |
                                     descriptors [B, H, W, descriptor_dim]

    :param depths: List[int], number of ConvNeXt V2 blocks per stage (3 stages). Default
        `[3, 3, 9]` (tiny). Length must equal `len(dims)`.
    :param dims: List[int], channel width per stage (3 stages). Default `[96, 192, 384]`.
    :param input_shape: Tuple[int, int, int], spatial+channel input shape
        `(height, width, channels)`. Default `(256, 256, 1)` (grayscale). H and W
        should be divisible by 8 so the semi-dense maps are exactly `H/8 x W/8`.
    :param descriptor_dim: int, descriptor channel count (and neck width). Default `256`.
    :param drop_path_rate: float, stochastic-depth rate forwarded to the encoder. Default `0.0`.
    :param kernel_size: int or tuple, ConvNeXt V2 block kernel size. Default `7`.
    :param activation: str or callable, ConvNeXt V2 block activation. Default `"gelu"`.
    :param use_bias: bool, whether convolutions use bias (encoder + heads). Default `True`.
    :param kernel_regularizer: Optional regularizer applied to encoder and head kernels.
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

    # Detector grid: 8x8 cell + 1 dustbin = 65 classes.
    DETECTOR_CHANNELS = 65
    # ConvNeXt V2 must run at strides=2 so 3 stages yield H/8 (see DECISION D-001).
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

        # --- Validate configuration ---
        if len(depths) != len(dims):
            raise ValueError(
                f"Length of depths ({len(depths)}) must equal length of dims ({len(dims)})"
            )
        if input_shape is None or len(input_shape) != 3:
            raise ValueError(f"input_shape must be a 3-tuple (H, W, C), got {input_shape}")
        if descriptor_dim <= 0:
            raise ValueError(f"descriptor_dim must be positive, got {descriptor_dim}")

        # --- Store configuration (all ctor params) ---
        self.depths = list(depths)
        self.dims = list(dims)
        self._input_shape = tuple(input_shape)
        self.descriptor_dim = descriptor_dim
        self.drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.activation = deserialize_activation(activation)
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

        # Unpack static spatial dims (used as graph-safe resize target).
        self.input_height, self.input_width, self.input_channels = self._input_shape

        # --- Build sublayers, all of them, unconditionally ---
        # DECISION plan_2026-06-18_e1411ebf/D-001: hold a whole nested ConvNeXtV2 model at
        # strides=2, not the default 4 (which gives /4,/16,/64, never H/8). See decisions.md.
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

        # Shared 1x1 neck: dims[-1] -> descriptor_dim, feeding both heads.
        self.proj = keras.layers.Conv2D(
            filters=self.descriptor_dim,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            name="proj",
        )

        # Detector head: 1x1 conv -> 65 raw logits (no softmax here).
        self.detector_head = keras.layers.Conv2D(
            filters=self.DETECTOR_CHANNELS,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=self.kernel_regularizer,
            name="detector_head",
        )

        # Descriptor head: 1x1 conv -> descriptor_dim semi-dense map (H/8).
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
        """Explicitly build each sublayer in forward order (anti-lazy-build guard).

        Building the nested encoder, neck, and both heads here (rather than relying on a
        deferred first call) ensures all weights exist before `.keras` weight restore,
        which otherwise silently drops lazily-created sublayer weights.
        """
        # 1. Encoder (its own build runs a dummy-forward over its sublayers).
        self.encoder.build(input_shape)
        encoder_out_shape = self.encoder.compute_output_shape(input_shape)

        # 2. Shared neck.
        self.proj.build(encoder_out_shape)
        neck_shape = self.proj.compute_output_shape(encoder_out_shape)

        # 3. Both heads consume the neck.
        self.detector_head.build(neck_shape)
        self.descriptor_head.build(neck_shape)

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass: encoder -> neck -> {detector logits, descriptor field}.

        :param inputs: 4D tensor `(batch, height, width, channels)`.
        :param training: bool or None, training-mode flag forwarded to sublayers.

        :return: Dict with `"keypoints"` (raw logits, `(B, H/8, W/8, 65)`) and `"descriptors"`
            (unit-L2 along channels, `(B, H, W, descriptor_dim)`).
        """
        feat = self.encoder(inputs, training=training)          # (B, H/8, W/8, dims[-1])
        neck = self.proj(feat, training=training)               # (B, H/8, W/8, descriptor_dim)

        keypoints = self.detector_head(neck, training=training)  # (B, H/8, W/8, 65) logits
        desc_coarse = self.descriptor_head(neck, training=training)  # (B, H/8, W/8, descriptor_dim)

        # Upsample to full (static) resolution; static target keeps this graph-safe.
        #
        # DECISION plan-2026-08-19T163559-499b6f0e/D-060: resize runs in float32 and casts
        # back -- TensorFlow's ResizeBicubic returns a silent None gradient at float16. See decisions.md.
        desc = keras.ops.image.resize(
            keras.ops.cast(desc_coarse, "float32"),
            size=(self.input_height, self.input_width),
            interpolation="bicubic",
        )
        desc = keras.ops.cast(desc, self.compute_dtype)

        # L2-normalize along the channel axis at every spatial location.
        # DECISION plan-2026-08-19T163559-499b6f0e/D-050: floor is max(1e-12, finfo(dtype).tiny)
        # -- np.float16(1e-12) is exactly 0.0, so a zero descriptor gave 0/0 = NaN. See decisions.md.
        norm_eps = max(1e-12, float(np.finfo(self.compute_dtype).tiny))
        desc = desc / (keras.ops.norm(desc, axis=-1, keepdims=True) + norm_eps)

        return {"keypoints": keypoints, "descriptors": desc}

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple]:
        """Compute the output shapes for both heads.

        :param input_shape: input shape tuple `(batch, H, W, C)`.

        :return: Dict mapping `"keypoints"` and `"descriptors"` to their output shapes.
        """
        # DECISION plan-2026-08-19T163559-499b6f0e/D-119: only the detector grid reads
        # `input_shape` -- the descriptor head resizes to the construction-time size. See decisions.md.
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
        """Return the full serialization config (all ctor params)."""
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
        """Reconstruct a SuperPoint instance from a config dict."""
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
    """Convenience factory for SuperPoint models.

    :param variant: one of `"tiny"`, `"base"`, `"large"`. Default `"base"`.
    :param input_shape: `(height, width, channels)`. Default `(256, 256, 1)`.
    :param **kwargs: forwarded to `SuperPoint.from_variant`.

    :return: A `SuperPoint` instance.
    """
    return SuperPoint.from_variant(variant, input_shape=input_shape, **kwargs)

# ---------------------------------------------------------------------
