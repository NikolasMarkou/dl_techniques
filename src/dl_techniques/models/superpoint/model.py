"""
Joint interest-point detection and description in a single forward pass.

This model embodies the principle of shared-encoder multi-task prediction, a
design paradigm that computes two structurally different outputs from one
representation rather than running separate detection and description pipelines.
The core idea addresses a redundancy in classical feature matching: a detector
(Harris, DoG, FAST) and a descriptor (SIFT, ORB) traditionally operate as
independent stages, with the descriptor recomputing local image structure that
the detector has already measured. Predicting both from a common encoder makes
the two tasks share their evidence, and, more importantly, allows them to be
trained jointly under a single self-supervised objective derived from homographic
warps rather than from human-annotated keypoints.

The detector's formulation is the non-obvious part. Rather than regressing a
dense per-pixel score, the detection map is treated as classification over an
8x8 cell:

`logits in R^65 = 64 pixel positions + 1 dustbin`

Each cell at resolution `H/8 x W/8` predicts which of its 64 pixels holds an
interest point, or, via the dustbin class, that it holds none. This buys three
things. Detection is resolved at full pixel resolution without any decoder or
upsampling path, since the channel index encodes sub-cell position. Non-maximum
suppression within a cell is implicit, because softmax over the 65 classes forces
competition. And the "no keypoint here" case has an explicit representation
rather than being expressed as a low score, which is what makes the classification
framing well-posed on mostly-featureless images. The head emits raw logits; the
softmax lives in the loss, per repo convention.

The descriptor head is semi-dense by construction. It predicts a
`descriptor_dim`-channel field at `H/8 x W/8` and interpolates bicubically to full
resolution, then normalizes along channels at every location:

`d(x, y) = f(x, y) / (||f(x, y)||_2 + eps)`

Interpolating a coarse field rather than predicting a dense one keeps memory and
compute tractable while relying on the fact that descriptors vary smoothly in
space. Unit-L2 normalization means descriptor similarity reduces to a dot
product, so matching is a single matrix multiply and the cosine and Euclidean
orderings coincide.

Decoding that 65-channel tensor back into pixel coordinates is deliberately *not*
part of this model, and expecting it here is the usual mistake. The forward pass
returns raw logits at `H/8 x W/8 x 65`. Recovering a full-resolution heatmap is
the consumer's job and is three steps: softmax over the 65 channels, drop the
dustbin channel to leave 64, then reinterpret those 64 channels as the `8x8`
pixels of their cell — a depth-to-space / pixel-shuffle from `(H/8, W/8, 64)` to
`(H, W, 1)`. Keeping that outside the model is what lets the detector be trained
as a plain 65-way cross-entropy against a `(H/8, W/8)` integer label map, which is
exactly what `losses/superpoint_loss.py` expects and what
`datasets/synthetic_shapes.py` emits, with class index 64 meaning "no keypoint in
this cell". Nothing in this model performs non-maximum suppression or thresholding
either; the within-cell competition the softmax provides is not a substitute for
cross-cell NMS at inference.

Architecturally, this implementation replaces the original VGG-style encoder with
a nested three-stage ConvNeXt V2 backbone — a deliberate substitution, not a
transcription of the 2018 network — followed by a shared 1x1 projection neck that
both heads consume:

`encoder -> proj (1x1) -> {detector_head (1x1), descriptor_head (1x1)}`

The neck is where the two tasks diverge; everything before it is shared. One
constraint is load-bearing: the encoder must run at `strides=2`, not the ConvNeXt
default of 4. At stride 4 three stages produce `/4, /16, /64` and never the `H/8`
the 8x8 cell decomposition requires; at stride 2 they produce exactly `H/8` at
`dims[-1]`. The encoder is held as a whole nested `ConvNeXtV2` model rather than
hand-walked stage by stage, which reuses its tested `get_config`/`from_config`
path.

Sublayers are built explicitly in forward order rather than lazily on first call.
This matters for serialization: weights created lazily during a deferred first
call may not exist when a `.keras` restore runs, and the resulting mismatch is
silent rather than an error. Similarly, the bicubic resize targets the static
input dimensions stored at construction, which keeps the operation graph-safe
under compilation but also means the model is tied to the `input_shape` it was
built with — it is not resolution-agnostic at call time. `H` and `W` should be
divisible by 8 so the semi-dense maps land exactly on `H/8 x W/8`.

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
from typing import List, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.models.convnext.convnext_v2 import ConvNeXtV2

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SuperPoint(keras.Model):
    """SuperPoint interest-point detector + descriptor with a ConvNeXt V2 encoder.

    Intent:
        Produce, in a single forward pass, a dense keypoint-detection heatmap (as raw
        logits over a 65-class 8x8-cell + dustbin grid) and a full-resolution, unit-L2
        descriptor field, sharing one ConvNeXt V2 encoder and a 1x1 projection neck. The
        detector head emits LOGITS (softmax lives in the loss, per repo convention); the
        descriptor field is L2-normalized along the channel axis at every pixel.

    Architecture:
        ```
        Input (B, H, W, C)
              │
              ▼
        ConvNeXtV2(strides=2, include_top=False, depths[:3], dims[:3])
              │   stem /2 → stage-1 down /4 → stage-2 down /8
              ▼
        feat (B, H/8, W/8, dims[2])
              │
              ▼
        proj  Conv2D 1x1 → (B, H/8, W/8, descriptor_dim)        [shared neck]
              ├──────────────────────────────┐
              ▼                               ▼
        detector_head Conv2D 1x1      descriptor_head Conv2D 1x1
        → (B, H/8, W/8, 65) LOGITS    → (B, H/8, W/8, descriptor_dim)
                                              │
                                              ▼  resize bicubic → (H, W)
                                              ▼  L2-normalize (axis=-1)
                                       descriptors (B, H, W, descriptor_dim)
        ```

    Args:
        depths: List[int], number of ConvNeXt V2 blocks per stage (3 stages). Default
            `[3, 3, 9]` (tiny). Length must equal `len(dims)`.
        dims: List[int], channel width per stage (3 stages). Default `[96, 192, 384]`.
        input_shape: Tuple[int, int, int], spatial+channel input shape
            `(height, width, channels)`. Default `(256, 256, 1)` (grayscale). H and W
            should be divisible by 8 so the semi-dense maps are exactly `H/8 x W/8`.
        descriptor_dim: int, descriptor channel count (and neck width). Default `256`.
        drop_path_rate: float, stochastic-depth rate forwarded to the encoder. Default `0.0`.
        kernel_size: int or tuple, ConvNeXt V2 block kernel size. Default `7`.
        activation: str or callable, ConvNeXt V2 block activation. Default `"gelu"`.
        use_bias: bool, whether convolutions use bias (encoder + heads). Default `True`.
        kernel_regularizer: Optional regularizer applied to encoder and head kernels.
        **kwargs: forwarded to `keras.Model`.

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
            depths: List[int] = [3, 3, 9],
            dims: List[int] = [96, 192, 384],
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

        # --- Store configuration (ALL ctor params) ---
        self.depths = list(depths)
        self.dims = list(dims)
        self._input_shape = tuple(input_shape)
        self.descriptor_dim = descriptor_dim
        self.drop_path_rate = drop_path_rate
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_regularizer = kernel_regularizer

        # Unpack static spatial dims (used as graph-safe resize target).
        self.input_height, self.input_width, self.input_channels = self._input_shape

        # --- Build sublayers (ALL of them, unconditionally) ---
        # DECISION plan_2026-06-18_e1411ebf/D-001: hold a whole nested ConvNeXtV2 Model at
        # strides=2 (NOT the default strides=4, which gives /4,/16,/64 and never H/8; and NOT
        # a hand-walked .stem_conv/.stages_list traversal). strides=2 over 3 stages yields
        # exactly H/8 @ dims[-1]. Reusing the tested ConvNeXtV2 + its get_config/from_config
        # is the chosen path; the stage-walking fallback is only for if the .keras round-trip
        # drops weights. Do NOT change strides or flatten the encoder. See decisions.md D-001.
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

        # Shared 1x1 neck: dims[-1] -> descriptor_dim, feeding BOTH heads.
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

        Args:
            inputs: 4D tensor `(batch, height, width, channels)`.
            training: bool or None, training-mode flag forwarded to sublayers.

        Returns:
            Dict with `"keypoints"` (raw logits, `(B, H/8, W/8, 65)`) and `"descriptors"`
            (unit-L2 along channels, `(B, H, W, descriptor_dim)`).
        """
        feat = self.encoder(inputs, training=training)          # (B, H/8, W/8, dims[-1])
        neck = self.proj(feat, training=training)               # (B, H/8, W/8, descriptor_dim)

        keypoints = self.detector_head(neck, training=training)  # (B, H/8, W/8, 65) LOGITS
        desc_coarse = self.descriptor_head(neck, training=training)  # (B, H/8, W/8, descriptor_dim)

        # Upsample to full (static) resolution; static target keeps this graph-safe.
        desc = keras.ops.image.resize(
            desc_coarse,
            size=(self.input_height, self.input_width),
            interpolation="bicubic",
        )

        # L2-normalize along the channel axis at every spatial location.
        desc = desc / (keras.ops.norm(desc, axis=-1, keepdims=True) + 1e-12)

        return {"keypoints": keypoints, "descriptors": desc}

    def compute_output_shape(self, input_shape: Tuple[int, ...]) -> Dict[str, Tuple]:
        """Compute the output shapes for both heads.

        Args:
            input_shape: input shape tuple `(batch, H, W, C)`.

        Returns:
            Dict mapping `"keypoints"` and `"descriptors"` to their output shapes.
        """
        batch = input_shape[0] if len(input_shape) == 4 else None
        grid_h = self.input_height // 8
        grid_w = self.input_width // 8
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
            "activation": self.activation,
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

        Args:
            variant: one of `"tiny"`, `"base"`, `"large"`.
            **kwargs: forwarded to the constructor (e.g. `input_shape`, `descriptor_dim`).

        Returns:
            A `SuperPoint` instance.

        Raises:
            ValueError: if `variant` is not a known variant.
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

    Args:
        variant: one of `"tiny"`, `"base"`, `"large"`. Default `"base"`.
        input_shape: `(height, width, channels)`. Default `(256, 256, 1)`.
        **kwargs: forwarded to `SuperPoint.from_variant`.

    Returns:
        A `SuperPoint` instance.
    """
    return SuperPoint.from_variant(variant, input_shape=input_shape, **kwargs)

# ---------------------------------------------------------------------
