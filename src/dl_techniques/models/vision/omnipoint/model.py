"""`OmniPoint`: a camera-agnostic-by-construction monocular metric point-cloud
model -- ViT encoder, three per-feature heads, plus an OPTIONAL geometric
conditioning path (intrinsics ray map + sparse depth).

Step 4 built the minimal forward pass: the shared encoder and the three
output heads (`heads.py`) over ONE shared feature map, with no conditioning
path. Step 5 (`conditioning.py`) adds an opt-in `enable_conditioning=True`
path, wired in additively -- with `enable_conditioning=False` (the default,
unchanged from Step 4), this model's behavior is bit-for-bit identical to
before Step 5 existed. See `conditioning.py`'s module docstring for the
fusion-point design (decisions.md D-013) and the mixed-batch per-sample-flag
contract.

Architecture:

.. code-block:: text

    Input [B, H, W, 3]
          |
          v
    +--------------------------------+
    | encoder: ViT (patch_size,      |
    | include_top=False,             |
    | pooling=None)                  |
    +---------------+----------------+
                    |  [B, N+1, D] (CLS-prefixed patch sequence)
        +-----------+-----------+
        |                       |
        v                       v
    cls_token = seq[:, 0]   _features_to_spatial:
        |                   drop CLS, reshape -> [B, h, w, D]
        |                       |
        v                 +-----+-----+
    MetricScaleHead        |           |
    (pooled MLP)           v           v
        |            RayDistanceHead  MaskHead
        |            (DPTDecoder x4   (DPTDecoder x1
        |             + combinator)    logits)
        v                  |           |
      scale        (ray, distance,   mask_logit
                     point)
                        |
                        v
       Output: (ray, distance, point, mask_logit, scale)

Backbone choice (decisions.md D-009, resolved before this file was written):
plain `ViT`, not `DINOv2VisionTransformer`. `ViT.from_variant(..., include_top=False,
pooling=None)` returns the full LayerNorm-ed patch sequence `(B, N+1, D)`,
CLS token at index 0 (`vit/model.py` docstring, lines 298-302) -- this module
reads that CLS token directly as the "metric token" for `MetricScaleHead`,
rather than global-average-pooling the patch tokens. This is the MORE
faithful and CHEAPER choice than GAP: `ViT`'s CLS token already accumulates a
whole-image summary through attention (`vit/model.py` module docstring), so
no extra pooling op or extra supervision signal is needed to make it
meaningful, and it costs zero extra compute since the encoder already
computes it.

Per-pixel output resolution -- a documented simplification, not a bug: this
repository's `DPTDecoder` only accepts a power-of-2 `upsample_factor`
(`depth_anything/components.py`'s own validation), but `ViT.from_variant`
here uses `patch_size=14` (the paper's ViT-L/14 convention, D-009), and 14 is
not a power of 2. Rather than force a mismatched upsample factor or silently
switch to a power-of-2 patch size the task did not ask for, this module keeps
`upsample_factor=1`: both heads' outputs sit at the encoder's native
`(H // patch_size, W // patch_size)` patch-grid resolution, not full pixel
resolution (see the `# DECISION ... D-011` anchor in `__init__` below).
Upsampling to full image resolution, if ever needed, is a downstream/
visualization concern layered on top of this model, not a change to it.

Output order (needed verbatim by Step 5's conditioning wiring and Step 9's
training script): `call()` returns the 5-tuple
``(ray, distance, point, mask_logit, scale)``. This does NOT exactly match
`OmniPointCombinedLoss.__call__`'s documented `y_pred` convention -- a 4-tuple
``(pred_ray, pred_distance, pred_mask_logit, pred_scale)`` with no `point`
entry, since that loss recomputes the affine point itself
(`pred_ray * pred_distance`) internally. Step 9 must therefore select indices
``(0, 1, 3, 4)`` from this model's output tuple before calling the combined
loss, or drop `point` from `call()`'s public output at that time -- recorded
here rather than silently reconciled, per this plan's Pre-Mortem #3.

References:
    - Ye et al., OmniPoint: Universal Monocular Metric Pointcloud from Any
      Camera. (paper summary supplied to this plan; no public arXiv id
      captured in EXPLORE)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Ranftl et al., 2021. Vision Transformers for Dense Prediction (DPT).
      (https://arxiv.org/abs/2103.13413)
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

from dl_techniques.utils.logger import logger
from dl_techniques.models.vision.vit.model import ViT
from dl_techniques.utils.keras_registration import register_dl_technique

from .heads import RayDistanceHead, MaskHead, MetricScaleHead
from .conditioning import ConditioningInputEncoder, ConditioningStateEmbedding

# ---------------------------------------------------------------------

OmniPointOutput = Tuple[
    keras.KerasTensor,  # ray:         (B, h, w, 3), unit-norm
    keras.KerasTensor,  # distance:    (B, h, w, 1), > 0
    keras.KerasTensor,  # point:       (B, h, w, 3), distance * ray
    keras.KerasTensor,  # mask_logit:  (B, h, w, 1), linear logits
    keras.KerasTensor,  # scale:       (B,), > 0
]

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.omnipoint.model")
class OmniPoint(keras.Model):
    """ViT encoder + ray/distance/mask/scale heads, no conditioning input.

    See this module's docstring for the architecture diagram, the backbone
    choice, the per-pixel-resolution simplification and the exact output
    tuple order.

    :param image_shape: Input image shape ``(height, width, channels)``. Must
        be divisible by ``patch_size`` on both spatial axes (`ViT`'s own
        constraint).
    :type image_shape: Tuple[int, int, int]
    :param vit_scale: `ViT`'s `SCALE_CONFIGS` key (``"base"``, ``"large"``,
        ...).
    :type vit_scale: str
    :param patch_size: Square ViT patch size. Defaults to 14 (ViT-L/14
        convention).
    :type patch_size: int
    :param decoder_dims: Channel dims per `DPTDecoder` stage, shared by both
        `RayDistanceHead` and `MaskHead`.
    :type decoder_dims: Optional[list]
    :param metric_hidden_dim: Hidden width of `MetricScaleHead`'s MLP.
    :type metric_hidden_dim: int
    :param kernel_initializer: Initializer for every head conv/dense kernel
        (does not affect the encoder, which uses `ViT`'s own reference
        initializer).
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for every head
        conv/dense kernel.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Floor shared by `RayDistanceHead`'s combinator and
        `MetricScaleHead`'s scale activation.
    :type epsilon: float
    :param encoder: A pre-built `ViT` encoder. Supplied by `from_config` so a
        deserialized archive keeps its saved topology/weights instead of
        constructing a fresh, randomly initialized one; `None` (default)
        builds a fresh encoder lazily in `build()`.
    :type encoder: Optional[keras.Model]
    :param enable_conditioning: If `True`, build the Step-5 conditioning path
        (`conditioning.py`): the encoder's own `input_shape` channel count is
        widened by `intrinsics_channels + depth_channels`, and `call()`
        accepts the optional `intrinsics_ray_map`/`sparse_depth`/
        `sparse_depth_mask` keyword arguments (see `call()`'s docstring).
        Defaults to `False`, in which case this model behaves exactly as in
        Step 4 (no conditioning path is built at all).
    :type enable_conditioning: bool
    :param intrinsics_channels: Fusion channels the intrinsics ray-map
        encoder emits. Only used when `enable_conditioning` is `True`.
    :type intrinsics_channels: int
    :param depth_channels: Fusion channels the sparse-depth encoder emits.
        Only used when `enable_conditioning` is `True`.
    :type depth_channels: int
    :param conditioning_hidden_channels: Hidden width of the conditioning
        path's small `Conv2D` stacks. Only used when `enable_conditioning` is
        `True`.
    :type conditioning_hidden_channels: int
    :param splat_kernel_size: `SparseDepthSplat`'s Gaussian window size. Only
        used when `enable_conditioning` is `True`.
    :type splat_kernel_size: Tuple[int, int]
    :param splat_sigma: `SparseDepthSplat`'s Gaussian sigma. Only used when
        `enable_conditioning` is `True`.
    :type splat_sigma: float
    :param kwargs: Additional keyword arguments for the `Model` base class.

    :raises ValueError: If `vit_scale` is not a valid `ViT.SCALE_CONFIGS` key,
        or if `image_shape` is not divisible by `patch_size`.

    Input shape:
        4D tensor ``(batch_size, height, width, 3)``.

    Output shape:
        5-tuple ``(ray, distance, point, mask_logit, scale)`` -- see this
        module's docstring for exact per-entry shapes.

    Example:
        .. code-block:: python

            model = OmniPoint.from_variant("omnipoint_base", image_shape=(224, 224, 3))
            x = keras.random.normal([2, 224, 224, 3])
            ray, distance, point, mask_logit, scale = model(x)
    """

    #: Public-name registry: variant name -> constructor-kwarg dict, following
    #: `ViT`'s own "thin wrapper over a scale table" pattern
    #: (`vit/model.py:371-373`).
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "omnipoint_base": {"vit_scale": "base"},
        "omnipoint_large": {"vit_scale": "large"},
    }

    def __init__(
            self,
            image_shape: Tuple[int, int, int] = (224, 224, 3),
            vit_scale: str = "base",
            patch_size: int = 14,
            decoder_dims: Optional[list] = None,
            metric_hidden_dim: int = 256,
            kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            epsilon: float = 1e-8,
            encoder: Optional[keras.Model] = None,
            enable_conditioning: bool = False,
            intrinsics_channels: int = 4,
            depth_channels: int = 4,
            conditioning_hidden_channels: int = 16,
            splat_kernel_size: Tuple[int, int] = (9, 9),
            splat_sigma: float = 2.0,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if vit_scale not in ViT.SCALE_CONFIGS:
            raise ValueError(
                f"Unsupported vit_scale: {vit_scale}. Choose from "
                f"{list(ViT.SCALE_CONFIGS.keys())}"
            )

        img_h, img_w, img_c = image_shape
        if img_h % patch_size != 0 or img_w % patch_size != 0:
            raise ValueError(
                f"image_shape {image_shape} must be divisible by patch_size "
                f"({patch_size}) on both spatial axes."
            )

        self.image_shape = tuple(image_shape)
        self.vit_scale = str(vit_scale)
        self.patch_size = int(patch_size)
        self.decoder_dims = list(decoder_dims) if decoder_dims is not None else [256, 128, 64, 32]
        self.metric_hidden_dim = int(metric_hidden_dim)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.epsilon = epsilon

        self.embed_dim = ViT.SCALE_CONFIGS[self.vit_scale][0]
        self.grid_h = img_h // self.patch_size
        self.grid_w = img_w // self.patch_size

        self.enable_conditioning = bool(enable_conditioning)
        self.intrinsics_channels = int(intrinsics_channels)
        self.depth_channels = int(depth_channels)
        self.conditioning_hidden_channels = int(conditioning_hidden_channels)
        self.splat_kernel_size = (int(splat_kernel_size[0]), int(splat_kernel_size[1]))
        self.splat_sigma = float(splat_sigma)
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-013: the encoder's own
        # input channel count is widened here, at __init__ time, not chosen
        # per-call -- ViT's PatchEmbedding2D conv kernel shape is fixed at
        # build time, so a call-time-varying channel count is not an option.
        # Conditioning is therefore an __init__-time architectural choice
        # (enable_conditioning=True/False), not a per-call one; a per-call
        # absent modality still runs this same widened-channel encoder, fed
        # zero-valued conditioning channels (see conditioning.py). See
        # decisions.md.
        self._encoder_input_channels = img_c + (
            (self.intrinsics_channels + self.depth_channels)
            if self.enable_conditioning else 0
        )

        # If an encoder was supplied (typically by `from_config` after
        # deserialization), accept it directly so its saved topology/weights
        # survive the load, mirroring `DepthAnything.__init__`'s identical
        # convention. Otherwise `build()` creates one fresh.
        self.encoder: Optional[keras.Model] = encoder

        self.conditioning_input_encoder: Optional[ConditioningInputEncoder] = None
        self.conditioning_state_embedding: Optional[ConditioningStateEmbedding] = None
        if self.enable_conditioning:
            self.conditioning_input_encoder = ConditioningInputEncoder(
                intrinsics_channels=self.intrinsics_channels,
                depth_channels=self.depth_channels,
                conv_hidden_channels=self.conditioning_hidden_channels,
                splat_kernel_size=self.splat_kernel_size,
                splat_sigma=self.splat_sigma,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name="conditioning_input_encoder",
            )
            self.conditioning_state_embedding = ConditioningStateEmbedding(
                name="conditioning_state_embedding",
            )

        # Heads -- pure functions of the config above, safe to construct
        # eagerly in `__init__` (mirrors `heads.py`'s own eager-sublayer
        # style; none of them need `input_shape` to be constructed, only to
        # be called).
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-011: upsample_factor=1 on
        # both dense heads. patch_size=14 is not a power of 2, and DPTDecoder
        # only accepts a power-of-2 upsample_factor -- do not "round" this to
        # 8 or 16 to get closer to full resolution; that produces an output
        # grid that no longer corresponds to the input pixel grid at all. See
        # decisions.md.
        self.ray_distance_head = RayDistanceHead(
            dims=self.decoder_dims,
            upsample_factor=1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            epsilon=self.epsilon,
            name="ray_distance_head",
        )
        self.mask_head = MaskHead(
            dims=self.decoder_dims,
            upsample_factor=1,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name="mask_head",
        )
        self.metric_scale_head = MetricScaleHead(
            hidden_dim=self.metric_hidden_dim,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            epsilon=self.epsilon,
            name="metric_scale_head",
        )

        logger.info(
            f"Initialized OmniPoint (vit_scale={self.vit_scale}, "
            f"patch_size={self.patch_size}, image_shape={self.image_shape}, "
            f"grid={self.grid_h}x{self.grid_w})"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the ViT encoder if one was not supplied via `from_config`.

        :param input_shape: Shape of the input image tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.encoder is None:
            img_h, img_w, _ = self.image_shape
            self.encoder = ViT(
                input_shape=(img_h, img_w, self._encoder_input_channels),
                scale=self.vit_scale,
                patch_size=self.patch_size,
                include_top=False,
                pooling=None,
                name=f"encoder_vit_{self.vit_scale}",
            )
        super().build(input_shape)
        # DECISION plan-2026-09-11T050223-1b47bcf6/D-023: force-build every head via a
        # dummy forward pass here, not lazily at first real `call()`. `.keras` reload runs
        # `build_from_config` -> `build()` only, never `call()`; a lazily-built head is
        # still 0-weight afterward, so saved head weights have nowhere to land and the
        # first post-reload `call()` silently re-randomizes them (D-124's identical defect
        # in `DPTDecoder`, one layer up the call stack). See decisions.md.
        dummy_spatial = keras.ops.zeros((1, self.grid_h, self.grid_w, self.embed_dim))
        dummy_cls = keras.ops.zeros((1, self.embed_dim))
        _ = self.ray_distance_head(dummy_spatial)
        _ = self.mask_head(dummy_spatial)
        _ = self.metric_scale_head(dummy_cls)

        # DECISION plan-2026-09-11T050223-1b47bcf6/D-023 (extended, completion-fix
        # step 4.1): D-023's original fix above only covered the 3 heads -- it never
        # touched `conditioning_input_encoder`/`conditioning_state_embedding` (Step 5),
        # so a `.keras` round-trip with `enable_conditioning=True` silently lost exactly
        # those 2 layers' weights on reload (MEASURED: 8 of 212 weights mismatched --
        # the review's CRITICAL #3). Force-build both here too, via the same
        # dummy-forward-pass mechanism, whenever they exist -- they only exist at all
        # when `enable_conditioning=True` (see `__init__`), so this is unconditional
        # given existence, never gated on a second flag. See decisions.md.
        if self.conditioning_input_encoder is not None:
            img_h, img_w, img_c = self.image_shape
            dummy_image = keras.ops.zeros((1, img_h, img_w, img_c))
            _ = self.conditioning_input_encoder(dummy_image)
        if self.conditioning_state_embedding is not None:
            dummy_tokens = keras.ops.zeros(
                (1, self.grid_h * self.grid_w + 1, self.embed_dim)
            )
            dummy_flag = keras.ops.zeros((1,), dtype="bool")
            _ = self.conditioning_state_embedding(dummy_tokens, dummy_flag, dummy_flag)

    def _features_to_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Drop the CLS token and reshape ``(B, N+1, D)`` -> ``(B, h, w, D)``.

        Copies `DepthAnything._features_to_spatial`'s exact CLS-drop/reshape
        pattern (`depth_anything/model.py:449-460`) rather than re-deriving
        it, per `findings/model-house-patterns.md`'s explicit recommendation.

        :param x: Encoder output sequence, ``(B, N+1, D)``.
        :type x: keras.KerasTensor
        :return: Spatial feature map, ``(B, h, w, D)``.
        :rtype: keras.KerasTensor
        """
        x = x[:, 1:, :]
        return keras.ops.reshape(x, (-1, self.grid_h, self.grid_w, self.embed_dim))

    def call(
            self,
            inputs: keras.KerasTensor,
            intrinsics_ray_map: Optional[keras.KerasTensor] = None,
            intrinsics_present: Optional[keras.KerasTensor] = None,
            sparse_depth: Optional[keras.KerasTensor] = None,
            sparse_depth_mask: Optional[keras.KerasTensor] = None,
            sparse_depth_present: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None,
    ) -> OmniPointOutput:
        """Forward pass: encoder -> shared spatial map + CLS token -> heads.

        Conditioning arguments are only meaningful when this instance was
        built with ``enable_conditioning=True``; they are accepted (and
        ignored) otherwise, matching Problem Statement invariant 4 -- the
        model produces the same output shape/dtype whether or not intrinsics/
        sparse depth are supplied. A Python ``None`` for
        ``intrinsics_ray_map``/``sparse_depth`` means "absent for this whole
        call"; a concrete per-sample ``*_present`` flag (``(B,)``) then
        selects, WITHIN a call that does supply the tensor, which individual
        samples actually carry valid data for that modality (see
        ``conditioning.py``'s module docstring for the full contract).

        :param inputs: Input image batch, ``(B, H, W, 3)``.
        :type inputs: keras.KerasTensor
        :param intrinsics_ray_map: Optional per-pixel unit ray map,
            ``(B, H, W, 3)``. Ignored unless ``enable_conditioning=True``.
        :type intrinsics_ray_map: Optional[keras.KerasTensor]
        :param intrinsics_present: Optional per-sample flag, ``(B,)``.
            Defaults to "all present" when ``intrinsics_ray_map`` is given.
        :type intrinsics_present: Optional[keras.KerasTensor]
        :param sparse_depth: Optional sparse depth values, ``(B, H, W, 1)``.
            Ignored unless ``enable_conditioning=True``.
        :type sparse_depth: Optional[keras.KerasTensor]
        :param sparse_depth_mask: Required companion validity mask when
            ``sparse_depth`` is given, ``(B, H, W, 1)``.
        :type sparse_depth_mask: Optional[keras.KerasTensor]
        :param sparse_depth_present: Optional per-sample flag, ``(B,)``.
            Defaults to "all present" when ``sparse_depth`` is given.
        :type sparse_depth_present: Optional[keras.KerasTensor]
        :param training: Whether the model runs in training or inference
            mode.
        :type training: Optional[bool]
        :return: 5-tuple ``(ray, distance, point, mask_logit, scale)`` -- see
            this module's docstring for the exact contract.
        :rtype: OmniPointOutput
        """
        if self.enable_conditioning:
            batch_size = keras.ops.shape(inputs)[0]

            fused_inputs = self.conditioning_input_encoder(
                inputs,
                intrinsics_ray_map=intrinsics_ray_map,
                intrinsics_present=intrinsics_present,
                sparse_depth=sparse_depth,
                sparse_depth_mask=sparse_depth_mask,
                sparse_depth_present=sparse_depth_present,
                training=training,
            )
            sequence = self.encoder(fused_inputs, training=training)

            # Concrete per-sample flags for the post-encoder token-space
            # stage: absent-for-the-whole-call (tensor is None) means every
            # sample's flag is False; present-for-the-call but no explicit
            # per-sample flag means every sample's flag is True.
            if intrinsics_ray_map is None:
                intrinsics_present_flags = keras.ops.zeros((batch_size,), dtype="bool")
            elif intrinsics_present is None:
                intrinsics_present_flags = keras.ops.ones((batch_size,), dtype="bool")
            else:
                intrinsics_present_flags = keras.ops.reshape(
                    keras.ops.cast(intrinsics_present, "bool"), (-1,)
                )

            if sparse_depth is None:
                sparse_depth_present_flags = keras.ops.zeros((batch_size,), dtype="bool")
            elif sparse_depth_present is None:
                sparse_depth_present_flags = keras.ops.ones((batch_size,), dtype="bool")
            else:
                sparse_depth_present_flags = keras.ops.reshape(
                    keras.ops.cast(sparse_depth_present, "bool"), (-1,)
                )

            sequence = self.conditioning_state_embedding(
                sequence,
                intrinsics_present_flags,
                sparse_depth_present_flags,
                training=training,
            )
        else:
            sequence = self.encoder(inputs, training=training)

        cls_token = sequence[:, 0, :]
        spatial = self._features_to_spatial(sequence)

        ray, distance, point = self.ray_distance_head(spatial, training=training)
        mask_logit = self.mask_head(spatial, training=training)
        scale = self.metric_scale_head(cls_token, training=training)

        return ray, distance, point, mask_logit, scale

    def get_config(self) -> Dict[str, Any]:
        """Return the model configuration for serialization.

        :return: The base `Model` config plus every constructor argument. The
            encoder sub-Model is serialized so a `.keras` archive round-trips
            both its topology and its weights, mirroring
            `DepthAnything.get_config`'s identical convention.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "image_shape": self.image_shape,
            "vit_scale": self.vit_scale,
            "patch_size": self.patch_size,
            "decoder_dims": self.decoder_dims,
            "metric_hidden_dim": self.metric_hidden_dim,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "epsilon": self.epsilon,
            "encoder": (
                keras.saving.serialize_keras_object(self.encoder)
                if self.encoder is not None
                else None
            ),
            "enable_conditioning": self.enable_conditioning,
            "intrinsics_channels": self.intrinsics_channels,
            "depth_channels": self.depth_channels,
            "conditioning_hidden_channels": self.conditioning_hidden_channels,
            "splat_kernel_size": self.splat_kernel_size,
            "splat_sigma": self.splat_sigma,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "OmniPoint":
        """Create a model from its configuration.

        :param config: Dictionary containing the model configuration.
        :type config: Dict[str, Any]
        :return: An `OmniPoint` model instance.
        :rtype: OmniPoint
        """
        cfg = dict(config)
        if isinstance(cfg.get("kernel_initializer"), dict):
            cfg["kernel_initializer"] = keras.initializers.deserialize(
                cfg["kernel_initializer"]
            )
        if isinstance(cfg.get("kernel_regularizer"), dict):
            cfg["kernel_regularizer"] = keras.regularizers.deserialize(
                cfg["kernel_regularizer"]
            )
        enc_cfg = cfg.pop("encoder", None)
        if enc_cfg is not None:
            cfg["encoder"] = keras.saving.deserialize_keras_object(enc_cfg)
        return cls(**cfg)

    @classmethod
    def from_variant(
            cls,
            variant: str,
            pretrained: bool = False,
            **kwargs: Any,
    ) -> "OmniPoint":
        """Create an `OmniPoint` model from a predefined variant.

        :param variant: One of `MODEL_VARIANTS` (``"omnipoint_base"``,
            ``"omnipoint_large"``).
        :type variant: str
        :param pretrained: Must stay `False`. No pretrained OmniPoint weights
            (or pretrained weights for either candidate backbone, `ViT` or
            `DINOv2VisionTransformer`) are distributed with `dl_techniques`.
        :type pretrained: bool
        :param kwargs: Passthrough to the constructor, overriding the
            variant's own defaults.
        :type kwargs: Any
        :return: A configured `OmniPoint` instance.
        :rtype: OmniPoint
        :raises ValueError: If `variant` is not recognized.
        :raises NotImplementedError: If `pretrained` is `True`.
        """
        if pretrained:
            raise NotImplementedError(
                f"No pretrained OmniPoint weights are distributed with "
                f"dl_techniques (requested variant '{variant}'); no "
                f"pretrained weights exist for its ViT backbone either. "
                f"Pass pretrained=False (default, random init) and load a "
                f"local checkpoint yourself via "
                f"model.load_weights('/path/to/weights.keras') instead."
            )

        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        # DECISION plan-2026-08-19T163559-499b6f0e/D-127 (`ResNet.from_variant`'s
        # own precedent, restated here): `.copy()` the preset before splatting
        # kwargs on top -- splatting the shared dict directly would let
        # `config.update(kwargs)` permanently poison `MODEL_VARIANTS[variant]`
        # for every future caller. See decisions.md.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating OmniPoint model variant '{variant}'")
        return cls(**config)

# ---------------------------------------------------------------------

#: Module-level alias of `OmniPoint.MODEL_VARIANTS`, so
#: `from dl_techniques.models.vision.omnipoint import MODEL_VARIANTS` works
#: without reaching through the class. The class attribute remains the single
#: source of truth; this is a read-only alias, not a second copy to keep in
#: sync.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = OmniPoint.MODEL_VARIANTS

# ---------------------------------------------------------------------


def create_omnipoint(
        variant: str = "omnipoint_base",
        image_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs: Any,
) -> OmniPoint:
    """Create and build an `OmniPoint` model instance.

    :param variant: One of `OmniPoint.MODEL_VARIANTS`.
    :type variant: str
    :param image_shape: Input image shape ``(height, width, channels)``.
    :type image_shape: Tuple[int, int, int]
    :param kwargs: Passthrough to `OmniPoint.from_variant`.
    :type kwargs: Any
    :return: A built `OmniPoint` model instance.
    :rtype: OmniPoint

    Example:
        .. code-block:: python

            model = create_omnipoint("omnipoint_base", image_shape=(224, 224, 3))
            ray, distance, point, mask_logit, scale = model(
                keras.random.normal([2, 224, 224, 3])
            )
    """
    model = OmniPoint.from_variant(variant, image_shape=image_shape, **kwargs)
    dummy_input = keras.random.normal([1] + list(image_shape))
    _ = model(dummy_input)
    logger.info("Successfully created and built OmniPoint model")
    return model

# ---------------------------------------------------------------------
