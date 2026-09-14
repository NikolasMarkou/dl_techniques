"""Camera-agnostic monocular metric point-cloud model.

``OmniPoint`` pairs a ViT encoder with three heads: a ray/distance head and a
mask head over the patch-grid feature map, and a metric-scale head over the
encoder's CLS token. The CLS token is read directly as the metric token instead
of global-average-pooling the patch tokens, since the encoder already computes
it. An optional conditioning path widens the encoder's input channel count at
construction time so an intrinsics ray map and sparse depth can be fused in;
with ``enable_conditioning=False``, the default, none of it is built.

``call()`` returns the 5-tuple ``(ray, distance, point, mask_logit, scale)``.
``OmniPointCombinedLoss`` takes a 4-tuple without ``point``, which it recomputes
itself, so a training loop selects indices ``(0, 1, 3, 4)``. Both dense heads
run at ``upsample_factor=1``, so their outputs sit at the patch-grid resolution
``(H // patch_size, W // patch_size)`` rather than full pixel resolution:
``patch_size`` defaults to 14, and ``DPTDecoder`` only upsamples by a power of
2. No pretrained weights ship with this model.

References:
    - Ye et al. OmniPoint: Universal Monocular Metric Pointcloud from Any
      Camera.
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words: Transformers
      for Image Recognition at Scale. (https://arxiv.org/abs/2010.11929)
    - Ranftl et al., 2021. Vision Transformers for Dense Prediction (DPT).
      (https://arxiv.org/abs/2103.13413)
"""

import keras
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

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
    """Predict rays, distance, a validity mask and a metric scale from an image.

    A ViT encoder produces a CLS-prefixed patch sequence. The CLS row feeds
    :class:`MetricScaleHead`; the remaining tokens are reshaped to a patch grid
    and feed :class:`RayDistanceHead` and :class:`MaskHead`. When
    ``enable_conditioning=True``, an intrinsics ray map and sparse depth are
    encoded into extra input channels before the encoder, and a token-space
    embedding marks which samples actually carried each modality.

    Architecture:

    .. code-block:: text

                         input [B, H, W, C]
                                │
                  ┌─────────────┴───────────────┐
                  ▼                             ▼
             enable_conditioning             otherwise
        ┌───────────────────┐           ┌───────────────────┐
        │ ConditioningInput │           │ pass through      │
        │  Encoder          │           │                   │
        └─────────┬─────────┘           └─────────┬─────────┘
        [B, H, W, C+ic+dc]                  [B, H, W, C]
                  └─────────────┬───────────────┘
                                ▼
                  ┌───────────────────────────────┐
                  │ ViT encoder                   │
                  │  no top, no pooling           │
                  └───────────────┬───────────────┘
                   [B, N+1, D], CLS at index 0
                                ▼
                  ┌───────────────────────────────┐
                  │ ConditioningStateEmbedding    │
                  │  (enable_conditioning only)   │
                  └───────────────┬───────────────┘
                                ▼
                  ┌───────────────────────────────┐
                  │ split sequence                │
                  │  cls = seq[:, 0]              │
                  │  spatial = reshape(seq[:, 1:])│
                  └───────────────┬───────────────┘
          ┌─────────────────────┬─┴─────────────────┐
          ▼                     ▼                   ▼
       from cls            from spatial        from spatial
      ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
      │ MetricScaleHead│  │ RayDistanceHead│  │ MaskHead       │
      └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
              ▼                   ▼                   ▼
          scale [B]      ray, distance, point     mask_logit

    ``call()`` returns them in the order
    ``(ray, distance, point, mask_logit, scale)``.

    Conditioning present-flags, resolved per modality inside ``call()``:

    .. code-block:: text

        tensor        *_present arg   resolved per-sample flags
        ──────────    ─────────────   ─────────────────────────
        None          any             all False
        given         None            all True
        given         given           cast to bool, flattened

    Variants:

    .. code-block:: text

        variant            vit_scale
        ───────────────    ─────────
        omnipoint_base     base
        omnipoint_large    large

    :param image_shape: Input image shape ``(height, width, channels)``. Both
        spatial axes must be divisible by ``patch_size``.
    :type image_shape: Tuple[int, int, int]
    :param vit_scale: Key into ``ViT.SCALE_CONFIGS`` (``"base"``, ``"large"``,
        and so on).
    :type vit_scale: str
    :param patch_size: Square ViT patch size. Defaults to 14.
    :type patch_size: int
    :param decoder_dims: Channel dimensions per ``DPTDecoder`` stage, shared by
        :class:`RayDistanceHead` and :class:`MaskHead`. ``None`` gives
        ``[256, 128, 64, 32]``.
    :type decoder_dims: Optional[list]
    :param metric_hidden_dim: Hidden width of :class:`MetricScaleHead`'s MLP.
    :type metric_hidden_dim: int
    :param kernel_initializer: Initializer for every head and conditioning
        kernel. The encoder uses ``ViT``'s own initializer.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for those same kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param epsilon: Floor shared by :class:`RayDistanceHead`'s combinator and
        :class:`MetricScaleHead`'s scale activation.
    :type epsilon: float
    :param encoder: A pre-built ``ViT`` encoder, supplied by ``from_config`` so
        a deserialized archive keeps its saved topology and weights. ``None``
        (default) builds a fresh encoder in ``build()``.
    :type encoder: Optional[keras.Model]
    :param enable_conditioning: Build the conditioning path. The encoder's input
        channel count is widened by ``intrinsics_channels + depth_channels``,
        and ``call()`` reads its conditioning keyword arguments. Defaults to
        ``False``, which builds no conditioning layers at all.
    :type enable_conditioning: bool
    :param intrinsics_channels: Fusion channels emitted by the intrinsics
        ray-map encoder. Read only when ``enable_conditioning`` is ``True``.
    :type intrinsics_channels: int
    :param depth_channels: Fusion channels emitted by the sparse-depth encoder.
        Read only when ``enable_conditioning`` is ``True``.
    :type depth_channels: int
    :param conditioning_hidden_channels: Hidden width of the conditioning path's
        ``Conv2D`` stacks. Read only when ``enable_conditioning`` is ``True``.
    :type conditioning_hidden_channels: int
    :param splat_kernel_size: ``SparseDepthSplat``'s Gaussian window size. Read
        only when ``enable_conditioning`` is ``True``.
    :type splat_kernel_size: Tuple[int, int]
    :param splat_sigma: ``SparseDepthSplat``'s Gaussian sigma. Read only when
        ``enable_conditioning`` is ``True``.
    :type splat_sigma: float
    :param kwargs: Additional ``Model`` base-class arguments.

    :raises ValueError: If ``vit_scale`` is not a key of ``ViT.SCALE_CONFIGS``,
        or if ``image_shape`` is not divisible by ``patch_size``.

    Input shape:
        4D tensor ``(batch_size, height, width, channels)``, matching
        ``image_shape``.

    Output shape:
        5-tuple ``(ray, distance, point, mask_logit, scale)`` with shapes
        ``(B, h, w, 3)``, ``(B, h, w, 1)``, ``(B, h, w, 3)``, ``(B, h, w, 1)``
        and ``(B,)``, where ``h`` and ``w`` are the patch grid.

    Example:
        .. code-block:: python

            model = OmniPoint.from_variant("omnipoint_base", image_shape=(224, 224, 3))
            x = keras.random.normal([2, 224, 224, 3])
            ray, distance, point, mask_logit, scale = model(x)
    """

    #: Variant name to constructor-kwarg dict.
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
        # DECISION D-013: widen the encoder's input channels here, at __init__
        # time; the patch-embedding kernel shape is fixed at build. decisions.md.
        self._encoder_input_channels = img_c + (
            (self.intrinsics_channels + self.depth_channels)
            if self.enable_conditioning else 0
        )

        # A supplied encoder comes from from_config, and keeps the saved
        # topology and weights instead of being rebuilt fresh.
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

        # DECISION D-011: upsample_factor=1 on both dense heads; patch_size 14
        # is not a power of 2 and DPTDecoder needs one. See decisions.md.
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
        """Build the encoder if none was supplied, then force-build every head.

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
        # DECISION D-023: force-build the heads with a dummy pass here; a .keras
        # reload runs build() but never call(), so lazy heads lose their weights.
        dummy_spatial = keras.ops.zeros((1, self.grid_h, self.grid_w, self.embed_dim))
        dummy_cls = keras.ops.zeros((1, self.embed_dim))
        _ = self.ray_distance_head(dummy_spatial)
        _ = self.mask_head(dummy_spatial)
        _ = self.metric_scale_head(dummy_cls)

        # DECISION D-023: the conditioning layers need the same force-build;
        # without it a round-trip lost 8 of 212 weights. See decisions.md.
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
        """Drop the CLS token and reshape ``(B, N+1, D)`` to ``(B, h, w, D)``.

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
        """Run the encoder, split its output, and apply the three heads.

        The conditioning arguments are read only when this instance was built
        with ``enable_conditioning=True``; otherwise they are accepted and
        ignored, so the output shapes and dtypes do not depend on them. A
        ``None`` tensor means the modality is absent for the whole call, while a
        per-sample ``*_present`` flag selects which samples inside a call carry
        valid data.

        :param inputs: Input image batch, ``(B, H, W, C)``.
        :type inputs: keras.KerasTensor
        :param intrinsics_ray_map: Optional per-pixel unit ray map,
            ``(B, H, W, 3)``.
        :type intrinsics_ray_map: Optional[keras.KerasTensor]
        :param intrinsics_present: Optional per-sample flag, ``(B,)``. Defaults
            to all-present when ``intrinsics_ray_map`` is given.
        :type intrinsics_present: Optional[keras.KerasTensor]
        :param sparse_depth: Optional sparse depth values, ``(B, H, W, 1)``.
        :type sparse_depth: Optional[keras.KerasTensor]
        :param sparse_depth_mask: Companion validity mask, required when
            ``sparse_depth`` is given, ``(B, H, W, 1)``.
        :type sparse_depth_mask: Optional[keras.KerasTensor]
        :param sparse_depth_present: Optional per-sample flag, ``(B,)``.
            Defaults to all-present when ``sparse_depth`` is given.
        :type sparse_depth_present: Optional[keras.KerasTensor]
        :param training: Whether the model runs in training or inference mode.
        :type training: Optional[bool]
        :return: 5-tuple ``(ray, distance, point, mask_logit, scale)``.
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

            # An absent tensor makes every sample's flag False; a present tensor
            # with no explicit flag makes every sample's flag True.
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

        :return: The base ``Model`` config plus every constructor argument. The
            encoder sub-model is serialized, so a ``.keras`` archive round-trips
            both its topology and its weights.
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
        :return: An ``OmniPoint`` model instance.
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
        """Create an ``OmniPoint`` model from a predefined variant.

        :param variant: One of ``MODEL_VARIANTS`` (``"omnipoint_base"``,
            ``"omnipoint_large"``).
        :type variant: str
        :param pretrained: Must stay ``False``. No pretrained OmniPoint weights
            are distributed, and none exist for the ViT backbone either.
        :type pretrained: bool
        :param kwargs: Passthrough to the constructor, overriding the variant's
            own defaults.
        :type kwargs: Any
        :return: A configured ``OmniPoint`` instance.
        :rtype: OmniPoint
        :raises ValueError: If ``variant`` is not recognized.
        :raises NotImplementedError: If ``pretrained`` is ``True``.
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

        # DECISION D-127: copy the preset before updating it; splatting the
        # shared dict would poison MODEL_VARIANTS for later callers.
        config = cls.MODEL_VARIANTS[variant].copy()
        config.update(kwargs)

        logger.info(f"Creating OmniPoint model variant '{variant}'")
        return cls(**config)

#: Read-only alias of OmniPoint.MODEL_VARIANTS, so the name can be imported
#: directly from the package. The class attribute stays the source of truth.
MODEL_VARIANTS: Dict[str, Dict[str, Any]] = OmniPoint.MODEL_VARIANTS


def create_omnipoint(
        variant: str = "omnipoint_base",
        image_shape: Tuple[int, int, int] = (224, 224, 3),
        **kwargs: Any,
) -> OmniPoint:
    """Create an ``OmniPoint`` model and build it with one dummy forward pass.

    :param variant: One of ``OmniPoint.MODEL_VARIANTS``.
    :type variant: str
    :param image_shape: Input image shape ``(height, width, channels)``.
    :type image_shape: Tuple[int, int, int]
    :param kwargs: Passthrough to ``OmniPoint.from_variant``.
    :type kwargs: Any
    :return: A built ``OmniPoint`` model instance.
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
