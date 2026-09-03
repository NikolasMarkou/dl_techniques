"""`DepthAnything`, a monocular depth model built from a ViT encoder, a
`DPTDecoder` head, and an optional Mean-Teacher branch for semi-supervised
training. Implements the Depth Anything V1 recipe, not V2: no synthetic
training data and no distillation through a large teacher.

Depth from a single image is scale-ambiguous, so the model targets depth up
to an unknown affine transform rather than metric depth. The semi-supervised
branch trains the student on a strongly perturbed (color jitter, CutMix) copy
of an image against a target taken from the teacher's prediction on the clean
image, so the two disagree and there is a gradient to learn from. A second
term aligns student features to the teacher to preserve the encoder's
pretrained semantic structure. Augmentation runs inside `train_step`, not
`call`, because CutMix changes the correct target over the mixed region.

The encoder is this repository's plain `ViT` (patch size 16) or a
`Conv-BN-ReLU` placeholder stack, not DINOv2, and ships no pretrained
weights; `from_pretrained_encoder` loads one. The teacher starts as a clone
of the student and only becomes an EMA average once the caller attaches
`TeacherEMACallback` (`teacher_ema.py`); nothing in this module advances it
on its own. `compile()` with no `loss=` defaults to `AffineInvariantLoss`,
matching the relative-depth objective; `src/train/depth_anything/` passes
its own `DepthEstimationLoss` instead. `use_feature_alignment` defaults to
`True` and `enable_semi_supervised` to `False`, so a default-configured
model builds a teacher that the default `train_step` path never uses;
`__init__` warns when it detects this combination.

The model overrides `train_step`, which most models in this repository do
not: the semi-supervised path needs two batches with different augmentation
per batch and a teacher outside the loss graph, neither of which fits
`compute_loss`'s single-batch signature.

References:
    - Yang et al., 2024. Depth Anything: Unleashing the Power of Large-Scale
      Unlabeled Data. CVPR. (https://arxiv.org/abs/2401.10891)
    - Yang et al., 2024. Depth Anything V2. (https://arxiv.org/abs/2406.09414)
      Not implemented here; listed to mark which generation this is.
    - Ranftl et al., 2021. Vision Transformers for Dense Prediction (DPT).
      (https://arxiv.org/abs/2103.13413)
      The decoder this one is named after and simplifies.
    - Ranftl et al., 2020. Towards Robust Monocular Depth Estimation: Mixing
      Datasets for Zero-shot Cross-dataset Transfer (MiDaS).
      (https://arxiv.org/abs/1907.01341)
      Origin of the affine-invariant objective this pipeline is built around.
    - Tarvainen & Valpola, 2017. Mean Teachers Are Better Role Models.
      (https://arxiv.org/abs/1703.01780)
      The EMA teacher that `teacher_ema.py` drives.
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
      The paper's encoder, which this package does not ship.
"""

import keras
from keras import ops
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.strong_augmentation import StrongAugmentation
from dl_techniques.losses.affine_invariant_loss import AffineInvariantLoss
from dl_techniques.losses.feature_alignment_loss import FeatureAlignmentLoss
from dl_techniques.models.vision.vit.model import ViT

from .components import DPTDecoder, REFERENCE_BN_EPSILON
from dl_techniques.utils.keras_registration import register_dl_technique

# Map depth_anything encoder_type slugs to ViT scale names.
#
# Kept as a module-level constant because it is also the source of
# ``DepthAnything.MODEL_VARIANTS`` below: the three slugs are named in exactly
# one place, so the encoder-type validation list can never drift from the
# variant registry (it used to be a second hand-written literal in __init__).
_VIT_SCALE_MAP: Dict[str, str] = {
    "vit_s": "small",
    "vit_b": "base",
    "vit_l": "large",
}
_VIT_PATCH_SIZE: int = 16

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.depth_anything.model")
class DepthAnything(keras.Model):
    """Monocular depth model: ViT encoder, DPT-style decoder, optional teacher.

    Architecture:

    .. code-block:: text

        Input [B, H, W, 3]
              │
              ▼
        ┌──────────────────────────────┐
        │ encoder: ViT (patch 16) or   │────► frozen_encoder (EMA teacher,
        │ placeholder Conv-BN-ReLU     │       optional, weight-shared clone)
        └──────────────┬───────────────┘
                       │  [B, N+1, D] (ViT) or [B, h, w, D]
                       ▼
        ┌──────────────────────────────┐
        │ _features_to_spatial:        │  (ViT path only: drop CLS token,
        │ reshape to [B, h, w, D]      │   reshape on image_shape // stride)
        └──────────────┬───────────────┘
                       ▼
        ┌──────────────────────────────┐
        │ decoder: DPTDecoder          │
        │ upsample_factor = stride     │
        └──────────────┬───────────────┘
                       ▼
        Output [B, H, W, output_channels]  (unconstrained, linear)

    Training (semi-supervised, `enable_semi_supervised=True`):

    .. code-block:: text

        labeled batch ──► augment ──► student ──► decoder ──► labeled loss
        unlabeled batch ─┬─► augment(strong) ──► student ──► decoder ──► consistency loss
                         └─► clean ──► frozen_encoder/teacher ──► pseudo-target
        student features ──► FeatureAlignmentLoss ──► frozen_encoder features
                                                        (only if use_feature_alignment)

    :param encoder_type: One of ``"vit_s"``, ``"vit_b"``, ``"vit_l"``.
    :type encoder_type: str
    :param image_shape: Input image shape ``(height, width, channels)``.
    :type image_shape: Tuple[int, int, int]
    :param decoder_dims: Channel dimension per decoder stage.
    :type decoder_dims: Optional[List[int]]
    :param output_channels: Number of channels in the predicted depth map.
    :type output_channels: int
    :param kernel_initializer: Initializer for convolutional kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for convolutional kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param loss_weights: Weight per loss term, keys ``"labeled"``,
        ``"unlabeled"``, ``"feature"``.
    :type loss_weights: Optional[Dict[str, float]]
    :param cutmix_prob: Probability of applying CutMix. Runs inside
        `train_step`, not `call`, so the depth target is CutMixed by the same
        box as the image.
    :type cutmix_prob: float
    :param color_jitter_strength: Strength of the color-jitter augmentation.
    :type color_jitter_strength: float
    :param input_value_range: Declared value range of the input images; color
        jitter clips back into it. Pass ``None`` for standardized or
        ``[-1, +1]`` inputs — see `create_depth_anything` for the full
        contract.
    :type input_value_range: Optional[Tuple[float, float]]
    :param use_feature_alignment: Whether to add the feature-alignment loss
        term. Governs that term only, not whether the teacher encoder is
        built: `enable_semi_supervised`'s pseudo-label term needs a teacher
        of its own, so setting this `True` with `enable_semi_supervised`
        `False` builds a teacher nothing reads, and `__init__` warns.
    :type use_feature_alignment: bool
    :param kwargs: Additional keyword arguments for the ``Model`` base class.

    :raises ValueError: If `encoder_type` or `encoder_kind` is not recognized.

    Input shape:
        4D tensor ``(batch_size, height, width, 3)``, or a tuple of two such
        tensors for semi-supervised training with labeled and unlabeled data.

    Output shape:
        4D tensor ``(batch_size, height, width, output_channels)``.

    Example:
        .. code-block:: python

            model = DepthAnything(
                encoder_type='vit_l',
                image_shape=(384, 384, 3),
                decoder_dims=[256, 128, 64, 32]
            )
            x = keras.random.normal([2, 384, 384, 3])
            depth = model(x)
            print(depth.shape)  # (2, 384, 384, 1)
    """

    # DECISION plan-2026-08-19T070627-a616f581/D-009: keep this table derived from `_VIT_SCALE_MAP`, never re-inline the slug list.
    # A second hand-written `['vit_s', 'vit_b', 'vit_l']` copy in __init__ can drift from the map that resolves them. See decisions.md.
    #: Public-name registry of the three published Depth Anything encoder sizes, derived from `_VIT_SCALE_MAP`.
    #: This package's variant knob is `encoder_type`, not `variant`, with no `from_variant`, so it is invisible
    #: to the repo-wide `MODEL_VARIANTS` guard in `tests/test_models/test_package_api_contract.py`.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        slug: {"encoder_type": slug, "vit_scale": scale}
        for slug, scale in _VIT_SCALE_MAP.items()
    }

    def __init__(
        self,
        encoder_type: str = 'vit_l',
        image_shape: Tuple[int, int, int] = (384, 384, 3),
        decoder_dims: Optional[List[int]] = None,
        output_channels: int = 1,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        cutmix_prob: float = 0.5,
        color_jitter_strength: float = 0.2,
        use_feature_alignment: bool = True,
        encoder_kind: str = 'real',
        enable_semi_supervised: bool = False,
        encoder: Optional[keras.Model] = None,
        input_value_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        input_shape: Optional[Tuple[int, int, int]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Back-compat alias: legacy `input_shape=` kwarg maps to `image_shape`.
        if input_shape is not None:
            logger.info(
                "DepthAnything: 'input_shape' kwarg is deprecated; use 'image_shape'. "
                "Forwarding the value."
            )
            image_shape = input_shape

        # Validate encoder type
        self.supported_encoders = list(self.MODEL_VARIANTS)
        if encoder_type not in self.supported_encoders:
            raise ValueError(
                f"Unsupported encoder type: {encoder_type}. "
                f"Supported types: {self.supported_encoders}"
            )
        if encoder_kind not in ('real', 'placeholder'):
            raise ValueError(
                f"Unsupported encoder_kind: {encoder_kind}. Choose 'real' or 'placeholder'."
            )

        # Store configuration parameters
        self.encoder_type = encoder_type
        self.image_shape = tuple(image_shape)
        # Keep legacy attribute name for any external code reading it.
        self.input_shape_param = self.image_shape
        self.decoder_dims = decoder_dims if decoder_dims is not None else [256, 128, 64, 32]
        self.output_channels = output_channels
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.loss_weights = loss_weights if loss_weights is not None else {
            'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1
        }
        self.cutmix_prob = cutmix_prob
        self.color_jitter_strength = color_jitter_strength
        self.input_value_range = (
            None if input_value_range is None else tuple(input_value_range)
        )
        self.use_feature_alignment = use_feature_alignment
        self.encoder_kind = encoder_kind
        self.enable_semi_supervised = bool(enable_semi_supervised)

        if self.use_feature_alignment and not self.enable_semi_supervised:
            logger.warning(
                "DepthAnything: use_feature_alignment=True with "
                "enable_semi_supervised=False builds the teacher encoder — "
                "roughly doubling parameter count and memory — while train_step "
                "never routes through the branch that reads it. Set "
                "enable_semi_supervised=True to use it, or "
                "use_feature_alignment=False to stop paying for it."
            )

        # Encoder geometry: stride from patch_size for real ViT, 32 for placeholder
        # (initial stride-2 conv + 3 maxpools across 4 stages — last stage no pool).
        if self.encoder_kind == 'real':
            self.encoder_stride = _VIT_PATCH_SIZE  # 16
        else:
            # Placeholder Conv encoder: initial stride-2 conv => /2, then 3
            # stride-2 maxpools across stages 0..2 (stage 3 has no pool) => /8.
            # Total stride 16 (matches the 4-stage DPT decoder's upsample).
            self.encoder_stride = 16
        self.encoder_h = self.image_shape[0] // self.encoder_stride
        self.encoder_w = self.image_shape[1] // self.encoder_stride

        # If an encoder was supplied (typically by `from_config` after
        # deserialization), accept it directly so its saved topology + weights
        # survive the load. Otherwise build() will create one fresh.
        self.encoder: Optional[keras.Model] = encoder
        self.encoder_embed_dim: Optional[int] = None
        if self.encoder_kind == 'real':
            scale = _VIT_SCALE_MAP[self.encoder_type]
            self.encoder_embed_dim = ViT.SCALE_CONFIGS[scale][0]

        # Other components — initialized in build().
        self.decoder: Optional[keras.layers.Layer] = None
        self.frozen_encoder: Optional[keras.Model] = None
        self.augmentation: Optional[keras.layers.Layer] = None

        logger.info(
            f"Initialized DepthAnything (encoder_type={encoder_type}, "
            f"encoder_kind={encoder_kind}, image_shape={self.image_shape}, "
            f"semi_supervised={self.enable_semi_supervised})"
        )

    def build(self, input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]) -> None:
        """Build the encoder, decoder, optional teacher, and augmentation.

        :param input_shape: Shape of input tensor(s).
        :type input_shape: Union[Tuple[int, ...], List[Tuple[int, ...]]]
        """

        # Construct the encoder if not already provided via from_config (which
        # passes a deserialized sub-Model directly). Building lazily inside
        # build() keeps the inner sub-Model under DepthAnything's tracking only
        # for fresh instantiations; for loaded models the encoder is already a
        # deserialized keras.Model with the saved topology + weights.
        if self.encoder is None:
            if self.encoder_kind == 'real':
                scale = _VIT_SCALE_MAP[self.encoder_type]
                self.encoder = ViT(
                    input_shape=self.image_shape,
                    scale=scale,
                    patch_size=_VIT_PATCH_SIZE,
                    include_top=False,
                    pooling=None,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    name=f'encoder_{self.encoder_type}_real',
                )
            else:
                self.encoder = self._create_placeholder_encoder(trainable=True)

        # Decoder: pass upsample_factor so the spatial output matches image_shape.
        # For real ViT (stride=16) with len(decoder_dims)>=4, upsample_factor=16 is
        # representable as 4 stages of 2x. For placeholder (stride=16 here) ditto.
        upsample_factor = self.encoder_stride
        self.decoder = DPTDecoder(
            dims=self.decoder_dims,
            output_channels=self.output_channels,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            upsample_factor=upsample_factor,
            name='dpt_decoder',
        )

        # Build order: ensure the student encoder is built, clone its topology and
        # force-build the clone, then copy weights student -> teacher and freeze.
        # DECISION plan-2026-08-17T183311-79c63e38/D-033: build the teacher when either flag is set, not `use_feature_alignment` alone.
        # `enable_semi_supervised` needs a teacher for pseudo-label consistency too; narrowing this silently degraded to labeled-only training. See decisions.md.
        if self.use_feature_alignment or self.enable_semi_supervised:
            try:
                dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
                _ = self.encoder(dummy, training=False)
                self.frozen_encoder = keras.models.clone_model(self.encoder)
                _ = self.frozen_encoder(dummy, training=False)
                self.frozen_encoder.set_weights(self.encoder.get_weights())
                self.frozen_encoder.trainable = False
            except Exception as exc:  # pragma: no cover — diagnostic path
                logger.warning(
                    f"DepthAnything: clone_model(encoder) failed ({exc!r}); "
                    "disabling feature alignment for this run"
                    + (
                        ", and the pseudo-label consistency term with it — "
                        "semi-supervised training will run labeled-only."
                        if self.enable_semi_supervised
                        else "."
                    )
                )
                self.frozen_encoder = None
                self.use_feature_alignment = False

        # Strong augmentation pipeline (always available — module-level import).
        self.augmentation = StrongAugmentation(
            cutmix_prob=self.cutmix_prob,
            color_jitter_strength=self.color_jitter_strength,
            input_value_range=self.input_value_range,
            name='strong_augmentation',
        )

        super().build(input_shape)

    def update_teacher_ema(self, decay: float = 0.999) -> None:
        """Update the frozen teacher encoder via EMA over the student weights.

        Intended to be called from a Keras callback per training step. No-op when
        the frozen encoder was not built.

        The condition is the teacher's existence alone. It used to also require
        ``use_feature_alignment``, which pinned the teacher at its initial
        weights for the whole run under ``enable_semi_supervised=True,
        use_feature_alignment=False`` — the pseudo-label consistency term needs
        a *moving* teacher just as much as the alignment term does.

        :param decay: EMA decay factor in ``[0, 1]``. Higher values give a
            slower update.
        :type decay: float
        """
        if self.frozen_encoder is None:
            return
        student_w = self.encoder.get_weights()
        teacher_w = self.frozen_encoder.get_weights()
        if len(student_w) != len(teacher_w):
            logger.warning(
                "update_teacher_ema: student/teacher weight counts differ; skipping."
            )
            return
        new_w = [decay * t + (1.0 - decay) * s for t, s in zip(teacher_w, student_w)]
        self.frozen_encoder.set_weights(new_w)

    def from_pretrained_encoder(
        self,
        weights_path: str,
        skip_prefixes: Tuple[str, ...] = (),
    ) -> 'DepthAnything':
        """Load encoder weights from a saved ``.keras`` checkpoint and re-sync teacher.

        Wraps :func:`dl_techniques.utils.weight_transfer.load_weights_from_checkpoint`
        against ``self.encoder``. Force-builds the model first if needed. After
        a successful load, re-copies student → teacher when feature alignment
        is enabled, so the teacher starts from the pretrained snapshot.

        :param weights_path: Path to a ``.keras`` checkpoint produced by
            ``model.save(...)``. The checkpoint may itself be a
            DepthAnything snapshot or a standalone encoder snapshot; the
            weight-transfer helper matches by layer name.
        :type weights_path: str
        :param skip_prefixes: Layer-name prefixes to ignore during transfer.
        :type skip_prefixes: Tuple[str, ...]
        :return: ``self``, so calls can chain.
        :rtype: DepthAnything
        """
        if not self.built:
            dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
            _ = self(dummy, training=False)

        report = load_weights_from_checkpoint(
            target=self.encoder,
            ckpt_path=weights_path,
            skip_prefixes=skip_prefixes,
        )
        logger.info(
            f"from_pretrained_encoder: loaded={report.num_loaded} "
            f"shape_mismatch={report.num_shape_mismatch} "
            f"missing_in_source={len(report.missing_in_source)} "
            f"unused_in_source={len(report.unused_in_source)}"
        )
        # Re-sync the frozen teacher so it starts from the pretrained snapshot.
        if self.frozen_encoder is not None:
            try:
                self.frozen_encoder.set_weights(self.encoder.get_weights())
                logger.info("from_pretrained_encoder: teacher re-synced from student.")
            except Exception as exc:  # pragma: no cover — diagnostic path
                logger.warning(
                    f"from_pretrained_encoder: teacher re-sync failed ({exc!r})."
                )
        return self

    def _features_to_spatial(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Convert a ViT-style ``(B, N+1, D)`` sequence into ``(B, h, w, D)``.

        Drops the CLS token and reshapes using the encoder geometry derived from
        ``image_shape`` and ``patch_size``. 4-D inputs are returned unchanged.
        """
        if len(x.shape) == 4:
            return x
        # (B, N+1, D) → drop CLS → (B, N, D) → reshape (B, h, w, D)
        x = x[:, 1:, :]
        d = self.encoder_embed_dim or x.shape[-1]
        return ops.reshape(x, (-1, self.encoder_h, self.encoder_w, d))

    def _create_placeholder_encoder(self, trainable: bool = True) -> keras.Model:
        """Create the placeholder Conv-BN-ReLU encoder (legacy mode).

        Used when ``encoder_kind='placeholder'``. For ``encoder_kind='real'``
        the actual ViT backbone is constructed eagerly in ``__init__``.

        :param trainable: Whether the encoder should be trainable.
        :type trainable: bool
        :return: Encoder model instance.
        :rtype: keras.Model
        """
        inputs = keras.layers.Input(shape=self.image_shape, name='encoder_input')

        # Initial convolution with proper initialization and regularization
        x = keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name='initial_conv'
        )(inputs)
        # DECISION plan-2026-08-17T183311-79c63e38/D-028: pass epsilon explicitly on all three placeholder BatchNorms.
        # Justification is consistency with the downstream DPT head's 1e-5, not reference fidelity (this stack has no reference). See decisions.md.
        x = keras.layers.BatchNormalization(
            epsilon=REFERENCE_BN_EPSILON, name='initial_bn'
        )(x)
        x = keras.layers.ReLU(name='initial_relu')(x)
        # Note: removed legacy stride-2 'initial_pool' to keep placeholder
        # encoder stride at 16 (matches 4-stage DPT decoder upsample).

        # Progressive feature extraction blocks
        dims = [64, 128, 256, 512]
        for i, dim in enumerate(dims):
            # First conv block
            x = keras.layers.Conv2D(
                filters=dim,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
                name=f'conv_block_{i}_1'
            )(x)
            x = keras.layers.BatchNormalization(
                epsilon=REFERENCE_BN_EPSILON, name=f'bn_block_{i}_1'
            )(x)
            x = keras.layers.ReLU(name=f'relu_block_{i}_1')(x)

            # Second conv block
            x = keras.layers.Conv2D(
                filters=dim,
                kernel_size=3,
                padding='same',
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                use_bias=False,
                name=f'conv_block_{i}_2'
            )(x)
            x = keras.layers.BatchNormalization(
                epsilon=REFERENCE_BN_EPSILON, name=f'bn_block_{i}_2'
            )(x)
            x = keras.layers.ReLU(name=f'relu_block_{i}_2')(x)

            # Downsample (except for last block to maintain spatial resolution)
            if i < len(dims) - 1:
                x = keras.layers.MaxPooling2D(
                    pool_size=2,
                    strides=2,
                    padding='same',
                    name=f'pool_block_{i}'
                )(x)

        # Feature projection layer
        features = keras.layers.Conv2D(
            filters=self.decoder_dims[0],  # Match decoder input
            kernel_size=1,
            padding='same',
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            use_bias=False,
            name='feature_projection'
        )(x)

        encoder = keras.Model(
            inputs=inputs,
            outputs=features,
            name=f'encoder_{self.encoder_type}'
        )
        encoder.trainable = trainable

        return encoder

    def call(
        self,
        inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through the model.

        This is the plain encoder-decoder path. It does not augment: strong
        augmentation is applied by `train_step` (see `_augment_with_targets`),
        where the depth target is in scope and can be CutMixed by the same box.

        :param inputs: Input tensor, shape ``(batch_size, height, width, 3)``,
            or a tuple of ``(labeled, unlabeled)`` tensors for training.
        :type inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param training: Whether the model runs in training or inference mode.
        :type training: Optional[bool]
        :return: Predicted depth maps, shape ``(batch_size, height, width, output_channels)``.
        :rtype: keras.KerasTensor
        """
        # Handle both single input and tuple input for training
        if isinstance(inputs, tuple):
            x_labeled, x_unlabeled = inputs
            # For simplicity, process labeled data in forward pass
            # Complex training logic would be handled in train_step
            x = x_labeled
        else:
            x = inputs

        # Extract features. ViT returns (B, N+1, D); placeholder returns 4-D.
        features = self.encoder(x, training=training)

        # Reshape sequence features to spatial 4-D before the decoder.
        features = self._features_to_spatial(features)

        # Decode features to depth.
        depth = self.decoder(features, training=training)

        return depth

    def compile(
        self,
        optimizer: keras.optimizers.Optimizer,
        loss: Optional[keras.losses.Loss] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        **kwargs: Any
    ) -> None:
        """Configure the model for training.

        :param optimizer: Keras optimizer instance.
        :type optimizer: keras.optimizers.Optimizer
        :param loss: Primary loss function. If ``None``, uses
            `AffineInvariantLoss`, matching depth supervision that is only
            defined up to an unknown affine transform.
        :type loss: Optional[keras.losses.Loss]
        :param loss_weights: Optional custom loss weights, overriding the
            defaults.
        :type loss_weights: Optional[Dict[str, float]]
        :param kwargs: Additional arguments passed to the parent ``compile``.
        """
        # Set default loss if none provided
        if loss is None:
            loss = AffineInvariantLoss()

        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        # Update loss weights if provided. Specialized loss instances are not
        # stored on `self` — that previously dead state caused get_config drift.
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        logger.info(f"Compiled DepthAnything with loss weights: {self.loss_weights}")

    def _pseudo_label_depth(self, x_unlab: keras.KerasTensor) -> keras.KerasTensor:
        """Generate stop-gradient pseudo-depth labels from the EMA teacher.

        Runs the frozen teacher encoder + decoder in inference mode and
        returns ``stop_gradient(depth)``. The student's gradient flows
        through ``self(x_unlab, training=True)`` but never through this path
        — exactly the Mean-Teacher / consistency-regularization recipe.

        :param x_unlab: Unlabeled input batch ``(B, H, W, C)``.
        :type x_unlab: keras.KerasTensor
        :return: Pseudo-depth tensor ``(B, H, W, output_channels)``, no gradient.
        :rtype: keras.KerasTensor
        """
        feat = self.frozen_encoder(x_unlab, training=False)
        feat = self._features_to_spatial(feat)
        pseudo = self.decoder(feat, training=False)
        return ops.stop_gradient(pseudo)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-014: augmentation stays in the training path, never in `call()`.
    # `call()` has no target in scope, so a `call()`-side CutMix mixed the image but not the depth map on ~cutmix_prob of batches. See decisions.md.
    def _augment_with_targets(
        self,
        x: keras.KerasTensor,
        targets: List[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor]]:
        """Strongly augment ``x`` and apply the same CutMix box to each target.

        Interface contract (called from both training paths):

        * Returns ``(x_aug, mixed_targets)`` with ``len(mixed_targets) ==
          len(targets)``, each entry shaped like its input.
        * Every target must share ``x``'s batch and spatial axes; channel counts
          are free (depth alone, or depth + validity mask).
        * When ``self.augmentation`` is ``None`` — tests and callers that opt out
          of strong augmentation set it so — the inputs are returned unchanged.

        :param x: Image batch ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :param targets: Tensors to mix by the same box as ``x``.
        :type targets: List[keras.KerasTensor]
        :return: The augmented images and the identically mixed targets.
        :rtype: Tuple[keras.KerasTensor, List[keras.KerasTensor]]
        """
        if self.augmentation is None:
            return x, list(targets)
        x_aug, mix = self.augmentation.augment_with_mix(x, training=True)
        return x_aug, [self.augmentation.apply_mix_to_target(t, mix) for t in targets]

    # DECISION plan-2026-08-17T183311-79c63e38/D-033: both custom step methods must end by calling this.
    # `compute_loss` does not feed `_loss_tracker` the way the default `train_step` does; skipping this left `history["loss"]` at 0.0 every step. See decisions.md.
    def _finalize_train_step(
        self,
        y: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        loss: keras.KerasTensor,
    ) -> Dict[str, keras.KerasTensor]:
        """Update every metric and return a flat logs dict.

        Interface contract (called from both training paths, and only from
        inside a step method after the gradients have been applied):

        * Updates each compiled metric with ``(y, y_pred)`` and feeds
          ``self._loss_tracker`` the scalar ``loss`` that was actually
          optimized — including every auxiliary term, not just the labeled one.
        * Returns ``{metric_name: scalar}``. `CompileMetrics.result()` returns a
          nested dict and is spliced in flat, so no ``"compile_metrics"`` key
          survives. Keras' own `pythonify_logs` would flatten it for callbacks
          anyway, but a direct `model.train_step(batch)` caller sees the raw
          dict, and that one was nested.
        * Never raises for a missing tracker: `_loss_tracker` only exists after
          `compile()`.

        :param y: Ground-truth targets for the compiled metrics.
        :type y: keras.KerasTensor
        :param y_pred: Model predictions for the compiled metrics.
        :type y_pred: keras.KerasTensor
        :param loss: The scalar total loss that was backpropagated.
        :type loss: keras.KerasTensor
        :return: Flat mapping of metric name to scalar result.
        :rtype: Dict[str, keras.KerasTensor]
        """
        loss_tracker = getattr(self, "_loss_tracker", None)
        for metric in self.metrics:
            if metric is loss_tracker:
                continue
            metric.update_state(y, y_pred)
        if loss_tracker is not None:
            loss_tracker.update_state(loss)

        results: Dict[str, keras.KerasTensor] = {}
        for metric in self.metrics:
            value = metric.result()
            if isinstance(value, dict):
                results.update(value)
            else:
                results[metric.name] = value
        return results

    def _train_step_labeled(
        self,
        x: keras.KerasTensor,
        y: keras.KerasTensor,
    ) -> Dict[str, keras.KerasTensor]:
        """Labeled-only path: augment image+target together, forward, backprop.

        Both step methods differentiate ``scaled_loss`` and report ``loss``.
        See the D-034 anchor below for why the two must not be collapsed.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-034: differentiate the scaled loss, report the unscaled one.
        # Under mixed_float16 skipping `scale_loss` divides every gradient by the loss scale (measured |dW| ratio ~2.8e4); float32 is unaffected. See decisions.md.
        x, (y,) = self._augment_with_targets(x, [y])
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Keras-3 canonical train_step — replaces deprecated
            # compiled-loss / compiled-metrics calls.
            # See dl_techniques/models/language/masked_language_model/mlm.py:309-343.
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
            loss = loss * self.loss_weights.get('labeled', 1.0)
            scaled_loss = self.optimizer.scale_loss(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self._finalize_train_step(y, y_pred, loss)

    def _train_step_semi_supervised(
        self,
        x_lab: keras.KerasTensor,
        x_unlab: keras.KerasTensor,
        y: keras.KerasTensor,
    ) -> Dict[str, keras.KerasTensor]:
        """Semi-supervised path: labeled compute_loss + consistency + optional FAL.

        Both branches take the teacher's view from the *clean* batch and the
        student's from the *augmented* one — that asymmetry is the whole recipe.
        What is not part of the recipe is CutMix mixing across batch rows without
        the target following: the pseudo-label is therefore mixed by the same box
        as the student's input before the consistency term compares them.
        """
        x_lab, (y,) = self._augment_with_targets(x_lab, [y])
        with tf.GradientTape() as tape:
            # ---- labeled supervision ----
            y_pred = self(x_lab, training=True)
            loss = self.compute_loss(x=x_lab, y=y, y_pred=y_pred)
            loss = loss * self.loss_weights.get('labeled', 1.0)

            # ---- semi-sup branch: consistency (always) + FAL (opt-in) ----
            # DECISION plan-2026-08-17T183311-79c63e38/D-033: gate on the teacher's existence, not `use_feature_alignment`.
            # Coupling pseudo-label consistency to that flag made `enable_semi_supervised=True, use_feature_alignment=False` silently labeled-only. See decisions.md.
            if self.frozen_encoder is not None:
                # The teacher's pseudo-depth is read off the clean batch, then
                # mixed by the same box the student's input received.
                pseudo = self._pseudo_label_depth(x_unlab)
                x_unlab_aug, (pseudo,) = self._augment_with_targets(
                    x_unlab, [pseudo]
                )

                if self.use_feature_alignment:
                    # Feature-Alignment-Loss on unlabeled features. The student
                    # sees the augmented batch here too — reading
                    # `self.encoder(x_unlab)` directly would bypass
                    # `self.augmentation` and reduce the term to a
                    # train/eval-mode difference between two initially
                    # identical encoders.
                    feat_student = self.encoder(x_unlab_aug, training=True)
                    feat_teacher = self.frozen_encoder(x_unlab, training=False)
                    # Pool to (B, D). ViT seq output is (B, N+1, D); drop CLS.
                    if len(feat_student.shape) == 4:
                        feat_student = ops.mean(feat_student, axis=[1, 2])
                        feat_teacher = ops.mean(feat_teacher, axis=[1, 2])
                    elif len(feat_student.shape) == 3:
                        feat_student = ops.mean(feat_student[:, 1:, :], axis=1)
                        feat_teacher = ops.mean(feat_teacher[:, 1:, :], axis=1)
                    fal = FeatureAlignmentLoss()(feat_teacher, feat_student)
                    loss = loss + self.loss_weights.get('feature', 0.1) * fal

                # Pseudo-label consistency: L1 between student depth on the
                # augmented batch and the teacher's identically mixed,
                # stop-gradient pseudo-depth.
                y_pred_unlab = self(x_unlab_aug, training=True)
                consistency = ops.mean(ops.abs(y_pred_unlab - pseudo))
                loss = loss + self.loss_weights.get('unlabeled', 0.5) * consistency

            scaled_loss = self.optimizer.scale_loss(loss)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(scaled_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return self._finalize_train_step(y, y_pred, loss)

    def train_step(self, data: Any) -> Dict[str, keras.KerasTensor]:
        """Execute one training step.

        Two input shapes are accepted:

        * ``(x, y)`` — labeled-only path (default). Routed to
          :meth:`_train_step_labeled`.
        * ``((x_lab, x_unlab), y_lab)`` — semi-supervised path. Active only
          when ``self.enable_semi_supervised`` is True. Routed to
          :meth:`_train_step_semi_supervised`, which adds a stop-gradient
          pseudo-label L1-consistency term over ``x_unlab`` and, when
          ``use_feature_alignment`` is also set, a Feature-Alignment-Loss term.

        :param data: Training data batch.
        :type data: Any
        :return: Dictionary containing loss metrics.
        :rtype: Dict[str, keras.KerasTensor]
        """
        x, y = data
        if (
            self.enable_semi_supervised
            and isinstance(x, (tuple, list))
            and len(x) == 2
        ):
            return self._train_step_semi_supervised(x[0], x[1], y)
        return self._train_step_labeled(x, y)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :return: Dictionary containing the model configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "encoder_type": self.encoder_type,
            "image_shape": self.image_shape,
            "decoder_dims": self.decoder_dims,
            "output_channels": self.output_channels,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "loss_weights": self.loss_weights,
            "cutmix_prob": self.cutmix_prob,
            "color_jitter_strength": self.color_jitter_strength,
            "input_value_range": self.input_value_range,
            "use_feature_alignment": self.use_feature_alignment,
            "encoder_kind": self.encoder_kind,
            "enable_semi_supervised": self.enable_semi_supervised,
            # Serialize the encoder sub-Model so save/load round-trips both
            # topology and weights through `.keras` archives. Mirrors the
            # MaskedLanguageModel pattern in mlm.py.
            "encoder": (
                keras.saving.serialize_keras_object(self.encoder)
                if self.encoder is not None
                else None
            ),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DepthAnything':
        """Create model from configuration.

        Accepts both ``image_shape`` (current key) and ``input_shape``
        (legacy key from pre-bd098beb saved configs) for back-compat.

        :param config: Dictionary containing model configuration.
        :type config: Dict[str, Any]
        :return: DepthAnything model instance.
        :rtype: DepthAnything
        """
        cfg = dict(config)
        # Deserialize initializer/regularizer if present as serialized dicts.
        if isinstance(cfg.get("kernel_initializer"), dict):
            cfg["kernel_initializer"] = keras.initializers.deserialize(
                cfg["kernel_initializer"]
            )
        if isinstance(cfg.get("kernel_regularizer"), dict):
            cfg["kernel_regularizer"] = keras.regularizers.deserialize(
                cfg["kernel_regularizer"]
            )
        # Deserialize encoder sub-Model when present.
        enc_cfg = cfg.pop("encoder", None)
        if enc_cfg is not None:
            cfg["encoder"] = keras.saving.deserialize_keras_object(enc_cfg)
        return cls(**cfg)

    # ------------------------------------------------------------------
    # Load-time materialization of the nested sub-Models.
    # ------------------------------------------------------------------
    # DECISION plan-2026-08-22T035419-a11304c8/D-009: keep the force-build here; never add a `save_own_variables` override.
    # A flat `self.weights` dump duplicates Keras' own recursive save (measured 2.00x archive size) without replacing it. See decisions.md.
    def load_own_variables(self, store: Any) -> None:  # type: ignore[override]
        """Materialize the nested sub-Models before Keras restores them.

        Keras 3 calls ``load_own_variables`` on the outer model *before* it
        recurses into ``encoder`` / ``frozen_encoder`` / ``decoder``. Those
        sub-Models are constructed in :meth:`build` but their own sub-layers
        (and therefore their variables) only exist once something has run a
        forward pass through them, so the recursion would otherwise restore
        into an empty tree. A single dummy forward under the saved
        ``image_shape`` materializes the full variable set first; the actual
        restore is then the ordinary path-based one, delegated to
        ``keras.Model``.
        """
        if not self.built or any(
            sub is not None and not sub.built
            for sub in (self.encoder, self.frozen_encoder, self.decoder)
        ):
            dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
            _ = self(dummy, training=False)

        super().load_own_variables(store)

# ---------------------------------------------------------------------

def create_depth_anything(
    encoder_type: str = 'vit_l',
    image_shape: Tuple[int, int, int] = (384, 384, 3),
    decoder_dims: Optional[List[int]] = None,
    output_channels: int = 1,
    kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
    kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    cutmix_prob: float = 0.5,
    color_jitter_strength: float = 0.2,
    use_feature_alignment: bool = True,
    encoder_kind: str = 'real',
    enable_semi_supervised: bool = False,
    input_value_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    input_shape: Optional[Tuple[int, int, int]] = None,
) -> DepthAnything:
    """Create and build Depth Anything model instance.

    Input contract: the model does not normalize its inputs, but the strong
    augmentation applied during training needs to know their range, since
    color jitter scales brightness and contrast and clips its result back into
    `input_value_range`. The default `(0.0, 1.0)` says the caller feeds images
    in `[0, 1]`. Pass `input_value_range=None` for standardized (mean-zero) or
    `[-1, +1]` images; the trainer in `src/train/depth_anything/` does this,
    since `src/train/common/megadepth.py` emits RGB in `[-1, +1]` and clipping
    those to `[0, 1]` would flatten every negative pixel to zero on the
    training path only, while evaluation saw the untouched image.

    Augmentation runs inside `train_step`, not inside `call`: CutMix mixes
    across batch rows, so the depth target has to be mixed by the same
    rectangle, and only the training path has the target. Calling
    `model(x, training=True)` directly returns an un-augmented forward pass.

    :param encoder_type: One of ``"vit_s"``, ``"vit_b"``, ``"vit_l"``.
    :type encoder_type: str
    :param image_shape: Input image shape ``(height, width, channels)``.
    :type image_shape: Tuple[int, int, int]
    :param decoder_dims: Channel dimension per decoder stage.
    :type decoder_dims: Optional[List[int]]
    :param output_channels: Number of channels in the predicted depth map.
    :type output_channels: int
    :param kernel_initializer: Initializer for convolutional kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for convolutional kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param loss_weights: Weight per loss term, keys ``"labeled"``,
        ``"unlabeled"``, ``"feature"``.
    :type loss_weights: Optional[Dict[str, float]]
    :param cutmix_prob: Probability of applying CutMix augmentation.
    :type cutmix_prob: float
    :param color_jitter_strength: Strength of the color-jitter augmentation.
    :type color_jitter_strength: float
    :param use_feature_alignment: Whether to use the feature-alignment loss.
    :type use_feature_alignment: bool
    :param encoder_kind: Encoder implementation, ``"real"`` (ViT) or
        ``"placeholder"`` (Conv-BN-ReLU stack).
    :type encoder_kind: str
    :param enable_semi_supervised: Whether to enable the semi-supervised
        training branch.
    :type enable_semi_supervised: bool
    :param input_value_range: Declared value range of the input images (see
        the input contract above).
    :type input_value_range: Optional[Tuple[float, float]]
    :param input_shape: Deprecated alias for `image_shape`.
    :type input_shape: Optional[Tuple[int, int, int]]
    :return: Configured and built DepthAnything model instance.
    :rtype: DepthAnything
    :raises ValueError: If `encoder_type` is not recognized.

    Example:
        .. code-block:: python

            model = create_depth_anything(
                encoder_type='vit_l',
                image_shape=(384, 384, 3),
                kernel_regularizer=keras.regularizers.L2(0.01)
            )
            model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=5e-6),
                loss=keras.losses.MeanSquaredError()
            )
    """
    logger.info(f"Creating DepthAnything model with encoder: {encoder_type}")

    # Resolve image_shape (legacy 'input_shape' alias).
    if input_shape is not None:
        image_shape = input_shape

    # Create model with specified configuration
    model = DepthAnything(
        encoder_type=encoder_type,
        image_shape=image_shape,
        decoder_dims=decoder_dims,
        output_channels=output_channels,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        loss_weights=loss_weights,
        cutmix_prob=cutmix_prob,
        color_jitter_strength=color_jitter_strength,
        input_value_range=input_value_range,
        use_feature_alignment=use_feature_alignment,
        encoder_kind=encoder_kind,
        enable_semi_supervised=enable_semi_supervised,
    )

    # Build model with dummy input to initialize all components
    dummy_input = keras.random.normal([1] + list(image_shape))
    _ = model(dummy_input)

    logger.info("Successfully created and built DepthAnything model")

    return model

# ---------------------------------------------------------------------
