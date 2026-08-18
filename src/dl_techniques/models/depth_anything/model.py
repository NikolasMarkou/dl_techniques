"""Monocular depth estimation following the Depth Anything (V1) semi-supervised
recipe: a ViT encoder, a convolutional dense-prediction head, and an optional
teacher branch supplying feature alignment and pseudo-label consistency.

Predicting depth from a single image is ill-posed — an image is consistent with a
whole family of scenes related by scale — so the only supervision that
generalizes across datasets is *relative*: depth up to an unknown affine
transform. That framing makes heterogeneous data usable, which is the real
bottleneck, because metric depth labels require LiDAR or stereo rigs and exist
for a few million images at most while unlabeled photographs exist by the
billion. Depth Anything's contribution is the observation that naive
pseudo-labeling on that unlabeled pool does nothing: a student trained on its
teacher's predictions of unperturbed images has no gradient to learn from,
because it can already produce those predictions. The fix is to make the
student's job strictly harder than the teacher's. The teacher sees a clean image;
the student sees the same image after strong perturbation (color jitter, CutMix)
and must still agree. The residual difficulty is what carries information. A
second term keeps the encoder from drifting away from the semantic structure of
its pretrained initialization by aligning student features to a frozen
counterpart's — semantic priors are what let a depth model reason about object
boundaries it never saw supervised.

Color jitter is a per-image photometric perturbation, so "the same image" holds
under it and the asymmetry above is exactly the intended one. CutMix is not: it
pastes a rectangle taken from *another row of the batch*, which changes what the
correct answer is over that rectangle. Augmentation therefore does not happen in
`call()`, where no target is in scope; `train_step` calls
`_augment_with_targets`, which mixes the image and every target — the labeled
depth map, and the teacher's pseudo-depth on the unlabeled branch — by one
shared box. `call(x, training=True)` is a plain un-augmented forward pass.

`DepthAnything` implements the **V1** recipe. V2's defining moves — replacing
real labeled data with synthetic renderings and distilling through a large
teacher onto pseudo-labeled real images — are not present.

Three things about the encoder differ from the paper and change what the model
inherits. First, the backbone is this repository's plain supervised `ViT` (patch
size 16), or a `Conv-BN-ReLU` stack when `encoder_kind='placeholder'`, not
DINOv2; no pretrained weights are downloaded by this package, so a freshly
constructed model has no semantic prior to transfer and the feature-alignment
term is aligning a random encoder to a copy of itself until
`from_pretrained_encoder` supplies a real checkpoint. Second, the "frozen"
encoder is *not* frozen in the V1 sense of a fixed pretrained reference; it is a
`clone_model` of the student, initialized from the student's own weights, and it
is intended to be advanced as a Mean-Teacher exponential moving average through
`update_teacher_ema`. That method is a plain public method and nothing in this
module calls it: the driver is `TeacherEMACallback` in this package's
`teacher_ema.py`, attached by the trainer in `src/train/depth_anything/`. Without
that callback the teacher stays pinned at the student's initial weights forever.
Third, `compile()` with no `loss=` defaults to `AffineInvariantLoss`, not to
mean-squared error. This is the objective the whole recipe is built around: the
supervision is relative, so a prediction that is structurally correct but
globally scaled or shifted is *right*, and an MSE default would penalize it for
choosing its own scale — which also contradicts the decoder's deliberately
linear, unclamped output. The default is only what a caller who compiles with an
optimizer alone gets; the trainer in `src/train/depth_anything/` passes its own
`DepthEstimationLoss` explicitly and is unaffected by it.

The decoder is named `DPTDecoder` but is not the DPT of the paper. It performs no
multi-scale "reassemble" from intermediate transformer layers and has no residual
fusion blocks or skip connections; it is a linear stack of `3x3 Conv - BatchNorm
- ReLU` stages with bilinear 2x upsampling inserted after the first
`log2(upsample_factor)` of them, consuming only the encoder's final feature map.
`upsample_factor` is set to the encoder stride so the output resolution returns
to the input's, which is why the placeholder encoder was deliberately built to
stride 16 (its legacy stride-2 initial pool was removed) rather than the 32 a
four-stage conv tower would naturally give — the decoder's four stages can only
express `2**4`. The final projection is linear by default, leaving the output
unconstrained; that is the correct choice for affine-invariant or scale-shift
losses, which need the network free to choose its own scale, and it is why no
sigmoid clamps the depth to `[0, 1]`.

A ViT emits `(B, N+1, D)` sequences while the decoder needs `(B, h, w, D)`, so
`_features_to_spatial` drops the CLS token and reshapes on the patch grid derived
from `image_shape // patch_size`. This ties the model to a fixed input
resolution: `encoder_h` and `encoder_w` are computed once in `__init__`, so
calling the model at a resolution other than the configured `image_shape` will
reshape against the wrong grid.

Two behavioural choices are worth stating as choices. `use_feature_alignment`
defaults to `True` while `enable_semi_supervised` defaults to `False`, so the
default configuration *builds* the teacher — roughly doubling parameter count and
memory — while the custom `train_step` never routes through the semi-supervised
branch that would use it. Enable the second rather than paying for a teacher that
does nothing; `__init__` warns about exactly this combination. Note that
*disabling the first* is no longer a way to avoid the teacher when semi-supervision
is on: the pseudo-label consistency term needs a teacher regardless of whether the
feature-alignment term is wanted, so the teacher is built whenever either flag is
set and `use_feature_alignment` now governs only the FAL term. And the clone is
wrapped in a `try/except` that degrades to `use_feature_alignment = False` with a
warning instead of raising, so an encoder that cannot be cloned yields a working
but silently unregularized model — and, when semi-supervision was requested, one
whose consistency term is inactive too, which the same warning says.

This model overrides `train_step`, against this repository's standing rule that
models use stock `fit()` and feed extra signals through `tf.data`. The exception
is deliberate and was measured before it was granted: the semi-supervised path
consumes two batches per step with *different* augmentation applied to each
(teacher on the clean unlabeled batch, student on the perturbed one — the
asymmetry above is the entire recipe), and it reads a teacher network that is not
part of the loss graph. `compute_loss(x, y, y_pred, sample_weight, training)`
receives neither the second batch nor the teacher, so the rule's prescribed shape
cannot express this step. The price of the exception is that everything Keras'
default `train_step` does for free must be done by hand here — in particular
feeding `self._loss_tracker`, which the default `compute_loss` does *not* do; see
`_finalize_train_step`. Do not treat this file as precedent for an ordinary
single-batch model.

Serialization overrides `save_own_variables` / `load_own_variables` to write the
entire ordered `self.weights` list into flat numeric slots at the top level.
Keras 3's default behaviour recurses into child models and maps weights by
attribute path, and for a `keras.Model` nested as an attribute of another
`keras.Model` that path mapping was measured to drift between save and load —
55 of 172 weights restored re-initialized, moving the forward output by 1 to 2.8.
Writing one flat, path-free record bypasses the framework's path walking
entirely; `load_own_variables` force-builds via a dummy forward first so the
weight list it indexes matches the one that was written.

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
from dl_techniques.models.vit.model import ViT

from .components import DPTDecoder, REFERENCE_BN_EPSILON

# Map depth_anything encoder_type slugs to ViT scale names.
_VIT_SCALE_MAP: Dict[str, str] = {
    "vit_s": "small",
    "vit_b": "base",
    "vit_l": "large",
}
_VIT_PATCH_SIZE: int = 16

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class DepthAnything(keras.Model):
    """Depth Anything model implementation.

    Implements the complete Depth Anything architecture for monocular depth estimation.
    The model combines a feature encoder (placeholder for DINOv2) with a DPT decoder
    to produce dense depth predictions from RGB images.

    The architecture includes:
    - Feature encoder for extracting multi-scale representations
    - DPT decoder for dense prediction
    - Optional feature alignment with frozen encoder
    - Strong augmentation pipeline for robust training

    Args:
        encoder_type: String, type of ViT encoder to use.
            Supported values: ['vit_s', 'vit_b', 'vit_l'].
            Defaults to 'vit_l'.
        input_shape: Tuple of integers, input image shape as (height, width, channels).
            Defaults to (384, 384, 3).
        decoder_dims: List of integers, dimensions for decoder layers.
            Defaults to [256, 128, 64, 32].
        output_channels: Integer, number of output channels for depth prediction.
            Defaults to 1.
        kernel_initializer: String or Initializer, initializer for convolutional kernels.
            Defaults to "he_normal".
        kernel_regularizer: Regularizer or None, regularizer for convolutional kernels.
            Defaults to None.
        loss_weights: Dict of strings to floats, weights for different loss components.
            Keys: 'labeled', 'unlabeled', 'feature'.
            Defaults to {'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1}.
        cutmix_prob: Float, probability of applying CutMix augmentation. The
            augmentation runs inside `train_step`, not inside `call`, so that
            the depth target is CutMixed by the same box as the image.
            Defaults to 0.5.
        color_jitter_strength: Float, strength of color jittering augmentation.
            Defaults to 0.2.
        input_value_range: Tuple of two floats or None, the declared value range
            of the input images. Color jitter clips its result back into this
            range. Pass None for standardized or `[-1, +1]` inputs — see
            `create_depth_anything` for the full contract.
            Defaults to (0.0, 1.0).
        use_feature_alignment: Boolean, whether to add the feature-alignment
            loss term during semi-supervised training. This governs the FAL
            *term* only; it no longer governs whether the teacher encoder is
            built, because `enable_semi_supervised`'s pseudo-label consistency
            term needs a teacher of its own. Setting it True with
            `enable_semi_supervised` False builds a teacher nothing reads, and
            warns. Defaults to True.
        **kwargs: Additional keyword arguments for the Model base class.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, 3)`
        Or tuple of two 4D tensors for training with labeled/unlabeled data.

    Output shape:
        4D tensor with shape: `(batch_size, height, width, output_channels)`

    Returns:
        A 4D tensor representing predicted depth maps.

    Raises:
        ValueError: If unsupported encoder type is specified.

    Example:
        >>> model = DepthAnything(
        ...     encoder_type='vit_l',
        ...     input_shape=(384, 384, 3),
        ...     decoder_dims=[256, 128, 64, 32]
        ... )
        >>> x = keras.random.normal([2, 384, 384, 3])
        >>> depth = model(x)
        >>> print(depth.shape)
        (2, 384, 384, 1)
    """

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
        self.supported_encoders = ['vit_s', 'vit_b', 'vit_l']
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
        """Build the model components.

        Args:
            input_shape: Shape of input tensor(s).
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

        # Frozen weight-shared teacher. Order of operations matters:
        #   1) ensure the student encoder is built (so it has weights to copy).
        #   2) clone topology and force-build the clone.
        #   3) copy weights student → teacher and freeze.
        # Wrapped in try/except — if cloning fails on an exotic subclass we
        # disable feature alignment for the run rather than crash the model.
        #
        # DECISION plan-2026-08-17T183311-79c63e38/D-033
        # The teacher is built whenever EITHER flag is set, not just under
        # `use_feature_alignment`. `enable_semi_supervised` needs a teacher for
        # the pseudo-label consistency term, which is not a feature-space term
        # and has nothing to do with feature alignment. When this condition was
        # `if self.use_feature_alignment:` alone, the documented combination
        # `enable_semi_supervised=True, use_feature_alignment=False` — exposed
        # as two INDEPENDENT CLI flags by `src/train/depth_anything/` — built no
        # teacher, so `_train_step_semi_supervised` unpacked `x_unlab`, used it
        # for nothing, and silently degraded to labeled-only training at half
        # the throughput, with `update_teacher_ema` a no-op as well. Do not
        # narrow this back to the FAL knob; `use_feature_alignment` governs the
        # FAL *term* (see `_train_step_semi_supervised`), not the teacher's
        # existence.
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

        Args:
            decay: EMA decay factor in ``[0,1]``. Higher values → slower update.
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

        Args:
            weights_path: Path to a ``.keras`` checkpoint produced by
                ``model.save(...)``. The checkpoint may itself be a
                DepthAnything snapshot or a standalone encoder snapshot — the
                weight-transfer helper matches by layer name.
            skip_prefixes: Layer-name prefixes to ignore during transfer.
                Defaults to ``()`` (transfer everything).

        Returns:
            ``self`` (so calls can chain).
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

        Args:
            trainable: Boolean indicating whether the encoder should be trainable.

        Returns:
            Encoder model instance.
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
        # DECISION plan-2026-08-17T183311-79c63e38/D-028
        # `epsilon` is EXPLICIT on all three BatchNorms of this placeholder
        # encoder, and the justification here is CONSISTENCY, not fidelity:
        # this Conv-BN-ReLU stack is an in-repo stand-in for DINOv2 with no
        # reference implementation, so it has nothing to be faithful to. What it
        # does have is a downstream DPT head normalizing at 1e-5; running the
        # encoder at Keras' 1e-3 put a 100x spread inside one forward pass for
        # no stated reason. Do NOT restate the literal here.
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

        This is the plain encoder → decoder path. It does **not** augment: strong
        augmentation is applied by `train_step` (see `_augment_with_targets`),
        where the depth target is in scope and can be CutMixed by the same box.

        Args:
            inputs: Input tensor with shape (batch_size, height, width, 3)
                or tuple of (labeled, unlabeled) tensors for training.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode.

        Returns:
            Predicted depth maps with shape (batch_size, height, width, output_channels).
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

        Args:
            optimizer: Keras optimizer instance.
            loss: Primary loss function. If None, uses `AffineInvariantLoss` —
                depth is supervised only up to an unknown affine transform, so
                that is the objective this recipe is built around.
            loss_weights: Optional custom loss weights to override defaults.
            **kwargs: Additional arguments passed to parent compile method.
        """
        # Set default loss if none provided
        if loss is None:
            loss = AffineInvariantLoss()

        super().compile(optimizer=optimizer, loss=loss, **kwargs)

        # Update loss weights if provided. Specialized loss instances are NOT
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

        Args:
            x_unlab: Unlabeled input batch ``(B, H, W, C)``.

        Returns:
            Pseudo-depth tensor ``(B, H, W, output_channels)`` with no gradient.
        """
        feat = self.frozen_encoder(x_unlab, training=False)
        feat = self._features_to_spatial(feat)
        pseudo = self.decoder(feat, training=False)
        return ops.stop_gradient(pseudo)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-014
    # Strong augmentation lives here, in the training path, and NOT in `call()`.
    # CutMix pastes a rectangle taken from another batch row; when the call sat
    # inside `call()` no target was in scope, so the image was mixed and the
    # depth map was not — every cut region supervised one scene's pixels against
    # another scene's depth, on ~`cutmix_prob` of all batches. Do not move the
    # augmentation back into `call()` and do not add a second augmentation call
    # anywhere: every consumer of an augmented image must receive it from this
    # method together with its identically-mixed targets.
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

        Args:
            x: Image batch ``(B, H, W, C)``.
            targets: Tensors to mix by the same box as ``x``.

        Returns:
            The augmented images and the identically mixed targets.
        """
        if self.augmentation is None:
            return x, list(targets)
        x_aug, mix = self.augmentation.augment_with_mix(x, training=True)
        return x_aug, [self.augmentation.apply_mix_to_target(t, mix) for t in targets]

    # DECISION plan-2026-08-17T183311-79c63e38/D-033
    # Both custom step methods MUST end here. Two things Keras' default
    # `train_step` does for free are not free once `train_step` is overridden,
    # and both were missing:
    #   1. `self._loss_tracker` is fed by the DEFAULT `train_step`, NOT by
    #      `compute_loss`. Both methods previously ran
    #      `for m in self.metrics: if m.name != "loss": m.update_state(y, y_pred)`
    #      and never fed the tracker anywhere, so `history.history["loss"]` was
    #      the `Mean` metric's reset default `0.0` on every step of every run —
    #      MEASURED `[0.0, 0.0]` across two epochs — and every
    #      `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau` monitoring
    #      `"loss"` was dead. `test_step` is not overridden, which is why
    #      `val_loss` was real and the shipped trainer never surfaced this.
    #   2. `self.metrics` yields the `CompileMetrics` CONTAINER, whose
    #      `.result()` is a dict. Do NOT skip the tracker by name: the tracker
    #      is identified by identity here, as in `capsnet/model.py`, because
    #      `Mean.update_state(y, y_pred)` accepts `(y, y_pred)` as
    #      `(values, sample_weight)` without raising and would silently
    #      accumulate garbage.
    def _finalize_train_step(
        self,
        y: keras.KerasTensor,
        y_pred: keras.KerasTensor,
        loss: keras.KerasTensor,
    ) -> Dict[str, keras.KerasTensor]:
        """Update every metric and return a FLAT logs dict.

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

        Args:
            y: Ground-truth targets for the compiled metrics.
            y_pred: Model predictions for the compiled metrics.
            loss: The scalar total loss that was backpropagated.

        Returns:
            Flat mapping of metric name to scalar result.
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
        # DECISION plan-2026-08-17T183311-79c63e38/D-034
        # `self.optimizer.scale_loss(loss)` MUST be called inside the tape, and
        # the SCALED value is what `tape.gradient` differentiates while the
        # UNSCALED value is what gets reported. Keras' own default TF
        # `train_step` does exactly this (`backend/tensorflow/trainer.py:72`),
        # and overriding `train_step` silently opts out of it.
        # Under `mixed_float16` Keras wraps the optimizer in a
        # `LossScaleOptimizer` whose `apply()` DIVIDES every gradient by
        # `dynamic_scale` (2**15 initially) unconditionally
        # (`optimizers/loss_scale_optimizer.py:172-177`). Skipping `scale_loss`
        # therefore does not merely lose fp16 precision -- it divides the whole
        # update by the loss scale. MEASURED on this exact step shape: total
        # |dW| over 10 steps was 8.74e-05 without the call vs 2.44e+00 with it,
        # a ratio of 2.79e+04 (~32768), against a float32 control of 2.74e+00.
        # Nothing warns; training simply does not move.
        # In float32 this is a provable no-op: the base
        # `Optimizer.scale_loss` returns `loss` unchanged unless
        # `loss_scale_factor` is set (`optimizers/base_optimizer.py:605-614`).
        # Do not "simplify" this back to `tape.gradient(loss, ...)`.
        x, (y,) = self._augment_with_targets(x, [y])
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Keras-3 canonical train_step — replaces deprecated
            # compiled-loss / compiled-metrics calls.
            # See dl_techniques/models/masked_language_model/mlm.py:309-343.
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
            # DECISION plan-2026-08-17T183311-79c63e38/D-033
            # The gate is the TEACHER's existence, not `use_feature_alignment`.
            # Both terms used to sit under `self.use_feature_alignment`, while
            # `train_step` routes here on `self.enable_semi_supervised` alone —
            # so the documented `enable_semi_supervised=True,
            # use_feature_alignment=False` configuration executed this method,
            # skipped the whole block, and was labeled-only training that paid
            # to unpack and discard `x_unlab`. Pseudo-label consistency is an
            # L1 between depth maps; it is not a feature-space term and must
            # not be coupled to the feature-alignment knob. Only the FAL term
            # below reads that knob.
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

        Args:
            data: Training data batch.

        Returns:
            Dictionary containing loss metrics.
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

        Returns:
            Dictionary containing the model configuration.
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

        Args:
            config: Dictionary containing model configuration.

        Returns:
            DepthAnything model instance.
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
    # Save / load delegation for nested sub-Models.
    # ------------------------------------------------------------------
    # DECISION plan_2026-05-10_bd098beb/D-004
    # Keras 3 walks weight paths inside `.keras` archives via attribute
    # tracking on the outer `keras.Model` subclass. When `self.encoder`
    # is itself a Functional/subclassed `keras.Model` (here, ViT), the
    # path mapping for its inner FFN/attention Dense kernels can drift
    # between save and load — 55/172 weights round-trip with
    # re-initialised values (forward diff ≈ 1-2.8). The MLM serialization
    # pattern fixes topology round-trip but not weight-path round-trip.
    # The canonical Keras-3 fix is to override `save_own_variables` /
    # `load_own_variables` and persist the full ordered weight list of
    # each sub-Model into a deterministic keyed slot in the store. This
    # bypasses Keras' path-walking for these sub-Models entirely.
    def save_own_variables(self, store: Any) -> None:  # type: ignore[override]
        """Persist all of DepthAnything's variables in one flat store.

        The default Keras 3 implementation only persists ``self``'s own
        direct variables and lets the framework recurse into children. For
        ViT-as-encoder that recursion has been observed to drop kernel
        arrays during load (see D-004). We instead serialize the full,
        ordered ``self.weights`` list under flat numeric keys at the
        DepthAnything level. ``self.weights`` already includes every
        variable of every nested layer (encoder, frozen_encoder, decoder,
        augmentation), so this is one canonical, path-free record.
        """
        all_vars = list(self.weights)
        for i, v in enumerate(all_vars):
            store[str(i)] = keras.ops.convert_to_numpy(v)

    def load_own_variables(self, store: Any) -> None:  # type: ignore[override]
        """Restore all of DepthAnything's variables from the flat store.

        Mirrors :meth:`save_own_variables` — assigns ``self.weights[i]``
        from ``store[str(i)]`` in deterministic order. If sub-layers
        haven't been built yet (Keras 3 may call ``load_own_variables``
        before recursing into children), force-build by running a
        single dummy forward pass under the saved ``image_shape`` so
        ``self.weights`` matches what was written at save time.
        """
        if not self.built or any(
            sub is not None and not sub.built
            for sub in (self.encoder, self.frozen_encoder, self.decoder)
        ):
            dummy = keras.ops.zeros((1,) + tuple(self.image_shape))
            _ = self(dummy, training=False)

        all_vars = list(self.weights)
        n_store = len(store.keys()) if hasattr(store, "keys") else len(all_vars)
        if n_store != len(all_vars):
            raise ValueError(
                f"DepthAnything.load_own_variables: store has {n_store} "
                f"entries but model has {len(all_vars)} weights."
            )
        for i, v in enumerate(all_vars):
            v.assign(store[str(i)])

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

    **Input contract.** The model does not normalize its inputs, but the strong
    augmentation it applies during training does need to know their range: color
    jitter scales brightness and contrast and its result is clipped back into
    `input_value_range`. The default `(0.0, 1.0)` says "the caller feeds images
    in `[0, 1]`". Pass `input_value_range=None` for standardized (mean-zero) or
    `[-1, +1]` images — the trainer in `src/train/depth_anything/` does, because
    `src/train/common/megadepth.py` emits RGB in `[-1, +1]`, and clipping those
    to `[0, 1]` would flatten every negative pixel to zero on the training path
    only, while evaluation saw the untouched image.

    Augmentation runs inside `train_step`, not inside `call`: CutMix mixes across
    batch rows, so the depth target has to be mixed by the same rectangle, and
    only the training path has the target. Calling `model(x, training=True)`
    directly therefore returns an *un-augmented* forward pass.

    Args:
        encoder_type: String, type of ViT encoder to use.
            Supported values: ['vit_s', 'vit_b', 'vit_l'].
            Defaults to 'vit_l'.
        input_shape: Tuple of integers, input image shape as (height, width, channels).
            Defaults to (384, 384, 3).
        decoder_dims: List of integers, dimensions for decoder layers.
            Defaults to [256, 128, 64, 32].
        output_channels: Integer, number of output channels for depth prediction.
            Defaults to 1.
        kernel_initializer: String or Initializer, initializer for convolutional kernels.
            Defaults to "he_normal".
        kernel_regularizer: Regularizer or None, regularizer for convolutional kernels.
            Defaults to None.
        loss_weights: Dict of strings to floats, weights for different loss components.
            Keys: 'labeled', 'unlabeled', 'feature'.
            Defaults to {'labeled': 1.0, 'unlabeled': 0.5, 'feature': 0.1}.
        cutmix_prob: Float, probability of applying CutMix augmentation.
            Defaults to 0.5.
        color_jitter_strength: Float, strength of color jittering augmentation.
            Defaults to 0.2.
        input_value_range: Tuple of two floats or None, the declared value range
            of the input images (see "Input contract" above).
            Defaults to (0.0, 1.0).
        use_feature_alignment: Boolean, whether to use feature alignment loss.
            Defaults to True.

    Returns:
        Configured and built DepthAnything model instance.

    Raises:
        ValueError: If unsupported encoder type is specified.

    Example:
        >>> model = create_depth_anything(
        ...     encoder_type='vit_l',
        ...     input_shape=(384, 384, 3),
        ...     kernel_regularizer=keras.regularizers.L2(0.01)
        ... )
        >>> model.compile(
        ...     optimizer=keras.optimizers.AdamW(learning_rate=5e-6),
        ...     loss=keras.losses.MeanSquaredError()
        ... )
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
