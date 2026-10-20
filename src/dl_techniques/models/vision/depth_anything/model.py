"""`DepthAnything`, a monocular depth model built from a DINOv2 encoder, a
DPT-style decoder head, and an optional Mean-Teacher branch for
semi-supervised training. Implements the Depth Anything V1 recipe, not V2: no
synthetic training data and no distillation through a large teacher.

The decoder is `DPTHead` (`components.py`), a port of the reference V1 head
(LiheYoung/Depth-Anything `depth_anything/dpt.py`, `use_bn=False`): it reads
the patch tokens of the encoder's last four layers, each through the final
norm, reassembles them at four scales, fuses them coarse to fine with
`refinenet4..1` and predicts at the input resolution. Its widths default to
the published checkpoint of each encoder size. Its output ends in a ReLU, as
both official relative-depth heads do (V1 `depth_anything/dpt.py`, V2
`depth_anything_v2/dpt.py`), so every prediction is non-negative and has no
upper bound.

Depth from a single image is scale-ambiguous, so the model targets depth up
to an unknown affine transform rather than metric depth. The semi-supervised
branch trains the student on a strongly perturbed (color jitter, CutMix) copy
of an image against a target taken from the teacher's prediction on the clean
image, so the two disagree and there is a gradient to learn from. A second
term aligns student features to the teacher to preserve the encoder's
pretrained semantic structure. Augmentation runs inside `train_step`, not
`call`, because CutMix changes the correct target over the mixed region. Only
the unlabeled batch is strongly augmented; the labeled batch trains as loaded
(the paper's horizontal flip happens in the data loader).

The encoder is the reference's: this repository's DINOv2 backbone
(`DINOv2VisionTransformer`, ViT-S/B/L at patch 14, pre-norm, LayerScale,
the official square position table resampled to the input grid), or a
`Conv-BN-ReLU` placeholder stack. The image sides must be multiples of 14,
as in the reference (518 by default). `from_variant(..., pretrained=True)`
loads the official DINOv2 weights (`facebook/dinov2-{small,base,large}`,
local download) into the encoder, strictly; the head starts from its
initialization, as in the paper. `from_pretrained_encoder` loads a saved
`.keras` encoder instead. The teacher starts as a clone
of the student and only becomes an EMA average once the caller attaches
`TeacherEMACallback` (`teacher_ema.py`); nothing in this module advances it
on its own. `compile()` with no `loss=` defaults to `AffineInvariantLoss`,
matching the relative-depth objective; `src/train/depth_anything/` passes
`AffineInvariantLoss(masked=True)`, the same objective over the valid pixels
of sparse ground truth. `use_feature_alignment` defaults to
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
      The reassemble-and-fuse decoder `DPTHead` follows.
    - Ranftl et al., 2020. Towards Robust Monocular Depth Estimation: Mixing
      Datasets for Zero-shot Cross-dataset Transfer (MiDaS).
      (https://arxiv.org/abs/1907.01341)
      Origin of the affine-invariant objective this pipeline is built around.
    - Tarvainen & Valpola, 2017. Mean Teachers Are Better Role Models.
      (https://arxiv.org/abs/1703.01780)
      The EMA teacher that `teacher_ema.py` drives.
    - Oquab et al., 2023. DINOv2: Learning Robust Visual Features without
      Supervision. (https://arxiv.org/abs/2304.07193)
      The paper's encoder; its official weights load from a local download.
"""

import keras
import tensorflow as tf
from typing import Dict, Tuple, Optional, Union, Any, List

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.weight_transfer import load_weights_from_checkpoint
from dl_techniques.layers.signal_processing.strong_augmentation import StrongAugmentation
from dl_techniques.losses.affine_invariant_loss import AffineInvariantLoss
from dl_techniques.losses.feature_alignment_loss import FeatureAlignmentLoss
from dl_techniques.models.vision.dino.dino_v2 import DINOv2VisionTransformer
from dl_techniques.models.vision.dino.official_weights import (
    OFFICIAL_DINOV2_POS_GRID as _OFFICIAL_POS_GRID,
    official_dinov2_weights_path,
    load_official_dinov2_file,
)

from .components import REFERENCE_BN_EPSILON, DPTHead, resize_bilinear_align_corners
from dl_techniques.utils.keras_registration import register_dl_technique

# Map depth_anything encoder_type slugs to DINOv2 variant names.
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
#: The reference encoder's patch size (DINOv2 ViT-*/14); image sides are multiples of it.
_DINOV2_PATCH_SIZE: int = 14

#: The placeholder Conv encoder's stride (a stride-2 conv and three stride-2 pools).
_PLACEHOLDER_STRIDE: int = 16

#: ``DPTHead`` widths ``(features, out_channels)`` per encoder size, from the
#: ``config.json`` of the published checkpoints ``LiheYoung/depth_anything_vit{s,b,l}14``
#: (all three: ``use_bn=False``, ``use_clstoken=False``).
_DPT_HEAD_CONFIGS: Dict[str, Tuple[int, Tuple[int, int, int, int]]] = {
    "vit_s": (64, (48, 96, 192, 384)),
    "vit_b": (128, (96, 192, 384, 768)),
    "vit_l": (256, (256, 512, 1024, 1024)),
}

#: How many encoder layers the head reads (``get_intermediate_layers(x, 4)``).
_DPT_NUM_LAYERS: int = 4

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.depth_anything.model")
class DepthAnything(keras.Model):
    """Monocular depth model: DINOv2 encoder, DPT-style decoder, optional teacher.

    Architecture:

    .. code-block:: text

        Input [B, H, W, 3]
              │
              ▼
        ┌──────────────────────────────┐
        │ encoder: DINOv2 (patch 14)   │────► frozen_encoder (EMA teacher,
        │ or placeholder Conv-BN-ReLU  │       optional, weight-shared clone)
        └──────────────┬───────────────┘
                       │  DINOv2: patch tokens of the last 4 layers, each
                       │  through the final norm, reshaped to [B, h, w, D];
                       │  placeholder: its one [B, h, w, D] map, 4 times
                       ▼
        ┌──────────────────────────────┐
        │ decoder: DPTHead (reference  │
        │ V1 head, no BN): reassemble, │
        │ refinenet4..1, output convs  │
        └──────────────┬───────────────┘
                       ▼  resize to (H, W) if needed, ReLU
        Output [B, H, W, output_channels]  (non-negative)

    Training (semi-supervised, `enable_semi_supervised=True`):

    .. code-block:: text

        labeled batch ──► augment ──► student ──► decoder ──► labeled loss
        unlabeled batch ─┬─► augment(strong) ──► student ──► decoder ──► consistency loss
                         └─► clean ──► frozen_encoder/teacher ──► pseudo-target
        student features ──► FeatureAlignmentLoss ──► frozen_encoder features
                                                        (only if use_feature_alignment)

    :param encoder_type: One of ``"vit_s"``, ``"vit_b"``, ``"vit_l"``
        (DINOv2 ViT-S/14, ViT-B/14, ViT-L/14).
    :type encoder_type: str
    :param image_shape: Input image shape ``(height, width, channels)``. With
        the DINOv2 encoder both sides must be multiples of 14 (the
        reference's 518 by default).
    :type image_shape: Tuple[int, int, int]
    :param dpt_features: Fusion width of the ``DPTHead`` (``features`` in the
        reference). ``None`` takes the published checkpoint's value for
        ``encoder_type`` (64 / 128 / 256).
    :type dpt_features: Optional[int]
    :param dpt_out_channels: The four projection widths of the ``DPTHead``
        (``out_channels`` in the reference). ``None`` takes the published
        checkpoint's values for ``encoder_type``.
    :type dpt_out_channels: Optional[List[int]]
    :param output_channels: Number of channels in the predicted depth map.
    :type output_channels: int
    :param kernel_initializer: Initializer for the encoder's and the
        decoder's kernels.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for the encoder's and the
        decoder's kernels.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param loss_weights: Weight per loss term, keys ``"labeled"``,
        ``"unlabeled"``, ``"feature"``.
    :type loss_weights: Optional[Dict[str, float]]
    :param cutmix_prob: Probability of applying CutMix to an UNLABELED batch
        in the semi-supervised step (the paper's perturbation for unlabeled
        images). Runs inside `train_step`, not `call`, so the teacher's
        pseudo-depth is CutMixed by the same box as the image. The labeled
        batch is never mixed.
    :type cutmix_prob: float
    :param color_jitter_strength: Strength of the color-jitter augmentation
        of an unlabeled batch (the labeled batch is never jittered).
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

    :raises ValueError: If `encoder_type` or `encoder_kind` is not recognized,
        or if a side of `image_shape` is not a multiple of 14 with the DINOv2
        encoder.

    Input shape:
        4D tensor ``(batch_size, height, width, 3)``, or a tuple of two such
        tensors for semi-supervised training with labeled and unlabeled data.

    Output shape:
        4D tensor ``(batch_size, height, width, output_channels)``, every
        value non-negative (the ReLU output).

    Example:
        .. code-block:: python

            model = DepthAnything(
                encoder_type='vit_l',
                image_shape=(518, 518, 3),
            )
            x = keras.random.normal([2, 518, 518, 3])
            depth = model(x)
            print(depth.shape)  # (2, 518, 518, 1)
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
        image_shape: Tuple[int, int, int] = (518, 518, 3),
        dpt_features: Optional[int] = None,
        dpt_out_channels: Optional[List[int]] = None,
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
        include_top: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

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
        default_features, default_out_channels = _DPT_HEAD_CONFIGS[encoder_type]
        self.dpt_features = int(default_features if dpt_features is None else dpt_features)
        self.dpt_out_channels = [int(c) for c in (
            default_out_channels if dpt_out_channels is None else dpt_out_channels)]
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
        self.include_top = include_top

        if self.use_feature_alignment and not self.enable_semi_supervised:
            logger.warning(
                "DepthAnything: use_feature_alignment=True with "
                "enable_semi_supervised=False builds the teacher encoder — "
                "roughly doubling parameter count and memory — while train_step "
                "never routes through the branch that reads it. Set "
                "enable_semi_supervised=True to use it, or "
                "use_feature_alignment=False to stop paying for it."
            )

        # Encoder geometry: the DINOv2 patch grid, or the placeholder's stride-16 map.
        if self.encoder_kind == 'real':
            self.encoder_stride = _DINOV2_PATCH_SIZE
            if self.image_shape[0] % _DINOV2_PATCH_SIZE or self.image_shape[1] % _DINOV2_PATCH_SIZE:
                raise ValueError(
                    f"image_shape {self.image_shape}: with the DINOv2 encoder both "
                    f"sides must be multiples of {_DINOV2_PATCH_SIZE} (the reference "
                    f"uses 518)")
            self.encoder_h = self.image_shape[0] // self.encoder_stride
            self.encoder_w = self.image_shape[1] // self.encoder_stride
        else:
            # Placeholder Conv encoder: initial stride-2 conv => /2, then 3
            # stride-2 'same' maxpools across stages 0..2 (stage 3 has no pool),
            # each rounding up => ceil(side / 16).
            self.encoder_stride = _PLACEHOLDER_STRIDE
            self.encoder_h = -(-self.image_shape[0] // self.encoder_stride)
            self.encoder_w = -(-self.image_shape[1] // self.encoder_stride)

        # If an encoder was supplied (typically by `from_config` after
        # deserialization), accept it directly so its saved topology + weights
        # survive the load. Otherwise build() will create one fresh.
        self.encoder: Optional[keras.Model] = encoder
        # The DINOv2 width; the placeholder projects its features to the same
        # width, so the head sees one input width per encoder_type.
        self.encoder_embed_dim: int = DINOv2VisionTransformer.MODEL_VARIANTS[
            _VIT_SCALE_MAP[self.encoder_type]]['embed_dim']

        # Other components — initialized in build().
        self.decoder: Optional[keras.layers.Layer] = None
        self.frozen_encoder: Optional[keras.Model] = None
        self.augmentation: Optional[keras.layers.Layer] = None

        logger.info(
            f"Initialized DepthAnything (encoder_type={encoder_type}, "
            f"encoder_kind={encoder_kind}, image_shape={self.image_shape}, "
            f"semi_supervised={self.enable_semi_supervised}, include_top={self.include_top})"
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
                # DECISION plan-2026-10-14T062616-81938e81/D-136
                # Do NOT go back to the in-tree ViT (patch 16, post-norm by
                # default, no LayerScale): the reference encoder is DINOv2
                # ViT-*/14, and only this backbone takes the official weights.
                # Its position table is the official square 37 x 37 grid,
                # resampled to this image's patch grid as the reference does.
                self.encoder = DINOv2VisionTransformer.from_variant(
                    _VIT_SCALE_MAP[self.encoder_type],
                    image_size=self.image_shape[:2],
                    patch_size=_DINOV2_PATCH_SIZE,
                    num_register_tokens=0,
                    pos_embed_grid_size=_OFFICIAL_POS_GRID,
                    name=f'encoder_{self.encoder_type}_real',
                )
            else:
                self.encoder = self._create_placeholder_encoder(trainable=True)
        if isinstance(self.encoder, DINOv2VisionTransformer):
            # The iBOT mask token never enters a depth forward (no masks), so
            # it gets no gradient; it stays a loaded, frozen weight.
            self.encoder.mask_token_layer.trainable = False

        # DECISION plan-2026-10-14T062616-81938e81/D-031
        # The output is ReLU'd, as in the reference: both official relative
        # heads (Depth-Anything depth_anything/dpt.py, Depth-Anything-V2
        # depth_anything_v2/dpt.py) end the head in nn.ReLU and apply F.relu
        # after the resize, so a prediction is never negative. Do NOT pass
        # min_depth/max_depth/use_log_depth: no reference maps this output
        # into a [min, max] or log range (only V2's metric head does
        # sigmoid * max_depth, which this relative-depth model does not
        # claim), and DPTHead takes none of them. See decisions.md D-031.
        # Without the head, call() returns before the decoder, so none is
        # created: an unrun, unbuilt decoder is a sub-layer with no weights.
        # DECISION plan-2026-10-14T062616-81938e81/D-128
        # Do NOT bring back a decoder like the shared head's dpt mode (deleted,
        # D-222): it decoded ONE feature map through Conv-BN-ReLU stages, while the
        # reference head (Depth-Anything DPTHead, use_bn=False) fuses four
        # encoder layers through refinenet4..1 and has no BatchNorm. The BN
        # was where the old decoder's train/inference gap sat (step-150 loss
        # 1.06 inference vs 0.39 training, D-127).
        if self.include_top:
            self.decoder = DPTHead(
                features=self.dpt_features,
                out_channels=self.dpt_out_channels,
                patch_size=self.encoder_stride,
                output_channels=self.output_channels,
                kernel_initializer=self.kernel_initializer,
                kernel_regularizer=self.kernel_regularizer,
                name='depth_head',
            )
            self.decoder.build([
                (None, self.encoder_h, self.encoder_w, self.encoder_embed_dim)
            ] * _DPT_NUM_LAYERS)

        # Build order: the student encoder is built; clone its topology, copy
        # weights student -> teacher and freeze.
        # DECISION plan-2026-08-17T183311-79c63e38/D-033: build the teacher when either flag is set, not `use_feature_alignment` alone.
        # `enable_semi_supervised` needs a teacher for pseudo-label consistency too; narrowing this silently degraded to labeled-only training. See decisions.md.
        if self.use_feature_alignment or self.enable_semi_supervised:
            try:
                if isinstance(self.encoder, DINOv2VisionTransformer):
                    # The class rebuilds its own graph from its config. clone_model
                    # would return a plain Functional without get_intermediate_layers,
                    # and under load_model's safe mode it refuses the graph's Lambdas.
                    self.frozen_encoder = DINOv2VisionTransformer.from_config(
                        self.encoder.get_config())
                else:
                    self.frozen_encoder = keras.models.clone_model(self.encoder)
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
        self._sync_teacher()
        return self

    def load_official_encoder_weights(self) -> 'DepthAnything':
        """Load the official DINOv2 weights of ``encoder_type`` into the encoder.

        Interface contract (callers: ``from_variant(pretrained=True)`` and the
        trainer's ``--pretrained-encoder``): strict, as
        `dl_techniques.models.vision.dino.official_weights.load_official_dinov2_file`
        (raises ``FileNotFoundError`` without the local download, ``ValueError``
        when a tensor does not fit). Then re-copies student -> teacher. The
        head keeps its initialization (the paper's randomly initialized decoder).

        :return: ``self``, so calls can chain.
        :rtype: DepthAnything
        :raises ValueError: If the encoder is the placeholder.
        """
        if self.encoder_kind != 'real':
            raise ValueError("official DINOv2 weights need encoder_kind='real'")
        if not self.built:
            self.build((None,) + tuple(self.image_shape))
        load_official_dinov2_file(self.encoder, _VIT_SCALE_MAP[self.encoder_type])
        self._sync_teacher()
        return self

    def _sync_teacher(self) -> None:
        """Copy the student encoder's weights into the frozen teacher, if one exists.

        Raises whatever ``set_weights`` raises: a teacher left at other weights
        than the loaded student would feed the consistency term silently.
        """
        if self.frozen_encoder is not None:
            self.frozen_encoder.set_weights(self.encoder.get_weights())
            logger.info("DepthAnything: teacher re-synced from student.")

    def _encoder_layers(
            self, encoder: keras.Model, x: keras.KerasTensor, training: Optional[bool],
            n: int = _DPT_NUM_LAYERS,
    ) -> List[keras.KerasTensor]:
        """The last ``n`` ``(B, h, w, D)`` encoder maps, from ``encoder``.

        Interface contract (callers: the student path in `call`, the teacher
        in `_pseudo_label_depth`, both with ``n = 4`` for the head, and the
        encoder-feature paths with ``n = 1``): the DINOv2 encoder gives the
        patch tokens of its last ``n`` blocks, each through its final norm
        (``get_intermediate_layers``), reshaped to the patch grid; the
        placeholder Conv encoder has no intermediate layers, so its one
        feature map is returned ``n`` times.

        :param encoder: The student encoder or the frozen teacher.
        :type encoder: keras.Model
        :param x: Images ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :param training: Forwarded to the encoder.
        :type training: Optional[bool]
        :param n: How many of the last layers.
        :type n: int
        :return: ``n`` feature maps, shallowest first.
        :rtype: List[keras.KerasTensor]
        """
        if hasattr(encoder, "get_intermediate_layers"):
            return [
                keras.ops.reshape(t, (-1, self.encoder_h, self.encoder_w, t.shape[-1]))
                for t in encoder.get_intermediate_layers(x, n, training=training)]
        return [encoder(x, training=training)] * n

    def _decode(self, layers: List[keras.KerasTensor]) -> keras.KerasTensor:
        """Run the head, resize to ``image_shape`` if needed, ReLU (the reference's tail).

        :param layers: The four maps from :meth:`_encoder_layers`.
        :type layers: List[keras.KerasTensor]
        :return: Depth ``(B, H, W, output_channels)``, non-negative.
        :rtype: keras.KerasTensor
        """
        depth = self.decoder(layers)
        depth = resize_bilinear_align_corners(depth, self.image_shape[0], self.image_shape[1])
        return keras.ops.relu(depth)

    def _create_placeholder_encoder(self, trainable: bool = True) -> keras.Model:
        """Create the placeholder Conv-BN-ReLU encoder (legacy mode).

        Used when ``encoder_kind='placeholder'``. For ``encoder_kind='real'``
        :meth:`build` constructs the DINOv2 backbone.

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
        # Justification is consistency with the repo's DPTDecoder BatchNorm 1e-5 (REFERENCE_BN_EPSILON; DPTHead has no BN since D-128 of plan-2026-10-14T062616-81938e81), not reference fidelity (this stack has no reference). See decisions.md.
        x = keras.layers.BatchNormalization(
            epsilon=REFERENCE_BN_EPSILON, name='initial_bn'
        )(x)
        x = keras.layers.ReLU(name='initial_relu')(x)
        # Note: removed legacy stride-2 'initial_pool' to keep placeholder
        # encoder stride at 16 (DPTHead reads it as a patch-16 grid).

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

        # Feature projection layer, to the ViT width of this encoder_type
        features = keras.layers.Conv2D(
            filters=self.encoder_embed_dim,
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
    ) -> Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]:
        """Forward pass through the model.

        This is the plain encoder-decoder path. It does not augment: strong
        augmentation is applied by the semi-supervised `train_step` to the
        unlabeled batch (see `_augment_with_targets`), where the teacher's
        pseudo-depth is in scope and can be CutMixed by the same box.

        :param inputs: Input tensor, shape ``(batch_size, height, width, 3)``,
            or a tuple of ``(labeled, unlabeled)`` tensors for training.
        :type inputs: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        :param training: Whether the model runs in training or inference mode.
        :type training: Optional[bool]
        :return: Predicted depth maps, shape ``(batch_size, height, width, output_channels)``,
            or encoder features dict when ``include_top=False``.
        :rtype: Union[keras.KerasTensor, Dict[str, keras.KerasTensor]]
        """
        # Handle both single input and tuple input for training
        if isinstance(inputs, tuple):
            x_labeled, x_unlabeled = inputs
            # For simplicity, process labeled data in forward pass
            # Complex training logic would be handled in train_step
            x = x_labeled
        else:
            x = inputs

        if not self.include_top:
            # Return encoder features for custom heads: the last layer's
            # normalized patch tokens, as a sequence and on the patch grid.
            spatial_features = self._encoder_layers(self.encoder, x, training, n=1)[0]
            features = keras.ops.reshape(
                spatial_features, (-1, self.encoder_h * self.encoder_w, spatial_features.shape[-1]))
            return {
                "encoder_features": features,
                "spatial_features": spatial_features
            }

        return self._decode(self._encoder_layers(self.encoder, x, training))

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
        pseudo = self._decode(self._encoder_layers(self.frozen_encoder, x_unlab, False))
        return keras.ops.stop_gradient(pseudo)

    # DECISION plan-2026-08-14T233721-d4f9beb2/D-014: augmentation stays in the training path, never in `call()`.
    # `call()` has no target in scope, so a `call()`-side CutMix mixed the image but not the depth map on ~cutmix_prob of batches. See decisions.md.
    def _augment_with_targets(
        self,
        x: keras.KerasTensor,
        targets: List[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, List[keras.KerasTensor], Optional[keras.KerasTensor]]:
        """Strongly augment ``x`` and apply the same CutMix box to each target.

        Interface contract (called by the semi-supervised step, for the
        unlabeled batch and its pseudo-depth; the labeled batch is not
        augmented, D-131):

        * Returns ``(x_aug, mixed_targets, box)`` with ``len(mixed_targets) ==
          len(targets)``, each entry shaped like its input. ``box`` is the
          ``(H, W, 1)`` CutMix mask (1 where a pixel came from the donor image,
          all 0 when CutMix was skipped), or ``None`` when nothing was mixed.
        * Every target must share ``x``'s batch and spatial axes; channel counts
          are free (depth alone, or depth + validity mask).
        * When ``self.augmentation`` is ``None`` — tests and callers that opt out
          of strong augmentation set it so — the inputs are returned unchanged,
          with ``box`` ``None``.

        :param x: Image batch ``(B, H, W, C)``.
        :type x: keras.KerasTensor
        :param targets: Tensors to mix by the same box as ``x``.
        :type targets: List[keras.KerasTensor]
        :return: The augmented images, the identically mixed targets and the box.
        :rtype: Tuple[keras.KerasTensor, List[keras.KerasTensor], Optional[keras.KerasTensor]]
        """
        if self.augmentation is None:
            return x, list(targets), None
        x_aug, mix = self.augmentation.augment_with_mix(x, training=True)
        box = None if mix is None else mix[0]
        return x_aug, [self.augmentation.apply_mix_to_target(t, mix) for t in targets], box

    @staticmethod
    def _unlabeled_consistency(
        y_pred: keras.KerasTensor,
        pseudo: keras.KerasTensor,
        box: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """The unlabeled loss of the paper, eq. 6-8 (arXiv 2401.10891, section 3.2).

        The affine-invariant loss ``rho`` (eq. 1-3, ``AffineInvariantLoss``
        with a mask) runs separately over the pixels the student's input kept
        from its own image (``1 - box``) and over the CutMix box (eq. 6, 7),
        each against the identically mixed pseudo-depth, and the two are
        averaged by their pixel shares (eq. 8). Without a mix ``box`` is
        ``None`` (or all 0) and the result is ``rho`` over the whole map.
        Each output channel is scored as its own map.

        :param y_pred: Student depth on the augmented batch ``(B, H, W, C)``.
        :type y_pred: keras.KerasTensor
        :param pseudo: The teacher's pseudo-depth, mixed by ``box``, same shape.
        :type pseudo: keras.KerasTensor
        :param box: ``(H, W, 1)`` CutMix mask, or ``None``.
        :type box: Optional[keras.KerasTensor]
        :return: Scalar loss, the batch mean of eq. 8.
        :rtype: keras.KerasTensor
        """
        # DECISION plan-2026-10-14T062616-81938e81/D-113
        # Do NOT score this term with a plain L1 (the old `mean(|S - T|)`):
        # the paper uses the affine-invariant loss here as for the labels, per
        # CutMix region, and a raw L1 against an unaligned teacher map
        # penalises scale and shift the labeled loss deliberately ignores.
        shape = keras.ops.shape(y_pred)
        channels = shape[3]
        # Each channel becomes its own (H, W, 1) map: (B, H, W, C) -> (B*C, H, W, 1).
        def per_channel(t):
            t = keras.ops.transpose(t, (0, 3, 1, 2))
            return keras.ops.reshape(t, (-1, shape[1], shape[2], 1))
        pred, target = per_channel(y_pred), per_channel(keras.ops.cast(pseudo, y_pred.dtype))
        if box is None:
            donor = keras.ops.zeros_like(pred)
        else:
            donor = keras.ops.broadcast_to(
                keras.ops.cast(box, pred.dtype), keras.ops.shape(pred))
        rho = AffineInvariantLoss(masked=True)
        total = keras.ops.zeros(keras.ops.shape(pred)[:1], dtype=pred.dtype)
        for region in (1.0 - donor, donor):
            share = keras.ops.mean(region, axis=(1, 2, 3))
            mask_and_target = keras.ops.concatenate([target, region], axis=-1)
            total = total + share * rho.call(mask_and_target, pred)
        return keras.ops.mean(keras.ops.reshape(total, (-1, channels)))

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
        """Labeled-only path: forward the batch as given, backprop.

        The labeled batch is not strongly augmented (see the D-131 anchor);
        its only augmentation, a horizontal flip, happens in the data loader.
        Both step methods differentiate ``scaled_loss`` and report ``loss``.
        See the D-034 anchor below for why the two must not be collapsed.
        """
        # DECISION plan-2026-08-17T183311-79c63e38/D-034: differentiate the scaled loss, report the unscaled one.
        # Under mixed_float16 skipping `scale_loss` divides every gradient by the loss scale (measured |dW| ratio ~2.8e4); float32 is unaffected. See decisions.md.
        # DECISION plan-2026-10-14T062616-81938e81/D-131
        # Do NOT CutMix or colour-jitter the labeled batch: the paper applies
        # only horizontal flipping to labeled images (arXiv 2401.10891, 4.1,
        # done by the MegaDepth loader); the strong perturbations are its
        # recipe for unlabeled images (3.2), applied in the semi-supervised step.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Keras-3 canonical train_step — replaces deprecated
            # compiled-loss / compiled-metrics calls.
            # See dl_techniques/models/common/masked_language_model/mlm.py:309-343.
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

        Both unlabeled branches take the teacher's view from the *clean* batch
        and the student's from the *augmented* one — that asymmetry is the whole
        recipe. What is not part of the recipe is CutMix mixing across batch rows
        without the target following: the pseudo-label is therefore mixed by the
        same box as the student's input before the consistency term compares
        them. The labeled batch is not augmented here (D-131, as in
        :meth:`_train_step_labeled`).
        """
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
                x_unlab_aug, (pseudo,), box = self._augment_with_targets(
                    x_unlab, [pseudo]
                )

                if self.use_feature_alignment:
                    # Feature-Alignment-Loss on unlabeled features. The student
                    # sees the augmented batch here too — reading
                    # `self.encoder(x_unlab)` directly would bypass
                    # `self.augmentation` and reduce the term to a
                    # train/eval-mode difference between two initially
                    # identical encoders.
                    # Pool the last layer's patch tokens to (B, D).
                    feat_student = keras.ops.mean(self._encoder_layers(
                        self.encoder, x_unlab_aug, True, n=1)[0], axis=[1, 2])
                    feat_teacher = keras.ops.mean(self._encoder_layers(
                        self.frozen_encoder, x_unlab, False, n=1)[0], axis=[1, 2])
                    fal = FeatureAlignmentLoss()(feat_teacher, feat_student)
                    loss = loss + self.loss_weights.get('feature', 0.1) * fal

                # Pseudo-label consistency: the affine-invariant loss per
                # CutMix region (eq. 6-8) between student depth on the
                # augmented batch and the teacher's identically mixed,
                # stop-gradient pseudo-depth.
                y_pred_unlab = self(x_unlab_aug, training=True)
                consistency = self._unlabeled_consistency(y_pred_unlab, pseudo, box)
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
          pseudo-label consistency term over ``x_unlab`` (the affine-invariant
          loss per CutMix region, :meth:`_unlabeled_consistency`) and, when
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

    @classmethod
    def from_variant(
        cls,
        variant: str,
        pretrained: bool = False,
        **kwargs: Any
    ) -> 'DepthAnything':
        """Build a built Depth Anything of one published encoder size.

        GUIDE section 1 path: with ``pretrained=True`` the official DINOv2
        weights of the variant's encoder are located first (raising
        ``FileNotFoundError`` before anything is built), then the model is
        built and the encoder loaded strictly
        (:meth:`load_official_encoder_weights`). The head keeps its
        initialization: the paper fine-tunes a randomly initialized decoder
        on a pretrained encoder.

        :param variant: ``"vit_s"``, ``"vit_b"`` or ``"vit_l"`` (the
            ``encoder_type``).
        :type variant: str
        :param pretrained: Load the official DINOv2 encoder weights.
        :type pretrained: bool
        :param kwargs: Any other constructor argument.
        :return: The built model.
        :rtype: DepthAnything
        :raises ValueError: If ``variant`` is unknown, ``kwargs`` names an
            ``encoder_type``, or ``pretrained`` is combined with the
            placeholder encoder.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(cls.MODEL_VARIANTS)}")
        if 'encoder_type' in kwargs:
            raise ValueError("pass the encoder size as `variant`, not encoder_type")
        if pretrained:
            if kwargs.get('encoder_kind', 'real') != 'real':
                raise ValueError("pretrained=True loads DINOv2 weights; encoder_kind must be 'real'")
            official_dinov2_weights_path(_VIT_SCALE_MAP[variant])
        model = cls(encoder_type=variant, **kwargs)
        model.build((None,) + tuple(model.image_shape))
        if pretrained:
            model.load_official_encoder_weights()
        return model

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization.

        :return: Dictionary containing the model configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "encoder_type": self.encoder_type,
            "image_shape": self.image_shape,
            "dpt_features": self.dpt_features,
            "dpt_out_channels": list(self.dpt_out_channels),
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
            "include_top": self.include_top,
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
    image_shape: Tuple[int, int, int] = (518, 518, 3),
    dpt_features: Optional[int] = None,
    dpt_out_channels: Optional[List[int]] = None,
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
    include_top: bool = True,
    pretrained: bool = False,
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

    Augmentation runs inside `train_step`, not inside `call`, and only on the
    unlabeled batch of the semi-supervised step: CutMix mixes across batch
    rows, so the teacher's pseudo-depth has to be mixed by the same rectangle,
    and only the training path has it. The labeled batch is not strongly
    augmented (the paper flips it horizontally, in the data loader). Calling
    `model(x, training=True)` directly returns an un-augmented forward pass.

    :param encoder_type: One of ``"vit_s"``, ``"vit_b"``, ``"vit_l"``.
    :type encoder_type: str
    :param image_shape: Input image shape ``(height, width, channels)``;
        sides are multiples of 14 with the DINOv2 encoder.
    :type image_shape: Tuple[int, int, int]
    :param dpt_features: Fusion width of the ``DPTHead``; ``None`` takes the
        published checkpoint's value for ``encoder_type``.
    :type dpt_features: Optional[int]
    :param dpt_out_channels: The head's four projection widths; ``None``
        takes the published checkpoint's values for ``encoder_type``.
    :type dpt_out_channels: Optional[List[int]]
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
    :param include_top: Whether to include the depth estimation head.
        When False, returns encoder features for custom heads. Defaults to True.
    :type include_top: bool
    :param pretrained: Load the official DINOv2 weights of ``encoder_type``
        into the encoder (`DepthAnything.from_variant`).
    :type pretrained: bool
    :return: Configured and built DepthAnything model instance.
    :rtype: DepthAnything
    :raises ValueError: If `encoder_type` is not recognized.

    Example:
        .. code-block:: python

            model = create_depth_anything(
                encoder_type='vit_l',
                image_shape=(518, 518, 3),
                pretrained=True,
            )
            model.compile(
                optimizer=keras.optimizers.AdamW(learning_rate=5e-6),
                loss=keras.losses.MeanSquaredError()
            )
    """
    logger.info(f"Creating DepthAnything model with encoder: {encoder_type}")

    # Create model with specified configuration
    model = DepthAnything.from_variant(
        encoder_type,
        pretrained=pretrained,
        image_shape=image_shape,
        dpt_features=dpt_features,
        dpt_out_channels=dpt_out_channels,
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
        include_top=include_top,
    )

    # Build model with dummy input to initialize all components
    dummy_input = keras.random.normal([1] + list(image_shape))
    _ = model(dummy_input)

    logger.info("Successfully created and built DepthAnything model")

    return model

# ---------------------------------------------------------------------
