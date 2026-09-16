"""
CLIP-style contrastive dual encoder whose towers are CliffordNet
geometric-algebra blocks instead of attention, with a selectable Clifford-aware
projection head.

Defines :class:`CliffordCLIP`, which maps an image and a caption to the same unit
sphere, supervised only by which pairing in the batch is the true one, as in standard
CLIP. What differs is how each tower mixes information: a CliffordNet block reads
features as multivectors over the channel axis and combines channel pairs at a fixed
shift through the geometric product ``a b = <a, b> + a ^ b``, whose inner part behaves
like a dot-product score and whose wedge part carries the orientation a symmetric
similarity discards. The shift set is small and fixed rather than all-pairs, so cost
is linear in sequence or spatial size. The product mixes channels and not positions,
so spatial and sequential context comes from a depthwise convolution inside each
block, bidirectional in the vision tower and causal in the text tower. The vision
tower is hierarchical, a patch stem then stages linked by ``PatchMerging``; the text
tower is isotropic; both wrap each transform-only block in an external residual. The
contrastive loss is not defined here, and
``dl_techniques.losses.CLIPContrastiveLoss`` matches this model's output schema.

References:
    - Ji, 2026. CliffordNet: All You Need is Geometric Algebra.
      arXiv:2601.06793v2.
    - Radford et al., 2021. Learning Transferable Visual Models From Natural
      Language Supervision. (https://arxiv.org/abs/2103.00020)
    - Zhang et al., 2026. Penguin-VL: Exploring the Efficiency Limits of VLM
      with LLM-based Vision Encoders. arXiv:2603.06569v2.
    - Liu et al., 2021. Swin Transformer: Hierarchical Vision Transformer using
      Shifted Windows. (https://arxiv.org/abs/2103.14030)
    - Touvron et al., 2021. Going deeper with Image Transformers.
      (https://arxiv.org/abs/2103.17239)
    - Huang et al., 2016. Deep Networks with Stochastic Depth.
      (https://arxiv.org/abs/1603.09382)
"""

from __future__ import annotations

import keras
from keras import initializers, ops, regularizers
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------------

from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CliffordNetBlock,
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
)
from dl_techniques.layers.regularization.layer_scale import LayerScale
from dl_techniques.layers.pooling.patch_merging import PatchMerging
from dl_techniques.layers.sequence_pooling import SequencePooling
from dl_techniques.layers.regularization.stochastic_depth import StochasticDepth
from dl_techniques.utils.clip_utils import (
    apply_clifford_head,
    compute_clip_logits,
    last_non_pad_token,
)
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.utils.logger import logger
from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------------

def _head_shifts_for(channels: int, requested: Optional[List[int]]) -> List[int]:
    """Return a valid non-empty list of head shifts given the channel size.

    ``SparseRollingGeometricProduct`` filters out shifts ``>= channels`` and raises
    when none remain, so this falls back to ``[1]`` for a head whose requested set
    would be filtered empty.

    :param channels: Channel count the head's geometric product runs at.
    :param requested: Requested shifts, or ``None`` for ``[1, 2]``.
    :return: The shifts that survive the filter.
    :raises ValueError: If ``channels`` is too small for any shift.
    """
    base = list(requested) if requested else [1, 2]
    kept = [s for s in base if s < channels]
    if not kept:
        kept = [1] if channels > 1 else []
    if not kept:
        raise ValueError(
            f"Cannot build head geometric product: channels={channels} is too "
            "small to support any shift (need channels >= 2)."
        )
    return kept

# ---------------------------------------------------------------------------

# DECISION plan-2026-08-19T163559-499b6f0e/D-072: one shared Initializer instance;
# pass clone_initializer(...) to every consumer, or the two towers come out with 763
# bit-identical weight pairs. See decisions.md.
_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)

# DECISION plan-2026-08-23T091307-9a110062/D-480: BatchNorm momentum is 0.9, the torch
# convention CliffordNet's stem uses; keras_momentum = 1 - torch_momentum, so this is
# not a typo for 0.1. See decisions.md.
_VISION_STEM_BN_MOMENTUM = 0.9
_LN_EPS: float = 1e-6


# ===========================================================================
# CliffordCLIP
# ===========================================================================


# DECISION plan-2026-09-16T085926-41908265/D-001: registration key fixed to
# dl_techniques.models.clifford_clip.model (convention-correct, per models/CLAUDE.md).
# Do NOT assume any .keras archive exists under the old "dl_techniques.models.clip.clifford_clip"
# key -- none does (results/ scan confirmed empty) -- and do not revert this without re-checking
# results/ first. See decisions.md D-001.
@register_dl_technique("dl_techniques.models.clifford_clip.model")
class CliffordCLIP(keras.Model):
    """CLIP-style dual-encoder model with Clifford geometric blocks.

    Both towers produce L2-normalized features in a shared ``embed_dim`` space, and
    ``call`` returns those features together with the temperature-scaled similarity
    matrices. The symmetric image-text and text-image cross-entropy over them is the
    training objective, computed outside this class.

    Architecture:

    .. code-block:: text

        image [B, H, W, C]              text [B, L] int32
                 │                              │
                 ▼                              ▼
        ┌──────────────────────┐   ┌──────────────────────┐
        │ vision tower         │   │ text tower           │
        └──────────────────────┘   └──────────────────────┘
                 │                              │
                 ▼                              ▼
           projection head                projection head
                 │  L2 normalize                │  L2 normalize
                 ▼                              ▼
          image_features [B, E]          text_features [B, E]
                 └───────────► logits ◄──────────────┘
                                 │  * exp(logit_scale), capped
                                 ▼
                logits_per_image, logits_per_text  [B, B]

    logit_scale stays float32 even under a mixed policy.

    Vision tower:

    .. code-block:: text

        image
                 │
                 ▼
        ┌──────────────────────┐
        │ stem  conv, stride p │
        └──────────────────────┘
                 │  [B, H/p, W/p, stage_channels[0]]
                 ▼  + vision_pos_embed (optional)
        ┌──────────────────────┐
        │ stage i blocks       │  depths[i] blocks, external residual
        └──────────────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ patch_merge_i        │  halves H and W, emits 2 * src
        └──────────────────────┘
                 │
                 ▼
        ┌──────────────────────┐
        │ merge_proj_i         │  only when target != 2 * src
        └──────────────────────┘
                 │  repeats at every stage boundary
                 ▼
        last stage [B, H', W', stage_channels[-1]]

    The text tower is isotropic: embeddings, then text_depth causal blocks.

    Projection heads:

    .. code-block:: text

        head_kind               z_det       z_ctx            output
        plain                   -           -                anchor
        mean_max                mean pool   max or last      geo(z_det, z_ctx)
        learned_query           mean pool   attention pool   geo(z_det, z_ctx)
        learned_query_residual  mean pool   attention pool   anchor + scale *
                                                             geo(z_det, z_ctx)

    The anchor is the mean pool for vision and the last non-pad token for text.

    Variants:

    .. code-block:: text

        variant  vision channels      depths      text  t.depth  embed
        nano     128,128,256,256      3,3,3,3     128   12       256
        nano_g   128,128,256,256      3,3,3,3     128   12       256
        mini     192,192,384,384      3,3,3,3     192   12       384
        small    192,192,384,384      4,4,4,3     192   15       384
        base     256,256,512,512      4,4,4,4     256   12       512
        large    384,384,768,768      5,5,5,5     384   16       768

    nano_g is nano with the vision global-context branch switched on.

    :param image_size: Input image resolution (``H == W``). Must be positive
        and divisible by ``vision_patch_size``.
    :param image_channels: Number of input image channels. Defaults to 3.
    :param vision_patch_size: Patch stem stride. Accepts
        ``1``/``2``/``4`` (matching :class:`CliffordNet` stem variants) or
        any positive integer for a generic single-conv stem.
    :param vision_stage_channels: Channel count per vision stage. This and
        ``vision_stage_depths`` are the preferred configuration, and every shipped
        variant uses them; both must be given together.
    :param vision_stage_depths: Number of :class:`CliffordNetBlock` layers per stage.
    :param vision_stage_shifts: Sparse rolling product shifts per stage. ``None``
        broadcasts ``vision_shifts`` (or ``[1, 2]``) to every stage. Each stage's
        largest shift must be below that stage's channel count.
    :param vision_channels: Legacy single-stage channel count, used only when no
        stage lists are given. Defaults to 192 on that path. After construction the
        attribute of this name holds the last stage's channels.
    :param vision_depth: Legacy single-stage block count, used only when no stage
        lists are given. Defaults to 12 on that path. The attribute afterwards holds
        the summed depth.
    :param vision_shifts: Legacy single-stage shifts, also the broadcast source for
        ``vision_stage_shifts=None``.
    :param vision_cli_mode: Clifford components ``"inner"``, ``"wedge"``,
        or ``"full"`` (default).
    :param vision_ctx_mode: Vision context mode ``"diff"`` or ``"abs"``.
    :param vision_use_global_context: Add global GAP context branch to
        vision blocks.
    :param vision_stochastic_depth_rate: Max DropPath rate for vision blocks,
        scheduled linearly across the summed depth of all stages.
    :param vision_positional_encoding: Add a learned 2D positional weight over the
        post-stem map. Off by default, in which case no such weight exists.
    :param vocab_size: Text vocabulary size. Must match the tokenizer used to
        produce ``input_ids``; the shipped trainer defaults to tiktoken
        ``gpt2`` (50257 tokens). Any tokenizer works as long as ``vocab_size``
        covers its maximum token ID.
    :param context_length: Maximum text sequence length.
    :param text_channels: Feature dim ``D_t`` for text blocks.
    :param text_depth: Number of :class:`CausalCliffordNetBlock` layers.
    :param text_shifts: Sparse rolling product shifts for text blocks.
    :param text_cli_mode: Clifford components for text.
    :param text_ctx_mode: Text context mode.
    :param text_use_global_context: Add global GAP context branch to
        text blocks (default ``False`` preserves the original text tower).
    :param text_stochastic_depth_rate: Max DropPath rate for text blocks.
    :param embed_dim: Shared projection dimension.
    :param layer_scale_init: Initial LayerScale gamma for all blocks.
    :param dropout_rate: Dropout probability applied to the text embedding
        and to the optional pre-projection ``head_dropout`` on both towers
        (vision: after ``vision_head_norm`` on ``(B, D_v)``; text: after
        ``text_head_norm`` on ``(B, L, D_t)``). When ``0.0`` the
        head-dropout sublayers are not materialised, matching the gate in
        :class:`CliffordNet` / :class:`CliffordNetLM`.
    :param pad_token_id: ID used as padding in text inputs (the last-token
        gather finds the last position where ``input_ids != pad_token_id``).
        Defaults to 0.
    :param logit_scale_init: Initial value for the learnable log-temperature
        (``exp(logit_scale)`` scales similarities). Defaults to 2.6592, i.e.
        ``exp(2.6592) ≈ 14.3``, matching the CLIP paper.
    :param logit_scale_max: Upper clip on ``exp(logit_scale)`` (matches
        OpenCLIP). Defaults to 100.0.
    :param head_shifts: Channel-shift offsets used by the Clifford-aware
        projection head's :class:`SparseRollingGeometricProduct`. Shifts
        ``>= channels`` are filtered out; if all are filtered, defaults
        to ``[1]``. Defaults to ``[1, 2]``.
    :param head_cli_mode: Clifford components used in the projection head
        (``"inner"``, ``"wedge"``, or ``"full"``). Defaults to ``"full"``
        so both scalar and bivector terms enter the embedding.
    :param head_kind: Which projection head to use. One of:

        - ``"plain"`` — Standard CLIP head: single pooled view
          (mean for vision, last-token for text) → LayerNorm → Dense
          (embed_dim). Clifford blocks stay in the backbone only.
        - ``"mean_max"`` — Clifford-aware head with two pooling views
          combined via :class:`SparseRollingGeometricProduct`. Vision
          uses (mean, max); text uses (masked-mean, last-token).
        - ``"learned_query"`` — Clifford-aware head where the second
          pooling view is a single learned attention query over the
          sequence of features. Vision ``z_det=mean``; text
          ``z_det=masked-mean``; both pair with an attention-pooled
          ``z_ctx`` from a learnable ``(1, D)`` query.
        - ``"learned_query_residual"`` *(default)* — Same two pooling
          views as ``learned_query``, but the geometric product output
          is injected as a LayerScale-gated residual on top of the
          canonical CLIP anchor (``z_det`` for vision, ``last_feat``
          for text). LayerScale gamma starts near zero, so the head
          begins as ``plain`` and introduces wedge/inner content as
          training proceeds, the way
          :class:`GatedGeometricResidual` works inside the backbone.
    :param use_bias: Whether Dense layers use bias.
    :param kernel_initializer: Kernel initializer for all Dense/projection.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    :param **kwargs: Forwarded to :class:`keras.Model`.

    :raises ValueError: If ``image_size`` or ``vision_patch_size`` is non-positive or
        they do not divide, if a channel, depth, embed or vocabulary size is
        non-positive, if the vision stage lists differ in length or one is given
        without the other, if a stage's largest shift is not below its channel count,
        if the post-stem map is smaller than ``2 ** n_stages`` so the final stage
        would be 1x1, or if ``head_cli_mode`` or ``head_kind`` is unknown.

    Output:
        A dict with ``image_features`` and ``text_features`` at ``(B, embed_dim)``,
        ``logits_per_image`` and ``logits_per_text`` at ``(B, B)``, and the scalar
        ``logit_scale``.

    Example:
        .. code-block:: python

            model = CliffordCLIP.from_variant(
                "nano", vocab_size=100352, image_size=96, context_length=64
            )
            images = keras.random.normal((4, 96, 96, 3))
            tokens = keras.random.randint((4, 64), 0, 100277, dtype="int32")
            out = model({"image": images, "text": tokens})
            # out has keys: image_features, text_features,
            # logits_per_image, logits_per_text, logit_scale
    """

    LAYERNORM_EPSILON: float = _LN_EPS

    # Scaling ladder; both towers share depth and channels so the contrastive
    # temperature update sees balanced gradient magnitudes. nano and nano_g match
    # CliffordNet / CliffordNetLM nano (channels=128, depth=12, shifts=[1,2]).
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # Hierarchical vision tower: 4 stages with PatchMerging between them and
        # the channel progression [D, D, 2D, 2D] (D-002). Total vision depth
        # matches the isotropic ladder; the text tower stays isotropic (D-001).
        "nano": dict(
            vision_stage_channels=[128, 128, 256, 256],
            vision_stage_depths=[3, 3, 3, 3],          # sum 12
            vision_stage_shifts=[[1, 2], [1, 2], [1, 2, 4], [1, 2, 4]],
            text_channels=128,
            text_depth=12,
            text_shifts=[1, 2],
            embed_dim=256,
            vision_stochastic_depth_rate=0.05,
            text_stochastic_depth_rate=0.05,
        ),
        # nano with the vision global-context branch (gFFN-G), mirroring
        # CliffordNet.lite_g. The flag currently reaches every stage.
        "nano_g": dict(
            vision_stage_channels=[128, 128, 256, 256],
            vision_stage_depths=[3, 3, 3, 3],
            vision_stage_shifts=[[1, 2], [1, 2], [1, 2, 4], [1, 2, 4]],
            vision_use_global_context=True,
            text_channels=128,
            text_depth=12,
            text_shifts=[1, 2],
            embed_dim=256,
            vision_stochastic_depth_rate=0.05,
            text_stochastic_depth_rate=0.05,
        ),
        "mini": dict(
            vision_stage_channels=[192, 192, 384, 384],
            vision_stage_depths=[3, 3, 3, 3],          # sum 12
            vision_stage_shifts=[[1, 2, 4], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8]],
            text_channels=192,
            text_depth=12,
            text_shifts=[1, 2, 4],
            embed_dim=384,
            vision_stochastic_depth_rate=0.1,
            text_stochastic_depth_rate=0.1,
        ),
        "small": dict(
            vision_stage_channels=[192, 192, 384, 384],
            vision_stage_depths=[4, 4, 4, 3],          # sum 15
            vision_stage_shifts=[[1, 2, 4], [1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8]],
            text_channels=192,
            text_depth=15,
            text_shifts=[1, 2, 4],
            embed_dim=384,
            vision_stochastic_depth_rate=0.1,
            text_stochastic_depth_rate=0.1,
        ),
        "base": dict(
            vision_stage_channels=[256, 256, 512, 512],
            vision_stage_depths=[4, 4, 4, 4],          # sum 16
            vision_stage_shifts=[[1, 2, 4], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
            text_channels=256,
            text_depth=12,
            text_shifts=[1, 2, 4],
            embed_dim=512,
            vision_stochastic_depth_rate=0.15,
            text_stochastic_depth_rate=0.1,
        ),
        "large": dict(
            vision_stage_channels=[384, 384, 768, 768],
            vision_stage_depths=[5, 5, 5, 5],          # sum 20
            vision_stage_shifts=[[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8, 16], [1, 2, 4, 8, 16]],
            text_channels=384,
            text_depth=16,
            text_shifts=[1, 2, 4, 8],
            embed_dim=768,
            vision_stochastic_depth_rate=0.2,
            text_stochastic_depth_rate=0.15,
        ),
    }

    def __init__(
        self,
        # Vision
        image_size: int = 224,
        image_channels: int = 3,
        vision_patch_size: int = 4,
        vision_channels: Optional[int] = None,
        vision_depth: Optional[int] = None,
        vision_shifts: Optional[List[int]] = None,
        # Hierarchical vision config, preferred over the scalar fields above.
        vision_stage_channels: Optional[List[int]] = None,
        vision_stage_depths: Optional[List[int]] = None,
        vision_stage_shifts: Optional[List[List[int]]] = None,
        vision_cli_mode: CliMode = "full",
        vision_ctx_mode: CtxMode = "diff",
        vision_use_global_context: bool = False,
        vision_stochastic_depth_rate: float = 0.1,
        vision_positional_encoding: bool = False,
        # Text
        vocab_size: int = 100352,
        context_length: int = 77,
        text_channels: int = 192,
        text_depth: int = 12,
        text_shifts: Optional[List[int]] = None,
        text_cli_mode: CliMode = "full",
        text_ctx_mode: CtxMode = "diff",
        text_use_global_context: bool = False,
        text_stochastic_depth_rate: float = 0.1,
        # Shared
        embed_dim: int = 384,
        layer_scale_init: float = 1e-5,
        dropout_rate: float = 0.1,
        pad_token_id: int = 0,
        logit_scale_init: float = 2.6592,
        logit_scale_max: float = 100.0,
        # Clifford-aware projection head
        head_shifts: Optional[List[int]] = None,
        head_cli_mode: CliMode = "full",
        head_kind: str = "learned_query_residual",
        use_bias: bool = True,
        kernel_initializer: Any = _DEFAULT_KERNEL_INIT,
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        if vision_patch_size <= 0:
            raise ValueError(
                f"vision_patch_size must be positive, got {vision_patch_size}"
            )
        if image_size % vision_patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"vision_patch_size ({vision_patch_size})"
            )
        if text_channels <= 0 or embed_dim <= 0:
            raise ValueError("channels and embed_dim must be positive")
        if text_depth <= 0:
            raise ValueError("depth must be positive")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {context_length}"
            )

        if vision_stage_channels is not None or vision_stage_depths is not None:
            if (
                vision_stage_channels is None
                or vision_stage_depths is None
            ):
                raise ValueError(
                    "vision_stage_channels and vision_stage_depths must both "
                    "be provided when using the staged vision config."
                )
            stage_channels = list(vision_stage_channels)
            stage_depths = list(vision_stage_depths)
            if vision_stage_shifts is None:
                # Broadcast scalar `vision_shifts` (or the default) to all stages.
                base_shifts = (
                    list(vision_shifts) if vision_shifts is not None else [1, 2]
                )
                stage_shifts = [list(base_shifts) for _ in stage_channels]
            else:
                stage_shifts = [list(s) for s in vision_stage_shifts]
        else:
            # Legacy single-stage path: no PatchMerging, scalar fields only.
            scalar_channels = vision_channels if vision_channels is not None else 192
            scalar_depth = vision_depth if vision_depth is not None else 12
            scalar_shifts = (
                list(vision_shifts) if vision_shifts is not None else [1, 2]
            )
            stage_channels = [scalar_channels]
            stage_depths = [scalar_depth]
            stage_shifts = [scalar_shifts]

        n_stages = len(stage_channels)
        if not (len(stage_depths) == n_stages == len(stage_shifts)):
            raise ValueError(
                f"vision stage lists must have equal length; got "
                f"channels={len(stage_channels)}, depths={len(stage_depths)}, "
                f"shifts={len(stage_shifts)}"
            )
        if n_stages == 0:
            raise ValueError("Need at least one vision stage")
        for i, c in enumerate(stage_channels):
            if c <= 0:
                raise ValueError(f"vision_stage_channels[{i}] must be positive, got {c}")
        for i, d in enumerate(stage_depths):
            if d <= 0:
                raise ValueError(f"vision_stage_depths[{i}] must be positive, got {d}")
        for i, sh in enumerate(stage_shifts):
            if not sh:
                raise ValueError(f"vision_stage_shifts[{i}] is empty")
            if max(sh) >= stage_channels[i]:
                raise ValueError(
                    f"vision_stage_shifts[{i}] has shift >= channels "
                    f"({max(sh)} >= {stage_channels[i]})"
                )
        # DECISION plan-2026-07-15T114613-5add9baa/D-001: require post_stem >= 2^n_stages
        # so the final stage keeps a 2x2 map; at 1x1 the attention pool is a
        # softmax over one element and its gradient is dead. See decisions.md.
        post_stem = image_size // vision_patch_size
        if post_stem < (1 << n_stages):
            raise ValueError(
                f"Hierarchical vision tower requires post-stem spatial dim "
                f"({post_stem}) >= 2^n_stages = {1 << n_stages} so the final stage "
                f"keeps a >=2x2 map (a 1x1 map makes the attention pool a "
                f"zero-gradient no-op); got image_size={image_size}, "
                f"patch_size={vision_patch_size}, n_stages={n_stages}"
            )

        self.image_size = image_size
        self.image_channels = image_channels
        self.vision_patch_size = vision_patch_size
        self.vision_stage_channels = stage_channels
        self.vision_stage_depths = stage_depths
        self.vision_stage_shifts = stage_shifts
        # Derived scalars for downstream sizing and legacy introspection.
        self.vision_channels = stage_channels[-1]
        self.vision_depth = sum(stage_depths)
        self.vision_shifts = list(stage_shifts[0])
        # Stem produces stage-0 channels.
        self._vision_stem_channels = stage_channels[0]
        self.vision_cli_mode = vision_cli_mode
        self.vision_ctx_mode = vision_ctx_mode
        self.vision_use_global_context = vision_use_global_context
        self.vision_stochastic_depth_rate = vision_stochastic_depth_rate
        self.vision_positional_encoding = vision_positional_encoding
        # Materialised in build() only when the flag is on, so off means no weight.
        self.vision_pos_embed = None

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.text_channels = text_channels
        self.text_depth = text_depth
        self.text_shifts = (
            list(text_shifts) if text_shifts is not None else [1, 2]
        )
        self.text_cli_mode = text_cli_mode
        self.text_ctx_mode = text_ctx_mode
        self.text_use_global_context = text_use_global_context
        self.text_stochastic_depth_rate = text_stochastic_depth_rate

        self.embed_dim = embed_dim
        self.layer_scale_init = layer_scale_init
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id
        self.logit_scale_init = logit_scale_init
        self.logit_scale_max = logit_scale_max
        self.head_shifts = (
            list(head_shifts) if head_shifts is not None else [1, 2]
        )
        if head_cli_mode not in ("inner", "wedge", "full"):
            raise ValueError(
                f"head_cli_mode must be 'inner', 'wedge', or 'full', got "
                f"{head_cli_mode!r}"
            )
        self.head_cli_mode = head_cli_mode
        valid_kinds = (
            "plain", "mean_max", "learned_query", "learned_query_residual",
        )
        if head_kind not in valid_kinds:
            raise ValueError(
                f"head_kind must be one of {valid_kinds}, got {head_kind!r}"
            )
        self.head_kind = head_kind
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self._build_vision_tower()
        self._build_text_tower()
        self._build_projections()

        # Weight placeholder; created in build().
        self.logit_scale = None

        logger.info(
            f"Created CliffordCLIP (image_size={image_size}, "
            f"vision_stage_channels={self.vision_stage_channels}, "
            f"vision_stage_depths={self.vision_stage_depths}, "
            f"text_channels={text_channels}, text_depth={text_depth}, "
            f"embed_dim={embed_dim}, vocab_size={vocab_size}, "
            f"context_length={context_length})"
        )

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _dense_kwargs(self) -> Dict[str, Any]:
        """Return the shared Dense arguments, with a fresh kernel initializer."""
        return dict(
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

    def _build_vision_tower(self) -> None:
        """Create the patch stem, the staged blocks, the merges and the head norm.

        Each stage holds ``vision_stage_depths[i]`` shape-preserving
        :class:`CliffordNetBlock` layers at ``vision_stage_channels[i]`` channels
        and ``vision_stage_shifts[i]`` shifts. Between adjacent stages a
        :class:`PatchMerging` halves the spatial resolution and emits
        ``2 * src_channels``, and a Dense projects to the next stage's channel count
        when that differs.
        """
        _conv_kw: Dict[str, Any] = dict(
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        stem_channels = self._vision_stem_channels

        # Stem: mirrors CliffordNet.model._build_stem semantics.
        if self.vision_patch_size == 1:
            self.vision_stem_conv1 = keras.layers.Conv2D(
                filters=stem_channels // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="vision_stem_conv1",
                **_conv_kw,
            )
            self.vision_stem_bn1 = keras.layers.BatchNormalization(
                name="vision_stem_bn1", momentum=_VISION_STEM_BN_MOMENTUM
            )
            self.vision_stem_conv2 = keras.layers.Conv2D(
                filters=stem_channels,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="vision_stem_conv2",
                **_conv_kw,
            )
            self._vision_stem_kind = "two_stage"
        elif self.vision_patch_size == 2:
            self.vision_stem_conv = keras.layers.Conv2D(
                filters=stem_channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=self.use_bias,
                name="vision_stem_conv",
                **_conv_kw,
            )
            self._vision_stem_kind = "single"
        elif self.vision_patch_size == 4:
            self.vision_stem_conv1 = keras.layers.Conv2D(
                filters=stem_channels // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="vision_stem_conv1",
                **_conv_kw,
            )
            self.vision_stem_bn1 = keras.layers.BatchNormalization(
                name="vision_stem_bn1", momentum=_VISION_STEM_BN_MOMENTUM
            )
            self.vision_stem_conv2 = keras.layers.Conv2D(
                filters=stem_channels,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="vision_stem_conv2",
                **_conv_kw,
            )
            self._vision_stem_kind = "two_stage"
        else:
            self.vision_stem_conv = keras.layers.Conv2D(
                filters=stem_channels,
                kernel_size=self.vision_patch_size,
                strides=self.vision_patch_size,
                padding="same",
                use_bias=self.use_bias,
                name="vision_stem_conv",
                **_conv_kw,
            )
            self._vision_stem_kind = "single"

        self.vision_stem_norm = keras.layers.BatchNormalization(
            name="vision_stem_norm", momentum=_VISION_STEM_BN_MOMENTUM
        )

        # One linear DropPath schedule across the total depth, not per stage.
        total_depth = sum(self.vision_stage_depths)
        global_drop_rates = linear_drop_path_rates(
            total_depth, self.vision_stochastic_depth_rate
        )

        self.vision_blocks: List[CliffordNetBlock] = []
        # Index in `vision_blocks` of each stage's first block.
        self._vision_stage_offsets: List[int] = []
        block_idx = 0
        for stage_idx, (stage_c, stage_d, stage_sh) in enumerate(
            zip(
                self.vision_stage_channels,
                self.vision_stage_depths,
                self.vision_stage_shifts,
            )
        ):
            self._vision_stage_offsets.append(block_idx)
            block_kw: Dict[str, Any] = dict(
                channels=stage_c,
                shifts=stage_sh,
                cli_mode=self.vision_cli_mode,
                ctx_mode=self.vision_ctx_mode,
                use_global_context=self.vision_use_global_context,
                layer_scale_init=self.layer_scale_init,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
            )
            for _ in range(stage_d):
                self.vision_blocks.append(
                    CliffordNetBlock(
                        name=f"vision_clifford_block_{block_idx}",
                        **block_kw,
                    )
                )
                block_idx += 1

        # The blocks are transform-only, so the residual lives here: one
        # StochasticDepth per flat vision block.
        self.vision_drop_paths: List[StochasticDepth] = [
            StochasticDepth(
                drop_path_rate=global_drop_rates[i],
                name=f"vision_drop_path_{i}",
            )
            for i in range(total_depth)
        ]

        # PatchMerging always emits 2 * src, so a Dense follows only where the
        # progression is not pure doubling.
        self.vision_merge_layers: List[PatchMerging] = []
        self.vision_merge_projections: List[Optional[keras.layers.Dense]] = []
        for i in range(len(self.vision_stage_channels) - 1):
            src_c = self.vision_stage_channels[i]
            dst_c = self.vision_stage_channels[i + 1]
            self.vision_merge_layers.append(
                PatchMerging(
                    dim=src_c,
                    use_bias=self.use_bias,
                    kernel_initializer=clone_initializer(self.kernel_initializer),
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    name=f"vision_patch_merge_{i}",
                )
            )
            if dst_c != 2 * src_c:
                self.vision_merge_projections.append(
                    keras.layers.Dense(
                        dst_c,
                        use_bias=self.use_bias,
                        kernel_initializer=clone_initializer(self.kernel_initializer),
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        name=f"vision_merge_proj_{i}",
                    )
                )
            else:
                self.vision_merge_projections.append(None)

        self.vision_head_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="vision_head_norm"
        )
        # Gated on dropout_rate > 0, matching CliffordNet.head_dropout.
        self.vision_head_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="vision_head_dropout")
            if self.dropout_rate > 0.0
            else None
        )

    def _build_text_tower(self) -> None:
        """Text tower: embeddings -> CausalCliffordNetBlocks -> LN."""
        self.token_embedding = keras.layers.Embedding(
            self.vocab_size,
            self.text_channels,
            embeddings_initializer=clone_initializer(_DEFAULT_KERNEL_INIT),
            name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            self.context_length,
            self.text_channels,
            embeddings_initializer=clone_initializer(_DEFAULT_KERNEL_INIT),
            name="position_embedding",
        )
        self.text_embed_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="text_embed_norm"
        )
        self.text_embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="text_embed_dropout"
        )

        text_drop_rates = linear_drop_path_rates(
            self.text_depth, self.text_stochastic_depth_rate
        )
        _t_block_kw: Dict[str, Any] = dict(
            channels=self.text_channels,
            shifts=self.text_shifts,
            cli_mode=self.text_cli_mode,
            ctx_mode=self.text_ctx_mode,
            use_global_context=self.text_use_global_context,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.text_blocks: List[CausalCliffordNetBlock] = [
            CausalCliffordNetBlock(
                name=f"text_clifford_block_{i}",
                **_t_block_kw,
            )
            for i in range(self.text_depth)
        ]
        # External residual + drop_path per text block (transform-only blocks).
        self.text_drop_paths: List[StochasticDepth] = [
            StochasticDepth(
                drop_path_rate=text_drop_rates[i],
                name=f"text_drop_path_{i}",
            )
            for i in range(self.text_depth)
        ]

        self.text_head_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="text_head_norm"
        )
        # Drops on the (B, L, D) sequence, matching CliffordNetLM.head_dropout.
        self.text_head_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="text_head_dropout")
            if self.dropout_rate > 0.0
            else None
        )

    def _build_projections(self) -> None:
        """Create the pooling views, geometric products and projections per tower.

        ``plain`` projects a single pooled view. The Clifford variants combine two
        pooled views through a :class:`SparseRollingGeometricProduct` so the
        embedding carries explicit bivector content, and differ only in how the
        second view ``z_ctx`` is produced.
        """
        _dk = self._dense_kwargs()

        # Clifford sub-layers: only materialised when the head uses them.
        self.vision_head_geo = None
        self.text_head_geo = None

        # DECISION plan-2026-07-15T140843-168d5bac/D-001: pool through generic
        # SequencePooling with attention_hidden_dim=channels, since its own default
        # is 256; do not re-wire the global pooling layers. See decisions.md.
        self.vision_det_pool = SequencePooling(
            strategy="mean", name="vision_det_pool"
        )
        if self.head_kind == "mean_max":
            self.vision_ctx_pool = SequencePooling(
                strategy="max", name="vision_ctx_pool"
            )
        elif self.head_kind in ("learned_query", "learned_query_residual"):
            self.vision_ctx_pool = SequencePooling(
                strategy="attention",
                attention_hidden_dim=self.vision_channels,
                attention_num_heads=1,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                name="vision_ctx_pool",
            )
        # plain: the vision anchor is z_det, so there is no context pool.
        else:
            self.vision_ctx_pool = None

        if self.head_kind == "plain":
            # Text plain uses last_non_pad_token only; no pooled views.
            self.text_det_pool = None
            self.text_ctx_pool = None
        else:
            self.text_det_pool = SequencePooling(
                strategy="mean", name="text_det_pool"
            )
            if self.head_kind in ("learned_query", "learned_query_residual"):
                self.text_ctx_pool = SequencePooling(
                    strategy="attention",
                    attention_hidden_dim=self.text_channels,
                    attention_num_heads=1,
                    kernel_initializer=clone_initializer(self.kernel_initializer),
                    name="text_ctx_pool",
                )
            # mean_max: text z_ctx is last_feat, so there is no context pool.
            else:
                self.text_ctx_pool = None

        if self.head_kind != "plain":
            v_head_shifts = _head_shifts_for(
                self.vision_channels, self.head_shifts
            )
            t_head_shifts = _head_shifts_for(
                self.text_channels, self.head_shifts
            )
            self.vision_head_geo = SparseRollingGeometricProduct(
                channels=self.vision_channels,
                shifts=v_head_shifts,
                cli_mode=self.head_cli_mode,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="vision_head_geo",
            )
            self.text_head_geo = SparseRollingGeometricProduct(
                channels=self.text_channels,
                shifts=t_head_shifts,
                cli_mode=self.head_cli_mode,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="text_head_geo",
            )

        # LayerScale gate for the residual Clifford head.
        self.vision_head_scale = None
        self.text_head_scale = None
        if self.head_kind == "learned_query_residual":
            self.vision_head_scale = LayerScale(
                multiplier_type="CHANNEL",
                initializer=keras.initializers.Constant(1e-5),
                constraint=None,
                name="vision_head_scale",
            )
            self.text_head_scale = LayerScale(
                multiplier_type="CHANNEL",
                initializer=keras.initializers.Constant(1e-5),
                constraint=None,
                name="text_head_scale",
            )

        self.vision_projection = keras.layers.Dense(
            self.embed_dim, name="vision_projection", **_dk
        )
        self.text_projection = keras.layers.Dense(
            self.embed_dim, name="text_projection", **_dk
        )

    # ------------------------------------------------------------------
    # build() — create the learnable temperature and run a symbolic pass
    # ------------------------------------------------------------------

    def build(
        self,
        input_shape: Union[
            Dict[str, Tuple[Optional[int], ...]],
            Tuple[Optional[int], ...],
        ],
    ) -> None:
        """Create weights and build sub-layers.

        Accepts either a dict with ``image`` and ``text`` keys, or a tuple
        shape for a single modality. When only one shape is provided, the
        other tower is built from the configured defaults so the model
        remains fully serializable.

        :param input_shape: Dict of per-modality shapes, or a single shape.
        """
        if self.built:
            return

        # DECISION plan_2026-05-31_76981d58/D-001: pin logit_scale to float32; under a
        # bf16 policy the learned temperature drifts the contrastive logits.
        # See decisions.md.
        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=initializers.Constant(self.logit_scale_init),
            trainable=True,
            dtype="float32",
        )

        # Created before the symbolic forward below, so the gated add in
        # _apply_vision_body finds it materialised and a reload keeps it.
        if self.vision_positional_encoding:
            post_stem = self.image_size // self.vision_patch_size
            self.vision_pos_embed = self.add_weight(
                name="vision_pos_embed",
                shape=(1, post_stem, post_stem, self._vision_stem_channels),
                initializer=initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
            )

        # Symbolic passes materialise every nested weight before super().build().
        image_shape = (
            None,
            self.image_size,
            self.image_size,
            self.image_channels,
        )
        text_shape = (None, self.context_length)
        if isinstance(input_shape, dict):
            image_shape = input_shape.get("image", image_shape)
            text_shape = input_shape.get("text", text_shape)

        img_dummy = keras.KerasTensor(image_shape, dtype="float32")
        txt_dummy = keras.KerasTensor(text_shape, dtype="int32")

        _ = self.encode_image(img_dummy)
        _ = self.encode_text(txt_dummy)

        super().build(input_shape)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _apply_vision_stem(
        self,
        images: keras.KerasTensor,
        training: Optional[bool],
    ) -> keras.KerasTensor:
        """Patch stem with optional two-stage conv + BN + SiLU."""
        if self._vision_stem_kind == "two_stage":
            x = keras.activations.silu(
                self.vision_stem_bn1(
                    self.vision_stem_conv1(images), training=training
                )
            )
            x = self.vision_stem_conv2(x)
        else:
            x = self.vision_stem_conv(images)
        return self.vision_stem_norm(x, training=training)

    def _apply_vision_body(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Apply stem + staged vision body, returning the last-stage map.

        Walks ``vision_blocks`` in order and inserts a
        :class:`PatchMerging` (and optional Dense projection) at the
        boundary between each pair of stages.

        :param images: Image tensor ``(B, H, W, C)``.
        :param training: Whether in training mode.
        :return: ``(B, H', W', vision_stage_channels[-1])``.
        """
        x = self._apply_vision_stem(images, training=training)
        # DECISION plan-2026-07-15T114613-5add9baa/D-004: positional encoding is
        # opt-in and off by default, which keeps old checkpoints loadable; the
        # geometric product mixes channels, not space. See decisions.md.
        if self.vision_positional_encoding:
            x = x + self.vision_pos_embed
        n_stages = len(self.vision_stage_channels)
        # Stage end indices in the flat `vision_blocks` list.
        stage_ends: List[int] = []
        running = 0
        for d in self.vision_stage_depths:
            running += d
            stage_ends.append(running)
        next_boundary_stage = 0
        for i, block in enumerate(self.vision_blocks):
            x = x + self.vision_drop_paths[i](
                block(x, training=training), training=training
            )
            if (
                next_boundary_stage < n_stages - 1
                and i + 1 == stage_ends[next_boundary_stage]
            ):
                merge = self.vision_merge_layers[next_boundary_stage]
                proj = self.vision_merge_projections[next_boundary_stage]
                x = merge(x, training=training)
                if proj is not None:
                    x = proj(x)
                next_boundary_stage += 1
        return x

    def encode_image(
        self,
        images: keras.KerasTensor,
        training: Optional[bool] = None,
        normalize: bool = True,
    ) -> keras.KerasTensor:
        """Encode a batch of images to the shared embedding space.

        :param images: Image tensor ``(B, H, W, C)``.
        :param training: Whether in training mode.
        :param normalize: L2-normalize the output. Set ``False`` to retrieve
            raw projected features (useful for auxiliary losses).
        :return: Embedding tensor ``(B, embed_dim)``.
        """
        x = self._apply_vision_body(images, training=training)

        # Flattened to a token sequence so the generic SequencePooling layers
        # handle every pooling view; z_det, the mean, is the CLIP anchor.
        b, h, w, d = (
            ops.shape(x)[0],
            ops.shape(x)[1],
            ops.shape(x)[2],
            ops.shape(x)[3],
        )
        seq = ops.reshape(x, (b, h * w, d))
        z_det = self.vision_det_pool(seq)
        z_ctx = (
            self.vision_ctx_pool(seq, training=training)
            if self.vision_ctx_pool is not None
            else None
        )
        mixed = apply_clifford_head(
            self.head_kind,
            anchor=z_det,
            z_det=z_det,
            z_ctx=z_ctx,
            geo_layer=self.vision_head_geo,
            scale_layer=self.vision_head_scale,
        )

        mixed = self.vision_head_norm(mixed)
        if self.vision_head_dropout is not None:
            mixed = self.vision_head_dropout(mixed, training=training)
        mixed = self.vision_projection(mixed)
        if normalize:
            # The 1e-8 epsilon underflows to 0.0 in fp16, so normalize in float32
            # and cast back; at float32 the casts are identities.
            mixed_f = ops.cast(mixed, "float32")
            mixed_f = mixed_f / (ops.norm(mixed_f, axis=-1, keepdims=True) + 1e-8)
            mixed = ops.cast(mixed_f, mixed.dtype)
        return mixed

    def encode_text(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
        normalize: bool = True,
    ) -> keras.KerasTensor:
        """Encode a batch of tokenized text to the shared embedding space.

        The text tower uses causal depthwise convolutions, so position *i*
        only sees positions ``<= i``. The final embedding is extracted from
        the last non-pad token (``input_ids != pad_token_id``).

        :param input_ids: Token ID tensor ``(B, context_length)``.
        :param training: Whether in training mode.
        :param normalize: L2-normalize the output.
        :return: Embedding tensor ``(B, embed_dim)``.
        """
        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(seq_len)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.text_embed_norm(x)
        x = self.text_embed_dropout(x, training=training)

        # `x` stays (B, L, D_t) — see `layers/geometric/clifford_block.py`.
        for block, drop_path in zip(self.text_blocks, self.text_drop_paths):
            x = x + drop_path(block(x, training=training), training=training)
        x = self.text_head_norm(x)
        if self.text_head_dropout is not None:
            x = self.text_head_dropout(x, training=training)

        # Pad mask (1 = real token, 0 = pad).
        non_pad = ops.cast(
            ops.not_equal(input_ids, self.pad_token_id), x.dtype
        )

        # Last-non-pad-token index (the canonical CLIP text anchor).
        last_feat = last_non_pad_token(
            x, input_ids, self.pad_token_id
        )

        # For the Clifford variants z_det is the masked mean, and z_ctx is either
        # last_feat (mean_max) or the attention pool over the masked sequence.
        anchor = last_feat
        if self.head_kind == "plain":
            mixed = anchor
        else:
            z_det = self.text_det_pool(x, mask=non_pad)
            if self.head_kind == "mean_max":
                z_ctx = last_feat
            # learned_query or learned_query_residual.
            else:
                z_ctx = self.text_ctx_pool(
                    x, mask=non_pad, training=training
                )
            mixed = apply_clifford_head(
                self.head_kind,
                anchor=anchor,
                z_det=z_det,
                z_ctx=z_ctx,
                geo_layer=self.text_head_geo,
                scale_layer=self.text_head_scale,
            )

        mixed = self.text_projection(mixed)
        if normalize:
            # Same fp16-safe normalize as encode_image.
            mixed_f = ops.cast(mixed, "float32")
            mixed_f = mixed_f / (
                ops.norm(mixed_f, axis=-1, keepdims=True) + 1e-8
            )
            mixed = ops.cast(mixed_f, mixed.dtype)
        return mixed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _get_logit_scale(self) -> keras.KerasTensor:
        """Return ``exp(logit_scale)`` clipped to ``logit_scale_max``."""
        # DECISION plan-2026-07-15T114613-5add9baa/D-001: exponentiate in float32, as
        # OpenCLIP does; in fp16 exp() overflows past log(65504). See decisions.md.
        ls = ops.cast(self.logit_scale, "float32")
        scale = ops.exp(ls)
        return ops.minimum(scale, ops.cast(self.logit_scale_max, "float32"))

    def call(
        self,
        inputs: Union[
            Dict[str, keras.KerasTensor],
            Tuple[keras.KerasTensor, keras.KerasTensor],
        ],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Encode both modalities and return features with their similarity matrices.

        :param inputs: Dict with ``"image"`` and ``"text"`` keys, or a
            tuple ``(images, input_ids)``.
        :param training: Whether in training mode.
        :return: Dict with ``image_features``, ``text_features``,
            ``logits_per_image``, ``logits_per_text``, and the float32
            ``logit_scale``.
        """
        if isinstance(inputs, dict):
            images = inputs["image"]
            input_ids = inputs["text"]
        else:
            images, input_ids = inputs[0], inputs[1]

        image_features = self.encode_image(images, training=training)
        text_features = self.encode_text(input_ids, training=training)

        scale = self._get_logit_scale()
        # Cast to the features' dtype, or an fp16 matmul raises on mixed dtypes.
        scale_c = ops.cast(scale, image_features.dtype)
        logits_per_image, logits_per_text = compute_clip_logits(
            image_features, text_features, scale_c
        )

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "logit_scale": scale,
        }

    # ------------------------------------------------------------------
    # Shape inference
    # ------------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Union[
            Dict[str, Tuple[Optional[int], ...]],
            Tuple[Optional[int], ...],
        ],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        """Compute the shapes of all five outputs.

        :param input_shape: Dict of per-modality shapes, or ``(image, text)``.
        :return: Dict of output name to shape; the logits are ``(B, B)`` and
            ``logit_scale`` is scalar.
        """
        if isinstance(input_shape, dict):
            image_shape = input_shape.get("image")
            text_shape = input_shape.get("text")
        else:
            image_shape, text_shape = input_shape[0], input_shape[1]
        batch = image_shape[0] if image_shape is not None else text_shape[0]
        return {
            "image_features": (batch, self.embed_dim),
            "text_features": (batch, self.embed_dim),
            "logits_per_image": (batch, batch),
            "logits_per_text": (batch, batch),
            "logit_scale": (),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the constructor arguments, with the staged vision fields resolved.

        :return: Configuration dictionary.
        """
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "image_channels": self.image_channels,
                "vision_patch_size": self.vision_patch_size,
                # Staged vision config (preferred over scalar legacy fields).
                "vision_stage_channels": self.vision_stage_channels,
                "vision_stage_depths": self.vision_stage_depths,
                "vision_stage_shifts": self.vision_stage_shifts,
                "vision_cli_mode": self.vision_cli_mode,
                "vision_ctx_mode": self.vision_ctx_mode,
                "vision_use_global_context": self.vision_use_global_context,
                "vision_stochastic_depth_rate": (
                    self.vision_stochastic_depth_rate
                ),
                "vision_positional_encoding": self.vision_positional_encoding,
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "text_channels": self.text_channels,
                "text_depth": self.text_depth,
                "text_shifts": self.text_shifts,
                "text_cli_mode": self.text_cli_mode,
                "text_ctx_mode": self.text_ctx_mode,
                "text_use_global_context": self.text_use_global_context,
                "text_stochastic_depth_rate": self.text_stochastic_depth_rate,
                "embed_dim": self.embed_dim,
                "layer_scale_init": self.layer_scale_init,
                "dropout_rate": self.dropout_rate,
                "pad_token_id": self.pad_token_id,
                "logit_scale_init": self.logit_scale_init,
                "logit_scale_max": self.logit_scale_max,
                "head_shifts": self.head_shifts,
                "head_cli_mode": self.head_cli_mode,
                "head_kind": self.head_kind,
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordCLIP":
        """Rebuild a model, deserializing the two regularizers.

        :param config: Dict as returned by :meth:`get_config`.
        :return: A new model instance.
        """
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        image_size: int = 224,
        context_length: int = 77,
        **kwargs: Any,
    ) -> "CliffordCLIP":
        """Construct a CliffordCLIP from a predefined variant.

        :param variant: One of ``"nano"``, ``"nano_g"``, ``"mini"``, ``"small"``, ``"base"``, ``"large"``.
            ``nano_g`` is ``nano`` with a global-context branch on the vision
            tower (analogue of :class:`CliffordNet.lite_g`).
        :param vocab_size: Vocabulary size (use the tokenizer's
            ``vocab_size``).
        :param image_size: Image resolution.
        :param context_length: Maximum text sequence length.
        :param **kwargs: Override any default hyperparameter.
        :return: A configured model.
        :rtype: CliffordCLIP
        :raises ValueError: If ``variant`` is unknown, or a resolved argument fails
            the constructor's checks.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordCLIP-{variant.upper()}")
        return cls(
            vocab_size=vocab_size,
            image_size=image_size,
            context_length=context_length,
            **defaults,
        )


# ===========================================================================
# Contrastive loss
# ===========================================================================
#
# Training uses dl_techniques.losses.CLIPContrastiveLoss, which matches this
# model's output schema; see train/cliffordnet/train_clip.py for a usage example.