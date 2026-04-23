"""CliffordCLIP — CLIP-style contrastive model with Clifford geometric blocks.

A dual-encoder contrastive model where **both** the vision and text towers
are built from isotropic CliffordNet blocks (arXiv:2601.06793v2):

- **Vision tower**: ``Conv2D`` patch stem -> ``BatchNorm`` -> *L*
  :class:`CliffordNetBlock` layers (bidirectional depthwise context) ->
  Clifford-aware projection head.
- **Text tower**: token + positional embedding ->
  ``LayerNorm`` -> reshape to ``(B, 1, seq_len, D)`` -> *L*
  :class:`CausalCliffordNetBlock` layers (causal left-only depthwise
  context, identical to :class:`CliffordNetLM`) -> squeeze ->
  Clifford-aware projection head.

The Clifford-aware projection head -- the core distinction from standard
CLIP -- avoids collapsing the tower output to a single mean-pooled vector.
The default variant (``head_kind="learned_query_residual"``) uses the
canonical CLIP anchor (GAP for vision, last-non-pad token for text) and
adds a LayerScale-gated residual: the geometric product of ``z_det`` (the
anchor) and ``z_ctx`` (a learnable single-query attention pool over the
feature sequence) is computed via :class:`SparseRollingGeometricProduct`
and scaled by a per-channel gamma initialised near zero. Contrastive
gradients therefore flow through both the symmetric (inner) and the
antisymmetric (wedge) parts of the algebra, but only where they
demonstrably reduce the contrastive loss — elsewhere the LayerScale
keeps the Clifford content small and the head behaves like plain CLIP.
Other head variants (``plain``, ``mean_max``, ``learned_query``) are
kept for A/B comparisons; see the :class:`CliffordCLIP` ``head_kind``
docstring.

A learnable temperature (``logit_scale``) scales the cosine-similarity
matrix before the symmetric contrastive cross-entropy loss; the loss
itself is unchanged from CLIP.

.. note::

    The Penguin-VL paper (arXiv:2603.06569) argues *against* contrastive
    pretraining as the optimal initialization for VLM vision encoders,
    preferring LLM-initialised encoders with generative supervision. This
    model deliberately ignores that recommendation: the user asked for a
    CLIP model and we borrow only Penguin-VL's training-schedule ideas
    (cosine LR, 3% warmup ratio, low-to-high resolution curriculum) in
    the training script. The Penguin-VL reconstruction losses
    (amplitude/direction/relation) require a frozen teacher encoder and
    are intentionally left for the training script to plug in.

Architecture (default head_kind="learned_query_residual"):

.. code-block:: text

    Image (B,H,W,3)                         Tokens (B,seq_len)
        │                                         │
        ▼                                         ▼
    Conv2D stem + BN                        Token + Position Embedding
        │                                         │
        ▼                                         ▼
    CliffordNetBlock × L_vis                CausalCliffordNetBlock × L_txt
    (bidirectional)                         (causal, B,1,L,D layout)
        │                                         │
        ▼                                         ▼
    z_det = GAP(x)  (B,D)                   z_det = masked-mean(x)
    z_ctx = LearnedQueryPool(x) (B,D)       z_ctx = LearnedQueryPool(x,mask)
                                            z_anchor = last-non-pad-token(x)
        │                                         │
        ▼                                         ▼
    geo = SparseRollingGeometricProduct(z_det, z_ctx)      [wedge+inner]
        │                                         │
        ▼                                         ▼
    mixed = z_det    + γ_v ⊙ geo           mixed = z_anchor + γ_t ⊙ geo
    (γ init 1e-5: starts ≈ plain, learns to inject Clifford content)
        │                                         │
        ▼                                         ▼
    LayerNorm                               (text_head_norm applied earlier
        │                                    to the full (B,L,D) sequence)
        ▼                                         │
    Dense(embed_dim)                              ▼
        │                                   Dense(embed_dim)
        ▼                                         │
    L2 Normalize                                  ▼
        │                                   L2 Normalize
        ▼                                         │
    image_features (B,D)                          ▼
                                            text_features (B,D)

    Similarity = image_features @ text_features^T * exp(logit_scale)

    The LayerScale-gated residual means the Clifford geometric product
    only contributes where it demonstrably reduces the contrastive loss.
    Where it doesn't help, γ stays near zero and the head behaves like
    plain CLIP — this is the empirical winner in the A/B sweep.

References:
    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra. arXiv:2601.06793v2.

    Radford, A., et al. (2021). Learning Transferable Visual
    Representations from Natural Language Supervision. ICML.
    arXiv:2103.00020.

    Zhang, B., et al. (2026). Penguin-VL: Exploring the Efficiency
    Limits of VLM with LLM-based Vision Encoders. arXiv:2603.06569v2.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import keras
from keras import initializers, ops, regularizers

from dl_techniques.layers.geometric.clifford_block import (
    CausalCliffordNetBlock,
    CliffordNetBlock,
    CliMode,
    CtxMode,
    SparseRollingGeometricProduct,
)
from dl_techniques.utils.logger import logger


def _head_shifts_for(channels: int, requested: Optional[List[int]]) -> List[int]:
    """Return a valid non-empty list of head shifts given the channel size.

    The SparseRollingGeometricProduct filters out shifts ``>= channels``
    and raises if none remain. For the tower head we want a small, robust
    default that works across variants, so we fall back to ``[1]`` when the
    requested set would be filtered empty.
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

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)
_LN_EPS: float = 1e-6


def _linear_drop_path_rates(num_blocks: int, max_rate: float) -> List[float]:
    """Linearly spaced drop-path rates from 0 to ``max_rate``."""
    if num_blocks <= 1:
        return [0.0] * num_blocks
    step = max_rate / (num_blocks - 1)
    return [round(i * step, 6) for i in range(num_blocks)]


@keras.saving.register_keras_serializable()
class _LayerScale1D(keras.layers.Layer):
    """Per-channel learnable scale, init near zero.

    Multiplies an input ``(B, D)`` tensor elementwise by a learnable
    ``(D,)`` gamma. Used by the ``learned_query_residual`` head to gate
    the Clifford geometric product output on top of a plain residual
    path. Gamma starts at ``init_value`` (default 1e-5), so the head
    initially behaves like plain CLIP and gradually injects wedge/inner
    content as training progresses — the same LayerScale pattern GGR
    uses inside the Clifford backbone.
    """

    def __init__(
        self,
        channels: int,
        init_value: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.init_value = init_value

    def build(self, input_shape: Tuple) -> None:
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.channels,),
            initializer=initializers.Constant(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        return x * self.gamma

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"channels": self.channels, "init_value": self.init_value})
        return config


@keras.saving.register_keras_serializable()
class _LearnedQueryPool1D(keras.layers.Layer):
    """Single-query attention pool over a ``(B, N, D)`` sequence.

    A learnable ``(1, D)`` query computes attention scores against each
    position in the sequence and returns the value-weighted mean — the
    canonical "class-token-free" attention pool. An optional
    ``(B, N)`` ``mask`` zeroes out padded positions before softmax so
    the pool ignores pad tokens.

    This is the second pooling view used by the CliffordCLIP
    ``learned_query`` head: the geometric product of (mean, attn-pool)
    gives wedge content that (max, mean) could not capture because the
    attention pool is *content-dependent* — it adapts to where the
    salient features are instead of taking the global maximum.
    """

    def __init__(
        self,
        channels: int,
        kernel_initializer: Any = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape: Tuple) -> None:
        self.query = self.add_weight(
            name="query",
            shape=(self.channels,),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        features: keras.KerasTensor,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        scores = ops.einsum("bnd,d->bn", features, self.query)
        scores = scores / (float(self.channels) ** 0.5)
        if mask is not None:
            mask = ops.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * (-1e9)
        weights = keras.activations.softmax(scores, axis=-1)
        return ops.einsum("bn,bnd->bd", weights, features)

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


# ===========================================================================
# CliffordCLIP
# ===========================================================================


@keras.saving.register_keras_serializable()
class CliffordCLIP(keras.Model):
    """CLIP-style dual-encoder model with Clifford geometric blocks.

    The vision and text towers are both built from Clifford algebra blocks.
    Given a batch of ``(image, text)`` pairs, both encoders produce
    L2-normalized feature vectors in a shared ``embed_dim`` space. The
    symmetric image-text and text-image cross-entropy over the
    temperature-scaled similarity matrix is the contrastive training
    objective.

    :param image_size: Input image resolution (``H == W``). Must be positive
        and divisible by ``vision_patch_size``.
    :param image_channels: Number of input image channels. Defaults to 3.
    :param vision_patch_size: Patch stem stride. Accepts
        ``1``/``2``/``4`` (matching :class:`CliffordNet` stem variants) or
        any positive integer for a generic single-conv stem.
    :param vision_channels: Feature dim ``D_v`` for vision blocks.
    :param vision_depth: Number of :class:`CliffordNetBlock` layers.
    :param vision_shifts: Sparse rolling product shifts for vision blocks.
    :param vision_cli_mode: Clifford components ``"inner"``, ``"wedge"``,
        or ``"full"`` (default).
    :param vision_ctx_mode: Vision context mode ``"diff"`` or ``"abs"``.
    :param vision_use_global_context: Add global GAP context branch to
        vision blocks.
    :param vision_stochastic_depth_rate: Max DropPath rate for vision blocks.
    :param vocab_size: Text vocabulary size (must cover all tiktoken
        ``cl100k_base`` IDs, ≥100277).
    :param context_length: Maximum text sequence length.
    :param text_channels: Feature dim ``D_t`` for text blocks.
    :param text_depth: Number of :class:`CausalCliffordNetBlock` layers.
    :param text_shifts: Sparse rolling product shifts for text blocks.
    :param text_cli_mode: Clifford components for text.
    :param text_ctx_mode: Text context mode.
    :param text_stochastic_depth_rate: Max DropPath rate for text blocks.
    :param embed_dim: Shared projection dimension.
    :param layer_scale_init: Initial LayerScale gamma for all blocks.
    :param dropout_rate: Embedding/head dropout probability.
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
          (GAP for vision, last-token for text) → LayerNorm → Dense
          (embed_dim). Baseline; Clifford blocks are only in the
          backbone.
        - ``"mean_max"`` — Clifford-aware head with two pooling views
          combined via :class:`SparseRollingGeometricProduct`. Vision
          uses (GAP, GMP); text uses (masked-mean, last-token).
        - ``"learned_query"`` — Clifford-aware head where the second
          pooling view is a single learned attention query over the
          sequence of features. Vision ``z_det=GAP``; text
          ``z_det=masked-mean``; both pair with an attention-pooled
          ``z_ctx`` from a learnable ``(1, D)`` query.
        - ``"learned_query_residual"`` *(default)* — Same two pooling
          views as ``learned_query``, but the geometric product output
          is injected as a LayerScale-gated residual on top of the
          canonical CLIP anchor (``z_det`` for vision, ``last_feat``
          for text). LayerScale gamma initialises near zero so the
          head starts behaving like ``plain`` and gradually introduces
          wedge/inner content where it helps. This mirrors the
          :class:`GatedGeometricResidual` pattern used inside the
          Clifford backbone itself and is the empirical winner on
          CC3M-smoke at 12.5 k steps.
    :param use_bias: Whether Dense layers use bias.
    :param kernel_initializer: Kernel initializer for all Dense/projection.
    :param bias_initializer: Bias initializer.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.

    Example:
        .. code-block:: python

            model = CliffordCLIP.from_variant(
                "nano", vocab_size=100352, image_size=96, context_length=64
            )
            images = keras.random.normal((4, 96, 96, 3))
            tokens = keras.random.uniform((4, 64), 0, 100277, dtype="int32")
            out = model({"image": images, "text": tokens})
            # out has keys: image_features, text_features,
            # logits_per_image, logits_per_text, logit_scale
    """

    LAYERNORM_EPSILON: float = _LN_EPS

    # Scaling ladder; both towers share depth/channels so the contrastive
    # temperature update sees balanced gradient magnitudes.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            vision_channels=128,
            vision_depth=12,
            vision_shifts=[1, 2],
            text_channels=128,
            text_depth=12,
            text_shifts=[1, 2],
            embed_dim=256,
            vision_stochastic_depth_rate=0.05,
            text_stochastic_depth_rate=0.05,
        ),
        "mini": dict(
            vision_channels=192,
            vision_depth=12,
            vision_shifts=[1, 2, 4],
            text_channels=192,
            text_depth=12,
            text_shifts=[1, 2, 4],
            embed_dim=384,
            vision_stochastic_depth_rate=0.1,
            text_stochastic_depth_rate=0.1,
        ),
        "small": dict(
            vision_channels=192,
            vision_depth=15,
            vision_shifts=[1, 2, 4],
            text_channels=192,
            text_depth=15,
            text_shifts=[1, 2, 4],
            embed_dim=384,
            vision_stochastic_depth_rate=0.1,
            text_stochastic_depth_rate=0.1,
        ),
        "base": dict(
            vision_channels=256,
            vision_depth=16,
            vision_shifts=[1, 2, 4, 8],
            text_channels=256,
            text_depth=12,
            text_shifts=[1, 2, 4],
            embed_dim=512,
            vision_stochastic_depth_rate=0.15,
            text_stochastic_depth_rate=0.1,
        ),
        "large": dict(
            vision_channels=384,
            vision_depth=20,
            vision_shifts=[1, 2, 4, 8, 16],
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
        vision_channels: int = 192,
        vision_depth: int = 12,
        vision_shifts: Optional[List[int]] = None,
        vision_cli_mode: CliMode = "full",
        vision_ctx_mode: CtxMode = "diff",
        vision_use_global_context: bool = False,
        vision_stochastic_depth_rate: float = 0.1,
        # Text
        vocab_size: int = 100352,
        context_length: int = 77,
        text_channels: int = 192,
        text_depth: int = 12,
        text_shifts: Optional[List[int]] = None,
        text_cli_mode: CliMode = "full",
        text_ctx_mode: CtxMode = "diff",
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

        # --- Validation ---
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
        if vision_channels <= 0 or text_channels <= 0 or embed_dim <= 0:
            raise ValueError("channels and embed_dim must be positive")
        if vision_depth <= 0 or text_depth <= 0:
            raise ValueError("depth must be positive")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if context_length <= 0:
            raise ValueError(
                f"context_length must be positive, got {context_length}"
            )

        # --- Store config ---
        self.image_size = image_size
        self.image_channels = image_channels
        self.vision_patch_size = vision_patch_size
        self.vision_channels = vision_channels
        self.vision_depth = vision_depth
        self.vision_shifts = (
            list(vision_shifts) if vision_shifts is not None else [1, 2]
        )
        self.vision_cli_mode = vision_cli_mode
        self.vision_ctx_mode = vision_ctx_mode
        self.vision_use_global_context = vision_use_global_context
        self.vision_stochastic_depth_rate = vision_stochastic_depth_rate

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.text_channels = text_channels
        self.text_depth = text_depth
        self.text_shifts = (
            list(text_shifts) if text_shifts is not None else [1, 2]
        )
        self.text_cli_mode = text_cli_mode
        self.text_ctx_mode = text_ctx_mode
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

        # --- Build sub-layer groups ---
        self._build_vision_tower()
        self._build_text_tower()
        self._build_projections()

        # Weight placeholder; created in build()
        self.logit_scale = None

        logger.info(
            f"Created CliffordCLIP (image_size={image_size}, "
            f"vision_channels={vision_channels}, vision_depth={vision_depth}, "
            f"text_channels={text_channels}, text_depth={text_depth}, "
            f"embed_dim={embed_dim}, vocab_size={vocab_size}, "
            f"context_length={context_length})"
        )

    # ------------------------------------------------------------------
    # Builders (golden rule: create sub-layers in __init__)
    # ------------------------------------------------------------------

    def _dense_kwargs(self) -> Dict[str, Any]:
        return dict(
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

    def _build_vision_tower(self) -> None:
        """Vision tower: patch stem -> CliffordNetBlocks -> GAP -> LN."""
        _conv_kw: Dict[str, Any] = dict(
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )

        # Stem: mirrors CliffordNet.model._build_stem semantics.
        if self.vision_patch_size == 1:
            self.vision_stem_conv1 = keras.layers.Conv2D(
                filters=self.vision_channels // 2,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False,
                name="vision_stem_conv1",
                **_conv_kw,
            )
            self.vision_stem_bn1 = keras.layers.BatchNormalization(
                name="vision_stem_bn1"
            )
            self.vision_stem_conv2 = keras.layers.Conv2D(
                filters=self.vision_channels,
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
                filters=self.vision_channels,
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
                filters=self.vision_channels // 2,
                kernel_size=3,
                strides=2,
                padding="same",
                use_bias=False,
                name="vision_stem_conv1",
                **_conv_kw,
            )
            self.vision_stem_bn1 = keras.layers.BatchNormalization(
                name="vision_stem_bn1"
            )
            self.vision_stem_conv2 = keras.layers.Conv2D(
                filters=self.vision_channels,
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
                filters=self.vision_channels,
                kernel_size=self.vision_patch_size,
                strides=self.vision_patch_size,
                padding="same",
                use_bias=self.use_bias,
                name="vision_stem_conv",
                **_conv_kw,
            )
            self._vision_stem_kind = "single"

        self.vision_stem_norm = keras.layers.BatchNormalization(
            name="vision_stem_norm"
        )

        # CliffordNet blocks
        vision_drop_rates = _linear_drop_path_rates(
            self.vision_depth, self.vision_stochastic_depth_rate
        )
        _v_block_kw: Dict[str, Any] = dict(
            channels=self.vision_channels,
            shifts=self.vision_shifts,
            cli_mode=self.vision_cli_mode,
            ctx_mode=self.vision_ctx_mode,
            use_global_context=self.vision_use_global_context,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.vision_blocks: List[CliffordNetBlock] = [
            CliffordNetBlock(
                drop_path_rate=vision_drop_rates[i],
                name=f"vision_clifford_block_{i}",
                **_v_block_kw,
            )
            for i in range(self.vision_depth)
        ]

        self.vision_global_pool = keras.layers.GlobalAveragePooling2D(
            name="vision_global_pool"
        )
        self.vision_global_max_pool = (
            keras.layers.GlobalMaxPooling2D(name="vision_global_max_pool")
            if self.head_kind == "mean_max"
            else None
        )
        self.vision_head_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="vision_head_norm"
        )
        # Optional pre-projection dropout, gated on dropout_rate > 0 — mirrors
        # CliffordNet.head_dropout (model.py) so the CLIP vision head has the
        # same regularisation hook as the vanilla classifier head.
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
            embeddings_initializer=_DEFAULT_KERNEL_INIT,
            name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            self.context_length,
            self.text_channels,
            embeddings_initializer=_DEFAULT_KERNEL_INIT,
            name="position_embedding",
        )
        self.text_embed_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="text_embed_norm"
        )
        self.text_embed_dropout = keras.layers.Dropout(
            self.dropout_rate, name="text_embed_dropout"
        )

        text_drop_rates = _linear_drop_path_rates(
            self.text_depth, self.text_stochastic_depth_rate
        )
        _t_block_kw: Dict[str, Any] = dict(
            channels=self.text_channels,
            shifts=self.text_shifts,
            cli_mode=self.text_cli_mode,
            ctx_mode=self.text_ctx_mode,
            use_global_context=False,
            layer_scale_init=self.layer_scale_init,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
        )
        self.text_blocks: List[CausalCliffordNetBlock] = [
            CausalCliffordNetBlock(
                drop_path_rate=text_drop_rates[i],
                name=f"text_clifford_block_{i}",
                **_t_block_kw,
            )
            for i in range(self.text_depth)
        ]

        self.text_head_norm = keras.layers.LayerNormalization(
            epsilon=_LN_EPS, name="text_head_norm"
        )
        # Optional pre-projection dropout on the (B, L, D) sequence — mirrors
        # CliffordNetLM.head_dropout (lm.py) which also drops on the 3D
        # sequence after head_norm and before the output projection.
        self.text_head_dropout = (
            keras.layers.Dropout(self.dropout_rate, name="text_head_dropout")
            if self.dropout_rate > 0.0
            else None
        )

    def _build_projections(self) -> None:
        """Projection heads, one per tower. Shape depends on ``head_kind``.

        - ``plain``: LayerNorm → Dense(embed_dim) on a single pooled view.
        - ``mean_max`` / ``learned_query``: two pooled views are combined
          through a :class:`SparseRollingGeometricProduct` so the projected
          embedding carries explicit bivector (wedge) content. The
          difference between the two Clifford variants is only how the
          second pooled view (``z_ctx``) is produced.
        """
        _dk = self._dense_kwargs()

        # Clifford sub-layers: only materialised when the head uses them.
        self.vision_head_geo = None
        self.text_head_geo = None
        self.vision_query_pool = None
        self.text_query_pool = None

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
                kernel_initializer=self.kernel_initializer,
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
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="text_head_geo",
            )

        if self.head_kind in ("learned_query", "learned_query_residual"):
            # One learnable (1, D) query per tower attending to the
            # (B, N, D) feature sequence. Output: (B, D).
            self.vision_query_pool = _LearnedQueryPool1D(
                channels=self.vision_channels,
                kernel_initializer=self.kernel_initializer,
                name="vision_query_pool",
            )
            self.text_query_pool = _LearnedQueryPool1D(
                channels=self.text_channels,
                kernel_initializer=self.kernel_initializer,
                name="text_query_pool",
            )

        # LayerScale gate for the residual Clifford head.
        self.vision_head_scale = None
        self.text_head_scale = None
        if self.head_kind == "learned_query_residual":
            self.vision_head_scale = _LayerScale1D(
                channels=self.vision_channels,
                init_value=1e-5,
                name="vision_head_scale",
            )
            self.text_head_scale = _LayerScale1D(
                channels=self.text_channels,
                init_value=1e-5,
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
        """
        if self.built:
            return

        self.logit_scale = self.add_weight(
            name="logit_scale",
            shape=(),
            initializer=initializers.Constant(self.logit_scale_init),
            trainable=True,
        )


        # Trigger sub-layer builds via symbolic forward passes so every
        # nested weight is materialised before super().build() marks us
        # as built.
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
        x = self._apply_vision_stem(images, training=training)
        for block in self.vision_blocks:
            x = block(x, training=training)

        # x: (B, H, W, D_v)
        z_det = self.vision_global_pool(x)  # (B, D_v) — canonical CLIP anchor

        if self.head_kind == "plain":
            mixed = z_det
        else:
            if self.head_kind == "mean_max":
                z_ctx = self.vision_global_max_pool(x)  # (B, D_v)
            else:  # learned_query or learned_query_residual
                b, h, w, d = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], ops.shape(x)[3]
                seq = ops.reshape(x, (b, h * w, d))    # (B, H*W, D_v)
                z_ctx = self.vision_query_pool(seq)    # (B, D_v)
            geo = self.vision_head_geo(z_det, z_ctx)   # (B, D_v)
            if self.head_kind == "learned_query_residual":
                mixed = z_det + self.vision_head_scale(geo)
            else:
                mixed = geo

        mixed = self.vision_head_norm(mixed)
        if self.vision_head_dropout is not None:
            mixed = self.vision_head_dropout(mixed, training=training)
        mixed = self.vision_projection(mixed)
        if normalize:
            mixed = mixed / (ops.norm(mixed, axis=-1, keepdims=True) + 1e-8)
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

        # Reshape to 4D (B, 1, L, D) for causal depthwise Conv2D.
        x = ops.expand_dims(x, axis=1)
        for block in self.text_blocks:
            x = block(x, training=training)
        x = ops.squeeze(x, axis=1)             # (B, L, D_t)
        x = self.text_head_norm(x)
        if self.text_head_dropout is not None:
            x = self.text_head_dropout(x, training=training)

        # Pad mask (1 = real token, 0 = pad).
        non_pad = ops.cast(
            ops.not_equal(input_ids, self.pad_token_id), x.dtype
        )                                       # (B, L)

        # Last-non-pad-token index (the canonical CLIP text anchor).
        non_pad_i = ops.cast(
            ops.not_equal(input_ids, self.pad_token_id), "int32"
        )
        lengths_i = ops.sum(non_pad_i, axis=1)
        last_idx = ops.clip(lengths_i - 1, 0, seq_len - 1)
        one_hot = ops.one_hot(last_idx, num_classes=seq_len, dtype=x.dtype)
        last_feat = ops.squeeze(
            ops.matmul(ops.expand_dims(one_hot, axis=1), x), axis=1
        )                                       # (B, D_t)

        if self.head_kind == "plain":
            mixed = last_feat  # canonical CLIP text anchor
        else:
            # For Clifford variants, z_det is the masked mean and the
            # plain-equivalent anchor is last_feat (canonical CLIP). When
            # residual, the anchor is last_feat and the Clifford product
            # is layered on top.
            lengths_f = ops.sum(non_pad, axis=1, keepdims=True)  # (B, 1)
            lengths_f = ops.maximum(lengths_f, 1.0)
            mask_exp = ops.expand_dims(non_pad, axis=-1)         # (B, L, 1)
            z_det = ops.sum(x * mask_exp, axis=1) / lengths_f    # (B, D_t)

            if self.head_kind == "mean_max":
                # z_ctx = last non-pad token.
                z_ctx = last_feat
            else:  # learned_query or learned_query_residual
                z_ctx = self.text_query_pool(x, mask=non_pad)    # (B, D_t)

            geo = self.text_head_geo(z_det, z_ctx)               # (B, D_t)
            if self.head_kind == "learned_query_residual":
                # Start from canonical CLIP anchor and inject Clifford
                # content as a LayerScale-gated residual.
                mixed = last_feat + self.text_head_scale(geo)
            else:
                mixed = geo

        mixed = self.text_projection(mixed)
        if normalize:
            mixed = mixed / (
                ops.norm(mixed, axis=-1, keepdims=True) + 1e-8
            )
        return mixed

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _get_logit_scale(self) -> keras.KerasTensor:
        """Return ``exp(logit_scale)`` clipped to ``logit_scale_max``."""
        # Match OpenCLIP: log-temperature is learned, temperature is clipped.
        scale = ops.exp(self.logit_scale)
        return ops.minimum(scale, self.logit_scale_max)

    def call(
        self,
        inputs: Union[
            Dict[str, keras.KerasTensor],
            Tuple[keras.KerasTensor, keras.KerasTensor],
        ],
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param inputs: Dict with ``"image"`` and ``"text"`` keys, or a
            tuple ``(images, input_ids)``.
        :param training: Whether in training mode.
        :return: Dict with ``image_features``, ``text_features``,
            ``logits_per_image``, ``logits_per_text``, ``logit_scale``.
        """
        if isinstance(inputs, dict):
            images = inputs["image"]
            input_ids = inputs["text"]
        else:
            images, input_ids = inputs[0], inputs[1]

        image_features = self.encode_image(images, training=training)
        text_features = self.encode_text(input_ids, training=training)

        scale = self._get_logit_scale()
        logits_per_image = scale * ops.matmul(
            image_features, ops.transpose(text_features)
        )
        logits_per_text = ops.transpose(logits_per_image)

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
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "image_channels": self.image_channels,
                "vision_patch_size": self.vision_patch_size,
                "vision_channels": self.vision_channels,
                "vision_depth": self.vision_depth,
                "vision_shifts": self.vision_shifts,
                "vision_cli_mode": self.vision_cli_mode,
                "vision_ctx_mode": self.vision_ctx_mode,
                "vision_use_global_context": self.vision_use_global_context,
                "vision_stochastic_depth_rate": (
                    self.vision_stochastic_depth_rate
                ),
                "vocab_size": self.vocab_size,
                "context_length": self.context_length,
                "text_channels": self.text_channels,
                "text_depth": self.text_depth,
                "text_shifts": self.text_shifts,
                "text_cli_mode": self.text_cli_mode,
                "text_ctx_mode": self.text_ctx_mode,
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

        :param variant: One of ``"nano"``, ``"mini"``, ``"small"``, ``"base"``, ``"large"``.
        :param vocab_size: Vocabulary size (use the tokenizer's
            ``vocab_size``).
        :param image_size: Image resolution.
        :param context_length: Maximum text sequence length.
        :param kwargs: Override any default hyperparameter.
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
# Training uses :class:`dl_techniques.losses.CLIPContrastiveLoss` which
# matches this model's output schema (dict with ``logits_per_image`` and
# ``logits_per_text``). Import and configure it directly from the losses
# package; no wrapper is needed here. See ``train/cliffordnet/train_clip.py``
# for a usage example.
