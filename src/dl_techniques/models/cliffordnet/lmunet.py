"""CliffordNet causal U-Net language model.

A hierarchical U-Net variant of :class:`CliffordNetLM` that processes
token sequences through an encoder/bottleneck/decoder stack of
:class:`CausalCliffordNetBlock` and :class:`CausalCliffordNetBlockDSv2`
layers. The decoder up-path uses ``UpSampling2D(nearest)`` along the
sequence (W) axis followed by ``Concatenate`` + 1x1 ``Conv2D`` skip
fusion. End-to-end causality is preserved: every component along
the encoder/decoder/skip paths only mixes information from positions
``<= k`` for any output position ``k``.

Architecture:

.. code-block:: text

    Input IDs (B, seq_len)
         │
         ▼
    Token + Positional Embedding
    ─► LayerNorm ─► Dropout
         │
         ▼
    Reshape to (B, 1, seq_len, base_channels)
         │
         ▼
    [right-pad along W to multiple of total_stride]
         │
         ▼
    Encoder:
      for i in 0..num_levels-1:
        CausalCliffordNetBlock × blocks_per_stage[i]   (channels = base * 2**i)
        store skip[i]
        if i < num_levels-1:
          CausalCliffordNetBlockDSv2(strides=stride_per_stage[i],
                                      out_channels=base * 2**(i+1))
    Bottleneck:
      CausalCliffordNetBlock × bottleneck_blocks  (channels = base * 2**(num_levels-1))
    Decoder (reverse i):
      UpSampling2D(size=(1, stride_per_stage[i]), nearest)
      Concatenate([upsampled, skip[i]], axis=-1)
      Conv2D(filters=base * 2**i, kernel_size=1)        # 3*C_i -> C_i
      CausalCliffordNetBlock × blocks_per_stage[i]
         │
         ▼
    [crop along W back to seq_len]
         │
         ▼
    Squeeze H ─► LayerNorm ─► Dropout ─► tied/untied LM head
         │
         ▼
    Logits (B, seq_len, vocab_size)

Outputs (Matryoshka Representation Learning + auxiliary embedding head)
----------------------------------------------------------------------
The model returns a flat-keyed dict. With default ``mrl_widths=None`` the
dict reduces to ``{"logits"}`` (back-compat with the original single-head
behavior). When ``mrl_widths`` is supplied (or resolved via a variant
default), the dict becomes:

- ``"logits": (B, T, V)`` — the primary head at the largest width
  (``base_channels``). Always present, name unchanged.
- ``f"logits_w{w}": (B, T, V)`` — one extra logits tensor per smaller
  width in ``mrl_widths[1:]``. All ``logits_*`` keys share the same labels
  at training time; the trainer weights them via Keras's ``loss_weights``
  dict.
- ``f"embedding_w{w}": (B, w)`` — L2-normalized sentence embedding per
  width (only when an embedding configuration is enabled; default is to
  emit a single ``embedding_w{base_channels}`` only when ``mrl_widths``
  is None — see below). Side outputs; never participate in loss.

Width sequence — "Power-of-2 anchored, base preserved" (plan D-002):
The largest width is always ``base_channels`` as-is (even when it is
not a power of 2). Every subsequent width is a strict power of 2
strictly less than ``base_channels``, descending, terminating at the
floor (default 16). Example: nano (base=128) → ``[128, 64, 32, 16]``;
mini (base=192) → ``[192, 128, 64, 32, 16]``; base (base=384) →
``[384, 256, 128, 64, 32, 16]``.

Causality of the MRL heads is structural: slicing the last axis of
``h_top`` and projecting per-position preserves the causal order along
W at every width.

Causality rationale
-------------------
- Embeddings are per-token (no spatial mixing).
- :class:`CausalCliffordNetBlock` uses left-only padded depthwise
  convolutions along W (causal).
- :class:`CausalCliffordNetBlockDSv2` is causality-preserving along W
  (LESSONS L33, plan_2026-05-06_13a2df9e D-001).
- ``UpSampling2D(size=(1, s), interpolation="nearest")`` repeats each
  cell ``s`` times — no future information is mixed (D-001 in
  plan_2026-05-06_82749628 ``decisions.md``).
- ``Concatenate(axis=-1)`` followed by 1x1 ``Conv2D`` is per-position;
  it does not look across W.
- Right-padding along W with zeros + cropping at the end does not
  pollute real positions because every block is causal in W (D-002).

References:
    Brandstetter, J., et al. (2025). CliffordNet: All You Need is
    Geometric Algebra. arXiv:2601.06793v2.
"""

import keras
from keras import initializers, regularizers
from typing import Any, Dict, List, Literal, Optional, Tuple

from dl_techniques.utils.logger import logger
from dl_techniques.utils.drop_path import linear_drop_path_rates
from dl_techniques.layers.geometric.clifford_block import (
    CliMode,
    CtxMode,
    CausalCliffordNetBlock,
    CausalCliffordNetBlockDSv2,
)

# ---------------------------------------------------------------------------

_DEFAULT_KERNEL_INIT = initializers.TruncatedNormal(stddev=0.02)


def _causal_upsample(x: keras.KerasTensor, stride: int) -> keras.KerasTensor:
    """Causally right-shift an upsampled feature by ``stride - 1`` along W.

    # DECISION D-007: causal upsample via right-shift.
    # Plain ``UpSampling2D(size=(1, s), interpolation="nearest")`` is NOT
    # causal at fine resolution. Pooled cell ``j`` was computed by DSv2 from
    # input positions ``[j*s, j*s+s-1]``. Nearest-upsample maps output
    # position ``k`` to pooled cell ``k // s``, whose max-input-seen is
    # ``(k//s)*s + s - 1``. For ``k % s != s - 1`` that exceeds ``k`` — a
    # future leak. Empirically observed: nano (stride [2,2], total_stride=4)
    # leaked positions 4-6 when only position 7 was perturbed (D-006).
    #
    # Right-shifting the upsampled feature by ``s - 1`` along W (left-zero-
    # pad ``s - 1``, drop ``s - 1`` from the right) maps output ``k`` to
    # pooled cell ``(k - (s - 1)) // s`` whose max-input-seen is
    # ``((k - (s - 1)) // s) * s + s - 1 <= k``. Strictly causal for all k.
    # See plans/plan_2026-05-06_82749628/decisions.md (D-007).
    """
    if stride <= 1:
        return x
    shift = stride - 1
    w = keras.ops.shape(x)[2]
    x = keras.ops.pad(x, [[0, 0], [0, 0], [shift, 0], [0, 0]])
    return x[:, :, :w, :]


def _default_mrl_widths(base_channels: int, floor: int = 16) -> List[int]:
    """Compute the default MRL width sequence for ``base_channels``.

    Rule (plan_2026-05-12_13c70aed/D-002): "Power-of-2 anchored, base
    preserved". The largest width is ``base_channels`` preserved as-is
    (may be non-power-of-2). Every subsequent width is a strict power
    of 2 strictly less than ``base_channels``, descending, terminating
    at the floor (default 16).

    :param base_channels: Top-of-U channel count (model's embedding dim).
    :param floor: Smallest allowed power-of-2 width. Default 16.
    :return: List of widths in strictly decreasing order, first element
        equals ``base_channels``.

    Examples:
        >>> _default_mrl_widths(128)
        [128, 64, 32, 16]
        >>> _default_mrl_widths(192)
        [192, 128, 64, 32, 16]
        >>> _default_mrl_widths(384)
        [384, 256, 128, 64, 32, 16]
        >>> _default_mrl_widths(768)
        [768, 512, 256, 128, 64, 32, 16]
    """
    if base_channels < floor:
        return [base_channels]
    # Largest power of 2 strictly less than base_channels.
    p = 1
    while p * 2 < base_channels:
        p *= 2
    # If base_channels is itself a power of 2, p == base_channels // 2.
    widths: List[int] = [base_channels]
    while p >= floor:
        widths.append(p)
        p //= 2
    return widths


@keras.saving.register_keras_serializable()
class CliffordNetLMUNet(keras.Model):
    """CliffordNet causal U-Net language model.

    Hierarchical encoder / bottleneck / decoder stack of causal Clifford
    blocks for autoregressive sequence modeling. Channels grow with depth
    following ``channels_per_stage[i] = round(base_channels * (channel_multiplier ** i))``.
    The default ``channel_multiplier=1.5`` keeps the deepest stages
    tractable; pass ``2.0`` to recover a standard doubling U-Net.

    :param vocab_size: Vocabulary size (including special tokens).
    :param max_seq_length: Maximum sequence length for positional embeddings.
    :param base_channels: Channel count at level 0 (top of U).
        Equals the embedding dimensionality.
    :param channel_multiplier: Per-stage channel growth factor. Stage ``i``
        has ``round(base_channels * (channel_multiplier ** i))`` channels.
        Default ``1.5``; use ``2.0`` for classical U-Net doubling.
    :param stride_per_stage: List of strides applied between consecutive
        encoder levels. ``num_levels = len(stride_per_stage) + 1``. The
        product is the model's ``total_stride``.
    :param blocks_per_stage: List of length ``num_levels``: number of
        :class:`CausalCliffordNetBlock` layers per encoder level (and the
        symmetrical decoder level).
    :param bottleneck_blocks: Number of :class:`CausalCliffordNetBlock`
        layers at the deepest level.
    :param shifts: Channel-shift offsets for sparse rolling product. Must
        satisfy ``max(shifts) < base_channels`` (the smallest channel count).
    :param cli_mode: Algebraic components (``"inner"``, ``"wedge"``, ``"full"``).
    :param ctx_mode: Context calculation mode (``"diff"`` or ``"abs"``).
    :param use_global_context: Add causal cumulative-mean context branch.
    :param layer_scale_init: Initial LayerScale gamma value.
    :param stochastic_depth_rate: Maximum DropPath rate (linear schedule
        across all blocks: encoder + bottleneck + decoder).
    :param dropout_rate: Embedding and pre-output dropout rate.
    :param tie_word_embeddings: If True, the LM head reuses the (transposed)
        token embedding matrix instead of an independent Dense projection.
    :param use_bias: Whether Dense/projection layers use bias.
    :param kernel_initializer: Kernel initializer for all dense layers.
    :param bias_initializer: Bias initializer for all dense layers.
    :param kernel_regularizer: Optional kernel regularizer.
    :param bias_regularizer: Optional bias regularizer.
    :param mrl_widths: Matryoshka representation widths. ``None`` →
        ``[base_channels]`` (single-head back-compat). When supplied,
        ``mrl_widths[0]`` MUST equal ``base_channels`` (preserved as-is,
        may be non-power-of-2); every subsequent entry MUST be a strict
        power of 2 strictly less than ``base_channels``, descending,
        floor 16 (plan D-002 "Power-of-2 anchored, base preserved").
    :param mrl_head_norm: If True, apply a per-width ``LayerNorm`` to
        the sliced ``(B, T, w)`` tensor before projection. Default True.
    :param emb_head: If True, add a learnable ``Dense(base_channels,
        use_bias=False)`` projection on the pooled vector before L2
        normalization. Default False (identity).
    :param embedding_pool: Pooling rule for the sentence embedding.
        ``"last"`` (default) → array index ``T-1``; ``"cls"`` →
        position 0 (requires ``cls_token_id``); ``"auto"`` → per-sample
        choose position 0 if ``input_ids[:, 0] == cls_token_id`` else
        position ``T-1`` (requires ``cls_token_id``).
    :param cls_token_id: Token ID used to detect a leading CLS token
        for ``embedding_pool in {"cls","auto"}``.
    :param l2_eps: Epsilon under the sqrt for L2 normalization of the
        per-width embeddings. Default ``1e-12``.

    Example:
        .. code-block:: python

            model = CliffordNetLMUNet.from_variant("nano", vocab_size=50261)
            input_ids = keras.random.uniform((2, 511), 0, 50261, dtype="int32")
            outputs = model(input_ids)
            print(outputs["logits"].shape)  # (2, 511, 50261)
    """

    LAYERNORM_EPSILON: float = 1e-6

    # Pre-defined variant configurations for NLP.
    # Channel ladder uses ``channel_multiplier=1.5`` (default) with
    # base_channels matching the lm.py variants for fair comparison.
    # See plan_2026-05-06_82749628 plan.md "Channel ladder" for derivation.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        "nano": dict(
            base_channels=128,
            stride_per_stage=[2, 2],
            blocks_per_stage=[2, 2, 2],
            bottleneck_blocks=2,
            shifts=[1, 2],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.05,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            # MRL widths — D-002 "Power-of-2 anchored, base preserved".
            mrl_widths=[128, 64, 32, 16],
        ),
        "mini": dict(
            base_channels=192,
            stride_per_stage=[2, 2],
            blocks_per_stage=[2, 3, 3],
            bottleneck_blocks=2,
            shifts=[1, 2, 4],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.1,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            mrl_widths=[192, 128, 64, 32, 16],
        ),
        "base": dict(
            base_channels=384,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[3, 3, 3, 3],
            bottleneck_blocks=3,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.15,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            mrl_widths=[384, 256, 128, 64, 32, 16],
        ),
        "large": dict(
            base_channels=512,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[3, 4, 4, 4],
            bottleneck_blocks=4,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.2,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            mrl_widths=[512, 256, 128, 64, 32, 16],
        ),
        "xl": dict(
            base_channels=768,
            stride_per_stage=[2, 2, 2],
            blocks_per_stage=[4, 4, 4, 4],
            bottleneck_blocks=4,
            shifts=[1, 2, 4, 8, 16],
            cli_mode="full",
            ctx_mode="diff",
            use_global_context=False,
            layer_scale_init=1e-5,
            stochastic_depth_rate=0.25,
            kernel_initializer=_DEFAULT_KERNEL_INIT,
            mrl_widths=[768, 512, 256, 128, 64, 32, 16],
        ),
    }

    def __init__(
        self,
        vocab_size: int,
        max_seq_length: int = 512,
        base_channels: int = 64,
        channel_multiplier: float = 1.5,
        stride_per_stage: Optional[List[int]] = None,
        blocks_per_stage: Optional[List[int]] = None,
        bottleneck_blocks: int = 2,
        shifts: Optional[List[int]] = None,
        cli_mode: CliMode = "full",
        ctx_mode: CtxMode = "diff",
        use_global_context: bool = False,
        layer_scale_init: float = 1e-5,
        stochastic_depth_rate: float = 0.1,
        dropout_rate: float = 0.0,
        tie_word_embeddings: bool = True,
        use_bias: bool = True,
        kernel_initializer: Any = "glorot_uniform",
        bias_initializer: Any = "zeros",
        kernel_regularizer: Optional[Any] = None,
        bias_regularizer: Optional[Any] = None,
        mrl_widths: Optional[List[int]] = None,
        mrl_head_norm: bool = True,
        emb_head: bool = False,
        embedding_pool: Literal["last", "cls", "auto"] = "last",
        cls_token_id: Optional[int] = None,
        l2_eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # --- Defaults / validation -------------------------------------
        if stride_per_stage is None:
            stride_per_stage = [2, 2]
        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2]
        if shifts is None:
            shifts = [1, 2]

        num_levels = len(stride_per_stage) + 1
        if len(blocks_per_stage) != num_levels:
            raise ValueError(
                f"blocks_per_stage must have length len(stride_per_stage)+1 "
                f"(={num_levels}); got len(blocks_per_stage)="
                f"{len(blocks_per_stage)}"
            )
        if base_channels <= 0:
            raise ValueError(
                f"base_channels must be positive, got {base_channels!r}"
            )
        if channel_multiplier < 1.0:
            raise ValueError(
                f"channel_multiplier must be >= 1.0 (channels must not "
                f"shrink with depth), got {channel_multiplier!r}"
            )
        if any(s < 1 for s in stride_per_stage):
            raise ValueError(
                f"stride_per_stage entries must be >= 1, got {stride_per_stage!r}"
            )
        if any(b < 1 for b in blocks_per_stage):
            raise ValueError(
                f"blocks_per_stage entries must be >= 1, got {blocks_per_stage!r}"
            )
        if bottleneck_blocks < 1:
            raise ValueError(
                f"bottleneck_blocks must be >= 1, got {bottleneck_blocks!r}"
            )
        # Invariant I4b: shifts < smallest channel count (= base_channels,
        # which is the top-of-U level — every other level has strictly
        # more channels under the doubling schedule).
        if max(shifts) >= base_channels:
            raise ValueError(
                f"max(shifts)={max(shifts)} must be < base_channels="
                f"{base_channels} (smallest channel count). Either reduce "
                f"shifts or increase base_channels."
            )

        # --- MRL widths resolution + validation -----------------------
        # DECISION plan_2026-05-12_13c70aed/D-002:
        #   Width rule is "Power-of-2 anchored, base preserved":
        #     - widths[0] == base_channels (preserved as-is; may be non-power-of-2)
        #     - widths[1:] are strict powers of 2 strictly less than base_channels,
        #       descending, floor 16.
        # Default (None) -> [base_channels] (single-head back-compat).
        if mrl_widths is None:
            mrl_widths_resolved: List[int] = [base_channels]
        else:
            mrl_widths_resolved = list(mrl_widths)
        if len(mrl_widths_resolved) == 0:
            raise ValueError("mrl_widths must be non-empty (or None).")
        if any((not isinstance(w, int)) or w <= 0 for w in mrl_widths_resolved):
            raise ValueError(
                f"mrl_widths entries must be positive ints, got {mrl_widths_resolved!r}"
            )
        if any(w > base_channels for w in mrl_widths_resolved):
            raise ValueError(
                f"mrl_widths entries must be <= base_channels={base_channels}; "
                f"got {mrl_widths_resolved!r}"
            )
        if mrl_widths_resolved[0] != base_channels:
            raise ValueError(
                f"mrl_widths[0] must equal base_channels={base_channels} "
                f"(largest head preserved at base width); got {mrl_widths_resolved!r}"
            )
        # Strictly decreasing.
        for a, b in zip(mrl_widths_resolved[:-1], mrl_widths_resolved[1:]):
            if a <= b:
                raise ValueError(
                    f"mrl_widths must be strictly decreasing; got {mrl_widths_resolved!r}"
                )
        # Every subsequent element MUST be a strict power of 2 AND strictly
        # less than base_channels (D-002).
        for w in mrl_widths_resolved[1:]:
            if w >= base_channels:
                raise ValueError(
                    f"mrl_widths[1:] entries must be strictly less than "
                    f"base_channels={base_channels}; got {mrl_widths_resolved!r}"
                )
            if (w & (w - 1)) != 0:
                raise ValueError(
                    f"mrl_widths[1:] entries must each be a power of 2; "
                    f"got {mrl_widths_resolved!r} (offending: {w})"
                )

        # embedding_pool validation
        if embedding_pool not in ("last", "cls", "auto"):
            raise ValueError(
                f"embedding_pool must be one of {{'last','cls','auto'}}, "
                f"got {embedding_pool!r}"
            )
        if embedding_pool in ("cls", "auto") and cls_token_id is None:
            raise ValueError(
                f"embedding_pool={embedding_pool!r} requires cls_token_id "
                f"to be set (got None)."
            )

        # --- Persisted hyperparameters ---------------------------------
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.stride_per_stage = list(stride_per_stage)
        self.blocks_per_stage = list(blocks_per_stage)
        self.bottleneck_blocks = bottleneck_blocks
        self.shifts = list(shifts)
        self.cli_mode = cli_mode
        self.ctx_mode = ctx_mode
        self.use_global_context = use_global_context
        self.layer_scale_init = layer_scale_init
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = dropout_rate
        self.tie_word_embeddings = tie_word_embeddings
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # MRL + embedding-head hyperparameters.
        self.mrl_widths: List[int] = mrl_widths_resolved
        self.mrl_head_norm: bool = bool(mrl_head_norm)
        self.emb_head: bool = bool(emb_head)
        self.embedding_pool: str = embedding_pool
        self.cls_token_id: Optional[int] = cls_token_id
        self.l2_eps: float = float(l2_eps)

        # --- Derived quantities ----------------------------------------
        self.num_levels = num_levels
        self.channels_per_stage: List[int] = [
            int(round(base_channels * (channel_multiplier ** i)))
            for i in range(num_levels)
        ]
        self.total_stride: int = 1
        for s in self.stride_per_stage:
            self.total_stride *= s

        # --- Embeddings ------------------------------------------------
        self.token_embedding = keras.layers.Embedding(
            vocab_size, base_channels, name="token_embedding",
        )
        self.position_embedding = keras.layers.Embedding(
            max_seq_length, base_channels, name="position_embedding",
        )
        self.embed_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="embed_norm",
        )
        self.embed_dropout = keras.layers.Dropout(
            dropout_rate, name="embed_dropout",
        )

        # --- Encoder / bottleneck / decoder ----------------------------
        # Total block count for the linear DropPath schedule across the
        # entire encoder + bottleneck + decoder stack.
        total_blocks = (
            sum(self.blocks_per_stage)        # encoder
            + bottleneck_blocks               # bottleneck
            + sum(self.blocks_per_stage[:-1]) # decoder mirrors all but deepest
        )
        drop_rates = linear_drop_path_rates(total_blocks, stochastic_depth_rate)
        dr_idx = 0  # running index into ``drop_rates``

        _block_kw_common: Dict[str, Any] = dict(
            shifts=self.shifts,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            use_global_context=use_global_context,
            layer_scale_init=layer_scale_init,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )

        # Encoder: per-level blocks + downsampler (except the last level).
        self.encoder_blocks: List[List[CausalCliffordNetBlock]] = []
        self.encoder_downsamplers: List[Optional[CausalCliffordNetBlockDSv2]] = []
        for i in range(num_levels):
            level_channels = self.channels_per_stage[i]
            level_blocks = []
            for j in range(self.blocks_per_stage[i]):
                level_blocks.append(
                    CausalCliffordNetBlock(
                        channels=level_channels,
                        drop_path_rate=drop_rates[dr_idx],
                        name=f"enc_block_l{i}_b{j}",
                        **_block_kw_common,
                    )
                )
                dr_idx += 1
            self.encoder_blocks.append(level_blocks)

            if i < num_levels - 1:
                next_channels = self.channels_per_stage[i + 1]
                self.encoder_downsamplers.append(
                    CausalCliffordNetBlockDSv2(
                        channels=level_channels,
                        out_channels=next_channels,
                        strides=self.stride_per_stage[i],
                        shifts=self.shifts,
                        cli_mode=cli_mode,
                        ctx_mode=ctx_mode,
                        use_global_context=use_global_context,
                        kernel_size=7,
                        stream_pool="avg",
                        skip_pool="avg",
                        ctx_norm_type="bn",
                        ctx_activation="silu",
                        layer_scale_init=layer_scale_init,
                        # Downsamplers run between block stacks, so use the
                        # drop-path rate at the boundary (encoder block at
                        # the level we're leaving). Use 0.0 to keep them
                        # deterministic and let the blocks carry the
                        # stochastic-depth burden.
                        drop_path_rate=0.0,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        name=f"enc_downsample_l{i}",
                    )
                )
            else:
                self.encoder_downsamplers.append(None)

        # Bottleneck: stack of dim-preserving blocks at the deepest level.
        deepest_channels = self.channels_per_stage[-1]
        self.bottleneck_block_list: List[CausalCliffordNetBlock] = []
        for k in range(bottleneck_blocks):
            self.bottleneck_block_list.append(
                CausalCliffordNetBlock(
                    channels=deepest_channels,
                    drop_path_rate=drop_rates[dr_idx],
                    name=f"bottleneck_block_{k}",
                    **_block_kw_common,
                )
            )
            dr_idx += 1

        # Decoder: walk levels in reverse, excluding the deepest one
        # (i in num_levels-2 .. 0). At each level i:
        #   upsample (s = stride_per_stage[i])
        #   concat with encoder skip[i]  (channels: C_{i+1} + C_i = 3*C_i)
        #   1x1 Conv2D project to C_i
        #   blocks_per_stage[i] dim-preserving blocks
        # Lists are kept in *encoder order* (index = encoder level i).
        # Levels with index num_levels-1 hold None (no decoder stage there).
        self.decoder_upsamplers: List[Optional[keras.layers.UpSampling2D]] = []
        self.decoder_concats: List[Optional[keras.layers.Concatenate]] = []
        self.decoder_skip_projs: List[Optional[keras.layers.Conv2D]] = []
        self.decoder_blocks: List[List[CausalCliffordNetBlock]] = []
        for i in range(num_levels):
            self.decoder_upsamplers.append(None)
            self.decoder_concats.append(None)
            self.decoder_skip_projs.append(None)
            self.decoder_blocks.append([])

        for i in reversed(range(num_levels - 1)):
            level_channels = self.channels_per_stage[i]
            self.decoder_upsamplers[i] = keras.layers.UpSampling2D(
                size=(1, self.stride_per_stage[i]),
                interpolation="nearest",
                name=f"dec_upsample_l{i}",
            )
            self.decoder_concats[i] = keras.layers.Concatenate(
                axis=-1, name=f"dec_concat_l{i}",
            )
            self.decoder_skip_projs[i] = keras.layers.Conv2D(
                filters=level_channels,
                kernel_size=1,
                padding="same",
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                name=f"dec_skip_proj_l{i}",
            )
            level_blocks: List[CausalCliffordNetBlock] = []
            for j in range(self.blocks_per_stage[i]):
                level_blocks.append(
                    CausalCliffordNetBlock(
                        channels=level_channels,
                        drop_path_rate=drop_rates[dr_idx],
                        name=f"dec_block_l{i}_b{j}",
                        **_block_kw_common,
                    )
                )
                dr_idx += 1
            self.decoder_blocks[i] = level_blocks

        assert dr_idx == total_blocks, (
            f"drop-path index drift: dr_idx={dr_idx}, total_blocks={total_blocks}"
        )

        # --- Output head ----------------------------------------------
        # Head norm is applied after the decoder restores the level-0
        # resolution / channel count, so it operates on ``base_channels``.
        self.head_norm = keras.layers.LayerNormalization(
            epsilon=self.LAYERNORM_EPSILON, name="head_norm",
        )
        self.head_dropout = (
            keras.layers.Dropout(dropout_rate, name="head_dropout")
            if dropout_rate > 0.0
            else None
        )

        # Per-width MRL LayerNorms (operate on the sliced (B, T, w) tensor).
        # Only built when mrl_head_norm=True. Disabling skips the per-width
        # LayerNorm and projects the raw sliced features directly.
        if self.mrl_head_norm:
            self.mrl_head_norms: Optional[List[keras.layers.LayerNormalization]] = [
                keras.layers.LayerNormalization(
                    epsilon=self.LAYERNORM_EPSILON,
                    name=f"mrl_head_norm_w{w}",
                )
                for w in self.mrl_widths
            ]
        else:
            self.mrl_head_norms = None

        # Tied/untied per-width LM heads.
        # Tied path: per-width bias variables (only when use_bias). The first
        # entry corresponds to the largest width (the primary "logits" head).
        # The legacy ``self.output_bias`` / ``self.output_proj`` attributes are
        # kept (set to the largest-width entry) for back-compat with any
        # external code that introspected them — the old single-head behavior
        # corresponds to mrl_widths == [base_channels] (length 1).
        if tie_word_embeddings:
            self.output_proj = None
            self.mrl_output_projs: Optional[List[keras.layers.Dense]] = None
            if use_bias:
                self.mrl_output_biases: Optional[List[Any]] = [
                    self.add_weight(
                        name=f"output_bias_w{w}",
                        shape=(vocab_size,),
                        initializer=bias_initializer,
                        regularizer=bias_regularizer,
                        trainable=True,
                    )
                    for w in self.mrl_widths
                ]
                # Legacy alias for the largest-width bias.
                self.output_bias = self.mrl_output_biases[0]
            else:
                self.mrl_output_biases = None
                self.output_bias = None
        else:
            self.mrl_output_biases = None
            self.output_bias = None
            self.mrl_output_projs = [
                keras.layers.Dense(
                    vocab_size,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=("output_proj" if i == 0 else f"output_proj_w{w}"),
                )
                for i, w in enumerate(self.mrl_widths)
            ]
            # Legacy alias for the largest-width projection.
            self.output_proj = self.mrl_output_projs[0]

        # Optional learnable embedding projection on the pooled (B, C0)
        # vector. Identity-by-default (None) → pooled vector is used directly.
        if self.emb_head:
            self.embedding_proj: Optional[keras.layers.Dense] = keras.layers.Dense(
                base_channels,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                name="embedding_proj",
            )
        else:
            self.embedding_proj = None

        logger.info(
            f"Created CliffordNetLMUNet (vocab_size={vocab_size}, "
            f"max_seq_length={max_seq_length}, base_channels={base_channels}, "
            f"channel_multiplier={channel_multiplier}, "
            f"channels_per_stage={self.channels_per_stage}, "
            f"stride_per_stage={self.stride_per_stage}, "
            f"blocks_per_stage={self.blocks_per_stage}, "
            f"bottleneck_blocks={bottleneck_blocks}, "
            f"total_stride={self.total_stride}, shifts={self.shifts}, "
            f"cli_mode={cli_mode}, ctx_mode={ctx_mode}, "
            f"global_ctx={use_global_context}, "
            f"tie_word_embeddings={tie_word_embeddings}, "
            f"mrl_widths={self.mrl_widths}, "
            f"mrl_head_norm={self.mrl_head_norm}, "
            f"emb_head={self.emb_head}, "
            f"embedding_pool={self.embedding_pool})"
        )

    def call(
        self,
        input_ids: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> Dict[str, keras.KerasTensor]:
        """Forward pass.

        :param input_ids: Token IDs ``(B, seq_len)`` (int32).
        :param training: Whether in training mode.
        :return: Dict with output tensors. Always contains:

            - ``"logits": (B, seq_len, vocab_size)`` — primary head at the
              largest width (``base_channels``).
            - For each smaller width ``w`` in ``self.mrl_widths[1:]``:
              ``f"logits_w{w}": (B, seq_len, vocab_size)``.
            - For each width ``w`` in ``self.mrl_widths``:
              ``f"embedding_w{w}": (B, w)`` — L2-normalized pooled embedding.

            When ``self.mrl_widths == [base_channels]`` (default back-compat
            path) the dict reduces to ``{"logits", f"embedding_w{base_channels}"}``.
        """
        ops = keras.ops

        # --- Embeddings ------------------------------------------------
        seq_len = ops.shape(input_ids)[1]
        positions = ops.arange(0, seq_len, dtype="int32")
        tok = self.token_embedding(input_ids)              # (B, T, C0)
        pos = self.position_embedding(positions)           # (T, C0)
        x = tok + pos                                       # (B, T, C0)
        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        # Reshape to (B, 1, T, C0) for the 4D causal blocks.
        x = ops.expand_dims(x, axis=1)

        # --- Right-pad along W to multiple of total_stride (D-002) ----
        # ``pad_w`` is a Python int when ``seq_len`` is static; if dynamic
        # we still compute a tensor pad amount via keras.ops. We use the
        # symbolic form so this works under tf.function tracing too.
        total_stride = self.total_stride
        # ``(-seq_len) % total_stride`` via integer ops.
        pad_w = (-seq_len) % total_stride
        # ops.pad accepts a static spec; for the dynamic case we use a
        # conditional via ops.cond is overkill — instead we always pad and
        # rely on pad_w being 0 when aligned (a no-op pad).
        # Note: keras.ops.pad expects a static paddings spec. Build it as
        # a list of [int|tensor, int|tensor].
        x = ops.pad(x, [[0, 0], [0, 0], [0, pad_w], [0, 0]])

        # --- Encoder ---------------------------------------------------
        skip_features: List[keras.KerasTensor] = [None] * self.num_levels
        for i in range(self.num_levels):
            for blk in self.encoder_blocks[i]:
                x = blk(x, training=training)
            skip_features[i] = x
            ds = self.encoder_downsamplers[i]
            if ds is not None:
                x = ds(x, training=training)

        # --- Bottleneck ------------------------------------------------
        for blk in self.bottleneck_block_list:
            x = blk(x, training=training)

        # --- Decoder ---------------------------------------------------
        for i in reversed(range(self.num_levels - 1)):
            up = self.decoder_upsamplers[i]
            x = up(x)
            # DECISION D-007: causal upsample via right-shift; without this
            # the round trip pool->nearest-upsample leaks future info into
            # past positions (see plans/plan_2026-05-06_82749628/decisions.md).
            x = _causal_upsample(x, self.stride_per_stage[i])
            skip = skip_features[i]
            x = self.decoder_concats[i]([x, skip])
            x = self.decoder_skip_projs[i](x)
            for blk in self.decoder_blocks[i]:
                x = blk(x, training=training)

        # --- Crop along W back to original seq_len --------------------
        x = x[:, :, :seq_len, :]

        # --- Squeeze H, head norm/dropout, LM projection --------------
        x = ops.squeeze(x, axis=1)             # (B, T, C0)
        x = self.head_norm(x)
        if self.head_dropout is not None:
            x = self.head_dropout(x, training=training)

        # h_top is the post-head-norm hidden state shared by all MRL widths
        # and by the embedding pool. (B, T, C0).
        h_top = x

        outputs: Dict[str, keras.KerasTensor] = {}

        # --- Per-width LM heads (MRL) ---------------------------------
        if self.tie_word_embeddings:
            emb_kernel = self.token_embedding.embeddings  # (V, C0)
        for idx, w in enumerate(self.mrl_widths):
            x_w = h_top[..., :w]
            if self.mrl_head_norms is not None:
                x_w = self.mrl_head_norms[idx](x_w)
            if self.tie_word_embeddings:
                # (V, w) slice; transpose to (w, V); matmul (B, T, w)x(w, V).
                logits_w = ops.matmul(
                    x_w, ops.transpose(emb_kernel[:, :w], (1, 0)),
                )
                if self.mrl_output_biases is not None:
                    logits_w = logits_w + self.mrl_output_biases[idx]
            else:
                logits_w = self.mrl_output_projs[idx](x_w)
            key = "logits" if idx == 0 else f"logits_w{w}"
            outputs[key] = logits_w

        # --- Pooled embedding head ------------------------------------
        # Default "last" → array index T-1; "cls" → position 0;
        # "auto" → per-sample where input_ids[:, 0] == cls_token_id then
        # position 0 else position T-1.
        if self.embedding_pool == "cls":
            pooled = h_top[:, 0, :]                              # (B, C0)
        elif self.embedding_pool == "auto":
            cls_mask = ops.equal(
                input_ids[:, 0:1],
                ops.cast(self.cls_token_id, dtype=input_ids.dtype),
            )                                                    # (B, 1)
            cls_mask_f = ops.cast(cls_mask, dtype=h_top.dtype)   # (B, 1)
            first_pos = h_top[:, 0, :]                           # (B, C0)
            last_pos = h_top[:, -1, :]                           # (B, C0)
            pooled = cls_mask_f * first_pos + (1.0 - cls_mask_f) * last_pos
        else:  # "last"
            pooled = h_top[:, -1, :]                             # (B, C0)

        if self.embedding_proj is not None:
            pooled = self.embedding_proj(pooled)                 # (B, C0)

        # L2-normalized per-width embedding side outputs.
        # Cast to float32 for the norm reduction to keep the unit-norm
        # guarantee tight at fp16/bf16 compute dtypes.
        compute_dtype = pooled.dtype
        pooled_f32 = ops.cast(pooled, "float32")
        for w in self.mrl_widths:
            e_w = pooled_f32[..., :w]
            denom = ops.sqrt(
                ops.sum(e_w * e_w, axis=-1, keepdims=True) + self.l2_eps
            )
            e_norm = e_w / denom
            outputs[f"embedding_w{w}"] = ops.cast(e_norm, compute_dtype)

        return outputs

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...],
    ) -> Dict[str, Tuple[Optional[int], ...]]:
        batch_dim = input_shape[0]
        seq_dim = input_shape[1]
        shapes: Dict[str, Tuple[Optional[int], ...]] = {}
        for idx, w in enumerate(self.mrl_widths):
            key = "logits" if idx == 0 else f"logits_w{w}"
            shapes[key] = (batch_dim, seq_dim, self.vocab_size)
        for w in self.mrl_widths:
            shapes[f"embedding_w{w}"] = (batch_dim, w)
        return shapes

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "max_seq_length": self.max_seq_length,
            "base_channels": self.base_channels,
            "channel_multiplier": self.channel_multiplier,
            "stride_per_stage": self.stride_per_stage,
            "blocks_per_stage": self.blocks_per_stage,
            "bottleneck_blocks": self.bottleneck_blocks,
            "shifts": self.shifts,
            "cli_mode": self.cli_mode,
            "ctx_mode": self.ctx_mode,
            "use_global_context": self.use_global_context,
            "layer_scale_init": self.layer_scale_init,
            "stochastic_depth_rate": self.stochastic_depth_rate,
            "dropout_rate": self.dropout_rate,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_bias": self.use_bias,
            "kernel_initializer": initializers.serialize(self.kernel_initializer),
            "bias_initializer": initializers.serialize(self.bias_initializer),
            "kernel_regularizer": regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": regularizers.serialize(self.bias_regularizer),
            "mrl_widths": list(self.mrl_widths),
            "mrl_head_norm": self.mrl_head_norm,
            "emb_head": self.emb_head,
            "embedding_pool": self.embedding_pool,
            "cls_token_id": self.cls_token_id,
            "l2_eps": self.l2_eps,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CliffordNetLMUNet":
        for key in ("kernel_regularizer", "bias_regularizer"):
            if config.get(key) and isinstance(config[key], dict):
                config[key] = regularizers.deserialize(config[key])
        return cls(**config)

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        max_seq_length: int = 512,
        **kwargs: Any,
    ) -> "CliffordNetLMUNet":
        """Create a CliffordNetLMUNet from a predefined variant.

        :param variant: One of ``"nano"``, ``"mini"``, ``"base"``,
            ``"large"``, ``"xl"``.
        :param vocab_size: Vocabulary size.
        :param max_seq_length: Maximum sequence length.
        :param kwargs: Override any default hyperparameter.
        :return: Configured :class:`CliffordNetLMUNet` instance.
        """
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Available: {list(cls.MODEL_VARIANTS.keys())}"
            )
        defaults = dict(cls.MODEL_VARIANTS[variant])
        defaults.update(kwargs)
        logger.info(f"Creating CliffordNetLMUNet-{variant.upper()}")
        return cls(
            vocab_size=vocab_size,
            max_seq_length=max_seq_length,
            **defaults,
        )
