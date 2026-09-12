"""Zamba2-specific building blocks.

This module holds the layers that are genuinely new for Zamba2 -- the ones
that have no reusable precedent elsewhere in the repository -- plus the thin
per-block wrappers that compose Zamba2's decoder stack out of existing
primitives (``Mamba2ResidualBlock``, the attention factory, ``RMSNorm``,
``RotaryPositionEmbedding``). Classes are added incrementally; see
``plans/plan-2026-09-12T075714-035fd488/plan.md`` Steps 1-4 for the build
order. :class:`LoRAAdapter` (step 1) and :class:`Zamba2SharedAttentionBlock`
(step 2) exist so far.

References:
    - Glorioso, P. et al., 2024. Zamba2: A Compact and Fast Hybrid Model.
      (https://arxiv.org/abs/2411.15242)
    - Hu, E. J. et al., 2021. LoRA: Low-Rank Adaptation of Large Language
      Models. (https://arxiv.org/abs/2106.09685)
"""

from typing import Any, Dict, Optional, Tuple, Union

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.norms.rms_norm import RMSNorm
from dl_techniques.layers.embedding.rotary_position_embedding import RotaryPositionEmbedding
from dl_techniques.layers.attention.factory import create_attention_layer

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.zamba2.lora_adapter")
class LoRAAdapter(keras.layers.Layer):
    """
    Additive low-rank adapter with one independent ``A``/``B`` pair per occurrence.

    Computes a pure additive delta ``(x @ A[i]) @ B[i] * (alpha / rank)`` for a
    caller-selected occurrence index ``i``. It does **not** own or wrap the
    base projection it augments -- the caller (a shared block invoked at
    several depths) is responsible for adding this delta onto its own base
    ``Dense`` output. This is the mechanism Zamba2 uses to let ``num_mem_blocks``
    physical shared blocks, reused across many more depth positions, specialize
    per depth without multiplying the shared block's own parameter count: every
    depth position gets its own LoRA pair even though several depth positions
    route through the same physical block.

    Architecture:

    .. code-block:: text

        x  [..., input_dim]
              │
              ▼  select occurrence i (a plain Python int, fixed per call site)
        A[i]: (input_dim, rank)          -- unique per occurrence
              │
              ▼  [..., rank]
        B[i]: (rank, output_dim)         -- unique per occurrence, zero-init
              │
              ▼  [..., output_dim]
        * (alpha / rank)
              │
              ▼
        delta  [..., output_dim]   (added by the CALLER onto its base output)

        A is one weight tensor of shape (num_occurrences, input_dim, rank);
        B is one weight tensor of shape (num_occurrences, rank, output_dim).
        Each occurrence's (input_dim, rank) / (rank, output_dim) slice is
        initialized independently -- see the Note on initialization below.

    :param output_dim: Width of the delta this adapter produces. Must match
        the base projection's output width the caller will add this onto.
        Must be positive.
    :type output_dim: int
    :param rank: Bottleneck width shared by every occurrence's ``A``/``B``
        pair. Must be positive.
    :type rank: int
    :param alpha: LoRA scaling numerator; the applied scale is
        ``alpha / rank``. Must be positive.
    :type alpha: float
    :param num_occurrences: Number of independent ``A``/``B`` pairs to
        allocate -- one per depth position that will select into this
        adapter via ``call(..., occurrence_idx=...)``, NOT one per physical
        shared block. Must be positive.
    :type num_occurrences: int
    :param kernel_initializer: Initializer applied independently to each
        occurrence's ``A`` slice. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar rank: The stored bottleneck width.
    :vartype rank: int
    :ivar alpha: The stored scaling numerator.
    :vartype alpha: float
    :ivar num_occurrences: The stored occurrence count.
    :vartype num_occurrences: int
    :ivar scale: The resolved ``alpha / rank`` scale applied to every delta.
    :vartype scale: float
    :ivar kernel_initializer: The resolved initializer for ``A``.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar a: Weight of shape ``(num_occurrences, input_dim, rank)``.
    :vartype a: keras.Variable
    :ivar b: Weight of shape ``(num_occurrences, rank, output_dim)``.
    :vartype b: keras.Variable

    :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
        ``num_occurrences`` is not positive.
    :raises ValueError: If ``call()`` is given an ``occurrence_idx`` outside
        ``[0, num_occurrences)``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, with the last axis set to
        ``output_dim``.

    Example:
        .. code-block:: python

            adapter = LoRAAdapter(output_dim=256, rank=8, alpha=16.0, num_occurrences=4)
            x = keras.random.normal((2, 10, 256))
            delta_0 = adapter(x, occurrence_idx=0)
            delta_1 = adapter(x, occurrence_idx=1)
            # delta_0 != delta_1: each occurrence has its own A/B pair.

    Note:
        ``B`` is initialized to zeros, the standard LoRA convention: every
        occurrence's delta is exactly zero at construction, so attaching this
        adapter to an already-trained shared block does not perturb its
        output until training updates ``B`` away from zero. ``A`` is
        initialized by applying ``kernel_initializer`` independently to each
        occurrence's ``(input_dim, rank)`` slice -- NOT by initializing the
        full stacked ``(num_occurrences, input_dim, rank)`` tensor in one
        call. Keras' built-in initializers compute fan-in/fan-out from a
        3-D shape as if it were a convolution kernel's
        ``(receptive_field, fan_in, fan_out)``, which would read
        ``num_occurrences`` as a spatial extent and scale every slice
        incorrectly. Initializing slice-by-slice instead makes the stacked
        tensor equivalent to ``num_occurrences`` independent 2-D weights.
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        alpha: float,
        num_occurrences: int,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and resolve the LoRA scale.

        No weights are created here -- ``A``/``B`` need the input width,
        which is only known in ``build()``.

        :raises ValueError: If ``output_dim``, ``rank``, ``alpha``, or
            ``num_occurrences`` is not positive.
        """
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if num_occurrences <= 0:
            raise ValueError(f"num_occurrences must be positive, got {num_occurrences}")

        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.num_occurrences = num_occurrences
        self.scale = alpha / rank
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.a: Optional[keras.Variable] = None
        self.b: Optional[keras.Variable] = None

        logger.info(
            f"Initialized LoRAAdapter with output_dim={output_dim}, rank={rank}, "
            f"alpha={alpha}, num_occurrences={num_occurrences}, scale={self.scale}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Create the per-occurrence ``A``/``B`` weight tensors.

        :param input_shape: Shape tuple of the input tensor; only the last
            axis (``input_dim``) is used.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input's last axis is not statically known.
        """
        if self.built:
            return

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "LoRAAdapter requires a statically-known input feature "
                f"dimension; got input_shape={input_shape}"
            )

        num_occurrences = self.num_occurrences
        rank = self.rank
        output_dim = self.output_dim
        a_initializer_fn = self.kernel_initializer

        def _a_initializer(shape: Tuple[int, int, int], dtype: Any = None) -> Any:
            # Apply the 2-D initializer independently to every occurrence's
            # own (input_dim, rank) slice -- see the class Note on why the
            # stacked 3-D shape must not be handed to the initializer as-is.
            slices = [
                a_initializer_fn(shape=(shape[1], shape[2]), dtype=dtype)
                for _ in range(shape[0])
            ]
            return keras.ops.stack(slices, axis=0)

        self.a = self.add_weight(
            name="lora_a",
            shape=(num_occurrences, input_dim, rank),
            initializer=_a_initializer,
            trainable=True,
        )
        self.b = self.add_weight(
            name="lora_b",
            shape=(num_occurrences, rank, output_dim),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        occurrence_idx: int,
    ) -> keras.KerasTensor:
        """
        Compute the additive LoRA delta for one occurrence.

        :param inputs: Input tensor of shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param occurrence_idx: Which occurrence's ``A``/``B`` pair to use. A
            plain Python ``int`` -- the depth position an occurrence
            corresponds to is static (fixed by ``layer_mapping`` at model
            construction time), never a data-dependent tensor value.
        :type occurrence_idx: int
        :return: Delta tensor of shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        :raises ValueError: If ``occurrence_idx`` is outside
            ``[0, num_occurrences)``.
        """
        if not (0 <= occurrence_idx < self.num_occurrences):
            raise ValueError(
                f"occurrence_idx must be in [0, {self.num_occurrences}), "
                f"got {occurrence_idx}"
            )

        a_i = self.a[occurrence_idx]
        b_i = self.b[occurrence_idx]

        delta = keras.ops.matmul(inputs, a_i)
        delta = keras.ops.matmul(delta, b_i)
        return delta * self.scale

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with the last dimension set to
            ``output_dim``.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "rank": self.rank,
                "alpha": self.alpha,
                "num_occurrences": self.num_occurrences,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.zamba2.shared_attention_block")
class Zamba2SharedAttentionBlock(keras.layers.Layer):
    """
    Zamba2's shared attention mem-block: one physical instance, many depths.

    Built once per physical mem-block slot (``num_mem_blocks`` instances
    total) and called at every ``'g'`` position in the decoder's
    ``layer_mapping``, round-robin. Every call site shares the identical
    weight tensors by plain Keras construction (the same Python object is
    invoked repeatedly) -- no LoRA or other per-call-site divergence lives in
    this block (see ``decisions.md`` D-001: the LoRA delta is scoped to the
    shared MLP mem-block's up-projection only, step 3, to keep this block's
    already-risky mask/RoPE/concat composition free of a second novel
    mechanism).

    Each call receives both the running decoder ``hidden_state`` and the
    *original* token embedding (computed once, before the first decoder
    block) and concatenates them on the feature axis before projecting back
    down to ``d_model`` -- the mem-block re-reads the raw input at every
    occurrence rather than only ever seeing it transformed by prior blocks,
    per the Zamba2 paper's mem-block design.

    Architecture:

    .. code-block:: text

        hidden_state [B, S, d_model]   original_embedding [B, S, d_model]
               │                                   │
               ▼ RMSNorm                           │
          normed_hidden                            │
               │                                   │
               └──────────────┬────────────────────┘
                               ▼ concatenate (feature axis)
                        [B, S, 2*d_model]
                               │
                               ▼ Dense -> d_model
                        [B, S, d_model]
                               │
                               ▼ reshape to heads, RotaryPositionEmbedding,
                                 reshape back (see Note)
                        [B, S, d_model]
                               │
                               ▼ 'multi_head' attention (causal, rank-3 mask);
                                 its own internal output projection IS this
                                 block's output projection
                        [B, S, d_model]
                               │
                               ▼ + hidden_state (residual on the ORIGINAL
                                 hidden state, not the concatenated tensor)
                        [B, S, d_model]

    :param d_model: Width of the decoder's hidden state. Must be positive
        and divisible by ``num_heads``.
    :type d_model: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param max_seq_len: Largest sequence length the RoPE tables are built
        for. Must be positive.
    :type max_seq_len: int
    :param rope_theta: RoPE frequency base, forwarded to
        :class:`RotaryPositionEmbedding`. Defaults to 10000.0.
    :type rope_theta: float
    :param rope_percentage: Fraction of the per-head dimension RoPE rotates,
        forwarded to :class:`RotaryPositionEmbedding`. Defaults to 1.0.
    :type rope_percentage: float
    :param attention_dropout_rate: Dropout rate inside the attention
        probability computation. Must be in ``[0, 1)``. Defaults to 0.0.
    :type attention_dropout_rate: float
    :param norm_epsilon: Epsilon for the pre-attention :class:`RMSNorm`.
        Must be positive. Defaults to 1e-6.
    :type norm_epsilon: float
    :param use_bias: Whether the input projection and the attention's
        internal projections use a bias term. Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the input projection and the
        attention's internal projections. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param kwargs: Extra arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar d_model: The stored hidden width.
    :vartype d_model: int
    :ivar num_heads: The stored head count.
    :vartype num_heads: int
    :ivar head_dim: ``d_model // num_heads``.
    :vartype head_dim: int
    :ivar norm: Pre-attention :class:`RMSNorm`.
    :vartype norm: RMSNorm
    :ivar input_proj: ``Dense(d_model)`` projecting the concatenated
        ``[original_embedding, normed_hidden]`` tensor back down to
        ``d_model``.
    :vartype input_proj: keras.layers.Dense
    :ivar rope: :class:`RotaryPositionEmbedding` applied per-head.
    :vartype rope: RotaryPositionEmbedding
    :ivar attention: The ``'multi_head'`` attention layer built by
        :func:`create_attention_layer`.
    :vartype attention: keras.layers.Layer

    :raises ValueError: If ``d_model`` is not divisible by ``num_heads``, or
        if ``d_model``, ``num_heads``, ``max_seq_len``, or ``norm_epsilon``
        is not positive.

    Input shape:
        Two tensors, each ``(batch_size, seq_len, d_model)``: ``hidden_state``
        and ``original_embedding``.

    Output shape:
        ``(batch_size, seq_len, d_model)``, matching ``hidden_state``.

    Example:
        .. code-block:: python

            block = Zamba2SharedAttentionBlock(d_model=256, num_heads=8, max_seq_len=512)
            h = keras.random.normal((2, 32, 256))
            e = keras.random.normal((2, 32, 256))
            # Called at two different depths -- same instance, same weights:
            out_depth_2 = block(h, e)
            out_depth_5 = block(h, e)

    Note:
        RoPE is applied to the post-concatenation, post-projection tensor
        (reshaped to ``(batch, num_heads, seq_len, head_dim)``, rotated, then
        reshaped back to ``(batch, seq_len, d_model)``) immediately BEFORE
        the ``'multi_head'`` attention call, per this plan's step-2 spec --
        not to the attention layer's own internal Q/K projections (that
        layer exposes no hook for this since it owns its QKV projections
        internally; see ``findings/existing-layers-for-zamba2.md`` #6). The
        causal mask is built as a rank-3 ``(batch, seq_len, seq_len)`` keep
        predicate (``1`` = keep), never rank-2, per the v2 guide's rank-3
        mask invariant (plan.md Problem Statement invariant 3) -- a rank-2
        mask degrades to padding semantics and silently stops being causal
        per-row.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rope_percentage: float = 1.0,
        attention_dropout_rate: float = 0.0,
        norm_epsilon: float = 1e-6,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        **kwargs: Any,
    ) -> None:
        """Validate the configuration and build every sub-layer.

        Sub-layers are created here (unconditionally), not in ``build()`` --
        the v2 guide's "create unconditionally, use conditionally" rule --
        but are not yet built; each is explicitly built in :meth:`build`
        once this block's own input shape is known.

        :raises ValueError: If ``d_model`` is not divisible by ``num_heads``,
            or if ``d_model``, ``num_heads``, ``max_seq_len``, or
            ``norm_epsilon`` is not positive.
        """
        super().__init__(**kwargs)

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.rope_percentage = rope_percentage
        self.attention_dropout_rate = attention_dropout_rate
        self.norm_epsilon = norm_epsilon
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.norm = RMSNorm(epsilon=self.norm_epsilon, name="norm")
        self.input_proj = keras.layers.Dense(
            d_model,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="input_proj",
        )
        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            rope_percentage=self.rope_percentage,
            name="rope",
        )
        self.attention = create_attention_layer(
            "multi_head",
            dim=self.d_model,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="attention",
        )

        logger.info(
            f"Initialized Zamba2SharedAttentionBlock with d_model={d_model}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}, max_seq_len={max_seq_len}"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build every sub-layer against the shapes it will actually see.

        :param input_shape: Shape of ``hidden_state``, ``(batch, seq_len,
            d_model)`` -- ``original_embedding`` shares this shape, so no
            second shape argument is needed.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        batch, seq_len, _ = input_shape
        concat_shape = (batch, seq_len, 2 * self.d_model)
        rope_input_shape = (batch, self.num_heads, seq_len, self.head_dim)

        self.norm.build(input_shape)
        self.input_proj.build(concat_shape)
        self.rope.build(rope_input_shape)
        self.attention.build(input_shape)

        super().build(input_shape)

    def _build_causal_mask(
        self,
        batch_size: Any,
        seq_len: Any,
        dtype: Any,
    ) -> keras.KerasTensor:
        """
        Build a rank-3 ``(batch_size, seq_len, seq_len)`` causal keep-mask.

        ``1`` marks a position as attendable (key index <= query index,
        i.e. the standard lower-triangular causal predicate); ``0`` marks a
        future position. Built at rank 3 deliberately -- a rank-2
        ``(seq_len, seq_len)`` mask would be broadcast by the attention
        layer as a PADDING mask, not a per-query-row causal one, silently
        degrading the invariant this block exists to hold (plan.md Problem
        Statement invariant 3).

        :param batch_size: Dynamic batch size tensor/value.
        :type batch_size: Any
        :param seq_len: Dynamic sequence-length tensor/value.
        :type seq_len: Any
        :param dtype: Dtype the mask is cast to (the attention layer's
            compute dtype).
        :type dtype: Any
        :return: Causal keep-mask of shape ``(batch_size, seq_len, seq_len)``.
        :rtype: keras.KerasTensor
        """
        row_idx = keras.ops.arange(seq_len)[:, None]
        col_idx = keras.ops.arange(seq_len)[None, :]
        mask_2d = keras.ops.cast(col_idx <= row_idx, dtype=dtype)
        return keras.ops.broadcast_to(mask_2d[None, :, :], (batch_size, seq_len, seq_len))

    def call(
        self,
        hidden_state: keras.KerasTensor,
        original_embedding: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """
        Run one occurrence of the shared attention mem-block.

        :param hidden_state: Running decoder hidden state, shape ``(batch,
            seq_len, d_model)``.
        :type hidden_state: keras.KerasTensor
        :param original_embedding: The token embedding computed once before
            the first decoder block, shape ``(batch, seq_len, d_model)``.
        :type original_embedding: keras.KerasTensor
        :param training: Whether in training mode, forwarded to every
            sub-layer.
        :type training: Optional[bool]
        :return: Output of shape ``(batch, seq_len, d_model)``.
        :rtype: keras.KerasTensor
        """
        normed_hidden = self.norm(hidden_state, training=training)
        concatenated = keras.ops.concatenate([original_embedding, normed_hidden], axis=-1)
        projected = self.input_proj(concatenated)

        batch_size = keras.ops.shape(projected)[0]
        seq_len = keras.ops.shape(projected)[1]

        heads_first = keras.ops.reshape(
            projected, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        heads_first = keras.ops.transpose(heads_first, (0, 2, 1, 3))
        rotated = self.rope(heads_first, training=training)
        rotated = keras.ops.transpose(rotated, (0, 2, 1, 3))
        rotated = keras.ops.reshape(rotated, (batch_size, seq_len, self.d_model))

        causal_mask = self._build_causal_mask(batch_size, seq_len, dtype=rotated.dtype)
        attention_output = self.attention(
            rotated, attention_mask=causal_mask, training=training
        )

        return hidden_state + attention_output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape of ``hidden_state`` (and
            ``original_embedding``), ``(batch, seq_len, d_model)``.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Unchanged from ``hidden_state``'s shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing every constructor argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "max_seq_len": self.max_seq_len,
                "rope_theta": self.rope_theta,
                "rope_percentage": self.rope_percentage,
                "attention_dropout_rate": self.attention_dropout_rate,
                "norm_epsilon": self.norm_epsilon,
                "use_bias": self.use_bias,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            }
        )
        return config

# ---------------------------------------------------------------------
