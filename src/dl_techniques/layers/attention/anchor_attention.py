"""
Hierarchical anchor-based attention over long sequences.

This layer implements an information bottleneck over the standard multi-head
self-attention mechanism. Its fundamental purpose is to preserve global context
propagation across a sequence while removing the quadratic cost of letting every
element attend to every other element. It does so by electing a small, fixed set
of elements as *anchors* and routing all global information exchange through
them.

Architecturally, the all-to-all attention graph is replaced by a two-tier,
hub-and-spoke graph:

-   **Anchor tokens** are a small subset of the sequence (the first ``K``
    positions). They perform full, quadratic self-attention among themselves and
    therefore act as the hubs: they aggregate information from one another and
    form a compressed global summary of the sequence.
-   **Query tokens** are the remaining ``N - K`` positions. They do not attend to
    each other at all. Each one cross-attends only to the anchors, reading from
    the global summary rather than reconstructing it. This is the spoke half of
    the graph, and it is where the savings come from.

Both tiers share a single attention call. The anchor queries and the query-token
queries are concatenated along the sequence axis and scored against the anchor
keys, so the score matrix is ``(N, K)`` rather than ``(N, N)``. Queries for the
two tiers come from separate projections, which lets the layer learn a distinct
"read from summary" behaviour for spokes and a "build the summary" behaviour for
hubs.

Architecture:
    Q, K and V come from three ``Dense`` projections, plus a **fourth** ``Dense``
    (``query_token_proj``) used only by the spoke tokens. In hierarchical mode the
    anchor queries and the spoke queries are concatenated along the sequence axis in
    original token order and scored against the anchor keys alone, producing an
    ``(N, K)`` score matrix instead of ``(N, N)``. Because the concatenation
    preserves token order, the output needs no re-scatter.

    Two structural properties are load-bearing:

    -   The mode is a **call** argument (``num_anchor_tokens``), not constructor
        state, so it is absent from ``get_config()``. A reloaded model runs in
        standard mode unless the caller passes the argument again.
    -   Hierarchical mode requires a statically-known sequence length; see the
        ``plan_2026-06-14_ab855e7e/D-002`` anchor in ``_hierarchical_attention``.

Foundational Mathematics:
    Write ``A`` for the anchor block (the first ``K`` tokens) and ``Q`` for the
    remaining ``N - K``. Standard attention factorizes token ``i``'s output as a
    convex combination over all ``N`` values. Anchor attention restricts the support
    of that combination to the ``K`` anchor values::

        out_i = sum_{j in A} softmax_j( q_i . k_j / sqrt(d) ) v_j

    with ``q_i = W_q x_i`` for ``i in A`` and ``q_i = W_q' x_i`` for ``i in Q``.
    Every path between two non-anchor tokens is therefore length 2 through the
    anchor set, which is exactly the hub-and-spoke structure: the anchors form a
    rank-``K`` bottleneck through which all long-range information must pass. The
    layer's expressive limit follows directly — no interaction that cannot be
    represented in the span of ``K`` value vectors survives.

    The complexity reduction follows just as directly: standard self-attention is
    ``O(N^2 * d)``, whereas anchor attention is
    ``O(K^2 * d + (N - K) * K * d) ~ O(N * K * d)`` when ``K << N``. For ``K=32`` and
    ``N=4096`` this is roughly a 128x reduction in attention computation, at the cost
    of forcing all long-range interaction through that ``K``-dimensional bottleneck.

References:
    - Beltagy, I., et al. (2020). "Longformer: The Long-Document Transformer".
      (https://arxiv.org/abs/2004.05150)
    - Lee, J., et al. (2019). "Set Transformer: A Framework for Attention-based
      Permutation-Invariant Neural Networks". (https://arxiv.org/abs/1810.00825)
"""

# ---------------------------------------------------------------------

import keras
from keras import ops, initializers, regularizers
from typing import Optional, Any, Dict, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import compute_attention_scale, validate_head_divisibility
from ..activations import ProbabilityOutput

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class AnchorAttention(keras.layers.Layer):
    """
    Hierarchical attention mechanism with an anchor-based information bottleneck.

    This layer implements a memory-efficient attention mechanism that reduces
    computational complexity for long sequences while retaining global context
    through a two-tier structure. The mode is selected per call, not per instance:

    - **Standard mode** (``num_anchor_tokens=None``): full self-attention over all
      tokens with ``O(N^2)`` complexity, using the configured probability
      activation (softmax, sparsemax, etc.).
    - **Hierarchical mode** (``num_anchor_tokens=K > 0``): anchor tokens perform
      full self-attention among themselves while query tokens cross-attend only to
      the anchors, giving ``O(K^2 + N*K)`` complexity.

    The computation follows:

    Standard: ``Q = X @ W_q, K = X @ W_k, V = X @ W_v;
    Output = Probability(Q @ K^T / sqrt(d_k)) @ V @ W_o``

    Hierarchical: ``Q_combined = [Q_anchor ; Q_query];
    scores = Q_combined @ K_anchor^T / sqrt(d_k);
    Output = Probability(scores) @ V_anchor @ W_o``

    **Architecture Overview:**

    Shapes use ``B`` = batch, ``N`` = seq_len, ``K`` = num_anchor_tokens,
    ``H`` = num_heads, ``d`` = head_dim. The merged-head width ``H*d`` equals
    ``dim`` unless ``head_dim`` is set explicitly.

    .. code-block:: text

        Standard mode  (num_anchor_tokens=None)
        Every token supplies a key and a value, so the score matrix is N x N.

                 Input  [B, N, dim]
                          │
                ┌─────────┼─────────┐
                ▼         ▼         ▼
             Q Proj    K Proj    V Proj
                │         │         │
                ▼         ▼         ▼
            [B,H,N,d] [B,H,N,d] [B,H,N,d]
                │         │         │
                └────┬────┘         │
                     ▼              │
             Q @ K^T / sqrt(d)      │
                 [B,H,N,N]          │
                     │              │
                     ▼              │
          Probability activation    │
                     │              │
                     ▼              │
                  Dropout           │
                     │              │
                     └───────┬──────┘
                             ▼
                        weights @ V
                         [B,H,N,d]
                             │
                             ▼
                        merge heads
                         [B,N,H*d]
                             │
                             ▼
                     Output projection
                             │
                             ▼
                    Output  [B, N, dim]


        Hierarchical mode  (num_anchor_tokens=K > 0)
        Only anchors supply keys and values, so the score matrix is N x K.
        Query tokens contribute a query and nothing else: that is the bottleneck.

                          Input  [B, N, dim]
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
              Anchors  x[:, :K]         Queries  x[:, K:]
                      │                         │
            ┌─────────┼─────────┐               │
            ▼         ▼         ▼               ▼
          K Proj    V Proj    Q Proj       Q-token Proj
            │         │         │               │
            ▼         ▼         ▼               ▼
           K_a       V_a       Q_a             Q_q
        [B,H,K,d] [B,H,K,d] [B,H,K,d]      [B,H,N-K,d]
            │         │         │               │
            │         │         └───────┬───────┘
            │         │                 ▼
            │         │         Q_all  [B,H,N,d]
            │         │                 │
            └─────────┼─────────────┬───┘
                      │             ▼
                      │  Q_all @ K_a^T / sqrt(d)
                      │         [B,H,N,K]
                      │             │
                      │             ▼
                      │  Probability activation
                      │             │
                      │             ▼
                      │          Dropout
                      │             │
                      └──────┬──────┘
                             ▼
                       weights @ V_a
                         [B,H,N,d]
                             │
                             ▼
                        merge heads
                         [B,N,H*d]
                             │
                             ▼
                     Output projection
                             │
                             ▼
                    Output  [B, N, dim]

        Q_all concatenates [Q_a ; Q_q] in original token order, so the K rows
        of the output belong to the anchors and no re-scatter is needed.

    **[NO MASK ARGUMENT — intentional carve-out, not an omission]** ``call()`` takes
    ``(x, num_anchor_tokens=None, training=None)`` and has **no** ``attention_mask``
    parameter. This is a frozen part of the public signature, adjacent to the
    ``factory.py`` D-007 carve-out that pins the non-standard ``call()`` signatures in
    this package: adding a mask argument would change the call contract for every
    consumer and for the factory's dispatch. Do NOT "restore parity" with the MHA
    family by bolting one on. The consequence, stated plainly so no caller is
    surprised: anchors are chosen POSITIONALLY (the first ``K`` elements), so a
    right-padded batch is safe, while a **left-padded batch promotes padding tokens
    into the global summary** and silently corrupts every spoke's read. Pre-trim or
    right-pad. If real masking is needed, that is a new layer or a follow-up plan, not
    an in-place signature edit.

    **[REUSE]** The ``dim % num_heads`` check and the ``1 / sqrt(head_dim)``
    temperature come from :mod:`~dl_techniques.layers.attention.common`; score
    normalization is the shared
    :class:`~dl_techniques.layers.activations.ProbabilityOutput`.

    **Known limitations:**

    - No ``attention_mask`` support — see the carve-out block above.
    - ``num_anchor_tokens`` is a call argument, not constructor state, and is
      therefore absent from ``get_config()``. A reloaded model runs in standard
      mode unless the caller passes the argument again.
    - The probability activation is built once against a square score shape. This
      assumes the activation carries no weights tied to the key axis, which holds
      for softmax and sparsemax; a key-length-dependent variant would need the
      build shape reworked.

    :param dim: Integer, input/output dimension of the attention layer.
        Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads.
        Must be positive and divide dim evenly.
    :type num_heads: int
    :param head_dim: Optional integer, dimension of each attention head.
        If None, computed as ``dim // num_heads``. When set explicitly the
        internal width becomes ``num_heads * head_dim``, which the output
        projection maps back to ``dim``. Defaults to None.
    :type head_dim: Optional[int]
    :param dropout_rate: Float, dropout rate applied to attention weights.
        Must be in range [0.0, 1.0]. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to True.
    :type use_bias: bool
    :param probability_type: String, type of probability function for attention
        scores (e.g., 'softmax', 'sparsemax', 'adaptive'). Defaults to 'softmax'.
    :type probability_type: str
    :param probability_config: Optional dictionary containing configuration for
        the probability layer. Defaults to None.
    :type probability_config: Optional[Dict[str, Any]]
    :param kernel_initializer: String or Initializer instance for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: String or Initializer instance for bias vectors.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
        Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If dim, num_heads or head_dim are not positive.
    :raises ValueError: If dim is not divisible by num_heads.
    :raises ValueError: If dropout_rate is outside [0.0, 1.0].
    :raises ValueError: From ``build()``, if the input is not 3D or its last
        dimension does not match ``dim``.
    :raises ValueError: From ``call()``, if ``num_anchor_tokens`` is set but not
        positive.
    :raises ValueError: From ``call()`` in hierarchical mode, if the input's
        sequence dimension is not statically known (see the
        ``plan_2026-06-14_ab855e7e/D-002`` anchor).
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.0,
            use_bias: bool = True,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            kernel_initializer: Union[str, initializers.Initializer] = "glorot_uniform",
            bias_initializer: Union[str, initializers.Initializer] = "zeros",
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the AnchorAttention layer.

        Validates the configuration and creates every sub-layer up front so that
        ``build()`` only has to wire shapes, keeping weight creation deterministic
        for serialization.
        """
        super().__init__(**kwargs)

        # ---------------------------------------------------------------------
        # Parameter validation
        # ---------------------------------------------------------------------
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # R13/A4: adopts the shared validator. Its message is character-for-character
        # what stood here, so the regex pinned at `test_anchor_attention.py:96`
        # (`match="must be divisible"`) still matches and the diagnostic is
        # byte-unchanged. Checked before the swap, not assumed.
        validate_head_divisibility(dim, num_heads)
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # ---------------------------------------------------------------------
        # Store configuration
        # ---------------------------------------------------------------------
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Total width of the concatenated heads. This equals `dim` only in the
        # default case; every Q/K/V projection and every merge-heads reshape must
        # use this value, never `dim`, or a custom head_dim silently mis-packs.
        self.inner_dim = self.num_heads * self.head_dim

        # Scaling factor: 1/sqrt(d_k).
        # R13: was `1.0 / np.sqrt(float(self.head_dim))`. Verified rather than
        # assumed before swapping — `.hex()` compared against the helper across 27
        # realistic head dims (1..512), 0 mismatches, so this is a bit-identical
        # rename and not a numerics change. Adopting also removes this file's only
        # numpy dependency (the import is dropped above). Still a Python float
        # computed in `__init__`, never in `call()`, per
        # `plan_2026-06-14_33b77a7a/D-002`.
        self.scale = compute_attention_scale(self.head_dim)

        # ---------------------------------------------------------------------
        # Create sub-layers
        # ---------------------------------------------------------------------
        common_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }

        # Projections for anchor tokens, reused for all tokens in standard mode
        self.query_proj = keras.layers.Dense(
            self.inner_dim,
            name="query_proj",
            **common_kwargs
        )
        self.key_proj = keras.layers.Dense(
            self.inner_dim,
            name="key_proj",
            **common_kwargs
        )
        self.value_proj = keras.layers.Dense(
            self.inner_dim,
            name="value_proj",
            **common_kwargs
        )

        # Separate query projection for the spoke tokens in hierarchical mode.
        # Kept distinct from `query_proj` so the two tiers can learn different
        # read behaviours against the same anchor keys.
        self.query_token_proj = keras.layers.Dense(
            self.inner_dim,
            name="query_token_proj",
            **common_kwargs
        )

        # Output projection: maps the merged heads (inner_dim) back to `dim`
        self.output_proj = keras.layers.Dense(
            self.dim,
            name="output_proj",
            **common_kwargs
        )

        # Probability activation (softmax/sparsemax/etc.)
        self.score_activation = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="score_activation"
        )

        # Dropout layer (optional)
        if self.dropout_rate > 0.0:
            self.dropout_layer = keras.layers.Dropout(
                self.dropout_rate,
                name="dropout"
            )
        else:
            self.dropout_layer = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers.

        Validates the input shape and explicitly builds every sub-layer so that
        all weight variables exist before weight restoration during model loading.

        :param input_shape: Shape tuple of the input tensor, expected as
            ``(batch_size, sequence_length, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If input is not 3D or the last dimension does not
            match dim.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Input must be 3D, got shape {input_shape}")

        if input_shape[-1] is not None and input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) "
                f"must match dim ({self.dim})"
            )

        input_shape = tuple(input_shape)

        # Q/K/V projections consume the raw input
        self.query_proj.build(input_shape)
        self.key_proj.build(input_shape)
        self.value_proj.build(input_shape)
        self.query_token_proj.build(input_shape)

        # The output projection consumes the merged heads, not the raw input
        self.output_proj.build(input_shape[:-1] + (self.inner_dim,))

        # Build the probability layer against a representative score shape. The
        # key axis is `seq_len` here; in hierarchical mode it is `num_anchors`.
        # This is only sound because the supported activations are agnostic to
        # the length of that axis (see "Known limitations" in the class docstring).
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.score_activation.build(score_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(score_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: Optional[int] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply anchor-based attention to the input tensor.

        Routes to either standard self-attention or hierarchical anchor attention
        depending on ``num_anchor_tokens``.

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param num_anchor_tokens: Optional integer specifying how many leading
            tokens act as anchors. If None, applies standard self-attention to all
            tokens. Defaults to None.
        :type num_anchor_tokens: Optional[int]
        :param training: Boolean indicating training mode for dropout.
            Defaults to None.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor

        :raises ValueError: If num_anchor_tokens is not positive.
        """
        if num_anchor_tokens is None:
            return self._standard_attention(x, training)

        # A non-positive anchor count yields an empty key/value set, which makes
        # the normalized scores NaN rather than failing; reject it here instead.
        if num_anchor_tokens <= 0:
            raise ValueError(
                f"num_anchor_tokens must be positive when set, got "
                f"{num_anchor_tokens}. Pass None for standard self-attention."
            )

        return self._hierarchical_attention(x, num_anchor_tokens, training)

    def _standard_attention(
            self,
            x: keras.KerasTensor,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Apply standard multi-head self-attention over all tokens.

        Every token attends to every other token, at ``O(N^2)`` cost.

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]

        # Linear projections: (batch, seq, inner_dim)
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # Split heads: (batch, seq, num_heads, head_dim)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Move heads ahead of the sequence: (batch, num_heads, seq, head_dim)
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product scores: (batch, heads, seq, seq)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale

        # Normalize scores into attention weights
        attn_weights = self.score_activation(scores)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # Weighted sum of values: (batch, heads, seq, head_dim)
        out = ops.matmul(attn_weights, v)

        # Merge heads: (batch, seq, heads, head_dim) -> (batch, seq, inner_dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.inner_dim))

        return self.output_proj(out)

    def _hierarchical_attention(
            self,
            x: keras.KerasTensor,
            num_anchor_tokens: int,
            training: Optional[bool]
    ) -> keras.KerasTensor:
        """
        Apply the hierarchical anchor-query attention pattern.

        Anchor tokens perform full self-attention among themselves; query tokens
        cross-attend only to the anchors. Both tiers are scored in a single matmul
        against the shared anchor keys.

        :param x: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type x: keras.KerasTensor
        :param num_anchor_tokens: Number of leading tokens treated as anchors.
        :type num_anchor_tokens: int
        :param training: Boolean indicating training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor

        :raises ValueError: If the sequence dimension is not statically known.
        """
        batch_size = ops.shape(x)[0]
        # DECISION plan_2026-06-14_ab855e7e/D-002: hierarchical mode needs a static
        # sequence length — the `num_anchor_tokens >= seq_len` branch below is a
        # Python-bool decision that crashes under @tf.function when seq_len is a
        # dynamic ops.shape() tensor (static-shape defect class; capsule/PFA
        # precedent). Fail loud on None; batch stays dynamic. Do NOT revert to
        # ops.shape for the sequence dim here. _standard_attention (no branch on
        # seq_len) is intentionally left dynamic-safe.
        seq_len = x.shape[1]
        if seq_len is None:
            raise ValueError(
                "AnchorAttention hierarchical mode (num_anchor_tokens set) "
                "requires a statically-known sequence length; got None. Provide "
                "inputs with a fixed sequence dimension, or use standard mode "
                "(num_anchor_tokens=None)."
            )

        # Every token is an anchor: the bottleneck is vacuous, so take the
        # cheaper path that also gives query tokens their full attention.
        if num_anchor_tokens >= seq_len:
            return self._standard_attention(x, training)

        # Positional split into hubs and spokes
        anchor_tokens = x[:, :num_anchor_tokens, :]
        query_tokens = x[:, num_anchor_tokens:, :]
        num_query_tokens = seq_len - num_anchor_tokens

        # -----------------------------------------------------------------
        # Anchor tier: full Q, K, V. Anchors alone supply keys and values, so
        # this is the only tier that writes into the global summary.
        # -----------------------------------------------------------------
        anchor_q = self.query_proj(anchor_tokens)
        anchor_k = self.key_proj(anchor_tokens)
        anchor_v = self.value_proj(anchor_tokens)

        anchor_q = ops.reshape(
            anchor_q,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )
        anchor_k = ops.reshape(
            anchor_k,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )
        anchor_v = ops.reshape(
            anchor_v,
            (batch_size, num_anchor_tokens, self.num_heads, self.head_dim)
        )

        # (batch, heads, num_anchors, head_dim)
        anchor_q = ops.transpose(anchor_q, (0, 2, 1, 3))
        anchor_k = ops.transpose(anchor_k, (0, 2, 1, 3))
        anchor_v = ops.transpose(anchor_v, (0, 2, 1, 3))

        # -----------------------------------------------------------------
        # Query tier: Q only, from its own projection. No K/V is computed here,
        # which is where the (N-K)^2 term disappears.
        # -----------------------------------------------------------------
        query_q = self.query_token_proj(query_tokens)
        query_q = ops.reshape(
            query_q,
            (batch_size, num_query_tokens, self.num_heads, self.head_dim)
        )
        query_q = ops.transpose(query_q, (0, 2, 1, 3))

        # -----------------------------------------------------------------
        # Concatenate queries as [anchors ; queries]. This order matches the
        # original token order, so the output needs no re-scatter.
        # Shape: (batch, heads, seq_len, head_dim)
        # -----------------------------------------------------------------
        combined_q = ops.concatenate([anchor_q, query_q], axis=2)

        # -----------------------------------------------------------------
        # All tokens attend only to anchors: (batch, heads, seq_len, num_anchors)
        # -----------------------------------------------------------------
        scores = ops.matmul(
            combined_q,
            ops.transpose(anchor_k, (0, 1, 3, 2))
        ) * self.scale

        attn_weights = self.score_activation(scores)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights, training=training)

        # attn_weights: (batch, heads, seq_len, num_anchors)
        # anchor_v:     (batch, heads, num_anchors, head_dim)
        # out:          (batch, heads, seq_len, head_dim)
        out = ops.matmul(attn_weights, anchor_v)

        # Merge heads: (batch, seq, heads, head_dim) -> (batch, seq, inner_dim)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch_size, seq_len, self.inner_dim))

        return self.output_proj(out)

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute output shape, identical to the input shape in both modes.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization, includes all constructor parameters.

        Note that ``num_anchor_tokens`` is a call argument rather than constructor
        state and is therefore not captured here.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "probability_type": self.probability_type,
            "probability_config": self.probability_config,
            "kernel_initializer": keras.initializers.serialize(
                self.kernel_initializer
            ),
            "bias_initializer": keras.initializers.serialize(
                self.bias_initializer
            ),
            "kernel_regularizer": keras.regularizers.serialize(
                self.kernel_regularizer
            ),
            "bias_regularizer": keras.regularizers.serialize(
                self.bias_regularizer
            ),
        })
        return config

# ---------------------------------------------------------------------