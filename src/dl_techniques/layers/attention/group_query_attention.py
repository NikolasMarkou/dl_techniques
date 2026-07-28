"""
Grouped Query Attention (GQA) Implementation with Rotary Position Embeddings.

This module implements Grouped Query Attention, an efficient attention mechanism that
reduces the computational and memory requirements of multi-head attention by sharing
key and value projections across multiple query heads while maintaining model quality.

Architecture:
    Standard Multi-Head Attention uses ``num_heads`` query, key, and value
    projections. Grouped Query Attention reduces this to ``num_heads`` query
    projections but only ``num_kv_heads`` key and value projections (where
    ``num_kv_heads < num_heads``), creating groups where
    ``group_size = num_heads // num_kv_heads``. Each K,V pair is shared across
    ``group_size`` query heads by repeating along the head dimension.

    RoPE, score normalization and the optional QK-norm are shared components,
    not local implementations — see the ``[REUSE]`` note on the class below.
    This class is also the base of ``mobile_mqa.MobileMQA``.

Foundational Mathematics:
    The attention computation follows::

        Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V

    with optional Rotary Position Embeddings (RoPE) applied to Q and K before
    scoring, and ``d_k = dim // num_heads``. The KV cache shrinks by exactly
    ``num_heads / num_kv_heads`` because only the ``num_kv_heads`` distinct K,V
    heads are stored; the repeat to ``num_heads`` happens at score time and is
    never materialized in the cache.

References:
    - Ainslie, J., et al. (2023). "GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints." https://arxiv.org/abs/2305.13245

    - Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need."
      https://arxiv.org/abs/1911.02150 (Multi-Query Attention predecessor)

    - Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
      Embedding." https://arxiv.org/abs/2104.09864 (RoPE integration)
"""

# ---------------------------------------------------------------------

import keras
from keras import ops
from typing import Optional, Union, Any, Dict, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from .common import apply_attention_mask, compute_attention_scale, validate_head_divisibility
from ..activations import ProbabilityOutput
from ..embedding.rotary_position_embedding import RotaryPositionEmbedding
from ..norms.factory import create_normalization_layer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class GroupedQueryAttention(keras.layers.Layer):
    """
    Grouped Query Attention layer with optional rotary position embeddings.

    This layer implements an efficient attention mechanism that reduces computational
    and memory costs by sharing key-value projections across multiple query heads.
    It supports both 3D (sequence) and 4D (vision) inputs.

    The key insight is that ``num_heads % num_kv_heads == 0``, creating groups where
    each K,V pair is shared across ``group_size = num_heads // num_kv_heads`` query heads.
    This reduces memory usage for K,V caches in autoregressive generation while
    maintaining most of the representational power of full multi-head attention.

    **[REUSE]** Three responsibilities are delegated rather than reimplemented:

    -   Rotary position embeddings come from the shared
        :class:`~dl_techniques.layers.embedding.rotary_position_embedding.RotaryPositionEmbedding`.
    -   Score normalization goes through the shared
        :class:`~dl_techniques.layers.activations.ProbabilityOutput` layer, so
        ``probability_type`` selects softmax / sparsemax / adaptive without any
        branching in ``call()``.
    -   Optional QK-norm layers come from
        :func:`~dl_techniques.layers.norms.factory.create_normalization_layer`.

    ``mobile_mqa.MobileMQA`` subclasses this layer. It reuses ``w_q``/``w_k``/
    ``w_v``/``w_o``, ``self.scale``, ``self.attn_prob``, ``self.dropout`` and this
    class's ``compute_output_shape()``, and overrides only ``call()``. Changing
    any of those attribute NAMES is a breaking change for the subclass.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │                  GroupedQueryAttention                       │
        │                                                              │
        │  Input [B, seq, dim] or [B, H, W, dim]                       │
        │         │                                                    │
        │         ├──────────────────┬──────────────────┐              │
        │         ▼                  ▼                  ▼              │
        │  ┌────────────┐    ┌────────────┐    ┌────────────┐          │
        │  │  Q_proj    │    │  K_proj    │    │  V_proj    │          │
        │  │ (num_heads)│    │(num_kv_hds)│    │(num_kv_hds)│          │
        │  └─────┬──────┘    └─────┬──────┘    └─────┬──────┘          │
        │        │                 │                 │                 │
        │        ▼                 ▼                 │                 │
        │  ┌────────────────────────────┐            │                 │
        │  │  (Optional) RoPE(Q, K)     │            │                 │
        │  └────────────┬───────────────┘            │                 │
        │               │                            │                 │
        │        ┌──────┴──────┐                     │                 │
        │        ▼             ▼                     ▼                 │
        │     Q [H]      K,V repeated ──────► K,V [H]                  │
        │        │         (group broadcast)    │                      │
        │        └──────┬───────────────────────┘                      │
        │               ▼                                              │
        │     scores = Q @ K^T / sqrt(d_k)                             │
        │               │                                              │
        │               ▼                                              │
        │         [+ attention_mask]                                   │
        │               │                                              │
        │               ▼                                              │
        │        softmax ──► dropout ──► weights @ V                   │
        │               │                                              │
        │               ▼                                              │
        │        Output Projection                                     │
        │               │                                              │
        │               ▼                                              │
        │  Output [B, seq, dim] or [B, H, W, dim]                      │
        └──────────────────────────────────────────────────────────────┘

    :param dim: Integer, input/output dimension (embedding size). Must be positive and
        divisible by num_heads.
    :type dim: int
    :param num_heads: Integer, number of attention heads for queries. Must be positive.
    :type num_heads: int
    :param num_kv_heads: Integer, number of key-value heads. Must be positive and divide
        num_heads evenly for grouping.
    :type num_kv_heads: int
    :param max_seq_len: Integer, maximum sequence length for positional embeddings.
        Must be positive. Defaults to 2048.
    :type max_seq_len: int
    :param dropout_rate: Float, dropout rate applied to attention weights.
        Must be between 0.0 and 1.0. Defaults to 0.0.
    :type dropout_rate: float
    :param rope_percentage: Float, fraction of head dimensions to apply rotary
        embeddings to. If 0.0, RoPE is disabled. Defaults to 1.0.
    :type rope_percentage: float
    :param rope_theta: Float, base frequency for rotary position embeddings.
        Must be positive. Defaults to 10000.0.
    :type rope_theta: float
    :param use_bias: Boolean, whether to use bias in linear projections.
        Defaults to False.
    :type use_bias: bool
    :param kernel_initializer: String or initializer instance, initializer for
        kernel weights. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: String or initializer instance, initializer for
        bias weights. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :type kernel_regularizer: Optional[keras.regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: Optional[keras.regularizers.Regularizer]
    :param probability_type: String identifier for the attention-score
        normalization strategy, forwarded to :class:`ProbabilityOutput` as its
        ``probability_type``. One of ``"softmax"``, ``"sparsemax"``,
        ``"threshmax"``, ``"adaptive"`` and their aliases. Defaults to
        ``"softmax"``. Routing/hierarchical variants are rejected: they consume
        features rather than logits.
    :type probability_type: str
    :param probability_config: Optional dictionary forwarded to
        :class:`ProbabilityOutput` as ``type_config``. Defaults to ``None``.
    :type probability_config: Optional[Dict[str, Any]]
    :param qk_norm_type: Optional normalization type applied per-head to Q and K
        before scoring, forwarded to :func:`create_normalization_layer`. K is
        normalized in its native ``num_kv_heads`` shape, before grouping. ``None``
        disables QK-norm. Defaults to ``None``.
    :type qk_norm_type: Optional[str]
    :param qk_norm_kwargs: Optional keyword arguments forwarded to
        :func:`create_normalization_layer` for both Q and K norms.
        Defaults to ``None``.
    :type qk_norm_kwargs: Optional[Dict[str, Any]]
    :param kwargs: Additional keyword arguments for the Layer parent class.

    :raises ValueError: If ``dim``, ``num_heads``, ``num_kv_heads``,
        ``max_seq_len`` or ``rope_theta`` is not positive.
    :raises ValueError: If ``dim`` is not divisible by ``num_heads``, or if
        ``num_heads`` is not divisible by ``num_kv_heads``.
    :raises ValueError: If ``dropout_rate`` or ``rope_percentage`` is outside
        ``[0, 1]``.
    :raises ValueError: If ``probability_type`` is a routing/hierarchical variant.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int = 2048,
        dropout_rate: float = 0.0,
        rope_percentage: float = 1.0,
        rope_theta: float = 10000.0,
        use_bias: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        probability_type: str = "softmax",
        probability_config: Optional[Dict[str, Any]] = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        self._validate_inputs(dim, num_heads, num_kv_heads, max_seq_len,
                            dropout_rate, rope_percentage, rope_theta)

        # Validate probability_type — GQA expects logits, not features.
        _invalid_prob_types = (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        )
        if probability_type in _invalid_prob_types:
            raise ValueError(
                f"probability_type='{probability_type}' is not supported by "
                f"GroupedQueryAttention; these expect features not logits. "
                f"Use one of: 'softmax', 'sparsemax', 'adaptive', etc."
            )

        # Store ALL configuration parameters
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.rope_percentage = rope_percentage
        self.rope_theta = rope_theta
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs

        # Derived parameters
        self.head_dim = self.dim // self.num_heads
        self.num_groups = self.num_heads // self.num_kv_heads

        # DECISION plan_2026-06-14_ab855e7e/D-001: static attention scale as a
        # Python float (math.sqrt, NOT ops.sqrt on a cast scalar — D-002 pattern).
        # Inherited by MobileMQA. Do NOT revert to ops.sqrt.
        #
        # R13: the expression now lives in `common.compute_attention_scale`, which IS
        # `1.0 / math.sqrt(float(head_dim))` — verified repr-identical for every
        # realistic head_dim, so `self.scale` is bit-identical and `MobileMQA`, which
        # reads this attribute in its own `call()`, is unaffected. The anchor above
        # still governs: Python float, computed in `__init__`, never in `call()`.
        self.scale = compute_attention_scale(self.head_dim)

        # CREATE all sub-layers in __init__
        self.w_q = keras.layers.Dense(
            self.num_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_q'
        )

        self.w_k = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_k'
        )

        self.w_v = keras.layers.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_v'
        )

        self.w_o = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='w_o'
        )

        # Attention dropout layer
        self.dropout = keras.layers.Dropout(self.dropout_rate, name='attention_dropout')

        # Rotary position embeddings (only if percentage > 0)
        if self.rope_percentage > 0.0:
            self.rope = RotaryPositionEmbedding(
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                rope_theta=self.rope_theta,
                rope_percentage=self.rope_percentage,
                name='rope'
            )
        else:
            self.rope = None

        # Probability activation for attention scores
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # Optional QK normalization layers
        if self.qk_norm_type is not None:
            self.q_norm = create_normalization_layer(
                self.qk_norm_type, name="q_norm", **(self.qk_norm_kwargs or {})
            )
            self.k_norm = create_normalization_layer(
                self.qk_norm_type, name="k_norm", **(self.qk_norm_kwargs or {})
            )
        else:
            self.q_norm = None
            self.k_norm = None

        logger.info(f"GroupedQueryAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, groups={self.num_groups}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        max_seq_len: int,
        dropout_rate: float,
        rope_percentage: float,
        rope_theta: float
    ) -> None:
        """Validate initialization parameters.

        :param dim: Model dimension.
        :type dim: int
        :param num_heads: Number of query heads.
        :type num_heads: int
        :param num_kv_heads: Number of key-value heads.
        :type num_kv_heads: int
        :param max_seq_len: Maximum sequence length.
        :type max_seq_len: int
        :param dropout_rate: Dropout rate.
        :type dropout_rate: float
        :param rope_percentage: RoPE percentage.
        :type rope_percentage: float
        :param rope_theta: RoPE base frequency.
        :type rope_theta: float
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {rope_theta}")
        # R13: adopts the shared validator. Its message is character-for-character
        # what stood here, so the regex pinned at test_group_query_attention.py:113
        # (and, via inheritance, test_mobile_mqa.py:119) still matches.
        validate_head_divisibility(dim, num_heads)
        # The check below is a DIFFERENT invariant and stays local on purpose: it is
        # about how many query heads share one K,V head (the "grouping" in GQA), not
        # about splitting a model dimension into heads. `common.
        # validate_head_divisibility` documents a head-SPLIT precondition
        # (`(..., dim) -> (..., num_heads, dim // num_heads)`); routing this through
        # it would make that documented contract untrue for a saved line, and its
        # `*_name` kwargs would produce the same text anyway. Pinned at
        # test_group_query_attention.py:117.
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not 0.0 <= rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in [0, 1], got {rope_percentage}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds sub-layers for robust serialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Detect if we need to flatten for 4D inputs during build logic if needed,
        # but Dense layers are generally agnostic to outer dimensions.
        self.w_q.build(input_shape)
        self.w_k.build(input_shape)
        self.w_v.build(input_shape)
        self.w_o.build(input_shape)
        self.dropout.build(input_shape)

        if self.rope is not None:
            # RoPE expects (batch, num_heads, seq_len, head_dim)
            # We estimate shapes based on input rank
            batch_size = input_shape[0] if input_shape[0] is not None else 1

            if len(input_shape) == 4:
                # 4D Input: (B, H, W, C) -> seq_len = H*W
                h = input_shape[1] if input_shape[1] is not None else self.max_seq_len
                w = input_shape[2] if input_shape[2] is not None else 1
                seq_len = h * w
            else:
                # 3D Input: (B, S, C)
                seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len

            rope_input_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            self.rope.build(rope_input_shape)

        # Estimate seq_len for sub-layer build shapes
        batch_size = input_shape[0] if input_shape[0] is not None else 1
        if len(input_shape) == 4:
            h = input_shape[1] if input_shape[1] is not None else self.max_seq_len
            w = input_shape[2] if input_shape[2] is not None else 1
            seq_len = h * w
        else:
            seq_len = input_shape[1] if input_shape[1] is not None else self.max_seq_len

        # Build attention probability layer with score shape (B, num_heads, seq, seq)
        score_shape = (batch_size, self.num_heads, seq_len, seq_len)
        self.attn_prob.build(score_shape)

        # Build QK normalization layers if present.
        # Q has num_heads, K has num_kv_heads (group-query attention).
        if self.q_norm is not None:
            q_norm_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
            k_norm_shape = (batch_size, self.num_kv_heads, seq_len, self.head_dim)
            self.q_norm.build(q_norm_shape)
            self.k_norm.build(k_norm_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply grouped query attention.

        Supports 3D ``(B, S, D)`` and 4D ``(B, H, W, D)`` inputs. For 4D inputs,
        spatial dimensions are flattened before attention and restored afterward.

        :param inputs: Input tensor of shape ``(B, S, D)`` or ``(B, H, W, D)``.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training mode for dropout.
        :type training: Optional[bool]
        :param attention_mask: Optional attention mask tensor.
        :type attention_mask: Optional[keras.KerasTensor]
        :param return_attention_weights: If True, returns attention weights alongside output.
        :type return_attention_weights: bool
        :return: Output tensor with same shape as input, or tuple of
            (output, attention_weights) if ``return_attention_weights=True``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]
        """
        input_shape = ops.shape(inputs)
        rank = len(inputs.shape)
        batch_size = input_shape[0]

        # 1. Project to Q, K, V
        # Dense layers broadcast over spatial dims, so this works for 3D and 4D
        q = self.w_q(inputs, training=training)
        k = self.w_k(inputs, training=training)
        v = self.w_v(inputs, training=training)

        # 2. Flatten spatial dimensions if 4D
        if rank == 4:
            height, width = input_shape[1], input_shape[2]
            seq_len = height * width
            q = ops.reshape(q, (batch_size, seq_len, -1))
            k = ops.reshape(k, (batch_size, seq_len, -1))
            v = ops.reshape(v, (batch_size, seq_len, -1))
        else:
            seq_len = input_shape[1]

        # 3. Reshape for Multi-Head Attention
        # Q: (B, S, H, D_h)
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # Transpose to (B, H, S, D_h) for efficient attention
        q = ops.transpose(q, (0, 2, 1, 3))
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # 4. Apply RoPE (Optional)
        if self.rope is not None:
            q = self.rope(q, training=training)
            k = self.rope(k, training=training)

        # 4b. Optional QK normalization (applied per-head before scoring).
        # K is normalized in its native num_kv_heads shape, prior to grouping.
        if self.q_norm is not None:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # 5. Grouping: Repeat K, V to match Q head count
        if self.num_groups > 1:
            k = ops.repeat(k, self.num_groups, axis=1)
            v = ops.repeat(v, self.num_groups, axis=1)

        # 6. Scaled Dot-Product Attention
        # (B, H, S, D_h) @ (B, H, D_h, S) -> (B, H, S, S)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))
        scores = scores * ops.cast(self.scale, scores.dtype)  # D-001: precomputed

        if attention_mask is not None:
            scores = self._apply_mask(scores, attention_mask)

        attention_weights = self.attn_prob(scores)
        attention_weights = self.dropout(attention_weights, training=training)

        # 7. Apply weights to Values
        # (B, H, S, S) @ (B, H, S, D_h) -> (B, H, S, D_h)
        out = ops.matmul(attention_weights, v)

        # 8. Restore Output Shape
        out = ops.transpose(out, (0, 2, 1, 3))  # (B, S, H, D_h)
        out = ops.reshape(out, (batch_size, seq_len, self.dim))  # (B, S, D)

        # Final projection
        output = self.w_o(out, training=training)

        # 9. Reshape back to 4D if input was 4D
        if rank == 4:
            output = ops.reshape(output, (batch_size, height, width, self.dim))

        if return_attention_weights:
            return output, attention_weights
        return output

    def _apply_mask(
            self,
            scores: keras.KerasTensor,
            mask: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Broadcast and apply attention mask to scores.

        :param scores: Attention scores of shape ``(B, H, S, S)``.
        :type scores: keras.KerasTensor
        :param mask: Attention mask tensor.
        :type mask: keras.KerasTensor
        :return: Masked scores tensor.
        :rtype: keras.KerasTensor
        """
        # R13 cross-reference — this helper is deliberately NOT shared.
        #
        # It is the THIRD variant of mask broadcasting in this package and the most
        # different of the three:
        #   * `multi_head_cross_attention.py::_apply_attention_mask` — casts first,
        #     then a double `ops.expand_dims` for the 2D case, and relies on
        #     broadcasting over the head axis;
        #   * `multi_head_latent_attention.py::_apply_attention_mask` — expands
        #     first, casts after, probes rank with `len(ops.shape(mask))`;
        #   * HERE — `ops.reshape` (not `expand_dims`) for the 2D case, and then an
        #     explicit `ops.repeat` that MATERIALIZES the head axis to `num_heads`
        #     instead of broadcasting it. That repeat is a real extra op with a real
        #     memory cost; it exists because 4D inputs arrive with a flattened
        #     `H*W` sequence axis and broadcast alone proved fragile there.
        #
        # WHAT NOT TO DO: do not unify the three. Any single body must choose one
        # cast order, one rank probe, and either broadcast or repeat — silently
        # rewriting the traced graph of the two layers it did not come from.
        #
        mask_shape = ops.shape(mask)

        # Handle 2D padding mask (B, S)
        if len(mask_shape) == 2:
            mask = ops.reshape(mask, (mask_shape[0], 1, 1, mask_shape[1]))
        # Handle 3D causal/combined mask (B, S, S)
        elif len(mask_shape) == 3:
            mask = ops.expand_dims(mask, axis=1)

        # Broadcast head dim if necessary
        if len(ops.shape(mask)) == 4 and ops.shape(mask)[1] == 1:
            mask = ops.repeat(mask, self.num_heads, axis=1)

        # THIS SITE'S MASK POLARITY, passed through verbatim: `mask` is a `1 = keep`
        # predicate, so it IS the keep predicate `apply_attention_mask` wants. Do NOT
        # "normalize" it into a `> 0` comparison or invert it — the helper performs no
        # polarity inference by design, so an inversion here raises nothing, changes
        # no shape and stays finite; the layer would just attend to the padding.
        # `TestGroupedQueryAttentionMaskPolarity` is the only guard that can see it.
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-007
        # `out_dtype` is pinned to the SCORES' own dtype, so the biased scores return
        # in the compute dtype (fp16 under `mixed_float16`), where `MASK_BIAS_VALUE`
        # is `-inf` again. That is deliberate and is NOT the bug being fixed:
        #   * The bug is `0 * -inf = NaN` at every UNMASKED position, produced by the
        #     ARITHMETIC form this line replaces. `ops.where` inside `mask_dtype(...)`
        #     removes that product structurally, and a row keeping >= 1 key softmaxes
        #     correctly with `-inf` entries. MEASURED on unfixed HEAD
        #     (B=2, N=64, D=64, H=4, kv=2): an ALL-ONES mask — masking NOTHING — gave
        #     8192/8192 NaN.
        #   * Do NOT "improve" this to `out_dtype=None` (stay in float32) hoping to
        #     also rescue a FULLY-MASKED row. It cannot: the next consumer is
        #     `self.attn_prob`, a Keras layer with autocasting ON, MEASURED to see a
        #     float32 input inside its own `call()` as float16 — so the promotion is
        #     silently undone and all that remains is a wider, slower add. Pinned by
        #     `TestGroupedQueryAttentionMaskHazardIsReal::
        #     test_the_probability_sublayer_autocasts_a_float32_input`.
        #   * A FULLY-MASKED query row is a SEPARATE hazard that no `out_dtype` choice can
        #     touch. It is handled by the rescue below (D-009), not here.
        # See decisions.md D-007 (plan-2026-07-27T183600-b4ef45f0).
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-009
        # The fully-masked-row rescue IS applied here, and it supersedes the "not applied
        # here" note above: a query row that keeps NOTHING is treated as keeping EVERYTHING,
        # so the all-`-inf` row is never FORMED and no NaN gradient is created either. It
        # arrives via `apply_attention_mask`'s DEFAULT `rescue_axis=-1` — step 4c flipped
        # the step-4b opt-in default on the user's direction ("I care about correctness, not
        # backwards compatibility").
        #
        # DECISION plan-2026-07-27T183600-b4ef45f0/D-017
        # The axis is DERIVED from this layer's own `probability_config` rather than left
        # to the helper's `-1` default: `ProbabilityOutput` reads its softmax `axis` from
        # `type_config` (`activations/probability_output.py:180`) and this layer forwards
        # `probability_config` VERBATIM, so a caller can move the reduction axis and the
        # pre-step-10 "checked, not assumed" claim held only for the DEFAULT config.
        # MEASURED at the sibling `gated_attention` under `mixed_float16` with
        # `probability_config={"axis": -2}` and a dead KEY COLUMN: 8192/8192 non-finite.
        # WHAT NOT TO DO: do NOT restore a bare `-1` (correct only while the caller leaves
        # the config alone) and do NOT read this as the rank/shape INFERENCE the D-009
        # anchor in `common.py` forbids — this reads the site's own declared config.
        # The full argument lives at the D-017 anchors in `common.py` and
        # `gated_attention.py`. See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
        #
        # WHAT NOT TO DO: do NOT pass `rescue_axis=None` to "get the loud NaN back" — the
        # user ruled the finite-garbage semantics package-wide on 2026-07-28, and opting out
        # also restores the NaN GRADIENT on that row; do NOT move the rescue after the
        # softmax (`ops.where(row_keeps, w, 0)` still contributes `0 * NaN` in the backward
        # pass). The full argument lives at the D-009 / D-008 anchors in `common.py`.
        # See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0).
        scores_dtype = keras.backend.standardize_dtype(scores.dtype)
        return apply_attention_mask(
            scores,
            mask,
            out_dtype=scores_dtype,
            rescue_axis=(self.probability_config or {}).get("axis", -1),
        )

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape, same as input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Configuration dictionary.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'rope_percentage': self.rope_percentage,
            'rope_theta': self.rope_theta,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
        })
        return config

# ---------------------------------------------------------------------
