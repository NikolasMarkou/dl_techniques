"""
Gated multi-head attention with rotary position embeddings.

This module defines ``GatedAttention``. Its forward pass differs from plain
multi-head attention in three ways. Query, key and value are each normalized
independently with zero-centered RMSNorm before the head split, value
included, not just query and key. Partial rotary position embedding (RoPE)
rotates only the first ``rope_percentage`` fraction of each head's
dimensions, leaving the rest untouched. And the block gates its own output:
a sigmoid projection of the attention output multiplies that same output
elementwise, rather than being driven by the layer's input. Grouped-query
attention is supported through ``num_kv_heads``.

A caller needs ``dim`` divisible by ``num_heads`` unless ``head_dim`` is
given explicitly, and ``rope_percentage`` in ``(0, 1]``.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
    - Su et al., 2021. RoFormer: Enhanced Transformer with Rotary Position
      Embedding. (https://arxiv.org/abs/2104.09864)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization. (https://arxiv.org/abs/1910.07467)
    - Touvron et al., 2023. LLaMA: Open and Efficient Foundation Language
      Models. (https://arxiv.org/abs/2302.13971)
"""

import keras
from typing import Optional, Union, Tuple, Dict, Any

from dl_techniques.utils.logger import logger
from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.embedding import create_embedding_layer
from dl_techniques.layers.activations import ProbabilityOutput, resolve_activation_layer

from .common import apply_attention_mask, compute_attention_scale
from dl_techniques.utils.keras_registration import register_dl_technique

@register_dl_technique("dl_techniques.layers.attention.gated_attention")
class GatedAttention(keras.layers.Layer):
    """
    Gated multi-head attention with Zero-Centered RMSNorm, partial RoPE, and sigmoid output gating.

    Combines input linear projection, separate Q/K/V projections normalized
    with Zero-Centered RMSNorm, partial Rotary Position Embedding on Q and K,
    scaled dot-product attention, and a sigmoid gating mechanism. The forward
    pass computes ``output = sigma(W_gate(A')) * A'`` where
    ``A' = Attention(RoPE(RMSNorm(Q)), RoPE(RMSNorm(K)), RMSNorm(V))``.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │  Input [B, S, dim]  ►  input_linear Dense(dim)  ►  x         │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  q = q_linear(x)   [B, S, attention_dim]                     │
        │  k = k_linear(x)   [B, S, kv_dim]                            │
        │  v = v_linear(x)   [B, S, kv_dim]      kv_dim = H_kv * d     │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  q_norm, k_norm, v_norm — zero-centered RMSNorm on all       │
        │  three, before the head reshape. V is normed too, which      │
        │  the sibling layers do not do.                               │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  reshape   Q [B, S, H, d]      K, V [B, S, H_kv, d]          │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  partial RoPE on Q and K only, in the (B, H, S, d) frame.    │
        │  V is never rotated.                                         │
        │                                                              │
        │    head_dim d                                                │
        │    ├──── rope_dim ─────┼──── d - rope_dim ────┤              │
        │    │ rotated by R_m    │ passed through as-is │              │
        │                                                              │
        │  rope_dim = int(d * rope_percentage), rounded down to even   │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  GQA expand — only when num_kv_groups > 1                    │
        │  keras.ops.repeat(K, num_kv_groups, axis=2), V likewise      │
        │  repeat, not tile: copies of one K/V head stay adjacent      │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  scaled_dot_product_attention() — a method of this class     │
        │    transpose to [B, H, S, d]  ►  S = q kᵀ * (1/sqrt(d))      │
        │    if attention_mask: the 1 = keep predicate passes through  │
        │      unchanged; a row that keeps nothing is rescued on the   │
        │      axis probability_config declares                        │
        │    A = attn_prob(S)  ►  dropout if training  ►  out = A v    │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  reshape [B, S, attention_dim]  ►  output_proj, only when    │
        │  attention_dim != dim (otherwise no such sub-layer)          │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼ y
        ┌──────────────────────────────────────────────────────────────┐
        │  output gate — computed from y, not from the layer input     │
        │                                                              │
        │   y ──┬──► output_gate_linear ──► gate_activation ──► g      │
        │       │                           (sigmoid by default)       │
        │       └──────────────────────────────────────────► y         │
        │                                                              │
        │                    Output = g ⊗ y   (elementwise)            │
        └───────────────────────────────┬──────────────────────────────┘
                                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output [B, S, dim]                                          │
        └──────────────────────────────────────────────────────────────┘

    :param dim: Model dimension size. Must be positive and divisible by
        ``num_heads`` if ``head_dim`` is not specified.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param num_kv_heads: Number of key/value heads (grouped-query attention).
        ``None`` (default) means one K/V head per query head, i.e. plain
        multi-head attention. Must divide ``num_heads``; each K/V head is then
        shared by ``num_heads // num_kv_heads`` query heads, shrinking the K/V
        projections and the KV cache by that factor.
    :type num_kv_heads: int or None
    :param head_dim: Optional dimension per attention head. If ``None``,
        defaults to ``dim // num_heads``.
    :type head_dim: int or None
    :param max_seq_len: Maximum sequence length for RoPE precomputation.
        Defaults to 4096.
    :type max_seq_len: int
    :param rope_percentage: Fraction of head dimensions to apply RoPE to
        (partial RoPE). Must be in ``(0, 1]``. Defaults to 0.5.
    :type rope_percentage: float
    :param dropout_rate: Dropout rate for attention weights. Must be in
        ``[0, 1]``. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias terms in linear layers.
        Defaults to ``False``.
    :type use_bias: bool
    :param kernel_initializer: Initializer for linear layer weights.
        Defaults to ``'glorot_uniform'``.
    :type kernel_initializer: str or keras.initializers.Initializer
    :param bias_initializer: Initializer for bias weights (if used).
        Defaults to ``'zeros'``.
    :type bias_initializer: str or keras.initializers.Initializer
    :param kernel_regularizer: Optional regularizer for linear layer weights.
    :type kernel_regularizer: keras.regularizers.Regularizer or None
    :param bias_regularizer: Optional regularizer for bias weights.
    :type bias_regularizer: keras.regularizers.Regularizer or None
    :param probability_type: Strategy used by :class:`ProbabilityOutput` to turn
        raw attention scores into weights. Defaults to ``"softmax"``, which
        reproduces standard attention. The score-level routing/hierarchical
        variants (``"routing"``, ``"deterministic_routing"``, ``"hierarchical"``,
        ``"hierarchical_routing"``) are rejected at construction time: they
        alter the shape/semantics of their input in ways this layer does not
        support.
    :type probability_type: str
    :param probability_config: Optional keyword arguments forwarded to the
        :class:`ProbabilityOutput` constructor (as its ``type_config``). ``None``
        means the strategy's own defaults.
    :type probability_config: Dict[str, Any] or None
    :param qk_norm_type: Normalization applied independently to ``Q`` and ``K``
        before the score matmul, resolved through ``norms.factory``. Defaults to
        ``"zero_centered_rms_norm"`` — this layer normalizes Q/K/V rather than
        pre-normalizing the block input, so this is an architectural default, not
        a tweak. Pass a different registered normalization name to change it.
    :type qk_norm_type: str
    :param qk_norm_kwargs: Optional keyword arguments forwarded to the
        QK-normalization layer's constructor. ``None`` means that layer's
        defaults.
    :type qk_norm_kwargs: Dict[str, Any] or None
    :param gate_activation_type: Activation applied to the gate projection,
        resolved through ``activations.factory``. Defaults to ``"sigmoid"``,
        which is what makes the gate a ``[0, 1]`` per-feature multiplier; any
        substitute that leaves the ``[0, 1]`` range changes the gate from an
        attenuator into a rescaler.
    :type gate_activation_type: str
    :param gate_activation_args: Optional keyword arguments forwarded to the gate
        activation layer's constructor. ``None`` means that layer's defaults.
    :type gate_activation_args: Dict[str, Any] or None
    :param kwargs: Additional arguments for the ``Layer`` base class.

    :raises ValueError: If ``dim`` is not positive or not divisible by
        ``num_heads`` (when ``head_dim`` is ``None``).
    :raises ValueError: If ``num_heads`` is not positive.
    :raises ValueError: If ``head_dim`` is not positive (when specified).
    :raises ValueError: If ``rope_percentage`` is not in ``(0, 1]``.
    :raises ValueError: If ``dropout_rate`` is not in ``[0, 1]``.
    :raises ValueError: If ``max_seq_len`` is not positive.
    :raises ValueError: If ``probability_type`` is one of the four disallowed
        score-level routing/hierarchical strategies listed above.
    :raises ValueError: If ``num_kv_heads`` is not positive.
    :raises ValueError: If ``num_kv_heads`` exceeds ``num_heads``.
    :raises ValueError: If ``num_heads`` is not divisible by ``num_kv_heads``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            num_kv_heads: Optional[int] = None,
            head_dim: Optional[int] = None,
            max_seq_len: int = 4096,
            rope_percentage: float = 0.5,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            probability_type: str = "softmax",
            probability_config: Optional[Dict[str, Any]] = None,
            qk_norm_type: str = "zero_centered_rms_norm",
            qk_norm_kwargs: Optional[Dict[str, Any]] = None,
            gate_activation_type: str = "sigmoid",
            gate_activation_args: Optional[Dict[str, Any]] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create every sub-layer, unbuilt.

        Every argument is documented on the class. Validation runs first, so a
        rejected configuration leaves no half-built layer behind. ``output_proj``
        is the one conditional sub-layer: it exists only when
        ``attention_dim != dim``.
        """
        super().__init__(**kwargs)

        # Validate parameters
        self._validate_inputs(dim, num_heads, head_dim, max_seq_len,
                            rope_percentage, dropout_rate)

        # Validate probability_type: routing/hierarchical variants are not
        # compatible with attention-score normalization (they alter shape /
        # semantics in ways gated_attention does not support).
        _disallowed_prob_types = (
            "routing",
            "deterministic_routing",
            "hierarchical",
            "hierarchical_routing",
        )
        if probability_type in _disallowed_prob_types:
            raise ValueError(
                f"probability_type='{probability_type}' is not supported for "
                f"GatedAttention. Disallowed types: {_disallowed_prob_types}."
            )

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: num_kv_heads defaults
        # to None (plain MHA), never a concrete value -- any non-None default narrows weights and breaks pre-2026-08-15 checkpoints. See decisions.md.
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.num_kv_heads > num_heads:
            raise ValueError(
                f"num_kv_heads ({self.num_kv_heads}) cannot exceed num_heads "
                f"({num_heads})"
            )
        if num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads "
                f"({self.num_kv_heads}); each K/V head is shared by "
                f"num_heads // num_kv_heads query heads"
            )
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.max_seq_len = max_seq_len
        self.rope_percentage = rope_percentage
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.probability_type = probability_type
        self.probability_config = probability_config
        self.qk_norm_type = qk_norm_type
        self.qk_norm_kwargs = qk_norm_kwargs
        self.gate_activation_type = gate_activation_type
        self.gate_activation_args = gate_activation_args

        # A custom head_dim can leave attention_dim != dim, hence output_proj below.
        self.attention_dim = self.num_heads * self.head_dim
        # K and V project to num_kv_heads * head_dim, shrinking the KV cache by num_kv_groups.
        self.kv_dim = self.num_kv_heads * self.head_dim

        # DECISION plan_2026-06-14_ab855e7e/D-001: precompute the scale as a
        # Python float in __init__, not with keras.ops.sqrt in call() -- a cast static int returns a backend tensor that can leak into a scratch graph. See decisions.md.
        self.scale = compute_attention_scale(self.head_dim)

        # Sub-layers are created here, unbuilt; build() builds them.
        # DECISION plan-2026-08-22T035419-a11304c8/D-200: clone the initializer
        # per projection, never pass self.kernel_initializer directly -- a shared instance gives bit-identical Q/K/V kernels. See decisions.md.
        self.input_linear = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="input_linear"
        )

        # QKV projections - project to attention_dim, not dim
        self.q_linear = keras.layers.Dense(
            self.attention_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="q_linear"
        )
        # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: K and V project to
        # kv_dim, not attention_dim -- the narrower width is the GQA KV-cache saving. See decisions.md.
        self.k_linear = keras.layers.Dense(
            self.kv_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="k_linear"
        )
        self.v_linear = keras.layers.Dense(
            self.kv_dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="v_linear"
        )

        # Q/K normalization layers (parameterized via qk_norm_type)
        self.q_norm = create_normalization_layer(
            self.qk_norm_type,
            name="q_norm",
            **(self.qk_norm_kwargs or {}),
        )
        self.k_norm = create_normalization_layer(
            self.qk_norm_type,
            name="k_norm",
            **(self.qk_norm_kwargs or {}),
        )
        self.v_norm = create_normalization_layer(
            'zero_centered_rms_norm',
            epsilon=1e-6,
            use_scale=True,
            name='v_norm'
        )

        # RoPE has no trainable parameters, only precomputed sin/cos tables.
        self.rope = create_embedding_layer(
            'rope',
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            rope_percentage=rope_percentage,
            name='rope'
        )

        # Dropout for attention weights (conditional creation)
        if dropout_rate > 0.0:
            self.dropout = keras.layers.Dropout(dropout_rate, name="attention_dropout")
        else:
            self.dropout = None

        # output_proj exists only when a custom head_dim leaves attention_dim != dim.
        if self.attention_dim != self.dim:
            self.output_proj = keras.layers.Dense(
                self.dim,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                name="output_proj"
            )
        else:
            self.output_proj = None

        # Output gate - always projects to dim
        self.output_gate_linear = keras.layers.Dense(
            self.dim,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_gate_linear"
        )

        # Parameterized attention-probability layer (replaces hardcoded softmax)
        self.attn_prob = ProbabilityOutput(
            probability_type=self.probability_type,
            type_config=self.probability_config,
            name="attn_prob",
        )

        # Parameterized output-gate activation (defaults to sigmoid)
        self.gate_activation = resolve_activation_layer(
            self.gate_activation_type,
            name="gate_activation",
            **(self.gate_activation_args or {}),
        )

        logger.info(f"GatedAttention initialized: dim={dim}, "
                   f"num_heads={num_heads}, head_dim={self.head_dim}, "
                   f"attention_dim={self.attention_dim}")

    def _validate_inputs(
        self,
        dim: int,
        num_heads: int,
        head_dim: Optional[int],
        max_seq_len: int,
        rope_percentage: float,
        dropout_rate: float
    ) -> None:
        """
        Validate initialization parameters.

        :param dim: Model dimension to validate.
        :type dim: int
        :param num_heads: Number of attention heads to validate.
        :type num_heads: int
        :param head_dim: Head dimension to validate (can be ``None``).
        :type head_dim: int or None
        :param max_seq_len: Maximum sequence length to validate.
        :type max_seq_len: int
        :param rope_percentage: RoPE percentage to validate.
        :type rope_percentage: float
        :param dropout_rate: Dropout rate to validate.
        :type dropout_rate: float

        :raises ValueError: If any parameter is invalid.
        """
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        # Not routed through common.validate_head_divisibility: this check is
        # conditional on head_dim is None and its message names the fix (supply head_dim), which the shared helper's message cannot express.
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}) when head_dim is None")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not 0.0 < rope_percentage <= 1.0:
            raise ValueError(f"rope_percentage must be in (0, 1], got {rope_percentage}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization support,
        ensuring all weight variables exist before weight restoration.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input shape, got {input_shape}")

        batch_size, seq_len, features = input_shape

        if features != self.dim:
            raise ValueError(
                f"Input feature dimension ({features}) must match dim ({self.dim})"
            )

        # Build input linear
        self.input_linear.build(input_shape)

        # Compute intermediate shapes
        linear_output_shape = (batch_size, seq_len, self.dim)
        qkv_shape = (batch_size, seq_len, self.attention_dim)
        kv_shape = (batch_size, seq_len, self.kv_dim)

        # Sub-layers with lazy build must be built explicitly here for serialization.
        self.q_linear.build(linear_output_shape)
        self.k_linear.build(linear_output_shape)
        self.v_linear.build(linear_output_shape)

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-071: build k_norm/v_norm at
        # kv_shape, not qkv_shape -- qkv_shape gives scale vectors num_kv_groups times too wide and mis-scales the forward pass. See decisions.md.
        self.q_norm.build(qkv_shape)
        self.k_norm.build(kv_shape)
        self.v_norm.build(kv_shape)

        # RoPE is built in the (B, H, S, head_dim) frame call() actually uses;
        # axis 2 must be the sequence axis.
        rope_input_shape = (batch_size, self.num_heads, seq_len, self.head_dim)
        self.rope.build(rope_input_shape)

        if self.dropout is not None:
            self.dropout.build((batch_size, self.num_heads, seq_len, seq_len))

        if self.output_proj is not None:
            self.output_proj.build((batch_size, seq_len, self.attention_dim))

        # Build output gate
        self.output_gate_linear.build((batch_size, seq_len, self.dim))

        # Build the attention-probability layer with shape
        # (batch, num_heads, seq_len, seq_len).
        self.attn_prob.build(
            (batch_size, self.num_heads, seq_len, seq_len)
        )

        # Build the gate activation (applied on output_gate_linear output of shape
        # (batch, seq_len, dim))
        self.gate_activation.build((batch_size, seq_len, self.dim))

        # Always call parent build at the end
        super().build(input_shape)

    def scaled_dot_product_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Compute scaled dot-product attention.

        :param q: Query tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type q: keras.KerasTensor
        :param k: Key tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :type v: keras.KerasTensor
        :param attention_mask: Optional attention mask of shape
            ``[batch, seq_len]`` or ``[batch, seq_len, seq_len]``.
        :type attention_mask: keras.KerasTensor or None
        :param training: Training mode flag.
        :type training: bool or None
        :return: Attention output tensor of shape
            ``[batch, seq_len, num_heads, head_dim]``.
        :rtype: keras.KerasTensor
        """
        # Transpose to [batch, num_heads, seq_len, head_dim] for attention computation
        q = keras.ops.transpose(q, axes=[0, 2, 1, 3])
        k = keras.ops.transpose(k, axes=[0, 2, 1, 3])
        v = keras.ops.transpose(v, axes=[0, 2, 1, 3])

        # Compute attention scores
        matmul_qk = keras.ops.matmul(q, keras.ops.transpose(k, axes=[0, 1, 3, 2]))

        # Scale by 1/sqrt(head_dim) for numerical stability (precomputed Python
        # float self.scale; see D-001 anchor in __init__).
        scaled_attention_logits = matmul_qk * keras.ops.cast(self.scale, matmul_qk.dtype)

        if attention_mask is not None:
            # The mask can be (batch, seq_len) for padding or (batch, seq_len, seq_len) for causal.
            # We must broadcast it to (batch, num_heads, seq_len, seq_len).
            mask_ndim = keras.ops.ndim(attention_mask)
            if mask_ndim == 2:
                # Padding mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = keras.ops.expand_dims(keras.ops.expand_dims(attention_mask, 1), 1)
            elif mask_ndim == 3:
                # Causal/Combined mask: (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = keras.ops.expand_dims(attention_mask, 1)
            else:
                # Rank 4 or anything else: assume the caller already shaped it
                # to broadcast against (batch, num_heads, seq_len, seq_len).
                mask = attention_mask

            # `getattr(d, "name", None) or str(d)`, not `keras.backend.standardize_dtype`:
            # a Keras-2 residue banned across `src/`, and `str` alone mis-renders a
            # `tf.DType`. Full note and the measured equivalence at `common.py`; D-007.
            logits_dtype = getattr(
                scaled_attention_logits.dtype, "name", None
            ) or str(scaled_attention_logits.dtype)
            scaled_attention_logits = apply_attention_mask(
                scaled_attention_logits,
                mask,
                out_dtype=logits_dtype,
                rescue_axis=(self.probability_config or {}).get("axis", -1),
            )

        # Parameterized attention-probability transform over the key dimension
        attention_weights = self.attn_prob(scaled_attention_logits)

        # Apply dropout during training
        if training and self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Apply attention to values
        output = keras.ops.matmul(attention_weights, v)

        # Transpose back to [batch, seq_len, num_heads, head_dim]
        output = keras.ops.transpose(output, axes=[0, 2, 1, 3])

        return output

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the gated attention layer.

        :param inputs: Input tensor of shape
            ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional attention mask of shape
            ``(batch_size, seq_len)`` or ``(batch_size, seq_len, seq_len)``.
        :type attention_mask: keras.KerasTensor or None
        :param training: Whether in training or inference mode.
        :type training: bool or None
        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        # Input linear projection
        # Shape: (B, S, dim) -> (B, S, dim)
        x = self.input_linear(inputs, training=training)

        # Get batch and sequence dimensions dynamically
        batch_size = keras.ops.shape(x)[0]
        seq_len = keras.ops.shape(x)[1]

        # Generate Q, K, V projections
        # Shape: (B, S, dim) -> (B, S, attention_dim) each, where
        #        attention_dim = num_heads * head_dim (may differ from dim)
        # q is [batch, seq, attention_dim]; k and v are [batch, seq, kv_dim].
        q = self.q_linear(x, training=training)
        k = self.k_linear(x, training=training)
        v = self.v_linear(x, training=training)

        # Apply Zero-Centered RMS Normalization
        q_norm = self.q_norm(q, training=training)
        k_norm = self.k_norm(k, training=training)
        v_norm = self.v_norm(v, training=training)

        # Reshape for multi-head attention and RoPE
        # [batch, seq, attention_dim] -> [batch, seq, num_heads, head_dim]
        # Shape: (B, S, attention_dim) -> (B, S, H, head_dim)
        q_reshaped = keras.ops.reshape(q_norm, (batch_size, seq_len, self.num_heads, self.head_dim))
        k_reshaped = keras.ops.reshape(k_norm, (batch_size, seq_len, self.num_kv_heads, self.head_dim))
        v_reshaped = keras.ops.reshape(v_norm, (batch_size, seq_len, self.num_kv_heads, self.head_dim))

        # DECISION plan-2026-08-14T233721-d4f9beb2/D-083: apply RoPE in the
        # (B, H, S, D) frame via transpose, never pass (B, S, H, D) straight in -- rope reads seq length from axis 2. See decisions.md.
        q_rope = keras.ops.transpose(
            self.rope(keras.ops.transpose(q_reshaped, (0, 2, 1, 3)), training=training),
            (0, 2, 1, 3),
        )
        k_rope = keras.ops.transpose(
            self.rope(keras.ops.transpose(k_reshaped, (0, 2, 1, 3)), training=training),
            (0, 2, 1, 3),
        )

        # GQA expand runs after RoPE, once per K/V head. ops.repeat keeps the
        # copies of one head adjacent; ops.tile would interleave groups and pair each query head with the wrong K/V head.
        if self.num_kv_groups > 1:
            k_rope = keras.ops.repeat(k_rope, self.num_kv_groups, axis=2)
            v_reshaped = keras.ops.repeat(v_reshaped, self.num_kv_groups, axis=2)

        # Apply scaled dot-product attention
        # Shape: 3x (B, S, H, head_dim) -> (B, S, H, head_dim)
        attention_output = self.scaled_dot_product_attention(
            q_rope, k_rope, v_reshaped, attention_mask=attention_mask, training=training
        )

        # Reshape back to [batch, seq, attention_dim]
        # Shape: (B, S, H, head_dim) -> (B, S, attention_dim)
        attention_output = keras.ops.reshape(
            attention_output, (batch_size, seq_len, self.attention_dim)
        )

        if self.output_proj is not None:
            # Shape: (B, S, attention_dim) -> (B, S, dim)
            attention_output = self.output_proj(attention_output, training=training)

        # Output gating mechanism (parameterized activation, defaults to sigmoid)
        # Shape: (B, S, dim) -> (B, S, dim); elementwise, no shape change
        gate_logits = self.output_gate_linear(attention_output, training=training)
        gate = self.gate_activation(gate_logits, training=training)
        gated_output = gate * attention_output

        return gated_output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: tuple
        :return: Output shape tuple (same as input shape).
        :rtype: tuple
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer configuration for serialization.

        :return: Dictionary containing the complete layer configuration.
        :rtype: dict
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'num_kv_heads': self.num_kv_heads,
            'head_dim': self.head_dim,
            'max_seq_len': self.max_seq_len,
            'rope_percentage': self.rope_percentage,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'probability_type': self.probability_type,
            'probability_config': self.probability_config,
            'qk_norm_type': self.qk_norm_type,
            'qk_norm_kwargs': self.qk_norm_kwargs,
            'gate_activation_type': self.gate_activation_type,
            'gate_activation_args': self.gate_activation_args,
        })
        return config
