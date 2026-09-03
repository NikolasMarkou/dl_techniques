"""LinearAttention, a purely-linear (O(N)) self-attention layer that is
bias-free and degree-1 homogeneous, for the bias-free denoiser stack.

Three bias-free ``Dense`` projections produce Q, K and V. A
positively-homogeneous non-negative feature map ``phi`` is applied to Q
and K, then matmul associativity contracts the key side first into a
``(d, d)`` state, so no ``N x N`` attention matrix is ever formed. A
mandatory denominator normalizer divides the result, and the heads are
merged and projected back to ``dim``, again bias-free. There is no
softmax, no normalization layer, and no additive constant anywhere on
the path; that absence is what preserves the ``f(alpha x) = alpha f(x)``
degree-1 property this layer exists to guarantee (the derivation and the
epsilon design are in ``call``'s ``D-001`` comment and ``decisions.md``).

There is no masking stage. ``mask=`` is accepted, for signature
compatibility with sibling attention layers, and silently discarded:
padded tokens still contribute to both the running state and the
normalizer. This layer is for the fixed-size, fully-populated, non-causal
denoiser stack, not for variable-length or causal sequences.

References:
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.
    - Choromanski et al., 2020. Rethinking Attention with Performers.
    - Towards Robust Image Denoising with Scale Equivariance. (https://arxiv.org/abs/2508.02967)
"""

import keras
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# Feature maps allowed here are exactly the positively-homogeneous,
# non-negative ones; 'elu_plus_one'/'exp'/'softmax' break degree-1
# homogeneity and are rejected in __init__.
_SUPPORTED_FEATURE_MAPS = ('relu', 'relu_squared', 'abs')
_FORBIDDEN_FEATURE_MAPS = ('elu_plus_one', 'exp', 'softmax')


@register_dl_technique("dl_techniques.layers.attention.linear_attention")
class LinearAttention(keras.layers.Layer):
    """Bias-free, degree-1-homogeneous linear (O(N)) attention (Miyasawa-compliant).

    Multi-head non-causal linear attention with a positively-homogeneous,
    non-negative feature map ``phi`` and a mandatory normalizer, computed via
    matmul associativity so the ``N x N`` attention matrix is never formed. Both
    Miyasawa properties hold for every input: bias-free (all projections
    ``use_bias=False`` by default) and degree-1 homogeneous
    (``f(alpha x) = alpha f(x)`` for ``alpha > 0``). See the module docstring for
    the full derivation (F-W2) and the eps resolution (F-W3, decisions.md D-001).

    .. warning::

       ``mask=`` is accepted and silently ignored. ``call()`` takes a ``mask``
       argument only so this layer's signature matches its siblings', then
       discards it (``del mask`` on the first line). There is no masking stage
       anywhere on the forward path, so padded tokens contribute their full
       weight to both the ``kv`` state and the normalizer ``z``: a padded
       batch produces different outputs for the real tokens than the same
       batch without padding, silently. Do not use this layer for
       variable-length or padded sequence data, or where causality is
       required. Its intended home is the fixed-size, fully-populated,
       non-causal bias-free denoiser stack.

    Homogeneity scope and limitations:

    - Feature-map / scale band. Degree-1 homogeneity is exact for ``'relu'``
      / ``'abs'`` (degree ``p=1``) across a wide input-scale band.
      ``'relu_squared'`` (``p=2``) can degrade at extreme small scales
      (``alpha <= ~1e-6``): the doubled dynamic range underflows and the
      ``1e-20`` floor activates, so the property no longer holds bit-exactly
      there. Prefer ``'relu'`` for the strongest guarantee.
    - Training mode. Homogeneity is a ``training=False`` / ``dropout_rate=0``
      property. With ``dropout_rate>0`` at ``training=True`` the output is
      stochastic (dropout runs after ``output_proj``) and not per-sample
      homogeneous; the default ``dropout_rate=0.0`` is the exact mode.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │                       LinearAttention                        │
        │                                                              │
        │   Input [B, N, dim]                                          │
        │          │                                                   │
        │          ├─────────────┬─────────────┐                       │
        │          ▼             ▼             ▼                       │
        │   ┌────────────┐ ┌────────────┐ ┌────────────┐               │
        │   │ query_proj │ │  key_proj  │ │ value_proj │  bias-free    │
        │   │ Dense(inn) │ │ Dense(inn) │ │ Dense(inn) │  (no beta)    │
        │   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘               │
        │         ▼              ▼              ▼                      │
        │   reshape + transpose -> [B, H, N, d]  (d = head_dim)        │
        │         │              │              │                      │
        │         ▼              ▼              │                      │
        │   ┌────────────┐ ┌────────────┐       │                      │
        │   │   phi(Q)   │ │   phi(K)   │       │  phi = relu /        │
        │   │  degree p  │ │  degree p  │       │  relu^2 / abs        │
        │   └─────┬──────┘ └─────┬──────┘       │  (>= 0, homogeneous) │
        │         │              │              │                      │
        │         │              └──────┬───────┘                      │
        │         │                     ▼                              │
        │         │      kv    = einsum('bhnd,bhne->bhde', phi_k, v)   │
        │         │              [B, H, d, d]        degree p+1        │
        │         │      k_sum = sum(phi_k, axis=2)                    │
        │         │              [B, H, d]           degree p          │
        │         │                     │                              │
        │         ├─────────────────────┤                              │
        │         ▼                     ▼                              │
        │   num = einsum(         z = einsum(                          │
        │     'bhnd,bhde->bhne',    'bhnd,bhd->bhn',                   │
        │      phi_q, kv)            phi_q, k_sum)                     │
        │   [B, H, N, d]           [B, H, N]                           │
        │   degree 2p+1            degree 2p, >= 0                     │
        │         │                     │                              │
        │         │                     ▼                              │
        │         │        ┌──────────────────────────────┐            │
        │         │        │ input-scaled eps  (D-001)    │            │
        │         │        │ denom = z + eps * mean_j(z)  │            │
        │         │        │ (same degree as z -> exact   │            │
        │         │        │  degree-1; a fixed +1e-6     │            │
        │         │        │  would break it)             │            │
        │         │        └──────────────┬───────────────┘            │
        │         └────────────┬──────────┘                            │
        │                      ▼                                       │
        │            out = num / denom   [B, H, N, d]   degree 1       │
        │            (divide forced to float32, then cast back)        │
        │                      │                                       │
        │                      ▼                                       │
        │            merge heads -> [B, N, inner_dim]                  │
        │                      │                                       │
        │                      ▼                                       │
        │            ┌────────────────────┐                            │
        │            │ output_proj Dense  │  bias-free                 │
        │            └─────────┬──────────┘                            │
        │                      ▼                                       │
        │            Output [B, N, dim]                                │
        │                                                              │
        │   mask= is accepted and silently discarded -- there is no   │
        │   masking stage anywhere in this diagram.                    │
        └──────────────────────────────────────────────────────────────┘

    :param dim: Model dimensionality (input and output feature size). Must be
        positive. If ``head_dim`` is None, must be divisible by ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Must be positive.
    :type num_heads: int
    :param head_dim: Per-head dimension. If None, defaults to ``dim // num_heads``
        (requiring ``dim % num_heads == 0``) and the inner projection dim equals
        ``dim``. If given, the inner projection dim is ``num_heads * head_dim`` and
        ``output_proj`` maps it back to ``dim``.
    :type head_dim: Optional[int]
    :param dropout_rate: Dropout rate applied to the output, in ``[0, 1]``.
    :type dropout_rate: float
    :param use_bias: Whether the projections use a bias. Default ``False``
        (bias-free mode). Setting ``True`` breaks bias-freeness and is only
        for callers that do not need that property.
    :type use_bias: bool
    :param feature_map: Positively-homogeneous non-negative feature map ``phi``.
        One of ``'relu'`` (p=1), ``'relu_squared'`` (p=2), ``'abs'`` (p=1).
        ``'elu_plus_one'``/``'exp'``/``'softmax'`` are rejected (they break
        degree-1 homogeneity).
    :type feature_map: str
    :param epsilon: Relative denominator floor. The effective floor is
        ``epsilon * mean_over_tokens(z)`` (input-scaled, D-001), keeping degree-1
        exact. Must be ``>= 0``.
    :type epsilon: float
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for projection bias vectors (only used if
        ``use_bias=True``).
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for projection biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for the Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim`` or ``num_heads`` is not positive, or if
        ``head_dim`` is given and not positive.
    :raises ValueError: If ``head_dim`` is ``None`` and ``dim`` is not divisible by
        ``num_heads``.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1]`` or ``epsilon``
        is negative.
    :raises ValueError: If ``feature_map`` is one of the forbidden
        non-homogeneous maps, or is not a supported map.
    :raises ValueError: From ``build()``, if the input is not 3D or its trailing
        dimension does not equal ``dim``.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            feature_map: str = 'relu',
            epsilon: float = 1e-6,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the four bias-free projections.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.
        """
        super().__init__(**kwargs)

        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        # Not common.validate_head_divisibility(): this check is conditional
        # (only matters when head_dim is None) and names the head_dim escape hatch, which the shared helper's message does not.
        if head_dim is None and dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads}) "
                f"when head_dim is None"
            )
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive when given, got {head_dim}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if feature_map in _FORBIDDEN_FEATURE_MAPS:
            raise ValueError(
                f"feature_map '{feature_map}' is forbidden: it breaks degree-1 "
                f"homogeneity (the '+1' additive constant in elu_plus_one, or the "
                f"exp/softmax non-homogeneous kernel). Allowed values: "
                f"{list(_SUPPORTED_FEATURE_MAPS)}"
            )
        if feature_map not in _SUPPORTED_FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {list(_SUPPORTED_FEATURE_MAPS)}, "
                f"got '{feature_map}'"
            )
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")

        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.feature_map = feature_map
        self.epsilon = epsilon
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # head_dim==None -> square case (inner == dim); else inner == num_heads*head_dim.
        self._head_dim_arg = head_dim
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim

        # DECISION plan-2026-08-22T035419-a11304c8/D-200: clone the initializer per
        # projection, never pass self.kernel_initializer directly -- one shared instance gives bit-identical Q/K kernels. See decisions.md.
        self.query_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='query_proj'
        )
        self.key_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='key_proj'
        )
        self.value_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='value_proj'
        )
        self.output_proj = layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_proj'
        )

        # DECISION plan-2026-08-27T040114-580f8b63/D-016: Dropout is created
        # unconditionally and gated in call(), not behind `if dropout_rate > 0` -- that made the object graph depend on dropout_rate. See decisions.md.
        self.dropout = layers.Dropout(dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        :param input_shape: Shape tuple of the input ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input (B, N, dim), got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build Q/K/V projections on the input shape; output_proj on the merged-head
        # shape (last dim == inner_dim).
        self.query_proj.build(input_shape)
        self.key_proj.build(input_shape)
        self.value_proj.build(input_shape)

        inner_shape = tuple(input_shape[:-1]) + (self.inner_dim,)
        self.output_proj.build(inner_shape)

        super().build(input_shape)

    def _feature_map(self, x: keras.KerasTensor) -> keras.KerasTensor:
        """Apply the positively-homogeneous, non-negative feature map ``phi``.

        All supported maps satisfy ``phi(alpha x) = alpha^p phi(x)`` for
        ``alpha > 0`` (positive homogeneity of degree ``p``) AND ``phi(x) >= 0``
        (needed so the denominator ``phi(Q).Sum phi(K)`` is non-negative). This is
        what makes the normalized attention degree-1 (F-W2).

        Forbidden maps, rejected in ``__init__``: ``elu(x) + 1`` (the ``+1`` is
        an additive degree-0 constant, breaking ``f(alpha x) = alpha f(x)``), and
        ``exp`` / ``softmax`` (non-homogeneous; softmax is temperature-sensitive).

        :param x: Input tensor.
        :type x: keras.KerasTensor
        :return: Non-negative, positively-homogeneous features (same shape as ``x``).
        :rtype: keras.KerasTensor
        """
        if self.feature_map == 'relu':
            # relu(alpha x) = alpha relu(x) for alpha > 0 -> degree p=1.
            return ops.relu(x)
        if self.feature_map == 'relu_squared':
            # relu(x)^2 -> degree p=2 (FLatten-style focus); still positively
            # homogeneous: (alpha relu(x))^2 = alpha^2 relu(x)^2.
            return ops.square(ops.relu(x))
        # 'abs': |alpha x| = alpha |x| for alpha > 0 -> degree p=1, non-negative.
        return ops.abs(x)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            mask: Optional[keras.KerasTensor] = None
    ) -> keras.KerasTensor:
        """Apply non-causal linear attention.

        :param inputs: Input tensor of shape ``(batch, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode (affects dropout only).
        :type training: Optional[bool]
        :param mask: Accepted and silently discarded. Present only so this
            layer's signature matches its siblings'; there is no masking stage, so
            padded tokens still contribute to the ``kv`` state and to the
            normalizer ``z``. See the ``warning`` block in the class docstring.
        :type mask: Optional[keras.KerasTensor]
        :return: Output tensor of shape ``(batch, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        # `del mask` is the ONLY handling this argument gets. Binding it to
        # nothing makes the discard explicit at the top of the function, instead
        # of leaving a reader to scan the body for a use that does not exist.
        # v1 is non-causal and unmasked; the argument exists only so this layer's
        # signature matches its siblings'. Don't start honoring `mask` here
        # without also removing the warnings in the class docstring and the note
        # in the diagram. A half-implemented mask is worse than a documented
        # no-op.
        del mask

        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]

        # 1. Bias-free projections -> (B, N, inner_dim).
        q = self.query_proj(inputs)
        k = self.key_proj(inputs)
        v = self.value_proj(inputs)

        # 2. Reshape to multi-head format (B, H, N, head_dim).
        q = ops.transpose(
            ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        k = ops.transpose(
            ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )
        v = ops.transpose(
            ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )

        # 3. Positively-homogeneous, non-negative features on Q and K.
        # Both are (B, H, N, d) and degree p.
        phi_q = self._feature_map(q)
        phi_k = self._feature_map(k)

        # 4. Associativity (O(N) in seq: the (N x N) matrix is never formed).
        #    kv    = Sum_j phi(K_j) (x) V_j  -> (B, H, d, d),  degree p+1
        #    k_sum = Sum_j phi(K_j)          -> (B, H, d),     degree p
        #    num   = phi(Q_i) . kv           -> (B, H, N, d),  degree 2p+1
        #    z     = phi(Q_i) . k_sum        -> (B, H, N),     degree 2p, >= 0
        kv = ops.einsum('bhnd,bhne->bhde', phi_k, v)
        k_sum = ops.sum(phi_k, axis=2)
        num = ops.einsum('bhnd,bhde->bhne', phi_q, kv)
        z = ops.einsum('bhnd,bhd->bhn', phi_q, k_sum)

        # DECISION plan_2026-07-07_1cab8d7a/D-001: scale epsilon by z's own
        # degree-2p mean, not a fixed additive floor like Performer's +1e-6 -- a degree-0 constant added to z would break degree-1 exactness. See decisions.md.
        z_mean = ops.mean(z, axis=-1, keepdims=True)
        eps_eff = self.epsilon * z_mean
        denom = z + eps_eff
        # Divide runs in float32 then casts back: under mixed_float16 the
        # 1e-20 dead-batch guard rounds to 0.0 and 0/0 becomes NaN otherwise.
        out_dtype = num.dtype
        num_f32 = ops.cast(num, 'float32')
        denom_f32 = ops.maximum(ops.cast(denom, 'float32'), 1e-20)
        out = ops.cast(num_f32 / denom_f32[..., None], out_dtype)

        # 6. Merge heads -> (B, N, inner_dim) -> bias-free output projection.
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (batch_size, seq_len, self.inner_dim),
        )
        out = self.output_proj(out)

        if self.dropout_rate > 0.0:
            out = self.dropout(out, training=training)

        return out

    def compute_output_shape(
            self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape (identical to the input shape: dim in == dim out).

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Shape tuple of the output (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the full configuration of the layer for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self._head_dim_arg,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'feature_map': self.feature_map,
            'epsilon': self.epsilon,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config
