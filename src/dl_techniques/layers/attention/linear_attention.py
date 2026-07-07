"""
Purely-linear (O(N)) self-attention that is Miyasawa-compliant by construction.

This module implements ``LinearAttention``: a multi-head, non-causal linear /
kernel attention layer designed to satisfy this repo's two operational Miyasawa
properties so it can drop into the bias-free denoiser stack without breaking the
additive-Gaussian ``residual = sigma^2 * score`` identity:

1. **Bias-free** — every Q/K/V/output projection is ``Dense(use_bias=False)`` by
   default; there is NO additive constant anywhere on the forward path (no
   softmax, no LayerNorm/RMSNorm, no beta/center, no learned bias). The only
   nonlinearity is the positively-homogeneous feature map ``phi``.
2. **Degree-1 positive homogeneity** — ``f(alpha * x) = alpha * f(x)`` for
   ``alpha > 0`` (verified numerically by a seq-shaped probe; target rel-err
   ``< 1e-4``).

**The math.** Standard attention ``softmax(Q K^T / sqrt(d)) V`` is O(N^2) AND
non-homogeneous (softmax is temperature-sensitive: ``softmax(alpha z) !=
softmax(z)``). Linear attention replaces the softmax exponential kernel with an
explicit, non-negative feature map ``phi`` and exploits matmul associativity::

    O_i = phi(Q_i) . (Sum_j phi(K_j) (x) V_j)  /  (phi(Q_i) . Sum_j phi(K_j) + eps_eff)
          |________ numerator (d x d state) ________|   |____ scalar denominator ____|

Contracting ``Sum_j phi(K_j) (x) V_j`` first yields a ``(d, d)`` state, so the
whole thing is ``O(N * d^2) = O(N)`` in sequence length: the ``N x N`` attention
matrix is NEVER materialized. This reuses the non-causal associativity contraction
STYLE of ``performer_attention.py`` (``phi(K)^T V`` contracted first), but NOT its
feature map nor its epsilon (see below).

**Degree-1 proof (F-W2).** Let ``x -> c x`` (``c > 0``). With no bias in the
projections, ``Q -> c Q``, ``K -> c K``, ``V -> c V``. If ``phi`` is positively
homogeneous of degree ``p`` (``phi(c z) = c^p phi(z)`` for ``c > 0``):

    - numerator  ``Sum_j phi(cQ_i) phi(cK_j) (cV_j) = c^p c^p c . Num = c^(2p+1) Num``
    - denominator ``Sum_j phi(cQ_i) phi(cK_j)       = c^(2p) Den``
    - O_i(c x) = c^(2p+1) / c^(2p) . (Num/Den) = c . O_i(x)   -> degree-1, for ANY p.

The **denominator normalizer is load-bearing** for homogeneity, not just for
stability: an unnormalized linear attention ``Sum_j phi(Q_i) phi(K_j) V_j`` is
degree ``2p+1`` and is degree-1 only in the trivial ``p = 0`` case. It is therefore
mandatory here, never an optional flag.

**Why not Performer's feature map / epsilon.** Performer's FAVOR+ map
``cos/sin . exp(-||x||^2 / 2)`` uses a Gaussian factor that is NOT positively
homogeneous, so it breaks property (2); and Performer adds a bare fixed ``+1e-6``
denominator floor, an additive constant that also breaks EXACT degree-1 (F-W3).
Both are rejected here (decisions.md D-003). The eps tension is resolved with an
**input-scaled epsilon** (decisions.md D-001, see ``call``).

**Scope.** v1 is **non-causal only** (denoising, the target use, is non-causal).
A causal cumsum variant (per-position prefix state) is future work; cosFormer
cosine reweighting, PolaFormer polarity split, and Norm x Direction query-norm
restoration are deferred (decisions.md D-003).

References:
    - Katharopoulos et al. (2020), "Transformers are RNNs: Fast Autoregressive
      Transformers with Linear Attention" (kernel feature map + associativity).
    - Choromanski et al. (2020), "Rethinking Attention with Performers"
      (associativity skeleton reused; FAVOR+ map NOT reused).
    - "Towards Robust Image Denoising with Scale Equivariance" (arxiv 2508.02967)
      — softmax-free first-order-homogeneous denoiser via ratio-of-equal-degree.
"""

# ---------------------------------------------------------------------

import keras
from typing import Optional, Union, Tuple, Any, Dict
from keras import ops, layers, initializers, regularizers

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Feature maps allowed here are exactly the positively-homogeneous ones (they keep
# f(alpha x) = alpha f(x), the Miyasawa property). Most are ALSO non-negative, which
# keeps the denominator phi(Q).Sum phi(K) a valid non-negative normalizer:
#   - 'relu' (p=1), 'relu_squared' (p=2), 'abs' (p=1): homogeneous AND non-negative.
#   - 'leaky_relu' (p=1): homogeneous (leaky(alpha z) = alpha leaky(z) for alpha > 0)
#     but SIGNED -- it produces negatives, so the "attention weights" are no longer a
#     non-negative partition and the denominator can be small/negative. Kept because a
#     signed slope gives a NON-zero gradient on the negative side (fixes the dead-ReLU
#     half); the denominator is guarded by a sign-aware magnitude floor in call() so a
#     genuinely-negative denominator stays stable instead of being clamped to +eps.
# NON-homogeneous maps are FORBIDDEN and rejected in __init__:
#   - 'elu_plus_one' (Katharopoulos elu(x)+1): the "+1" is an additive degree-0
#     constant -> breaks f(alpha x) = alpha f(x)  (F-W2 / F-W3).
#   - 'exp' / 'softmax': exp is non-homogeneous (exp(alpha z) != alpha^p exp(z))
#     and softmax is temperature-sensitive (softmax(alpha z) != softmax(z)).
_SUPPORTED_FEATURE_MAPS = ('relu', 'relu_squared', 'abs', 'leaky_relu')
_FORBIDDEN_FEATURE_MAPS = ('elu_plus_one', 'exp', 'softmax')

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class LinearAttention(keras.layers.Layer):
    """Bias-free, degree-1-homogeneous linear (O(N)) attention (Miyasawa-compliant).

    Multi-head non-causal linear attention with a positively-homogeneous,
    non-negative feature map ``phi`` and a mandatory normalizer, computed via
    matmul associativity so the ``N x N`` attention matrix is never formed. Both
    Miyasawa properties hold by construction: bias-free (all projections
    ``use_bias=False`` by default) and degree-1 homogeneous
    (``f(alpha x) = alpha f(x)`` for ``alpha > 0``). See the module docstring for
    the full derivation (F-W2) and the eps resolution (F-W3, decisions.md D-001).

    **Homogeneity scope / limitations (honest caveats):**

    - **Feature-map / scale band.** Degree-1 homogeneity is *exact* for ``'relu'``
      / ``'abs'`` (degree ``p=1``) across a wide input-scale band. ``'relu_squared'``
      (``p=2``, degree-4 denominator) can *degrade* at extreme small scales
      (``alpha <= ~1e-6``): the doubled dynamic range underflows and the ``1e-20``
      floor activates, so the property no longer holds bit-exactly there. Prefer
      ``'relu'`` for the strongest guarantee; keep ``'relu_squared'`` to realistic
      scales.
    - **Masking.** ``mask=`` is currently **IGNORED** (v1 is non-causal and
      unmasked). Padded tokens still contribute to the ``kv`` state and the
      normalizer; do not rely on this layer for correct masked/padded results.
    - **Training mode.** Homogeneity is a ``training=False`` / ``dropout_rate=0``
      property. With ``dropout_rate>0`` at ``training=True`` the output is
      stochastic (Dropout is applied after ``output_proj``) and thus not
      per-sample homogeneous; the default ``dropout_rate=0.0`` is the Miyasawa mode.
    - **Signed feature map (``'leaky_relu'``).** ``'leaky_relu'`` is degree-1
      homogeneous (so it PRESERVES the Miyasawa property) and its non-zero negative
      slope removes the dead-ReLU zero-gradient half. BUT it is the only supported map
      that is NOT non-negative: it breaks the non-negative-kernel guarantee, so the
      per-token "attention weights" are no longer a convex partition and the
      denominator ``z`` can be small or negative. The denominator is guarded by a
      *sign-aware* magnitude floor (``call`` step 5) so this stays finite and stable
      (a negative ``z`` keeps its sign instead of being clamped to ``+1e-20``), and
      homogeneity remains exact wherever ``|z| > 1e-20`` — but use a SMALL
      ``negative_slope`` (default ``0.01``) and keep ``epsilon`` at its default; a
      large slope makes ``z`` cross zero more often and is numerically riskier.

    **Architecture Overview:**

    .. code-block:: text

        Input [B, N, dim]
          -> Dense query_proj/key_proj/value_proj (bias-free) -> Q,K,V [B, N, inner]
          -> reshape/transpose -> [B, H, N, head_dim]
          -> phi(Q), phi(K)  (positively homogeneous, non-negative)
          -> kv    = einsum('bhnd,bhne->bhde', phi_k, v)     [B, H, d, d]
             k_sum = sum(phi_k, axis=2)                      [B, H, d]
             num   = einsum('bhnd,bhde->bhne', phi_q, kv)    [B, H, N, d]
             z     = einsum('bhnd,bhd->bhn',  phi_q, k_sum)  [B, H, N]
          -> input-scaled eps (D-001): denom = z + epsilon * mean_j(z)
          -> out = num / denom                               [B, H, N, d]
          -> merge heads -> Dense output_proj (bias-free)    [B, N, dim]

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
        (**compliant / bias-free mode**). Setting ``True`` breaks bias-freeness
        and is only for non-Miyasawa callers.
    :type use_bias: bool
    :param feature_map: Positively-homogeneous feature map ``phi``. One of ``'relu'``
        (p=1), ``'relu_squared'`` (p=2, FLatten-style focus), ``'abs'`` (p=1) — all
        non-negative — or ``'leaky_relu'`` (p=1, SIGNED: homogeneous and
        dead-gradient-free but breaks the non-negative kernel; see the caveat above).
        ``'elu_plus_one'``/``'exp'``/``'softmax'`` are rejected (they break degree-1
        homogeneity).
    :type feature_map: str
    :param negative_slope: Negative-half slope for ``feature_map='leaky_relu'`` (the
        ``alpha`` in ``leaky_relu``); ignored by the other maps. Must be in
        ``[0, 1]``; ``0.0`` recovers plain ReLU. Default ``0.01`` (small, to keep the
        signed tail — hence the non-negativity break — small).
    :type negative_slope: float
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
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            head_dim: Optional[int] = None,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            feature_map: str = 'relu',
            negative_slope: float = 0.01,
            epsilon: float = 1e-6,
            kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[regularizers.Regularizer] = None,
            bias_regularizer: Optional[regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
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
                f"feature_map '{feature_map}' is FORBIDDEN: it breaks degree-1 "
                f"homogeneity (the '+1' additive constant in elu_plus_one, or the "
                f"exp/softmax non-homogeneous kernel). Allowed values: "
                f"{list(_SUPPORTED_FEATURE_MAPS)}"
            )
        if feature_map not in _SUPPORTED_FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {list(_SUPPORTED_FEATURE_MAPS)}, "
                f"got '{feature_map}'"
            )
        if not 0.0 <= negative_slope <= 1.0:
            raise ValueError(
                f"negative_slope must be in [0, 1], got {negative_slope}"
            )
        if epsilon < 0.0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.feature_map = feature_map
        self.negative_slope = negative_slope
        self.epsilon = epsilon
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Computed attributes.
        # head_dim==None -> square case (inner == dim); else inner == num_heads*head_dim.
        self._head_dim_arg = head_dim
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.inner_dim = self.num_heads * self.head_dim

        # Create sub-layers in __init__ (unbuilt).
        self.query_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='query_proj'
        )
        self.key_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='key_proj'
        )
        self.value_proj = layers.Dense(
            self.inner_dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='value_proj'
        )
        self.output_proj = layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_proj'
        )

        # Dropout layer (optional).
        if dropout_rate > 0.0:
            self.dropout = layers.Dropout(dropout_rate)
        else:
            self.dropout = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its sub-layers.

        :param input_shape: Shape tuple of the input ``(batch, seq_len, dim)``.
        :type input_shape: Tuple[Optional[int], ...]
        """
        if self.built:
            return

        # Validate input shape
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
        ``alpha > 0`` (positive homogeneity of degree ``p``) — this is what makes the
        normalized attention degree-1 (F-W2). ``'relu'``/``'relu_squared'``/``'abs'``
        are ALSO ``>= 0`` (so the denominator ``phi(Q).Sum phi(K)`` is non-negative);
        ``'leaky_relu'`` is homogeneous but SIGNED (see class caveat), handled by the
        sign-aware denominator floor in ``call``.

        FORBIDDEN maps (rejected in ``__init__``) and WHY:
          - ``elu(x) + 1``: the ``+1`` is an additive degree-0 constant; it makes
            ``phi(alpha x) != alpha^p phi(x)`` -> breaks ``f(alpha x) = alpha f(x)``.
          - ``exp`` / ``softmax``: exponential is non-homogeneous and softmax is
            temperature-sensitive (``softmax(alpha z) != softmax(z)``).

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
        if self.feature_map == 'leaky_relu':
            # leaky_relu(alpha x) = alpha leaky_relu(x) for alpha > 0 -> degree p=1.
            # SIGNED (produces negatives): non-zero gradient on the negative side
            # (no dead-ReLU half) at the cost of the non-negative-kernel guarantee.
            return ops.leaky_relu(x, negative_slope=self.negative_slope)
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
        :param mask: Unused in v1 (accepted for API uniformity).
        :type mask: Optional[keras.KerasTensor]
        :return: Output tensor of shape ``(batch, seq_len, dim)``.
        :rtype: keras.KerasTensor
        """
        del mask  # v1 is non-causal and unmasked; accepted only for API uniformity.

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
        phi_q = self._feature_map(q)  # (B, H, N, d), degree p
        phi_k = self._feature_map(k)  # (B, H, N, d), degree p

        # 4. Associativity (O(N) in seq: the (N x N) matrix is never formed).
        #    kv    = Sum_j phi(K_j) (x) V_j  -> (B, H, d, d),  degree p+1
        #    k_sum = Sum_j phi(K_j)          -> (B, H, d),     degree p
        #    num   = phi(Q_i) . kv           -> (B, H, N, d),  degree 2p+1
        #    z     = phi(Q_i) . k_sum        -> (B, H, N),     degree 2p
        #            (>= 0 for the non-negative maps; SIGNED for 'leaky_relu')
        kv = ops.einsum('bhnd,bhne->bhde', phi_k, v)
        k_sum = ops.sum(phi_k, axis=2)
        num = ops.einsum('bhnd,bhde->bhne', phi_q, kv)
        z = ops.einsum('bhnd,bhd->bhn', phi_q, k_sum)

        # 5. Input-scaled epsilon (THE CRUX; decisions.md D-001).
        #    A FIXED additive floor (Performer's bare +1e-6) is a degree-0 constant
        #    that breaks EXACT degree-1: num is degree 2p+1, z is degree 2p, so a
        #    constant added to z does NOT scale with z and the quotient stops being
        #    degree-1 (F-W3). Instead scale epsilon by z's OWN degree-2p mean, so the
        #    floor has the SAME degree as z: then num/(z + eps_eff) stays exactly
        #    degree-1. The maximum(., 1e-20) is ONLY a NaN guard for a fully-dead
        #    batch (all phi zero -> z_mean == 0); it is a negligible degree-0 floor
        #    that the homogeneity probe tolerance absorbs -- the single residual
        #    non-homogeneous corner.
        #    fp16-SAFETY: the divide + 1e-20 floor run in float32, then cast back to
        #    the compute dtype. Under a mixed_float16 policy the compute dtype is fp16,
        #    where 1e-20 rounds to 0.0 -> the dead-token guard would fail (0/0 NaN on an
        #    all-zero batch). Doing it in float32 is numerically identical to a pure-fp32
        #    run, so exact degree-1 on float32 is UNCHANGED. Do NOT drop the cast: a bare
        #    fp16 divide reintroduces the NaN (reviewer WARNING, review-iter-1.md:13).
        #    SIGN-AWARE FLOOR: for the non-negative maps z >= 0 and this reduces EXACTLY
        #    to maximum(denom, 1e-20). For SIGNED 'leaky_relu' a genuinely-negative denom
        #    keeps its sign+magnitude (min(denom, -1e-20)) instead of being clamped to
        #    +1e-20 -- clamping would flip the output sign and explode it. Only the
        #    |denom| < 1e-20 near-dead corner is floored; homogeneity stays exact
        #    wherever |denom| > 1e-20.
        # DECISION plan_2026-07-07_1cab8d7a/D-001
        z_mean = ops.mean(z, axis=-1, keepdims=True)          # (B, H, 1), degree 2p
        eps_eff = self.epsilon * z_mean                       # degree 2p -> keeps degree-1
        denom = z + eps_eff                                   # (B, H, N), degree 2p
        out_dtype = num.dtype                                 # compute dtype (fp16 under mixed policy)
        num_f32 = ops.cast(num, 'float32')
        denom_f32 = ops.cast(denom, 'float32')
        denom_f32 = ops.where(                                # sign-aware magnitude floor
            denom_f32 >= 0.0,
            ops.maximum(denom_f32, 1e-20),                    # == old path for z >= 0
            ops.minimum(denom_f32, -1e-20),                   # signed maps: preserve sign
        )
        out = ops.cast(num_f32 / denom_f32[..., None], out_dtype)   # (B, H, N, d), degree 1

        # 6. Merge heads -> (B, N, inner_dim) -> bias-free output projection.
        out = ops.reshape(
            ops.transpose(out, (0, 2, 1, 3)),
            (batch_size, seq_len, self.inner_dim),
        )
        out = self.output_proj(out)

        if self.dropout is not None:
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
            'negative_slope': self.negative_slope,
            'epsilon': self.epsilon,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
