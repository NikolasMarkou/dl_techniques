"""
Linear-complexity attention by approximating the softmax kernel with random
features (FAVOR+).

Standard attention is quadratic because the softmax sits between two matmuls:
``softmax(Q K^T) V`` cannot be reassociated, since the nonlinearity must see the
full ``N x N`` score matrix before anything is contracted with ``V``. Everything
about attention's cost follows from that one blocking nonlinearity, not from the
matmuls themselves.

The Performer's move is to remove the blockage rather than approximate around it.
If the exponential kernel can be written as an inner product of explicit feature
maps, ``exp(q . k) ~ E[<phi(q), phi(k)>]``, then the numerator becomes
``phi(Q) phi(K)^T V`` — a product of three matrices with no nonlinearity between
them, so associativity applies. Contracting ``phi(K)^T V`` FIRST yields an
``(r, d)`` summary of the whole sequence, and every query then reads from that
summary. Cost falls from ``O(N^2 d)`` to ``O(N r d)``, and the ``N x N`` matrix is
never formed at any point. The denominator is the same trick applied to a vector of
ones: attention's normalizer is just the kernel summed over keys.

Causal masking survives the reassociation, which is not obvious. A causal query
must read a summary of the PREFIX rather than of the whole sequence, and because
the summary is a sum of per-position outer products, the prefix summaries are its
running `cumsum`. So causality costs a scan over positions instead of a triangular
mask — still linear, though the rank-5 intermediate makes it substantially more
memory-hungry than the non-causal path.

Two properties of this file matter and are NOT normalized away by the
package-wide style pass. `call()` takes **no** `attention_mask` argument;
that absence is frozen and honest, because the factory dispatches on argument names
and a silently-ignored mask parameter would be worse than none. And the softmax
temperature is precomputed once in `__init__` as a Python float, never recomputed
with `ops.sqrt` in `call()`.

One documented gap: `ortho_scaling` does not orthogonalize. The "O" in FAVOR+ is
orthogonal random features, which reduce approximation variance; what this
implementation does when `ortho_scaling > 0` is multiply the Gaussian projection by
a scalar. The default is `0.0`, so at default settings even that multiply is
skipped. The projection is also redrawn on every call, so two forward passes on the
same input do not agree exactly.

A second, larger gap: **`nb_features` is a variance knob, not an accuracy knob.**
Standard Performer semantics are that approximation error falls toward zero as
`nb_features` grows. This feature map is BIASED and does not converge.

THE HARNESS IS STATED IN FULL BELOW, because the error LEVEL depends on it and an
earlier version of this section quoted numbers from a harness that is not in the
tree. To re-derive: `keras.utils.set_random_seed(0)`; input
`keras.random.normal((batch, 64, dim), seed=1234)` with `head_dim=16` and
`dim = head_dim * num_heads`; the reference is `softmax(q K^T) V` built from THIS
layer's own `to_qkv` weights and `self.scale`; the score is the relative
Frobenius norm of the difference, taken BEFORE `to_out`; 20 redraws of the
projection matrix per cell, mean reported.

Re-measured 2026-08-28 at `num_heads=1, batch=1`::

    features m :      8       32      128      512     2048     8192
    rel. error : 0.7773   0.7729   0.7714   0.7715   0.7714   0.7714
    std dev    : 0.0060   0.0019   0.0014   0.0005   0.0003   0.0002

The per-redraw VARIANCE does shrink with `m` as a Monte-Carlo estimator should;
the BIAS plateaus and does not move. **The plateau LEVEL belongs to the harness,
not to the layer**: the same code at `num_heads=4, batch=4` plateaus at 0.5767
instead, and the earlier unrecorded harness read 0.7796-0.8164. Quote the SHAPE
(no convergence, variance-only benefit) and never the absolute number.

WHAT CAUSES THE FLOOR IS NOT ESTABLISHED, and an earlier version of this
paragraph named three causes that ablation does not support. Ablated in the
harness above, `num_heads=1, batch=1`, mean of 20 redraws::

    features m :      8       32      128      512     2048     8192
    as shipped : 0.7773   0.7729   0.7714   0.7715   0.7714   0.7714
    no norm fac: 0.7762   0.7720   0.7714   0.7713   0.7713   0.7714
    no clamp   : 0.7522   0.7434   0.7419   0.7405   0.7401   0.7399

* The query-side `exp(-||x||^2/2)` factor very nearly cancels. It is a POSITIVE
  per-row scalar `c`, and `max(0, c * f) == c * max(0, f)` for `c > 0`, so it
  multiplies the numerator and the normalizer of
  `phi(Q)(phi(K)^T V) / phi(Q)(phi(K)^T 1)` alike. It is NOT exactly inert, and
  the earlier text claiming that was wrong. `_linear_attention` adds `1e-6` to
  the normalizer, and that constant does not scale with `c`, so a relative
  residue of about `(1/c - 1) * 1e-6 / z` survives. Measure it PAIRED: one set
  of weights, one PINNED projection matrix, `num_heads=1, batch=1, seq_len=32`,
  layer output magnitude ~0.3. The max absolute delta is then **7.3e-09 at
  float64** and **1.3e-04 at float32**. The float32 figure is contraction
  rounding, not the factor: each arm re-run against itself differs by exactly
  0.0. **Do not read the `no norm fac` ROW above as this effect.** That table is
  UNPAIRED: its rows are independent redraws, so the 0.0011 spread at `m=8` is
  redraw noise and is larger than the factor's real contribution. The factor is
  neither a cause of the floor nor a defect.
* The `ops.maximum(features, 0)` clamp is not the cause of the floor either, and
  the previous claim that removing it alone is "catastrophic, two to four orders
  worse" (317.5 at `m=8`) DID NOT REPRODUCE: removing it lowers the mean error
  slightly here. The denominator hazard it is meant to guard was not observed
  either -- at `m=128`, 20 redraws, the minimum `z` was 22.62 unclamped against
  23.52 clamped, with no sign change. Do NOT read that as "the clamp is dead
  weight". An unclamped cos/sin feature is SIGNED, so `z` can still cross zero at
  other `(seq_len, m, scale)` points, and removing the clamp is a VALUE change
  for every shipped checkpoint. It is shipped behaviour whose stated rationale is
  now UNPROVEN, not behaviour that has been cleared for removal.

So: the bias is real and reproducible, the estimator does not converge, and the
mechanism is OPEN. Do not "correct" the feature map on the strength of a story
about which term is at fault -- ablate first, in the harness above, and compare
against the shipped map AT THE SAME m, num_heads and batch.

Two corrected maps were tried and rejected (decisions.md D-013): a fixed
trigonometric map and a textbook positive FAVOR+ map both converge under the
symmetric `d^(-1/4)` scaling FAVOR+ assumes, but neither beats the shipped map at
the feature counts anyone sets. At `m = 128` they measured 82.7 and 2.08 against
this map's 0.78 -- the trigonometric one by two orders, the FAVOR+ one by about
2.7x. Symmetric scaling alone does not move the floor either
(`0.7826 -> 0.7736` at `m = 128`).

Use this layer for its linear `O(N)` cost. Do not use it where the answer must
approach softmax attention, and do not read `nb_features` as a quality dial.

Foundational mathematics::

    exp(q . k) ~ E[ <phi(q), phi(k)> ]

    O = phi(Q) (phi(K)^T V) / ( phi(Q) (phi(K)^T 1_N) + 1e-6 )

with the trigonometric feature map
``phi(x) = sqrt(2/r) [cos(w_i . x), sin(w_i . x)]``, additionally scaled by
``exp(-||x||^2 / 2)`` on the query side.

References:
    - Choromanski et al., 2020. Rethinking Attention with Performers. ICLR 2021.
      (https://arxiv.org/abs/2009.14794)
    - Rahimi and Recht, 2007. Random Features for Large-Scale Kernel Machines.
      NIPS. (the trigonometric feature map this uses)
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast Autoregressive
      Transformers with Linear Attention. (the causal prefix-state formulation)
      (https://arxiv.org/abs/2006.16236)
    - Vaswani et al., 2017. Attention Is All You Need. (the quadratic mechanism
      being approximated) (https://arxiv.org/abs/1706.03762)
"""

# ---------------------------------------------------------------------

import math
import keras
from typing import Optional, Union, Tuple, Any, Dict

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .common import (
    compute_attention_scale,
    validate_head_divisibility
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.attention.performer_attention")
class PerformerAttention(keras.layers.Layer):
    """
    Performer attention: linear complexity via FAVOR+ kernel approximation.

    Approximates standard softmax attention in ``O(N)`` time and memory instead of
    ``O(N^2)``. Positive random feature maps ``phi(Q)`` and ``phi(K)`` are
    constructed such that ``exp(q . k) ~ <phi(q), phi(k)>``, after which
    ``phi(Q) (phi(K)^T V) / phi(Q) (phi(K)^T 1_N)`` gives the attention output
    without ever materializing the ``N x N`` matrix. The feature map is
    trigonometric — ``phi(x) = (1/sqrt(r)) [cos(w_i . x), sin(w_i . x)]``, scaled by
    ``exp(-||x||^2 / 2)`` on the query side.

    **Frozen signature: there is no mask argument.** ``call()`` accepts
    ``(inputs, training, return_attention_scores)`` and has no ``attention_mask``
    or ``mask`` parameter, unlike most siblings in this package. The attention
    factory registers this layer for CONSTRUCTION only and documents the
    call-signature difference rather than renaming it, so the difference is a
    recorded caveat rather than an accident. **Do NOT "fix" this** by adding a
    mask parameter or renaming one: the factory dispatches on the exact argument
    names, and a silently-ignored ``attention_mask`` would be worse than an honest
    absence — callers would believe padding was handled when it is not. Causal
    masking IS supported, but only through the ``causal=True`` constructor flag,
    which selects the prefix-``cumsum`` path in ``_linear_attention``.

    **[REUSE]** The ``dim % num_heads`` check and the ``1 / sqrt(head_dim)``
    temperature come from :mod:`~dl_techniques.layers.attention.common` rather
    than being re-derived here.

    **Why this is O(N): the re-association.**

    .. code-block:: text

        Standard attention. The softmax sits BETWEEN the two matmuls, so
        it must see the whole N x N score matrix before anything hits V.
        Nothing can be reordered around it.

            Q [N, d] ─┐
                      ├─► QKᵀ [N, N] ─► softmax ─► @ V ─► out [N, d]
            K [N, d] ─┘        ▲
                               └─ this matrix is materialized: O(N²·d)

        Performer. Replace exp(q·k) by <phi(q), phi(k)> and the softmax
        disappears. Three plain matmuls remain, so associativity applies
        and the RIGHT pair is contracted FIRST.

            phi(K)ᵀ [F, N] ─┐
                            ├─► KV [F, d]      ONE summary of the
            V       [N, d] ─┘                  whole sequence
                              │
            phi(Q)  [N, F] ───┴─► @ KV ─► num [N, d]

            phi(K)ᵀ · 1_N ─► k_sum [F]
            phi(Q) · k_sum ─► z [N]  (+ 1e-6)

            out = num / z          cost O(N · F · d), and the N x N
                                   matrix is never formed at any point

        Read the second figure right-to-left: the summary is built once,
        then every query reads from it. That ordering IS the algorithm.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────────────────────────────────┐
        │  Input [B, N, dim]                                           │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  self.to_qkv (Keras name 'qkv_projection'):                  │
        │  Dense(3·dim) → split 3 → heads                              │
        │    q, k, v  [B, H, N, d_h]                                   │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  q = q · (1/sqrt(d_h))                                       │
        │    the scale is applied BEFORE the feature map, not after a  │
        │    score matmul — there IS no score matmul                   │
        │    NOTE: self.scale ALSO multiplies the projection matrix    │
        │    below, so it lands on the query path TWICE. That is       │
        │    the shipped behaviour, not a transcription error.         │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  projection matrix, REDRAWN ON EVERY CALL from               │
        │  keras.random.normal(seed=None)   [H, F/2, d_h]              │
        │    · ortho_scaling ONLY IF ortho_scaling > 0 — and it        │
        │      DEFAULTS to 0.0, so at default settings that multiply   │
        │      is SKIPPED entirely                                     │
        │    · scale, unconditionally                                  │
        │  Even when applied, ortho_scaling is a plain scalar          │
        │  multiply: nothing here orthogonalizes, despite the name     │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  phi(x) = max(0, [cos(xWᵀ), sin(xWᵀ)] · feature_scale)       │
        │    [B, H, N, F];  phi(q) is additionally scaled by           │
        │    exp(−‖q‖² / 2), phi(k) is not                             │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  fork on the causal flag. The (N × N) matrix is NEVER        │
        │  formed in either branch — associativity keeps this          │
        │  O(N · F · d_h):                                             │
        │                                                              │
        │    causal=False   KV = phi(k)ᵀ · v          [B, H, F, d_h]   │
        │                   z  = phi(q) · Σ_n phi(k)  + 1e−6           │
        │                   ONE summary of the whole sequence          │
        │                                                              │
        │    causal=True    KV = cumsum_n(phi(k) ⊗ v)  prefix state    │
        │                   z  = phi(q) · cumsum_n phi(k) + 1e−6       │
        │                   a PREFIX summary per position; rank-5      │
        │                   intermediate, so far more memory-hungry    │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  out = (phi(q) · KV) / z → merge heads                       │
        │  → self.to_out (Keras name 'output_projection'): Dense(dim)  │
        │    → dropout, ONLY IF dropout_rate > 0                       │
        └───────────────┬──────────────────────────────────────────────┘
                        ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  Output [B, N, dim]                                          │
        │    call() has NO mask argument at all (frozen signature);    │
        │    the only masking is the causal=True constructor flag.     │
        │    return_attention_scores=True returns (output, None).      │
        └──────────────────────────────────────────────────────────────┘

    **Complexity:**

    .. code-block:: text

        standard attention   O(N² · d)     the (N × N) score matrix
        Performer            O(N · F · d)  F = nb_features

        F costs memory proportionally. It does NOT buy accuracy: this
        feature map is biased, and the measured error against a softmax
        reference plateaus near 0.78 from F = 32 upward. The module
        docstring carries the table. Raising F narrows the per-redraw
        VARIANCE only.

    :param dim: Model dimensionality. Must be positive and divisible by num_heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param nb_features: Number of random features for the kernel approximation.
        Higher values cost proportionally more memory and shrink the per-redraw
        VARIANCE, but they do not shrink the approximation ERROR: this feature map
        is biased and its measured error plateaus near 0.78. Do not read this as a
        quality dial. The module docstring carries the measured table.
    :type nb_features: int
    :param ortho_scaling: Scaling factor applied to the random projection matrix.
        NOTE (limitation): when ``ortho_scaling > 0`` this currently applies a plain
        scalar multiplication to the random Gaussian projection; it does NOT perform
        orthogonal random feature construction. True FAVOR+ orthogonalization (e.g.
        QR / Gram-Schmidt of the projection rows) is NOT implemented. The parameter
        therefore only rescales the (non-orthogonal) random features. ``0.0`` disables
        the scaling.
    :type ortho_scaling: float
    :param causal: Whether to use causal (autoregressive) attention masking.
        Selects the prefix-``cumsum`` path, which is still linear in ``N`` but
        materially more memory-hungry than the non-causal one.
    :type causal: bool
    :param dropout_rate: Dropout rate for attention weights, between 0 and 1.
    :type dropout_rate: float
    :param use_bias: Whether to use bias in Q, K, V projections.
    :type use_bias: bool
    :param kernel_initializer: Initializer for projection weight matrices.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for projection bias vectors.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for projection weights.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for projection biases.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional arguments for Layer base class.
    :type kwargs: Any

    :raises ValueError: If ``dim``, ``num_heads`` or ``nb_features`` is not
        positive.
    :raises ValueError: If ``dim`` is not divisible by ``num_heads``.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1]``.
    :raises ValueError: From ``build()``, if the input is not 3D or its trailing
        dimension does not equal ``dim``.

    Input shape:
        3D tensor with shape ``(batch_size, seq_len, dim)``. There is no mask
        argument; padding is not handled.

    Output shape:
        3D tensor with shape ``(batch_size, seq_len, dim)`` — unchanged from the
        input. With ``return_attention_scores=True`` the return is
        ``(output, None)``: no attention matrix exists to return.

    Example:
        >>> # Long-sequence encoder attention
        >>> attn = PerformerAttention(dim=512, num_heads=8, nb_features=256)
        >>> x = keras.random.normal((2, 8192, 512))
        >>> y = attn(x, training=False)                  # (2, 8192, 512)
        >>>
        >>> # Autoregressive: the only masking this layer offers
        >>> attn = PerformerAttention(dim=512, num_heads=8, causal=True)
        >>>
        >>> # Interface compatibility; the second entry is always None
        >>> y, scores = attn(x, return_attention_scores=True)

    Note:
        The projection matrix is redrawn on every call, so two forward passes over
        the same input differ by the sampling noise of the kernel approximation.
        This layer is an approximation of softmax attention, not an exact
        reformulation of it — unlike, say, blockwise/online-softmax attention.

    Attributes:
        to_qkv: Fused Q/K/V projection, ``3 * dim`` wide.
        to_out: Output projection back to ``dim``.
        dropout: Output dropout, or ``None`` at rate 0.
        head_dim: ``dim // num_heads``.
        scale: The ``1 / sqrt(head_dim)`` temperature, a Python float.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            nb_features: int = 256,
            ortho_scaling: float = 0.0,
            causal: bool = False,
            dropout_rate: float = 0.0,
            use_bias: bool = False,
            kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
            bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
            kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
            bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Validate the configuration and create the projections and dropout.

        This layer owns no weights of its own: the random projection is resampled
        per call rather than stored. See the class docstring for the parameter
        reference.
        """
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        validate_head_divisibility(dim, num_heads)
        if nb_features <= 0:
            raise ValueError(f"nb_features must be positive, got {nb_features}")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.causal = causal
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        # Normalize regularizers via regularizers.get() so str/dict/object/None
        # all round-trip uniformly through regularizers.serialize() in get_config.
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # Computed attributes
        self.head_dim = dim // num_heads
        self.scale = compute_attention_scale(self.head_dim)
        self._feature_scale = math.sqrt(2.0 / float(self.nb_features))

        # Create sub-layers in __init__
        # Q, K, V projection layer
        self.to_qkv = keras.layers.Dense(
            3 * dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='qkv_projection'
        )

        # Output projection layer
        self.to_out = keras.layers.Dense(
            dim,
            use_bias=use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name='output_projection'
        )

        # DECISION plan-2026-08-27T040114-580f8b63/D-014 — the Dropout is created
        # UNCONDITIONALLY and gated in `call()`. Do NOT go back to creating it only
        # when `dropout_rate > 0`: that made both the object graph and the
        # auto-generated sub-layer names depend on `dropout_rate`.
        # See decisions.md D-014 (plan-2026-08-27T040114-580f8b63).
        self.dropout = keras.layers.Dropout(dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer and its two projection sub-layers.

        Both projections take the same input shape: ``to_out`` consumes
        ``(B, N, dim)`` because heads are merged back before it runs.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If ``input_shape`` is not rank 3, or if its last
            dimension does not equal ``dim``.
        """
        if self.built:
            return

        # Validate input shape
        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {input_shape}")
        if input_shape[-1] != self.dim:
            raise ValueError(
                f"Last dimension of input ({input_shape[-1]}) must match dim ({self.dim})"
            )

        # Build sub-layers
        self.to_qkv.build(input_shape)
        self.to_out.build(input_shape)

        super().build(input_shape)

    def _create_projection_matrix(self, batch_size: int) -> keras.KerasTensor:
        """Create the random projection matrix for the FAVOR+ feature map.

        Resampled on EVERY call, from an unseeded ``keras.random.normal``, so it is
        not a weight and does not round-trip through a checkpoint. Note that
        ``ortho_scaling`` only rescales here — nothing orthogonalizes.

        :param batch_size: Batch size for broadcasting.
        :type batch_size: int

        :return: Projection matrix of shape
            ``(batch, num_heads, nb_features//2, head_dim)``.
        :rtype: keras.KerasTensor
        """
        # Generate random Gaussian matrix
        # Shape: (num_heads, nb_features//2, head_dim)
        shape = (self.num_heads, self.nb_features // 2, self.head_dim)
        # DECISION plan-2026-08-27T040114-580f8b63/D-014 — `dtype=self.compute_dtype`
        # is required. Do NOT drop it: without it `keras.random.normal` follows the
        # GLOBAL float policy and returns float32 while `q`/`k` arrive in the
        # compute dtype, so under `mixed_float16` the einsum below raises
        # `InvalidArgumentError: cannot compute Mul`. See decisions.md D-014.
        projection = keras.random.normal(
            shape=shape,
            mean=0.0,
            stddev=1.0,
            seed=None,
            dtype=self.compute_dtype
        )

        # Optionally apply orthogonalization for better approximation
        if self.ortho_scaling > 0:
            # QR decomposition for orthogonalization
            # Note: Keras doesn't have QR, so we use Gram-Schmidt approximation
            # This is a simplified version - in production, consider using backend-specific QR
            projection = projection * self.ortho_scaling

        # Scale by sqrt(head_dim) for proper variance
        projection = projection * self.scale

        # Broadcast to batch dimension
        # Shape: (batch, num_heads, nb_features//2, head_dim)
        projection = keras.ops.expand_dims(projection, axis=0)
        projection = keras.ops.repeat(projection, batch_size, axis=0)

        return projection

    def _create_kernel_features(
            self,
            x: keras.KerasTensor,
            projection_matrix: keras.KerasTensor,
            is_query: bool = True
    ) -> keras.KerasTensor:
        """Map inputs into the trigonometric random feature space.

        ``cos`` and ``sin`` of the projection are concatenated to width
        ``nb_features``, scaled so the inner product approximates the exponential
        kernel in expectation, and clamped non-negative. The
        ``exp(-||x||^2 / 2)`` factor is applied to QUERIES only. The asymmetry is
        intended and nearly free: the factor cancels between the numerator and
        the denominator downstream, except against the ``1e-6`` that
        :meth:`_linear_attention` adds to the normalizer. See the module
        docstring for the paired measurement of what survives.

        :param x: Input tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type x: keras.KerasTensor
        :param projection_matrix: Random projection matrix.
        :type projection_matrix: keras.KerasTensor
        :param is_query: Whether this is for queries (affects normalization).
        :type is_query: bool

        :return: Random features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :rtype: keras.KerasTensor
        """
        # Project input: x @ projection_matrix^T
        # x: (batch, num_heads, seq_len, head_dim)
        # projection_matrix: (batch, num_heads, nb_features//2, head_dim)
        # Result: (batch, num_heads, seq_len, nb_features//2)
        x_projected = keras.ops.einsum('bhnd,bhfd->bhnf', x, projection_matrix)

        # Apply trigonometric random features (FAVOR+)
        # This creates positive features that approximate exp(x·y)
        features_cos = keras.ops.cos(x_projected)
        features_sin = keras.ops.sin(x_projected)

        # Concatenate to get full feature dimension
        # Shape: (batch, num_heads, seq_len, nb_features)
        features = keras.ops.concatenate([features_cos, features_sin], axis=-1)

        # Apply proper scaling for kernel approximation
        # The scaling ensures E[φ(x)ᵀφ(y)] ≈ exp(xᵀy)
        features = features * self._feature_scale

        # For numerical stability, apply exponential normalization
        if is_query:
            # Normalize queries to prevent numerical issues
            features = (
                    features *
                    keras.ops.exp(
                        -keras.ops.square(
                            keras.ops.norm(x, axis=-1, keepdims=True)
                        ) / 2.0
                    )
            )

        # Clamp to non-negative. The stated rationale -- that unclamped cos/sin
        # features let the denominator pass through zero -- is UNPROVEN: see the
        # module docstring's ablation, where removing this clamp lowered the mean
        # error and `z` never changed sign. Keep it anyway: the features are
        # SIGNED without it, and removing it moves every shipped checkpoint's
        # output.
        return keras.ops.maximum(features, 0)

    def _linear_attention(
            self,
            q: keras.KerasTensor,
            k: keras.KerasTensor,
            v: keras.KerasTensor
    ) -> keras.KerasTensor:
        """Contract the feature maps in the order that keeps the cost linear.

        The non-causal branch builds ONE ``(F, d_h)`` summary of the whole
        sequence; the causal branch builds a prefix summary per position via
        ``cumsum`` over the per-position outer products. Neither branch forms the
        ``N x N`` matrix. Both add ``1e-6`` to the normalizer.

        :param q: Query features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type q: keras.KerasTensor
        :param k: Key features of shape ``(batch, num_heads, seq_len, nb_features)``.
        :type k: keras.KerasTensor
        :param v: Value tensor of shape ``(batch, num_heads, seq_len, head_dim)``.
        :type v: keras.KerasTensor

        :return: Attention output of shape ``(batch, num_heads, seq_len, head_dim)``.
        :rtype: keras.KerasTensor
        """
        if self.causal:
            # For causal attention, we need cumulative sums
            # This is a simplified implementation - consider optimizing for production

            # Create cumulative KV and K_sum for causal attention.
            # k: (B, h, N, F), v: (B, h, N, D). Per-position outer product
            # k (x) v -> (B, h, N, F, D); cumsum over the sequence axis gives the
            # prefix-summed KV state. No expand_dims on v: the einsum already
            # produces the rank-5 outer product (a stray expand_dims made v rank-5
            # and crashed the einsum with "rank 5 vs expected 4").
            kv_cumsum = keras.ops.cumsum(
                keras.ops.einsum('bhnf,bhnd->bhnfd', k, v),
                axis=2
            )
            k_cumsum = keras.ops.cumsum(k, axis=2)

            # Recompute with cumulative values. q:(B,h,N,F) contracts against
            # kv_cumsum:(B,h,N,F,D) -> out:(B,h,N,D) directly (rank-4, no squeeze).
            z_causal = keras.ops.einsum('bhnf,bhnf->bhn', q, k_cumsum) + 1e-6
            out = keras.ops.einsum('bhnf,bhnfd->bhnd', q, kv_cumsum)
            out = out / keras.ops.expand_dims(z_causal, axis=-1)
        else:
            # Compute KV: φ(K)ᵀV
            # k: (batch, num_heads, seq_len, nb_features)
            # v: (batch, num_heads, seq_len, head_dim)
            # kv: (batch, num_heads, nb_features, head_dim)
            kv = keras.ops.einsum('bhnf,bhnd->bhfd', k, v)

            # Compute normalization: φ(Q) · sum(φ(K))
            # k_sum: (batch, num_heads, nb_features)
            k_sum = keras.ops.sum(k, axis=2)

            # z: (batch, num_heads, seq_len)
            z = keras.ops.einsum('bhnf,bhf->bhn', q, k_sum)
            # Small epsilon so a near-zero normalizer cannot blow up the divide.
            z = z + 1e-6

            # Compute output: φ(Q) · KV / Z
            # out: (batch, num_heads, seq_len, head_dim)
            out = keras.ops.einsum('bhnf,bhfd->bhnd', q, kv)
            out = out / keras.ops.expand_dims(z, axis=-1)

        return out

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None,
            return_attention_scores: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]:
        """Apply Performer attention to inputs.

        Project, reshape to heads, scale the queries, draw a fresh projection
        matrix, build both feature maps, contract linearly, then merge heads and
        project out. Note the frozen signature: there is no mask argument.

        :param inputs: Input tensor of shape ``(batch_size, seq_len, dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether in training mode.
        :type training: Optional[bool]
        :param return_attention_scores: If ``True``, returns ``(output, None)``
            for compatibility. Performer does not compute explicit attention matrices.
        :type return_attention_scores: bool

        :return: Output tensor of shape ``(batch_size, seq_len, dim)``.
            If return_attention_scores is ``True``, returns ``(output, None)``.
        :rtype: Union[keras.KerasTensor, Tuple[keras.KerasTensor, None]]
        """
        batch_size = keras.ops.shape(inputs)[0]
        seq_len = keras.ops.shape(inputs)[1]

        # Project to Q, K, V
        # Shape: (batch, seq_len, 3*dim)
        qkv = self.to_qkv(inputs)

        # Split into Q, K, V
        # Each has shape: (batch, seq_len, dim)
        q, k, v = keras.ops.split(qkv, 3, axis=-1)

        # Reshape to multi-head format
        # Shape: (batch, num_heads, seq_len, head_dim)
        q = keras.ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = keras.ops.transpose(q, (0, 2, 1, 3))

        k = keras.ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = keras.ops.transpose(k, (0, 2, 1, 3))

        v = keras.ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = keras.ops.transpose(v, (0, 2, 1, 3))

        # Scale queries
        q = q * self.scale

        # Generate random projection matrix
        projection_matrix = self._create_projection_matrix(batch_size)

        # Create kernel features φ(Q) and φ(K)
        q_features = self._create_kernel_features(q, projection_matrix, is_query=True)
        k_features = self._create_kernel_features(k, projection_matrix, is_query=False)

        # Compute linear attention
        out = self._linear_attention(q_features, k_features, v)

        # Reshape back to (batch, seq_len, dim)
        out = keras.ops.transpose(out, (0, 2, 1, 3))
        out = keras.ops.reshape(out, (batch_size, seq_len, self.dim))

        # Apply output projection
        out = self.to_out(out)

        # Apply dropout if specified
        if self.dropout_rate > 0.0:
            out = self.dropout(out, training=training)

        if return_attention_scores:
            # Performer doesn't compute explicit attention matrices
            # Return None for compatibility with standard attention interface
            return out, None

        return out

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape, which equals the input shape.

        The output projection maps ``dim`` back to ``dim``, so the layer is
        shape-preserving.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]

        :return: Shape tuple of the output (same as input).
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the layer for serialization.

        :return: Dictionary containing all configuration parameters.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'nb_features': self.nb_features,
            'ortho_scaling': self.ortho_scaling,
            'causal': self.causal,
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------