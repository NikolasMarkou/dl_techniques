"""
Energy attention of the Energy Transformer (ET), in :class:`EnergyAttention`,
plus the module-level mask helpers ``_token_keep`` / ``_symmetric_token_keep``.

This is not standard attention: there is no value matrix. The layer defines a
scalar energy ``E_ATT(g)`` over the token state, and ``call()`` returns the
negative gradient of that energy, a descent direction rather than a weighted
sum of values. The gradient is hand-coded in closed form with ``keras.ops``
(no autodiff — ``keras.ops.grad`` does not exist in Keras 3.8, and a
backend-specific autodiff tape is not allowed in ``src/``); nothing proves the
derivation except the autodiff oracle test ``test_gradient_oracle`` in
``tests/test_layers/test_attention/test_energy_attention.py``. Edit
:meth:`EnergyAttention.energy` and :meth:`EnergyAttention.update` as a pair —
a wrong descent direction still runs, trains and looks finite.

For layer-normed tokens ``g`` (shape ``(B, N, D)``), heads ``h``, key/query
projections ``K = w_key g`` and ``Q = w_query g``, and inverse temperature
``beta = 1/sqrt(head_dim)``::

    E_ATT(g) = -(1/beta) * sum_{h,q} logsumexp_k ( beta * K[h,k] . Q[h,q] )

References:
    - Hoover, Liang, Pham, Panda, Strobelt, Zaki, Chau, Krotov (2023).
      "Energy Transformer". NeurIPS 2023. (https://arxiv.org/abs/2302.07253) —
      equations (3)-(4).
"""

import keras
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.logger import logger

# DECISION plan-2026-07-27T130643-38c5646a/D-007: keep these 3 names defined here,
# never delete or re-point them — energy_transformer.py imports them by name. See decisions.md.
#
# DECISION plan_2026-07-13_57c9833e/D-009: this bias only ever lives in mask dtype
# (>= float32); in fp16 it is -inf. See decisions.md.
from dl_techniques.layers.attention.common import (
    MASK_BIAS_VALUE as _MASK_BIAS_VALUE,
    mask_dtype as _mask_dtype,
    compute_attention_scale as _compute_attention_scale,
)
from dl_techniques.utils.keras_registration import register_dl_technique


def _token_keep(
    mask: keras.KerasTensor,
    dtype: str,
) -> keras.KerasTensor:
    """Validate and cast a Keras-propagated rank-2 ``(B, N)`` per-token validity mask.

    The one place the rank-2 contract for the Keras-propagated ``mask`` lives, with
    four call sites named here rather than line-numbered (line numbers go stale):

    1. :meth:`EnergyAttention._build_keep_mask`, which then expands the result
       symmetrically over the key and query axes (``_symmetric_token_keep``).
    2. ``HopfieldNetwork.energy`` in ``layers/transformers/energy_transformer.py``,
       per-token and un-expanded, to drop pad tokens from the energy sum.
    3. ``HopfieldNetwork.update`` in the same file, the matching gradient side; the
       pair must see the same keep, or ``update() != -dE/dg``.
    4. ``EnergyTransformer._hopfield_token_mask``, which ands the Keras mask with a
       rank-2 ``attention_mask`` (D-006).

    Do not re-implement the rank check at any call site.

    :param mask: Boolean (or 0/1) per-token validity mask of shape ``(B, N)``.
    :type mask: keras.KerasTensor
    :param dtype: Dtype to cast into — the mask compute dtype (>= float32) wherever the
        value feeds a ``-1e9`` bias or a reduction (D-009), the layer's compute dtype
        where it merely gates an already-computed tensor.
    :type dtype: str

    :return: 0/1 tensor of shape ``(B, N)`` in ``dtype``.
    :rtype: keras.KerasTensor

    :raises ValueError: If ``mask`` is not rank 2.
    """
    rank = len(mask.shape)
    if rank != 2:
        raise ValueError(
            "the Keras-propagated `mask` must have rank 2 (B, N) — a per-token "
            f"validity mask — got rank {rank}. Pass a rank-3/rank-4 mask as "
            "`attention_mask` instead."
        )
    return ops.cast(mask, dtype)


def _symmetric_token_keep(token_keep: keras.KerasTensor) -> keras.KerasTensor:
    """Expand a rank-2 ``(B, N)`` per-token validity mask to a ``(B, 1, N, N)`` keep tensor.

    ``keep[b, :, n, m] = token_keep[b, n] * token_keep[b, m]`` — the token is removed
    from both the key role ``n`` and the query role ``m``. This is D-008's semantics,
    factored out for its two call sites (the explicit rank-2 ``attention_mask`` and the
    Keras-propagated ``mask``); see the D-008 / D-002 anchors in :meth:`_build_keep_mask`.
    Do not duplicate it, or weaken either factor to a key-only mask.

    :param token_keep: 0/1 tensor of shape ``(B, N)`` in the mask compute dtype.
    :type token_keep: keras.KerasTensor

    :return: 0/1 keep tensor of shape ``(B, 1, N, N)``, axis 2 = key ``n``, axis 3 = query ``m``.
    :rtype: keras.KerasTensor
    """
    # key_keep is (B,1,N,1) and query_keep is (B,1,1,N), so the product broadcasts
    # to (B,1,N,N).
    key_keep = ops.expand_dims(ops.expand_dims(token_keep, axis=1), axis=-1)
    query_keep = ops.expand_dims(ops.expand_dims(token_keep, axis=1), axis=2)
    return key_keep * query_keep


@register_dl_technique("dl_techniques.layers.attention.energy_attention")
class EnergyAttention(keras.layers.Layer):
    """Energy Transformer multi-head energy attention (bias-free, no value matrix).

    Exposes a scalar token-mixing energy ``E_ATT(g)`` together with its exact
    closed-form negative gradient, so that an :class:`EnergyTransformer` block
    can perform gradient descent on ``E_ATT + E_HN`` instead of running an
    opaque ``attn -> FFN`` residual stream.

    Architecture:

    .. code-block:: text

                      Input  g   [B, N, D]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
        w_key  (Y, H, D)                w_query  (Y, H, D)
        einsum ► K  [B,Y,H,N]           einsum ► Q  [B,Y,H,N]
        (bias-free)                     (bias-free)
               └───────────────┬───────────────┘
                               ▼
                  A = sum_y K·Q      [B, H, N, N]
                  n = key index,  m = query index
                               ▼
                  logits = beta * A, cast to mask_dtype
                  (>= float32)
                               ▼
                  optional: logits = logits ⊙ Ŵ
                  (adjacency_weight, paper eq. 25)
                               ▼
                  keep [B,1,N,N] ► additive -1e9 bias.
                  The diagonal is dropped when attn_self
                  is False; a pad token drops in both the
                  key role and the query role.
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
        energy(g)                       update(g) == call(g)
        lse = logsumexp_n(logits)       omega = softmax_n(logits)
        gate columns that keep          omega = omega * keep
        no key at all                   omega = omega * Ŵ (optional)
               │                               │
               ▼                               ▼
        E = -(1/beta) *                 term_q = einsum(w_query,K,omega)
            sum_h,m lse                 term_k = einsum(w_key,  Q,omega)
        ► [B,]  a diagnostic            ► term_q + term_k  [B, N, D]
          trace; it never drives          == -dE_ATT/dg
          the state update

    There is no value matrix and no output projection in this layer. A
    consumer adds ``step_size * call(g)`` to ``g``.

    Mathematics (notation: ``B``=batch, ``N``=tokens, ``D``=``dim``, ``Y``=``head_dim``,
    ``H``=``num_heads``; ``n`` indexes a token in its key role, ``m`` in its query
    role):

    .. code-block:: text

        K_{y h n} = sum_d W^K_{y h d} g_{n d}     # keys    (no bias)
        Q_{y h m} = sum_d W^Q_{y h d} g_{m d}     # queries (no bias)
        A_{h n m} = sum_y K_{y h n} Q_{y h m}

        E_ATT = -(1/beta) * sum_h sum_m
                    log( sum_{n valid} exp(beta * A_{h n m}) )

    The ``n != m`` exclusion (``attn_self=False``, the paper's ET-Full image config) is a
    mask, not a separate code path; appendix eq. 13 permits ``attn_self=True``.

    Closed-form gradient, with ``omega`` the softmax of ``beta * A`` over the key
    index ``n`` (masked entries zeroed):

    .. code-block:: text

        -dE_ATT/dg_{i d} = sum_h sum_y
            [ W^Q_{y h d} * ( sum_n K_{y h n} omega_{h n i} )
            + W^K_{y h d} * ( sum_m Q_{y h m} omega_{h i m} ) ]

    Derivation sketch:

    - ``dE_ATT/dA_{h n m} = -(1/beta) * beta * omega_{h n m} = -omega_{h n m}``.
    - ``A_{h n m}`` depends on ``g_i`` through ``K`` when ``n == i`` and through ``Q``
      when ``m == i``.
    - ``dA_{h n m}/dg_{i d} = delta_{n,i} sum_y W^K_{y h d} Q_{y h m}
      + delta_{m,i} sum_y W^Q_{y h d} K_{y h n}``.
    - Chain rule, then negate, gives the two terms above. Both softmax normalizations
      are over the key index ``n``.

    The second term is the ET-specific contribution; vanilla attention has no
    equivalent. See the ``D-001`` anchor on ``term_k`` in :meth:`update`.

    :meth:`update` returns ``-dE/dg``, the descent direction, not the gradient.
    A consumer adds ``step_size * update``. Flipping this sign turns the
    block's dynamics into energy ascent, which still runs and produces finite
    outputs, so a wrong sign is silent.

    This layer and ``HopfieldNetwork`` both expose the trio ``energy(g, ...)
    -> (B,)`` / ``update(g, ...) -> (B, N, D)`` / ``call(...) ->
    update(...)`` as a duck-typed convention rather than a shared base class
    or ``Protocol``, since there are only two implementors and one consumer.

    :param dim: Token embedding dimension ``D``. The only required argument.
    :type dim: int
    :param num_heads: Number of attention heads ``H``. Defaults to ``8``.
    :type num_heads: int
    :param head_dim: Per-head key/query dimension ``Y``. ``None`` -> ``dim // num_heads``.
    :type head_dim: Optional[int]
    :param beta: Inverse temperature. ``None`` -> ``1 / sqrt(head_dim)``.
    :type beta: Optional[float]
    :param attn_self: If ``False`` (default, paper's ET-Full), a token is excluded from
        attending to itself (the diagonal ``n == m`` is masked out). If ``True``, the
        diagonal is kept (appendix eq. 13).
    :type attn_self: bool
    :param kernel_initializer: Initializer for ``w_key`` / ``w_query``. Defaults to
        ``TruncatedNormal(stddev=0.02)`` (the paper's ``N(0, 0.02)``).
    :type kernel_initializer: Union[str, initializers.Initializer]

    :raises ValueError: If ``dim <= 0``, ``num_heads <= 0``, the resolved ``head_dim <= 0``,
        or an explicitly-supplied ``beta <= 0``.

    Input shape:
        3D tensor ``(batch, num_tokens, dim)``.

    Output shape:
        Identical to the input shape — ``(batch, num_tokens, dim)``.

    Attributes:
        w_key: Bias-free key projection, shape ``(head_dim, num_heads, dim)``.
        w_query: Bias-free query projection, shape ``(head_dim, num_heads, dim)``.

    Example:
        >>> layer = EnergyAttention(dim=64, num_heads=4)
        >>> g = keras.random.normal((2, 16, 64))
        >>> layer.energy(g).shape       # scalar energy per batch element
        (2,)
        >>> layer(g).shape              # == layer.update(g) == -dE/dg
        (2, 16, 64)

    References:
        - Hoover et al., "Energy Transformer", NeurIPS 2023, arXiv:2302.07253, eq. (3)-(4).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        beta: Optional[float] = None,
        attn_self: bool = False,
        kernel_initializer: Union[str, initializers.Initializer] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and resolve ``head_dim`` and ``beta``.

        Every argument is documented on the class. The two projection weights are
        created in :meth:`build`, not here, because their shape needs ``dim``, which
        is a constructor argument, and Keras expects weight creation in ``build``.

        :param kwargs: Forwarded to ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``dim <= 0``, ``num_heads <= 0``, the resolved
            ``head_dim <= 0``, or an explicitly-supplied ``beta <= 0``.
        """
        super().__init__(**kwargs)

        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(f"num_heads must be a positive integer, got {num_heads}")

        resolved_head_dim = dim // num_heads if head_dim is None else head_dim
        if not isinstance(resolved_head_dim, int) or resolved_head_dim <= 0:
            raise ValueError(
                "head_dim must resolve to a positive integer, got "
                f"{resolved_head_dim} (dim={dim}, num_heads={num_heads}, "
                f"head_dim={head_dim})"
            )
        if beta is not None and (not isinstance(beta, (int, float)) or beta <= 0):
            raise ValueError(f"beta must be a positive number or None, got {beta}")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.beta = float(beta) if beta is not None else None
        self.attn_self = bool(attn_self)
        self.kernel_initializer = (
            initializers.TruncatedNormal(stddev=0.02)
            if kernel_initializer is None
            else initializers.get(kernel_initializer)
        )

        self._head_dim = int(resolved_head_dim)
        # Resolved in __init__, not call(), so self._beta stays a Python float
        # and folds into the traced graph as a constant (D-002).
        self._beta = (
            float(beta) if beta is not None
            else _compute_attention_scale(self._head_dim)
        )

        self.w_key: Optional[keras.Variable] = None
        self.w_query: Optional[keras.Variable] = None

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyAttention with dim={self.dim}, "
            f"num_heads={self.num_heads}, head_dim={self._head_dim}, "
            f"beta={self._beta:.6f}, attn_self={self.attn_self}"
        )


    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Create the bias-free ``(Y, H, D)`` key and query projections.

        :param input_shape: Input shape ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :raises ValueError: If the last axis of ``input_shape`` is not ``dim``.
        """
        if self.built:
            return

        feature_dim = input_shape[-1]
        if feature_dim is not None and int(feature_dim) != self.dim:
            raise ValueError(
                f"Input feature dimension {feature_dim} does not match dim={self.dim}"
            )

        # (Y, H, D) — per-head dim first, so the einsums below read 'yhd'.
        w_shape = (self._head_dim, self.num_heads, self.dim)

        # No bias: E_ATT is defined without one, and a bias term would not be
        # expressible in the closed-form gradient below.
        self.w_key = self.add_weight(
            name="w_key",
            shape=w_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )
        # DECISION plan-2026-08-19T163559-499b6f0e/D-068: clone_initializer here, a
        # shared instance made w_key/w_query bit-identical (symmetric K^T Q). See decisions.md.
        self.w_query = self.add_weight(
            name="w_query",
            shape=w_shape,
            initializer=clone_initializer(self.kernel_initializer),
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    def _build_keep_mask(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
        mask: Optional[keras.KerasTensor] = None,
    ) -> Tuple[keras.KerasTensor, bool]:
        """Normalize the user keep mask to a ``(b, h, n, m)``-broadcastable 0/1 tensor.

        ``attention_mask`` is a keep mask (``1`` = attend, ``0`` = masked), the same
        convention as the sibling ``MultiHeadCrossAttention``. A rank-2 ``(B, N)`` mask
        is applied symmetrically to both the key axis ``n`` and the query axis ``m``,
        unlike the sibling's key-only rank-2 reading: in ET a token masked only as a
        key still acts as a query, and the ``term_k`` gradient term would otherwise
        keep propagating it. When both ``attention_mask`` and the Keras-propagated
        ``mask`` are supplied, precedence is logical and — neither can resurrect a
        token the other hid. The ``attn_self=False`` diagonal exclusion is always-on
        and anded in separately, since ET can generate a fully-masked query column on
        its own (``N == 1``).

        # DECISION plan_2026-07-13_57c9833e/D-006: keep mask, not an additive bias;
        # do not re-interpret as a drop mask. See decisions.md.
        #
        # DECISION plan_2026-07-13_57c9833e/D-008: rank-2 mask applies symmetrically
        # to key AND query axes, not key-only. See decisions.md.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-002: the Keras-propagated mask merges
        # here via D-008's symmetric expansion, not its own path. See decisions.md.
        #
        # DECISION plan_2026-07-13_ca4f71a2/D-003: precedence when both masks are
        # given is logical AND; never let the explicit mask win. See decisions.md.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: Keep mask of shape ``(B, N)`` (a per-token validity mask,
            applied to both the key and the query axis — D-008), ``(B, N, N)``
            (interpreted ``(b, n, m)`` with ``n`` = key and ``m`` = query, broadcast over
            heads), or ``(B, H, N, N)``. ``None`` means attend everywhere.
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean per-token validity
            mask, anded with ``attention_mask`` (D-003).
        :type mask: Optional[keras.KerasTensor]

        :return: ``(keep, is_masked)``. ``keep`` is a 0/1 tensor in float32 (not the
            compute dtype — D-009), broadcastable to ``(B, H, N, N)`` with axis 2 = key
            index ``n`` and axis 3 = query index ``m``. ``is_masked`` is a Python bool:
            ``False`` means nothing is masked anywhere (no user mask and
            ``attn_self=True``), so the additive bias can be skipped entirely.
        :rtype: Tuple[keras.KerasTensor, bool]

        :raises ValueError: If ``attention_mask`` or ``mask`` has an unsupported rank.
        """
        num_tokens = ops.shape(g)[1]
        mask_dtype = _mask_dtype(self.compute_dtype)

        # is_masked is a Python bool, resolvable at trace time from attn_self and
        # whether a mask tensor was passed, never from tensor values, so it is graph-safe.
        is_masked = (
            (attention_mask is not None) or (mask is not None) or (not self.attn_self)
        )

        if attention_mask is None:
            keep = ops.ones((1, 1, 1, 1), dtype=mask_dtype)
        else:
            explicit = ops.cast(attention_mask, mask_dtype)
            rank = len(explicit.shape)
            if rank == 2:
                # (B, N) -> (B, 1, N, N) symmetrically: keep[b,:,n,m] = mask[b,n]*mask[b,m].
                keep = _symmetric_token_keep(explicit)
            elif rank == 3:
                # (B, N, N) read as (b, n, m) -> (B, 1, N, N): broadcast over heads.
                keep = ops.expand_dims(explicit, axis=1)
            elif rank == 4:
                # (B, H, N, N) already in the einsum layout.
                keep = explicit
            else:
                raise ValueError(
                    "attention_mask must have rank 2 (B, N), 3 (B, N, N) or "
                    f"4 (B, H, N, N), got rank {rank}"
                )

        if mask is not None:
            # Keras-propagated mask (D-002), cast to mask dtype so it never lands
            # in fp16 where -1e9 is -inf (D-009); multiplying is the logical AND (D-003).
            keras_mask = _token_keep(mask, mask_dtype)
            keep = keep * _symmetric_token_keep(keras_mask)

        if not self.attn_self:
            # Always-on diagonal exclusion (n == m), anded with the user mask.
            eye = ops.eye(num_tokens, dtype=mask_dtype)
            keep = keep * (1.0 - ops.reshape(eye, (1, 1, num_tokens, num_tokens)))

        return keep, is_masked

    def _project(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> Tuple[keras.KerasTensor, ...]:
        """Compute ``K``, ``Q``, the masked ``logits`` and the ``keep`` mask.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional keep mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated ``(B, N)`` mask, anded with
            ``attention_mask`` (see :meth:`_build_keep_mask`, D-002/D-003).
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional finite per-head weighted-adjacency ``Ŵ``
            broadcastable to ``(B, H, N, N)`` (paper eq.25, Branch A). When supplied it is
            folded multiplicatively into the raw score before the ``beta`` scaling and the
            keep-bias: ``logit = beta * (A ⊙ Ŵ) + M``. It is a per-block constant
            (``dŴ/dg == 0``) hoisted by the block, not a reinterpretation of the 0/1
            ``attention_mask`` (D-002/D-006). ``None`` (default) keeps this code path
            unreachable and the layer byte-identical to today.
        :type adjacency_weight: Optional[keras.KerasTensor]

        ``K`` and ``Q`` come back in the compute dtype (they feed the gradient einsums,
        which contract against the compute-dtype weights). ``logits`` and ``keep`` come back
        in float32, unconditionally — see the D-009 anchor at ``_MASK_BIAS_VALUE``.

        :return: ``(K, Q, logits, keep)`` with ``K``/``Q`` of shape ``(B, Y, H, N)`` in the
            compute dtype and ``logits``/``keep`` broadcastable to ``(B, H, N, N)`` in
            float32 (``n`` = key axis 2).
        :rtype: Tuple[keras.KerasTensor, ...]
        """
        keep, is_masked = self._build_keep_mask(g, attention_mask, mask=mask)

        # k and q are (B, Y, H, N) with n = key and m = query; a is (B, H, N, N).
        k = ops.einsum('yhd,bnd->byhn', self.w_key, g)
        q = ops.einsum('yhd,bmd->byhm', self.w_query, g)
        a = ops.einsum('byhn,byhm->bhnm', k, q)

        # DECISION plan_2026-07-13_57c9833e/D-009: logits->bias->logsumexp chain stays
        # in float32+ throughout, bias applied via ops.where, never (1-keep)*NEG. See decisions.md.
        # (B, H, N, N) in the mask dtype, i.e. float32 or wider.
        logits = ops.cast(a, _mask_dtype(self.compute_dtype)) * self._beta
        if adjacency_weight is not None:
            # Branch A (paper eq.25): fold the learned edge weight w_hat into the score
            # multiplicatively, beta * (A * w_hat); a per-block constant w.r.t. g.
            w_hat = ops.cast(adjacency_weight, _mask_dtype(self.compute_dtype))
            logits = logits * w_hat
        if is_masked:
            # Both where branches are tensors in the mask dtype, so a bare Python
            # 0.0/-1e9 pair does not get promoted and collide with a float64 logits.
            bias = ops.where(
                keep > 0.0,
                ops.zeros_like(keep),
                ops.full_like(keep, _MASK_BIAS_VALUE),
            )
            logits = logits + bias

        return k, q, logits, keep

    def energy(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Scalar attention energy ``E_ATT`` per batch element (paper eq. 4).

        .. code-block:: text

            E_ATT = -(1/beta) * sum_h sum_m
                        logsumexp_n( beta * A_{h n m} )

        The ``logsumexp`` is over the key axis ``n``. A query column ``m`` whose keys
        are all masked contributes exactly 0 (not ``-1e9 / beta``): the ``col_valid``
        indicator gates it out. This is what makes ``N == 1`` with ``attn_self=False``
        return ``0.0`` rather than a huge negative number or a NaN.

        This formula is the spec: :meth:`update` must match it exactly, never edited
        to make the gradient oracle pass.

        :param g: Token state ``(B, N, D)``, typically the output of ``EnergyLayerNorm``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional keep mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask.
            When both masks are supplied the rule is a logical and — a token is attended
            only if valid under both, and neither mask can un-mask what the other hid
            (decisions.md D-003). This parameter exists because ``energy()`` is part of the
            public duck-typed surface an ``EnergyTransformer`` block calls directly, outside
            ``__call__``, where Keras cannot inject the mask for us.
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional finite weighted adjacency ``Ŵ`` (paper eq.25,
            Branch A), folded into the score as ``beta * (A ⊙ Ŵ)`` (see :meth:`_project`).
            It must be the same tensor passed to :meth:`update`, or ``update() != -dE/dg``.
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        # Callable outside __call__, where Keras has not opened an autocast scope,
        # so cast explicitly or a float32 g meets float16 weights under mixed_float16.
        g = ops.cast(g, self.compute_dtype)

        # mask and adjacency_weight must reach this same _project call as in update(),
        # or update() stops being -dE/dg; test_gradient_oracle catches a drift here.
        _, _, logits, keep = self._project(
            g, attention_mask, mask=mask, adjacency_weight=adjacency_weight
        )

        # logsumexp over the key axis n (axis=2) -> (B, H, N) indexed by (b, h, m).
        lse = ops.logsumexp(logits, axis=2)

        # A fully-masked query column must contribute exactly 0 energy.
        col_valid = ops.cast(
            ops.sum(keep, axis=2) > 0.0, _mask_dtype(self.compute_dtype)
        )

        # DECISION plan_2026-07-13_ca4f71a2/D-005: energy stays in the reduce dtype
        # (>= float32); casting back to compute dtype hit -inf at N>=512 under mixed_float16. See decisions.md.
        return -(1.0 / self._beta) * ops.sum(lse * col_valid, axis=(1, 2))

    def update(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Return ``-dE_ATT/dg``, the descent direction, not the gradient.

        This is the negative gradient. The consumer adds ``step_size * update``
        to the token state. Flipping this sign at the call site silently
        inverts the dynamics into energy ascent, which still runs, trains, and
        produces finite outputs.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional keep mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask,
            anded with ``attention_mask`` (decisions.md D-003). It must be passed to the
            same ``_project`` call ``energy()`` makes, or ``update() != -dE/dg``.
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional finite weighted adjacency ``Ŵ`` (paper eq.25,
            Branch A). It must be the same tensor passed to :meth:`energy`, or
            ``update() != -dE/dg``.
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: ``-dE_ATT/dg`` of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        # Cast at the head of the public method — see the note in energy().
        g = ops.cast(g, self.compute_dtype)

        # Same masks and adjacency_weight as energy() — a weight landing in only
        # one of them makes update() != -dE/dg.
        k, q, logits, keep = self._project(
            g, attention_mask, mask=mask, adjacency_weight=adjacency_weight
        )

        # Softmax over the key axis n, then zero the masked keys. The post-softmax
        # * keep is not redundant with the additive bias: softmax of an all -1e9
        # row returns a uniform 1/N, which additive biasing alone cannot fix.
        omega = ops.softmax(logits, axis=2) * keep

        # DECISION plan-2026-07-15T053724-78001af1/D-001: when adjacency_weight is
        # given, both term_q and term_k must use omega_eff = omega * w_hat, never just one. See decisions.md.
        if adjacency_weight is not None:
            w_hat = ops.cast(adjacency_weight, _mask_dtype(self.compute_dtype))
            omega = omega * w_hat
        omega = ops.cast(omega, self.compute_dtype)

        # Term 1: token i in the query role. This is the only term vanilla
        # attention has (with an implied value matrix V = (W^Q)^T K).
        term_q = ops.einsum('yhd,byhn,bhnm->bmd', self.w_query, k, omega)

        # DECISION plan_2026-07-13_57c9833e/D-001: term_k (token i in the key role)
        # is the ET-specific gradient term; deleting it silently breaks -dE/dg. See decisions.md.
        term_k = ops.einsum('yhd,byhm,bhnm->bnd', self.w_key, q, omega)

        return term_q + term_k

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Return the energy descent direction ``-dE_ATT/dg`` for ``inputs``.

        Unlike a standard attention layer, this does not return a weighted sum of
        values (there is no value matrix); it returns :meth:`update`.

        ``supports_masking = True``, and this signature declares ``mask``, which is
        what makes Keras inject a propagated mask (e.g. from an upstream
        ``Embedding(mask_zero=True)``). Removing the ``mask`` parameter would silently
        drop the mask and let pad tokens influence every real token, even with
        ``supports_masking`` still set. When both ``mask`` and ``attention_mask`` are
        supplied the rule is a logical and (decisions.md D-003).

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional keep mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused; the layer is deterministic.
        :type training: Optional[bool]
        :param mask: Keras-propagated rank-2 ``(B, N)`` boolean per-token validity mask.
            Normally injected by Keras, not passed by hand.
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional FINITE weighted adjacency ``Ŵ`` (paper eq.25,
            Branch A), forwarded unchanged to :meth:`update` (see :meth:`_project`).
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: Tensor of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        return self.update(
            inputs, attention_mask=attention_mask, mask=mask,
            adjacency_weight=adjacency_weight,
        )

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identity — the update lives in the input's space).

        Uses only the passed shape and stored config, never a weight shape, so it is
        valid on an unbuilt layer.

        :param input_shape: Input shape ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape


    def get_config(self) -> Dict[str, Any]:
        """Return the full constructor configuration for serialization.

        :return: Dictionary containing every ``__init__`` argument.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'beta': self.beta,
            'attn_self': self.attn_self,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        })
        return config


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "EnergyAttention":
        """Reconstruct the layer from its serialized configuration.

        :param config: Configuration dictionary produced by :meth:`get_config`.
        :type config: Dict[str, Any]

        :return: A new ``EnergyAttention`` instance.
        :rtype: EnergyAttention
        """
        config = dict(config)
        if 'kernel_initializer' in config:
            config['kernel_initializer'] = initializers.deserialize(
                config['kernel_initializer']
            )
        return cls(**config)

