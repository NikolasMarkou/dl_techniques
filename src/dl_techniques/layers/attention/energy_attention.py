"""
Energy Attention for Keras 3.x.

Implements the multi-head *energy* attention of the Energy Transformer (ET), Hoover,
Liang, Pham, Panda, Strobelt, Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023
(https://arxiv.org/abs/2302.07253), equations (3)-(4).

**This is NOT standard attention.** There is **no value matrix**. The layer defines a
scalar energy ``E_ATT(g)`` over the token state, and its ``call()`` returns the
**negative gradient of that energy** — a descent direction — rather than a weighted sum
of values. The gradient is hand-coded in closed form (``keras.ops`` only, no autodiff:
``keras.ops.grad`` does not exist in keras 3.8, and a backend-specific autodiff tape is
forbidden in ``src/`` — see decisions.md D-001). Its correctness rests entirely on the
autodiff oracle test ``test_gradient_oracle`` in
``tests/test_layers/test_attention/test_energy_attention.py``.
"""

import keras
import math
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

# Finite (NOT -inf) additive bias for masked logits. Load-bearing: `logsumexp` of an
# all `-inf` row is `-inf`, and `0 * -inf = NaN`. Matches the house convention in
# `multi_head_cross_attention.py:406`.
_MASK_BIAS_VALUE = -1e9


@keras.saving.register_keras_serializable()
class EnergyAttention(keras.layers.Layer):
    """Energy Transformer multi-head energy attention (bias-free, no value matrix).

    **Intent**: expose a scalar token-mixing energy ``E_ATT(g)`` together with its exact
    closed-form negative gradient, so that an :class:`EnergyTransformer` block can perform
    *provable gradient descent* on ``E_ATT + E_HN`` instead of running an opaque
    ``attn -> FFN`` residual stream.

    **Mathematics** (notation: ``B``=batch, ``N``=tokens, ``D``=``dim``, ``Y``=``head_dim``,
    ``H``=``num_heads``; ``n`` indexes a token in its **KEY** role, ``m`` in its **QUERY**
    role):

    .. code-block:: text

        K_{y h n} = sum_d W^K_{y h d} g_{n d}            # keys   (no bias)
        Q_{y h m} = sum_d W^Q_{y h d} g_{m d}            # queries (no bias)
        A_{h n m} = sum_y K_{y h n} Q_{y h m}

        E_ATT = -(1/beta) * sum_h sum_m log( sum_{n valid} exp(beta * A_{h n m}) )

    The ``n != m`` exclusion (``attn_self=False``, the paper's ET-Full image config) is a
    mask, not a separate code path; appendix eq. 13 permits ``attn_self=True``.

    **Closed-form gradient.** With ``omega`` the softmax of ``beta * A`` over the **KEY**
    index ``n`` (masked entries zeroed):

    .. code-block:: text

        -dE_ATT/dg_{i d} = sum_h sum_y [ W^Q_{y h d} * ( sum_n K_{y h n} omega_{h n i} )
                                       + W^K_{y h d} * ( sum_m Q_{y h m} omega_{h i m} ) ]

    Derivation sketch:

    - ``dE_ATT/dA_{h n m} = -(1/beta) * beta * omega_{h n m} = -omega_{h n m}``.
    - ``A_{h n m}`` depends on ``g_i`` through ``K`` when ``n == i`` and through ``Q``
      when ``m == i``.
    - ``dA_{h n m}/dg_{i d} = delta_{n,i} sum_y W^K_{y h d} Q_{y h m}
      + delta_{m,i} sum_y W^Q_{y h d} K_{y h n}``.
    - Chain rule, then negate -> the two terms above. **Both** softmax normalizations are
      over the KEY index ``n``.

    The **second** term is the ET-specific contribution and is absent from vanilla
    attention. See the ``D-001`` anchor at :meth:`update`.

    **SIGN DISCIPLINE.** :meth:`update` returns ``-dE/dg`` — the **descent direction**,
    NOT the gradient. A consumer therefore *adds* ``step_size * update``. Do not "fix"
    this sign: flipping it silently turns the block's dynamics into energy *ascent*, which
    still runs and still produces finite outputs.

    **Duck-typed convention (NOT an ABC).** This layer and ``HopfieldNetwork`` both expose
    the trio ``energy(g, ...) -> (B,)`` / ``update(g, ...) -> (B, N, D)`` /
    ``call(...) -> update(...)``. Two implementors and one consumer earn the *convention*,
    not an inheritance hierarchy; there is deliberately no base class or ``Protocol``.

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
        super().__init__(**kwargs)

        # ----- validation -----
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

        # ----- store ALL configuration -----
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

        # ----- resolved (non-config) derived values -----
        self._head_dim = int(resolved_head_dim)
        self._beta = (
            float(beta) if beta is not None
            else 1.0 / math.sqrt(float(self._head_dim))
        )

        # ----- weights are created in build() -----
        self.w_key: Optional[keras.Variable] = None
        self.w_query: Optional[keras.Variable] = None

        self.supports_masking = True

        logger.debug(
            f"Initialized EnergyAttention with dim={self.dim}, "
            f"num_heads={self.num_heads}, head_dim={self._head_dim}, "
            f"beta={self._beta:.6f}, attn_self={self.attn_self}"
        )

    # -----------------------------------------------------------------

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

        w_shape = (self._head_dim, self.num_heads, self.dim)  # (Y, H, D)

        # NO BIAS, by construction: the paper's energy E_ATT is defined without one, and
        # a bias term would not be expressible in the closed-form gradient below.
        self.w_key = self.add_weight(
            name="w_key",
            shape=w_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )
        self.w_query = self.add_weight(
            name="w_query",
            shape=w_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )

        super().build(input_shape)

    # -----------------------------------------------------------------
    # Masking
    # -----------------------------------------------------------------

    def _build_keep_mask(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
    ) -> keras.KerasTensor:
        """Normalize the user KEEP mask to a ``(b, h, n, m)``-broadcastable 0/1 tensor.

        # DECISION plan_2026-07-13_57c9833e/D-006
        Mask convention follows the sibling ``multi_head_cross_attention.py:380-407``:
        ``attention_mask`` is a **KEEP** mask (``1`` = attend, ``0`` = masked), NOT an
        additive ``-inf`` bias. Do NOT re-interpret it as a boolean *drop* mask and do NOT
        accept an additive mask — every sibling attention layer in this package uses the
        keep convention, and flipping it here would silently invert every caller's mask.

        The ``attn_self=False`` diagonal exclusion is a SEPARATE, always-on mask ANDed
        with the user mask (not folded into it), because ET *generates* fully-masked query
        columns by construction (``attn_self=False`` with ``N == 1``). The two-stage
        treatment that follows in :meth:`energy` / :meth:`update` — a FINITE ``-1e9``
        additive bias PLUS a post-softmax ``* keep`` PLUS a ``col_valid`` gate on the
        energy — is load-bearing for exactly that degenerate case. See decisions.md D-006.

        # DECISION plan_2026-07-13_57c9833e/D-008
        A rank-2 ``(B, N)`` mask is applied **SYMMETRICALLY** (to the key axis ``n`` AND
        the query axis ``m``), which is a DELIBERATE DEVIATION from the sibling, where a
        rank-2 mask is key-only. Do NOT "restore" the key-only reading: in ET a token
        masked only as a KEY still acts as a QUERY, and the second gradient term
        (``term_k``, summed over query columns ``m``) then propagates that token's state
        into EVERY other token's update — so a padding token would still influence real
        tokens, and its query column would still be summed into ``E_ATT``. Vanilla
        attention has no ``term_k``, which is why key-only masking is sufficient THERE and
        insufficient HERE. Verified live: the key-only reading makes S8a
        (``test_masked_token_has_no_influence``) RED. Rank-3/rank-4 masks keep the
        sibling's ``(n = key, m = query)`` semantics untouched. See decisions.md D-008.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: KEEP mask of shape ``(B, N)`` (a per-token VALIDITY mask,
            applied to both the key and the query axis — see D-008 above), ``(B, N, N)``
            (interpreted ``(b, n, m)`` with ``n`` = key and ``m`` = query, broadcast over
            heads), or ``(B, H, N, N)``. ``None`` means "attend everywhere".
        :type attention_mask: Optional[keras.KerasTensor]

        :return: A 0/1 ``keep`` tensor broadcastable to ``(B, H, N, N)`` with axis 2 = key
            index ``n`` and axis 3 = query index ``m``.
        :rtype: keras.KerasTensor

        :raises ValueError: If ``attention_mask`` has an unsupported rank.
        """
        num_tokens = ops.shape(g)[1]

        if attention_mask is None:
            keep = ops.ones((1, 1, 1, 1), dtype=self.compute_dtype)
        else:
            mask = ops.cast(attention_mask, self.compute_dtype)
            rank = len(mask.shape)
            if rank == 2:
                # (B, N) token-validity mask -> (B, 1, N, N), applied SYMMETRICALLY:
                # keep[b, :, n, m] = mask[b, n] * mask[b, m]. An invalid token is removed
                # from BOTH the key role and the query role (D-008 above).
                key_keep = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=-1)   # (B,1,N,1)
                query_keep = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=2)  # (B,1,1,N)
                keep = key_keep * query_keep                                         # (B,1,N,N)
            elif rank == 3:
                # (B, N, N) read as (b, n, m) -> (B, 1, N, N): broadcast over heads.
                keep = ops.expand_dims(mask, axis=1)
            elif rank == 4:
                # (B, H, N, N) already in the einsum layout.
                keep = mask
            else:
                raise ValueError(
                    "attention_mask must have rank 2 (B, N), 3 (B, N, N) or "
                    f"4 (B, H, N, N), got rank {rank}"
                )

        if not self.attn_self:
            # Always-on diagonal exclusion (n == m), ANDed with the user mask.
            eye = ops.eye(num_tokens, dtype=self.compute_dtype)          # (N, N) = (n, m)
            keep = keep * (1.0 - ops.reshape(eye, (1, 1, num_tokens, num_tokens)))

        return keep

    # -----------------------------------------------------------------
    # Core: shared projections
    # -----------------------------------------------------------------

    def _project(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor],
    ) -> Tuple[keras.KerasTensor, ...]:
        """Compute ``K``, ``Q``, the masked ``logits`` and the ``keep`` mask.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]

        :return: ``(K, Q, logits, keep)`` with ``K``/``Q`` of shape ``(B, Y, H, N)`` and
            ``logits``/``keep`` broadcastable to ``(B, H, N, N)`` (``n`` = key axis 2).
        :rtype: Tuple[keras.KerasTensor, ...]
        """
        keep = self._build_keep_mask(g, attention_mask)

        k = ops.einsum('yhd,bnd->byhn', self.w_key, g)      # (B, Y, H, N)  n = KEY
        q = ops.einsum('yhd,bmd->byhm', self.w_query, g)    # (B, Y, H, N)  m = QUERY
        a = ops.einsum('byhn,byhm->bhnm', k, q)             # (B, H, N, N)

        # FINITE -1e9, never -inf: logsumexp of an all -inf row is -inf, and the
        # subsequent `0 * -inf` (a fully-masked column) would be NaN.
        mask_bias = (1.0 - keep) * _MASK_BIAS_VALUE
        logits = self._beta * a + mask_bias                 # (B, H, N, N)

        return k, q, logits, keep

    # -----------------------------------------------------------------
    # Public API: energy / update / call
    # -----------------------------------------------------------------

    def energy(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Scalar attention energy ``E_ATT`` per batch element (paper eq. 4).

        .. code-block:: text

            E_ATT = -(1/beta) * sum_h sum_m logsumexp_n( beta * A_{h n m} )

        The ``logsumexp`` is over the **KEY** axis ``n``. A query column ``m`` whose keys
        are ALL masked contributes **exactly 0** (not ``-1e9 / beta``): the ``col_valid``
        indicator gates it out. This is what makes ``N == 1`` with ``attn_self=False``
        return ``0.0`` rather than a huge negative number or a NaN.

        **This formula is the SPEC.** :meth:`update` must match *this*; never edit this to
        make the gradient oracle pass (plan STOP-IF 1).

        :param g: Token state ``(B, N, D)``, typically the output of ``EnergyLayerNorm``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        _, _, logits, keep = self._project(g, attention_mask)

        # logsumexp over the KEY axis n (axis=2) -> (B, H, N) indexed by (b, h, m).
        lse = ops.logsumexp(logits, axis=2)

        # col_valid: does query column m have AT LEAST ONE valid key? A fully-masked
        # column must contribute EXACTLY 0 energy. Independent of `g` -> contributes no
        # gradient path (the autodiff oracle sees it as a constant).
        col_valid = ops.cast(
            ops.sum(keep, axis=2) > 0.0, self.compute_dtype
        )                                                   # (B, H, N) or broadcastable

        return -(1.0 / self._beta) * ops.sum(lse * col_valid, axis=(1, 2))  # (B,)

    # -----------------------------------------------------------------

    def update(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Return ``-dE_ATT/dg`` — the DESCENT DIRECTION, **not** the gradient.

        **SIGN DISCIPLINE**: this is the *negative* gradient. The consumer *adds*
        ``step_size * update`` to the token state. A reader who assumes this returns
        ``+dE/dg`` and "fixes" the sign at the call site silently inverts the dynamics into
        energy ASCENT — which still runs, still trains, and still produces finite outputs.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]

        :return: ``-dE_ATT/dg`` of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        k, q, logits, keep = self._project(g, attention_mask)

        # Softmax over the KEY axis n, then ZERO the masked keys. The post-softmax `* keep`
        # is NOT redundant with the additive -1e9 bias: softmax of an ALL -1e9 row returns
        # a UNIFORM 1/N, which is WRONG, and ET generates such rows by construction
        # (attn_self=False with N == 1). Additive biasing alone cannot fix a fully-masked
        # row. See decisions.md D-006.
        omega = ops.softmax(logits, axis=2) * keep          # (B, H, N, N)

        # Term 1: token i in the QUERY role. This is the only term vanilla attention has
        # (with an implied value matrix V = (W^Q)^T K).
        term_q = ops.einsum('yhd,byhn,bhnm->bmd', self.w_query, k, omega)   # (B, N, D)

        # DECISION plan_2026-07-13_57c9833e/D-001
        # DO NOT DELETE `term_k`. It is the SECOND term of the closed-form gradient
        # -dE_ATT/dg (token i in the KEY role) and is the ET-specific contribution absent
        # from vanilla attention. Removing it leaves a layer that runs, produces
        # plausible finite outputs, and TRAINS — while no longer being the gradient of any
        # energy, so the block's descent guarantee silently evaporates. It is NOT
        # verifiable by inspection; the ONLY thing proving it correct is
        # `test_gradient_oracle` (S6a), and its necessity was verified LIVE by deleting it
        # once and confirming BOTH S6a and the energy-descent test S7 go RED (plan S10).
        # See decisions.md D-001.
        term_k = ops.einsum('yhd,byhm,bhnm->bnd', self.w_key, q, omega)     # (B, N, D)

        return term_q + term_k                              # == -dE_ATT/dg

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Return the energy descent direction ``-dE_ATT/dg`` for ``inputs``.

        Unlike a standard attention layer, this does NOT return a weighted sum of values
        (there is no value matrix); it returns :meth:`update`.

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param training: Unused; the layer is deterministic.
        :type training: Optional[bool]

        :return: Tensor of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        return self.update(inputs, attention_mask=attention_mask)

    # -----------------------------------------------------------------

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Return the output shape (identity — the update lives in the input's space).

        Uses only the passed shape and stored config, never a weight shape, so it is valid
        on an UNBUILT layer.

        :param input_shape: Input shape ``(batch, num_tokens, dim)``.
        :type input_shape: Tuple[Optional[int], ...]

        :return: The same shape as the input.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------

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

# ---------------------------------------------------------------------
