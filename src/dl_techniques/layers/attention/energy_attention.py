"""
Implements the multi-head *energy* attention of the Energy Transformer (ET), Hoover,
Liang, Pham, Panda, Strobelt, Zaki, Chau, Krotov, "Energy Transformer", NeurIPS 2023
(https://arxiv.org/abs/2302.07253), equations (3)-(4).

**This is NOT standard attention, and there is no value matrix.** The layer defines a
scalar energy ``E_ATT(g)`` over the token state. Its ``call()`` returns the **negative
gradient of that energy**, a descent direction, not a weighted sum of values.

The gradient is hand-coded in closed form, with ``keras.ops`` and no autodiff.
``keras.ops.grad`` does not exist in keras 3.8, and a backend-specific autodiff tape is
not allowed in ``src/``. Nothing in the file proves the derivation; the autodiff oracle
test ``test_gradient_oracle`` in
``tests/test_layers/test_attention/test_energy_attention.py`` is the only thing that
does.

Architecture:
    The module is a *descent-direction generator*, not a token mixer. Three free
    functions sit above the layer, and the layer itself has two public faces:

    1.  **Mask plumbing (module level).** ``_token_keep`` validates and casts the
        rank-2 ``(B, N)`` Keras-propagated per-token validity mask — the single home
        of that contract, with four call sites spanning this module and
        ``layers/transformers/energy_transformer.py``. ``_symmetric_token_keep``
        expands it to a ``(B, 1, N, N)`` keep tensor that removes a masked token from
        BOTH the key role and the query role. ``_MASK_BIAS_VALUE`` / ``_mask_dtype``
        are aliased re-exports of ``common.py`` (see the D-007 anchor below).

    2.  **:meth:`EnergyAttention.energy` — the forward face.** Layer-normed tokens
        ``g`` are projected by two bias-free weights, ``w_key`` and ``w_query``, into
        per-head keys and queries. Those are scored, masked, and reduced by a
        ``logsumexp`` over the key axis into one SCALAR energy per batch element.
        There is no value matrix and no output projection. Nothing is ever "read out"
        of the tokens.

    3.  **:meth:`EnergyAttention.call` — the gradient face.** Returns ``-dE_ATT/dg``,
        hand-derived in closed form with ``keras.ops`` only. The softmax here is the
        *derivative* of the ``logsumexp`` in ``energy()``, not a separately designed
        attention distribution. Edit the two methods as a pair. A wrong descent
        direction still runs, still trains and still looks finite, so
        ``test_gradient_oracle`` is the only thing that would catch it.

Foundational Mathematics:
    For layer-normed tokens ``g`` (shape ``(B, N, D)``), heads ``h``, key/query
    projections ``K = w_key g`` and ``Q = w_query g``, and inverse temperature
    ``beta = 1/sqrt(head_dim)``::

        E_ATT(g) = -(1/beta) * sum_{h,q} logsumexp_k ( beta * K[h,k] . Q[h,q] )

    The layer's ``call()`` returns ``-dE_ATT/dg``, so stacking blocks that each add
    this output to their token state performs gradient descent on the total energy.
    The ``logsumexp`` is the softmax's potential function: differentiating it is what
    produces the familiar attention weights, rather than the weights being posited
    first and an energy reverse-engineered afterwards.

References:
    - Hoover, Liang, Pham, Panda, Strobelt, Zaki, Chau, Krotov (2023).
      "Energy Transformer". NeurIPS 2023. (https://arxiv.org/abs/2302.07253) —
      equations (3)-(4).
"""

# ---------------------------------------------------------------------

import keras
from keras import ops, initializers
from typing import Any, Dict, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers import clone_initializer
from dl_techniques.utils.logger import logger

# DECISION plan-2026-07-27T130643-38c5646a/D-007
# These three names are a PUBLISHED PRIVATE CONTRACT, not local detail.
# `layers/transformers/energy_transformer.py:91` does
# `from dl_techniques.layers.attention.energy_attention import _mask_dtype, _token_keep`
# — a plain by-name import, verified at `energy_transformer.py:88-91`.
# WHAT NOT TO DO:
#   * Do NOT delete `_MASK_BIAS_VALUE` / `_mask_dtype` from this module's namespace just
#     because their bodies now live in `common.py`. Deleting one does not break a
#     forward pass; `EnergyTransformer` stops IMPORTING, taking every ET model and
#     trainer with it. That is now pinned by
#     `test_energy_attention.py::TestPrivateReExportContract`, which was proven RED by
#     removing one alias. Before that test existed the suite stayed green.
#   * Do NOT "clean up" the aliasing by rewriting the import in `energy_transformer.py`.
#     The alias is the cheaper contract.
#   * Do NOT re-point `_token_keep` at `common.py` — it was never extracted. It encodes
#     the rank-2 `(B, N)` Keras-mask contract specific to the Energy Transformer family,
#     not a generic attention primitive, and it stays DEFINED here, below.
# The D-007 entry itself lives in that plan's own decisions.md.
#
# DECISION plan_2026-07-13_57c9833e/D-009
# The originating plan directory is gone, so this comment is the record.
# Keep this anchor HERE, at `_MASK_BIAS_VALUE`: several docstrings in this file say
# "see the D-009 anchor at ``_MASK_BIAS_VALUE``", and this is that anchor.
# `-1e9` is NOT a dtype-independent "finite" number: `np.float16(-1e9) == -inf`. It is
# usable ONLY inside a float32-or-wider computation, and this module keeps it there: the
# whole logits -> bias -> softmax/logsumexp chain runs in
# `_mask_dtype(self.compute_dtype)` and is cast back to the compute dtype at the end.
# Three standing prohibitions, stated in full at the definition site
# (`common.py`'s `MASK_BIAS_VALUE`): never apply the bias in the compute dtype; never use
# the arithmetic form `(1 - keep) * _MASK_BIAS_VALUE`, which is `0 * -inf = NaN` at every
# UNMASKED position, and use `ops.where(keep > 0, 0.0, _MASK_BIAS_VALUE)` instead; never
# "simplify" to a per-dtype magic constant.
from dl_techniques.layers.attention.common import (
    MASK_BIAS_VALUE as _MASK_BIAS_VALUE,
    mask_dtype as _mask_dtype,
    compute_attention_scale as _compute_attention_scale,
)
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


def _token_keep(
    mask: keras.KerasTensor,
    dtype: str,
) -> keras.KerasTensor:
    """Validate and cast a Keras-propagated rank-2 ``(B, N)`` per-token validity mask.

    The SINGLE place the rank-2 contract for the Keras-propagated ``mask`` lives. It has
    **FOUR** call sites. Names are the stable identifier, so no line numbers are given here:
    the four that used to be were all stale within a month.

    1. :meth:`EnergyAttention._build_keep_mask`, which then expands the result
       symmetrically over the key and query axes (``_symmetric_token_keep``).
    2. ``HopfieldNetwork.energy`` in ``layers/transformers/energy_transformer.py``,
       per-token and un-expanded, to drop PAD tokens from the energy SUM.
    3. ``HopfieldNetwork.update`` in the same file — the matching gradient side. The pair
       MUST see the same keep, or ``update() != -dE/dg``.
    4. ``EnergyTransformer._hopfield_token_mask``, which ANDs the Keras mask with a
       rank-2 ``attention_mask`` (D-006).

    Do NOT re-implement the rank check at any call site. A second copy is a second error
    message to drift.

    :param mask: Boolean (or 0/1) per-token validity mask of shape ``(B, N)``.
    :type mask: keras.KerasTensor
    :param dtype: Dtype to cast into — the MASK compute dtype (>= float32) wherever the value
        feeds a ``-1e9`` bias or a reduction (D-009), the layer's compute dtype where it merely
        gates an already-computed tensor.
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
    """Expand a rank-2 ``(B, N)`` per-token VALIDITY mask to a ``(B, 1, N, N)`` keep tensor.

    ``keep[b, :, n, m] = token_keep[b, n] * token_keep[b, m]`` — the token is removed from
    BOTH the key role ``n`` and the query role ``m``. This is D-008's semantics, factored
    out because it now has two call sites (the explicit rank-2 ``attention_mask`` and the
    Keras-propagated ``mask``); see the D-008 / D-002 anchors in :meth:`_build_keep_mask`.
    Do NOT duplicate it, and do NOT weaken either factor to a key-only mask.

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

    **Intent**: expose a scalar token-mixing energy ``E_ATT(g)`` together with its exact
    closed-form negative gradient, so that an :class:`EnergyTransformer` block can perform
    *provable gradient descent* on ``E_ATT + E_HN`` instead of running an opaque
    ``attn -> FFN`` residual stream.

    **Architecture Overview:**

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
                  n = KEY index,  m = QUERY index
                               ▼
                  logits = beta * A, cast to mask_dtype
                  (>= float32)
                               ▼
                  optional: logits = logits ⊙ Ŵ
                  (adjacency_weight, paper eq. 25)
                               ▼
                  keep [B,1,N,N] ► additive -1e9 bias.
                  The diagonal is dropped when attn_self
                  is False; a PAD token drops in BOTH the
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

    There is NO value matrix and NO output projection in this layer. A consumer
    ADDS ``step_size * call(g)`` to ``g``.

    **Mathematics** (notation: ``B``=batch, ``N``=tokens, ``D``=``dim``, ``Y``=``head_dim``,
    ``H``=``num_heads``; ``n`` indexes a token in its **KEY** role, ``m`` in its **QUERY**
    role):

    .. code-block:: text

        K_{y h n} = sum_d W^K_{y h d} g_{n d}     # keys    (no bias)
        Q_{y h m} = sum_d W^Q_{y h d} g_{m d}     # queries (no bias)
        A_{h n m} = sum_y K_{y h n} Q_{y h m}

        E_ATT = -(1/beta) * sum_h sum_m
                    log( sum_{n valid} exp(beta * A_{h n m}) )

    The ``n != m`` exclusion (``attn_self=False``, the paper's ET-Full image config) is a
    mask, not a separate code path; appendix eq. 13 permits ``attn_self=True``.

    **Closed-form gradient.** With ``omega`` the softmax of ``beta * A`` over the **KEY**
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
    - Chain rule, then negate -> the two terms above. **Both** softmax normalizations are
      over the KEY index ``n``.

    The **second** term is the ET-specific contribution. Vanilla attention has no
    equivalent. See the ``D-001`` anchor on ``term_k`` in :meth:`update`.

    **SIGN DISCIPLINE.** :meth:`update` returns ``-dE/dg``, the **descent direction**,
    not the gradient. A consumer therefore *adds* ``step_size * update``. Do not "fix"
    this sign. Flipping it turns the block's dynamics into energy *ascent*, which still
    runs and still produces finite outputs.

    **Duck-typed convention, not an ABC.** This layer and ``HopfieldNetwork`` both expose
    the trio ``energy(g, ...) -> (B,)`` / ``update(g, ...) -> (B, N, D)`` /
    ``call(...) -> update(...)``. Two implementors and one consumer earn a *convention*,
    not an inheritance hierarchy. There is no base class and no ``Protocol`` here on
    purpose: one more implementor would be the point at which that changes.

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
        # `_compute_attention_scale` is `1.0 / math.sqrt(float(head_dim))` — byte-identical
        # to the expression it replaces. Called HERE, in `__init__`, never in `call()`:
        # `self._beta` must stay a Python float so it folds into the traced graph as a
        # constant (anchor `plan_2026-06-14_33b77a7a/D-002`).
        self._beta = (
            float(beta) if beta is not None
            else _compute_attention_scale(self._head_dim)
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

        # (Y, H, D) — per-head dim first, so the einsums below read 'yhd'.
        w_shape = (self._head_dim, self.num_heads, self.dim)

        # NO BIAS. The paper's energy E_ATT is defined without one, and a bias term
        # would not be expressible in the closed-form gradient below.
        self.w_key = self.add_weight(
            name="w_key",
            shape=w_shape,
            initializer=self.kernel_initializer,
            trainable=True,
            dtype=self.dtype,
        )
        # DECISION plan-2026-08-19T163559-499b6f0e/D-068 — `clone_initializer`, because one
        # shared instance handed to both `add_weight` calls made `w_key` and `w_query`
        # bit-identical, so the initial score matrix `K^T Q` was EXACTLY SYMMETRIC in all six
        # `energy_transformer` / `graph_energy_transformer` classes. `self.kernel_initializer`
        # itself is untouched, so `get_config` still reports it. See decisions.md D-068.
        self.w_query = self.add_weight(
            name="w_query",
            shape=w_shape,
            initializer=clone_initializer(self.kernel_initializer),
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
        mask: Optional[keras.KerasTensor] = None,
    ) -> Tuple[keras.KerasTensor, bool]:
        """Normalize the user KEEP mask to a ``(b, h, n, m)``-broadcastable 0/1 tensor.

        # DECISION plan_2026-07-13_57c9833e/D-006
        The originating plan directory is gone, so this comment is the record.
        Mask convention follows the sibling
        ``MultiHeadCrossAttention._apply_attention_mask``: ``attention_mask`` is a
        **KEEP** mask (``1`` = attend, ``0`` = masked), NOT an additive ``-inf`` bias.
        Do NOT re-interpret it as a boolean *drop* mask, and do NOT accept an additive
        mask. Every sibling attention layer in this package uses the keep convention, so
        flipping it here would silently invert every caller's mask.

        The ``attn_self=False`` diagonal exclusion is a SEPARATE, always-on mask ANDed
        with the user mask rather than folded into it, because ET *generates* fully-masked
        query columns on its own (``attn_self=False`` with ``N == 1``). The two-stage
        treatment that follows in :meth:`energy` / :meth:`update` — a FINITE ``-1e9``
        additive bias, PLUS a post-softmax ``* keep``, PLUS a ``col_valid`` gate on the
        energy — is what makes that degenerate case come out right. Do not collapse it.

        # DECISION plan_2026-07-13_57c9833e/D-008
        The originating plan directory is gone, so this comment is the record.
        A rank-2 ``(B, N)`` mask is applied **SYMMETRICALLY**, to the key axis ``n`` AND
        the query axis ``m``. That is a chosen deviation from the sibling, where a rank-2
        mask is key-only. Do NOT "restore" the key-only reading. In ET a token masked only
        as a KEY still acts as a QUERY, and the second gradient term (``term_k``, summed
        over query columns ``m``) then propagates that token's state into EVERY other
        token's update. A padding token would still influence real tokens, and its query
        column would still be summed into ``E_ATT``. Vanilla attention has no ``term_k``,
        which is why key-only masking is enough THERE and not enough HERE. Verified live:
        the key-only reading makes ``test_masked_token_has_no_influence`` RED.
        Rank-3 and rank-4 masks keep the sibling's ``(n = key, m = query)`` semantics.

        # DECISION plan_2026-07-13_ca4f71a2/D-002
        The originating plan directory is gone, so this comment is the record.
        The Keras-propagated ``mask`` (from e.g. ``Embedding(mask_zero=True)``) is merged
        HERE, as one extra multiplicative keep factor that reuses D-008's symmetric
        expansion (``_symmetric_token_keep``). The propagated ``mask`` IS exactly a rank-2
        ``(B, N)`` per-token validity mask — D-008's shape and D-008's semantics. Do NOT
        give it its own masking path: a second convention is a second thing to get wrong,
        and the symmetric key-AND-query treatment is exactly what the PAD-token defect
        needs, since a PAD token masked only as a KEY still propagates through ``term_k``.
        The merge only ADDS a factor; D-006 and D-008 semantics are unchanged.

        # DECISION plan_2026-07-13_ca4f71a2/D-003
        The originating plan directory is gone, so this comment is the record.
        Precedence when BOTH masks arrive is LOGICAL AND. The multiplication below IS that
        AND, and it composes with ANY rank (2, 3 or 4) of ``attention_mask``.
        WHAT NOT TO DO:
          * Do NOT raise on a conflict between the two masks. Detecting one compares mask
            VALUES — a tensor-valued condition, unresolvable at trace time and so
            graph-UNSAFE, unlike the Python-bool ``is_masked`` below.
          * Do NOT make the explicit ``attention_mask`` win. That would let an explicit
            mask silently UN-MASK a PAD token the framework had declared invalid, which is
            the PAD-token defect through the front door. AND is the only rule monotone in
            safety: neither mask can ever resurrect a token the other hid.

        :param g: Token state ``(B, N, D)``.
        :type g: keras.KerasTensor
        :param attention_mask: KEEP mask of shape ``(B, N)`` (a per-token VALIDITY mask,
            applied to both the key and the query axis — see D-008 above), ``(B, N, N)``
            (interpreted ``(b, n, m)`` with ``n`` = key and ``m`` = query, broadcast over
            heads), or ``(B, H, N, N)``. ``None`` means "attend everywhere".
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean per-token validity
            mask. ANDed with ``attention_mask`` (D-003).
        :type mask: Optional[keras.KerasTensor]

        :return: ``(keep, is_masked)``. ``keep`` is a 0/1 tensor in ``float32`` (NOT the
            compute dtype — see the D-009 anchor at ``_MASK_BIAS_VALUE``), broadcastable to
            ``(B, H, N, N)`` with axis 2 = key index ``n`` and axis 3 = query index ``m``.
            ``is_masked`` is a **Python** bool: ``False`` means nothing is masked anywhere
            (no user mask AND ``attn_self=True``), so the additive bias can be skipped
            entirely — the sibling's fast path, and the reason the sibling never hit the
            fp16 ``-inf`` bug.
        :rtype: Tuple[keras.KerasTensor, bool]

        :raises ValueError: If ``attention_mask`` or ``mask`` has an unsupported rank.
        """
        num_tokens = ops.shape(g)[1]
        mask_dtype = _mask_dtype(self.compute_dtype)

        # NOTE: `is_masked` is a PYTHON bool, resolvable at trace time — it depends only on
        # `attn_self` and on whether a mask tensor was passed, never on tensor VALUES. It is
        # therefore graph-safe.
        is_masked = (
            (attention_mask is not None) or (mask is not None) or (not self.attn_self)
        )

        if attention_mask is None:
            keep = ops.ones((1, 1, 1, 1), dtype=mask_dtype)
        else:
            explicit = ops.cast(attention_mask, mask_dtype)
            rank = len(explicit.shape)
            if rank == 2:
                # (B, N) token-validity mask -> (B, 1, N, N), applied SYMMETRICALLY:
                # keep[b, :, n, m] = mask[b, n] * mask[b, m]. An invalid token is removed
                # from BOTH the key role and the query role (D-008 above).
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
            # The Keras-propagated mask (D-002), cast into the mask compute dtype — it
            # arrives BOOLEAN, and it must NEVER land in fp16 where `-1e9` is `-inf` (D-009).
            # Multiplying the keep factors IS the logical AND (D-003) and composes with any
            # rank of `attention_mask` above.
            # `_token_keep` is the ONE place the rank-2 contract lives.
            keras_mask = _token_keep(mask, mask_dtype)
            keep = keep * _symmetric_token_keep(keras_mask)

        if not self.attn_self:
            # Always-on diagonal exclusion (n == m), ANDed with the user mask.
            # (N, N), read as (n = key, m = query).
            eye = ops.eye(num_tokens, dtype=mask_dtype)
            keep = keep * (1.0 - ops.reshape(eye, (1, 1, num_tokens, num_tokens)))

        return keep, is_masked

    # -----------------------------------------------------------------
    # Core: shared projections
    # -----------------------------------------------------------------

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
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
        :type attention_mask: Optional[keras.KerasTensor]
        :param mask: Optional Keras-propagated ``(B, N)`` mask, ANDed with
            ``attention_mask`` (see :meth:`_build_keep_mask`, D-002/D-003).
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional FINITE per-head weighted-adjacency ``Ŵ``
            broadcastable to ``(B, H, N, N)`` (paper eq.25, Branch A). When supplied it is
            folded MULTIPLICATIVELY into the raw score BEFORE the ``beta`` scaling and the
            keep-bias: ``logit = beta * (A ⊙ Ŵ) + M``. It is a per-block CONSTANT
            (``dŴ/dg == 0``) hoisted by the block, NOT a reinterpretation of the 0/1
            ``attention_mask`` (D-002/D-006). ``None`` (default) → the code path below is
            provably UNREACHABLE and the layer is byte-identical to today.
        :type adjacency_weight: Optional[keras.KerasTensor]

        ``K`` and ``Q`` come back in the **compute dtype** (they feed the gradient einsums,
        which contract against the compute-dtype weights). ``logits`` and ``keep`` come back
        in **float32**, unconditionally — see the D-009 anchor at ``_MASK_BIAS_VALUE``.

        :return: ``(K, Q, logits, keep)`` with ``K``/``Q`` of shape ``(B, Y, H, N)`` in the
            compute dtype and ``logits``/``keep`` broadcastable to ``(B, H, N, N)`` in
            float32 (``n`` = key axis 2).
        :rtype: Tuple[keras.KerasTensor, ...]
        """
        keep, is_masked = self._build_keep_mask(g, attention_mask, mask=mask)

        # k and q are (B, Y, H, N) with n = KEY and m = QUERY; a is (B, H, N, N).
        k = ops.einsum('yhd,bnd->byhn', self.w_key, g)
        q = ops.einsum('yhd,bmd->byhm', self.w_query, g)
        a = ops.einsum('byhn,byhm->bhnm', k, q)

        # DECISION plan_2026-07-13_57c9833e/D-009
        # The originating plan directory is gone, so this comment is the record.
        # (a) The ENTIRE logits -> bias -> softmax/logsumexp chain runs in float32, so
        #     `_MASK_BIAS_VALUE` stays finite under ANY global policy; in fp16 it would be
        #     `-inf`. It is also the right thing numerically for a `logsumexp` under mixed
        #     precision.
        # (b) The bias is applied via `ops.where`, never as `(1 - keep) * NEG`. `where`
        #     cannot produce `0 * inf`, so the NaN failure mode is gone structurally, not
        #     just by virtue of the dtype.
        # (c) When nothing is masked anywhere (no user mask AND `attn_self=True`) the bias
        #     is SKIPPED entirely — the sibling's fast path.
        # Do NOT collapse any of the three. They are three separate guarantees.
        # (B, H, N, N) in the mask dtype, i.e. float32 or wider.
        logits = ops.cast(a, _mask_dtype(self.compute_dtype)) * self._beta
        if adjacency_weight is not None:
            # Branch A (paper eq.25): fold the FINITE learned edge weight `Ŵ` into the score
            # MULTIPLICATIVELY, in the mask dtype (>= float32) so the whole logits -> bias ->
            # logsumexp chain stays out of fp16 (D-009). This is `beta * (A ⊙ Ŵ)`. `Ŵ` is a
            # per-block CONSTANT w.r.t. `g`, so it adds NO new gradient term; it only reweights
            # the existing delta-structure (see the omega_eff anchor in `update`). A real
            # Python branch — NOT a multiply-by-ones — so `None` is byte-identical to today.
            w_hat = ops.cast(adjacency_weight, _mask_dtype(self.compute_dtype))
            logits = logits * w_hat
        if is_masked:
            # Both `where` branches are tensors IN THE MASK DTYPE — a bare Python `0.0` /
            # `-1e9` pair gets promoted to float32 and then collides with a float64 `logits`.
            bias = ops.where(
                keep > 0.0,
                ops.zeros_like(keep),
                ops.full_like(keep, _MASK_BIAS_VALUE),
            )
            logits = logits + bias

        return k, q, logits, keep

    # -----------------------------------------------------------------
    # Public API: energy / update / call
    # -----------------------------------------------------------------

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
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask.
            When BOTH masks are supplied the rule is a **logical AND** — a token is attended
            only if valid under both, and neither mask can un-mask what the other hid
            (decisions.md D-003). This parameter exists because ``energy()`` is part of the
            public duck-typed surface an ``EnergyTransformer`` block calls DIRECTLY, outside
            ``__call__``, where Keras cannot inject the mask for us.
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional FINITE weighted adjacency ``Ŵ`` (paper eq.25,
            Branch A), folded into the score as ``beta * (A ⊙ Ŵ)`` (see :meth:`_project`).
            It MUST be the SAME tensor passed to :meth:`update`, or ``update() != -dE/dg``.
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: Energy of shape ``(B,)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        # C-2: this is a PUBLIC method, callable OUTSIDE `__call__` — where Keras has NOT
        # opened an autocast scope. Without this cast a float32 `g` meets float16 weights
        # under `mixed_float16` and the einsum raises InvalidArgumentError. The factory
        # registry ADVERTISES this method, so it must be safe standalone.
        g = ops.cast(g, self.compute_dtype)

        # I1/STOP-IF 2: `mask` MUST reach `_project` here AND in `update()`. If it lands in
        # only one of them, `update()` stops being `-dE/dg` — the layer still runs, still
        # trains, and the descent guarantee silently evaporates. `test_gradient_oracle` runs
        # WITH a Keras mask (`mask_kind` in {KERAS, BN+KERAS}) precisely to catch that. The
        # SAME discipline applies to `adjacency_weight`: it must reach the SAME `_project`
        # call that `update()` makes, or `update() != -dE/dg`.
        # `logits` and `keep` come back in the mask dtype (float32 or wider).
        _, _, logits, keep = self._project(
            g, attention_mask, mask=mask, adjacency_weight=adjacency_weight
        )

        # logsumexp over the KEY axis n (axis=2) -> (B, H, N) indexed by (b, h, m).
        lse = ops.logsumexp(logits, axis=2)

        # col_valid: does query column m have AT LEAST ONE valid key? A fully-masked
        # column must contribute EXACTLY 0 energy. Independent of `g` -> contributes no
        # gradient path (the autodiff oracle sees it as a constant).
        col_valid = ops.cast(
            ops.sum(keep, axis=2) > 0.0, _mask_dtype(self.compute_dtype)
        )

        # DECISION plan_2026-07-13_ca4f71a2/D-005
        # The originating plan directory is gone, so this comment is the record.
        # The energy is returned in the REDUCE dtype (`_mask_dtype`, i.e. >= float32) and is
        # NOT cast back to the compute dtype. WHAT NOT TO DO:
        #   * Do NOT "restore the boundary" with `ops.cast(energy, self.compute_dtype)`. That
        #     spelling is the bug this anchor replaces. The float32 reduction protects the
        #     ACCUMULATOR, and the cast then puts the O(-8e4) result back into fp16 (max
        #     65504) on the very last op. Measured WHEN THAT CAST WAS PRESENT, under
        #     `mixed_float16`: N=256 gave -32256, already within a factor of 2 of the limit;
        #     N=512 sat at the limit; and the energy went `-inf` at N >= 512, reaching a full
        #     `-inf` at N=1024. Those numbers describe the rejected spelling, not the code
        #     below. The cast also quantized the trace to ~1 part in 2048, which made the
        #     energy DESCENT invisible.
        #   * This is safe because NOTHING in the compute path consumes `energy()`. It is a
        #     REPORTED DIAGNOSTIC SCALAR: `EnergyTransformer.call()` only appends it to the
        #     `return_energy=True` trace, and the state update is driven by `update()` alone.
        #     Widening the energy's dtype therefore cannot change any layer's OUTPUT. If some
        #     future compute path contracts this against fp16 weights, cast AT THAT CALL
        #     SITE — do not re-narrow the energy here.
        return -(1.0 / self._beta) * ops.sum(lse * col_valid, axis=(1, 2))

    # -----------------------------------------------------------------

    def update(
        self,
        g: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
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
        :param mask: Optional Keras-propagated rank-2 ``(B, N)`` boolean validity mask,
            ANDed with ``attention_mask`` (decisions.md D-003). It MUST be passed to the
            SAME ``_project`` call ``energy()`` makes, or ``update() != -dE/dg``.
        :type mask: Optional[keras.KerasTensor]
        :param adjacency_weight: Optional FINITE weighted adjacency ``Ŵ`` (paper eq.25,
            Branch A). It MUST be the SAME tensor passed to :meth:`energy`, or
            ``update() != -dE/dg``.
        :type adjacency_weight: Optional[keras.KerasTensor]

        :return: ``-dE_ATT/dg`` of shape ``(B, N, D)``.
        :rtype: keras.KerasTensor
        """
        if not self.built:
            self.build(g.shape)

        # C-2: cast at the head of the public method — see the note in `energy()`.
        g = ops.cast(g, self.compute_dtype)

        # I1/STOP-IF 2: same masks AND same `adjacency_weight` as `energy()` — see the note
        # there. A weight landing in only one of them makes `update() != -dE/dg`.
        # `logits` and `keep` come back in the mask dtype (float32 or wider).
        k, q, logits, keep = self._project(
            g, attention_mask, mask=mask, adjacency_weight=adjacency_weight
        )

        # Softmax over the KEY axis n, then ZERO the masked keys. The post-softmax `* keep`
        # is NOT redundant with the additive -1e9 bias: softmax of an ALL -1e9 row returns
        # a UNIFORM 1/N, which is wrong, and ET generates such rows on its own
        # (attn_self=False with N == 1). Additive biasing alone cannot fix a fully-masked
        # row. See the D-006 anchor in `_build_keep_mask`.
        # The softmax itself is evaluated in float32 (D-009); `omega` is cast back to the
        # compute dtype at the boundary, to contract with the compute-dtype K/Q/weights.
        omega = ops.softmax(logits, axis=2) * keep

        # DECISION plan-2026-07-15T053724-78001af1/D-001
        # The originating plan directory is gone, so this comment is the record.
        # Branch A (paper eq.25) weighted adjacency. When `Ŵ` is supplied, the energy is
        # `E' = -(1/beta) Σ_h Σ_m logsumexp_n( beta·A·Ŵ + M )`, so
        #   -dE'/dg = -(1/beta) Σ omega'·d(beta·A·Ŵ)/dg = Σ (omega'·Ŵ)·dA/dg ,
        # where omega' = softmax_n(beta·A·Ŵ + M). `Ŵ` and `M` are per-block CONSTANTS
        # (dŴ/dg == 0, Branch A), so dA/dg keeps its EXACT delta structure — the two-term
        # (term_q + term_k) form below is reused verbatim with `omega_eff = omega·Ŵ` in place
        # of `omega`. WHAT NOT TO DO (this is the exact gradient the oracle at
        # test_energy_attention.py::TestGradientOracle checks, verified RED when the `·Ŵ`
        # factor is deleted, at N∈{64,1024}):
        #   * Do NOT recompute `Ŵ` per descent step. It is a per-block constant HOISTED by
        #     the block; recomputing it from the evolving `g_t` would add a `dŴ/dg` term that
        #     this closed form does not carry. That is Branch B1, Tier-3, and it is not this.
        #   * Do NOT drop the `·Ŵ` factor from `omega_eff`. Folding `Ŵ` into the logits ALONE
        #     (so only `omega'` sees it) is NOT enough: the chain rule pulls a SECOND `Ŵ` out
        #     of `d(beta·A·Ŵ)/dg`. Omitting it leaves a layer that runs, trains, and produces
        #     finite plausible output while no longer being `-dE/dg` — the descent guarantee
        #     silently evaporates. Only `test_gradient_oracle` catches it.
        #   * Do NOT feed `Ŵ` to only one of `term_q`/`term_k`; BOTH gradient terms carry it.
        # The sibling D-001 anchor on `term_k` below is the other half of this gradient.
        if adjacency_weight is not None:
            w_hat = ops.cast(adjacency_weight, _mask_dtype(self.compute_dtype))
            # This is omega_eff, still in the mask dtype.
            omega = omega * w_hat
        omega = ops.cast(omega, self.compute_dtype)

        # Term 1: token i in the QUERY role. This is the only term vanilla attention has
        # (with an implied value matrix V = (W^Q)^T K).
        term_q = ops.einsum('yhd,byhn,bhnm->bmd', self.w_query, k, omega)

        # DECISION plan_2026-07-13_57c9833e/D-001
        # The originating plan directory is gone, so this comment is the record.
        # DO NOT DELETE `term_k`. It is the SECOND term of the closed-form gradient
        # -dE_ATT/dg (token i in the KEY role), and it is the ET-specific contribution that
        # vanilla attention does not have. Removing it leaves a layer that runs, produces
        # plausible finite outputs, and TRAINS — while no longer being the gradient of any
        # energy, so the block's descent guarantee silently evaporates. It is NOT verifiable
        # by inspection. The only thing proving it correct is `test_gradient_oracle`, and its
        # necessity was verified LIVE: deleting it once turned both that oracle and the
        # energy-descent test RED.
        term_k = ops.einsum('yhd,byhm,bhnm->bnd', self.w_key, q, omega)

        # (B, N, D). This sum IS -dE_ATT/dg.
        return term_q + term_k

    # -----------------------------------------------------------------

    def call(
        self,
        inputs: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
        adjacency_weight: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Return the energy descent direction ``-dE_ATT/dg`` for ``inputs``.

        Unlike a standard attention layer, this does NOT return a weighted sum of values
        (there is no value matrix); it returns :meth:`update`.

        **Masking.** ``supports_masking = True``, and this signature DECLARES ``mask`` —
        which is what makes Keras inject a propagated mask (e.g. from an upstream
        ``Embedding(mask_zero=True)``). Do NOT remove the parameter: ``supports_masking``
        alone only suppresses Keras' "layer does not support masking" error; without a
        ``mask`` parameter here the mask is silently DROPPED and PAD tokens influence every
        real token (F-02). When BOTH ``mask`` and ``attention_mask`` are supplied the rule is
        a **logical AND** — a token is attended only if valid under both (decisions.md D-003).

        :param inputs: Token state ``(B, N, D)``.
        :type inputs: keras.KerasTensor
        :param attention_mask: Optional KEEP mask (see :meth:`_build_keep_mask`).
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
