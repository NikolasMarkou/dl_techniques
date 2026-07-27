"""
Shared primitives for the ``layers/attention`` package.

This module is the shared home for four small pieces that were previously re-derived
across the package: the additive attention-mask bias constant, the dtype in which a
masked softmax must be evaluated, the ``dim % num_heads`` divisibility check, and the
softmax temperature (``1 / sqrt(head_dim)``).

Adoption is **deliberately partial, and unevenly so** — read this before assuming a
number you see in a sibling module came from here:

-   ``validate_head_divisibility`` (11 importers) and ``compute_attention_scale``
    (14 importers) ARE genuinely consolidated. Where a module declines them, it says
    so in a comment naming the measurement — e.g. ``progressive_focused_attention.py``
    and ``mmdit_joint_attention.py`` both keep ``head_dim ** -0.5``, which is NOT
    bit-identical to ``compute_attention_scale`` (it mismatches in the last ULP for
    218 of head_dim 1..1024, including 32 and 128).

-   ``MASK_BIAS_VALUE`` / ``mask_dtype`` have exactly **ONE** importer:
    ``energy_attention.py``, under the legacy private aliases ``_MASK_BIAS_VALUE`` /
    ``_mask_dtype`` (see the D-007 anchor there). **Fourteen other modules** still
    carry a LOCAL ``-1e9``-family value, counted mechanically at step 10 with
    comments and docstrings stripped: ``attention_routing_capsule``,
    ``capsule_routing``, ``differential``, ``gated``, ``group_query``, ``hopfield``,
    ``ideogram4`` (as ``_MASK_NEG``), ``lighthouse`` (a per-dtype TABLE, the most
    careful form in the package), ``multi_head_cross``, ``multi_head_latent``,
    ``progressive_focused``, ``ring``, ``rpc``, ``single_window``.

    *(That count corrects the "9 other modules" figure recorded in D-011 at step 9,
    which was read off a grep that also matched prose. The direction of the error
    matters: consolidation here is even less complete than the decision log claimed.)*

    That is not an oversight and it is not laziness. Each of those sites was probed
    individually and found NOT behavior-identical to this pair: they differ in cast
    order, in the arithmetic-versus-``ops.where`` form, and several of them are the
    systemic ``0 * -inf = NaN`` fp16 sites documented in place as known defects
    (``hopfield_attention.py`` NaNs the ENTIRE batch under ``mixed_float16`` given an
    all-ones mask — measured, 512/512). Making them adopt this constant is therefore a
    *numerics change*, not a rename, and it is reserved for the follow-up plan that is
    allowed to change behavior. Until then, these two exports are a documentation
    consolidation with one consumer, honestly charged as such in that plan's
    ``decisions.md`` D-011.

Architecture:
    Deliberately **flat**: four module-level names, no classes, no registry, no Keras
    serialization registration. Nothing here is a layer, and nothing here holds state.
    A Keras layer imports these the same way it imports ``math`` — they are helpers,
    not a base class. Keeping the module free of classes is what stops a shared-helper
    file from quietly growing into a framework that every layer must inherit from.

    1.  **Masking** — ``MASK_BIAS_VALUE`` together with ``mask_dtype()`` form a pair.
        The constant is only safe inside the dtype the helper returns; using one
        without the other reintroduces the fp16 overflow documented below.

    2.  **Validation** — ``validate_head_divisibility()`` centralizes a check that was
        written 19 times with 6 slightly different message spellings. Its parameter
        names are configurable so that each call site keeps naming *its own*
        constructor arguments (``dim``/``hidden_size``/``embed_dim``).

    3.  **Scaling** — ``compute_attention_scale()`` returns a plain Python ``float``, so
        the softmax temperature is a graph constant folded at trace time rather than a
        per-call ``ops.sqrt`` node.

Foundational Mathematics:
    Masked scaled dot-product attention applies an additive bias ``b`` to the logits
    before the softmax::

        A = softmax( (Q K^T) * s + b ),   s = 1 / sqrt(d_head)

    with ``b = 0`` at kept positions and ``b = MASK_BIAS_VALUE`` at masked positions.
    Because ``exp(-1e9) == 0`` in float32, the masked positions receive exactly zero
    probability mass. This argument holds **only** while ``-1e9`` is representable —
    see the fp16 note on ``MASK_BIAS_VALUE``.

References:
    - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS. (The ``1/sqrt(d_k)``
      scaling and the additive mask bias.)
    - ``GUIDE.md`` in this package, section "What lives in ``common.py``".
"""

# ---------------------------------------------------------------------

import math

# ---------------------------------------------------------------------

# DECISION plan-2026-07-27T130643-38c5646a/D-002
# Promoted verbatim in substance from `energy_attention.py` (anchor
# `plan_2026-07-13_57c9833e/D-009`), which was the only module in the package carrying
# this guard. Every other `-1e9` mask site had the hazard and no documentation.
#
# `-1e9` is NOT a dtype-independent "finite" number: `np.float16(-1e9) == -inf`
# (verified empirically, not assumed). This constant is therefore ONLY usable inside a
# float32-or-wider computation. Callers MUST materialize the bias in `mask_dtype(...)`,
# run the logits -> bias -> softmax/logsumexp chain there, and cast back to the compute
# dtype at the end.
#
# WHAT NOT TO DO (this bug SHIPPED once and was caught by an adversarial reviewer):
#   * Do NOT apply this bias directly in the layer's compute dtype. Under
#     `mixed_precision.set_global_policy('mixed_float16')` it becomes `-inf`.
#   * Do NOT use the arithmetic form `mask_bias = (1 - keep) * MASK_BIAS_VALUE`. At
#     every UNMASKED position that is `0 * -inf = NaN` — which made EnergyAttention and
#     EnergyTransformer emit 512/512 NaN under mixed_float16 with NO mask supplied. Use
#     `ops.where(keep > 0, 0.0, MASK_BIAS_VALUE)`, which CANNOT produce `0 * inf` at
#     all: the failure mode is removed structurally, not merely numerically.
#   * Do NOT "simplify" this into a per-dtype magic constant (e.g. `finfo(dtype).min / 2`).
#     A dtype-dependent constant is a second thing to get wrong, and fp16's usable range
#     is too narrow to both zero a softmax AND keep comfortable headroom.
# See decisions.md D-002 (this plan) and energy_attention.py's D-009 anchor.
MASK_BIAS_VALUE = -1e9


def mask_dtype(compute_dtype: str) -> str:
    """Dtype in which a masked softmax / logsumexp chain must be evaluated.

    Always AT LEAST float32, regardless of the global mixed-precision policy, so that
    :data:`MASK_BIAS_VALUE` is finite by construction and so the ``logsumexp`` is not
    evaluated in fp16. A ``float64`` policy is honored rather than silently downcast.

    :param compute_dtype: The layer's compute dtype (e.g. ``'float16'`` under
        ``mixed_float16``).
    :type compute_dtype: str

    :return: ``'float64'`` if the layer already computes in float64, else ``'float32'``.
    :rtype: str
    """
    return "float64" if compute_dtype == "float64" else "float32"


def validate_head_divisibility(
        dim: int,
        num_heads: int,
        *,
        dim_name: str = "dim",
        num_heads_name: str = "num_heads",
) -> None:
    """Validate that a model dimension splits evenly across attention heads.

    Multi-head attention reshapes ``(..., dim)`` into ``(..., num_heads, dim //
    num_heads)``; an uneven split silently drops or duplicates features, so this is a
    hard constructor precondition rather than a warning. Call from ``__init__``.

    The ``*_name`` keyword arguments exist so each call site can name **its own**
    constructor arguments (``hidden_size``, ``embed_dim``, ...) in the error message
    while sharing one implementation.

    :param dim: Total model / embedding dimension to be split across heads.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dim_name: Name of the ``dim`` argument as spelled by the calling layer,
        used only in the error message. Defaults to ``'dim'``.
    :type dim_name: str
    :param num_heads_name: Name of the ``num_heads`` argument as spelled by the calling
        layer, used only in the error message. Defaults to ``'num_heads'``.
    :type num_heads_name: str

    :raises ValueError: If ``dim`` is not divisible by ``num_heads``. The message names
        both offending values.

    :return: ``None``. Raises on failure, returns silently on success.
    :rtype: None
    """
    if dim % num_heads != 0:
        raise ValueError(
            f"{dim_name} ({dim}) must be divisible by {num_heads_name} ({num_heads})"
        )


def compute_attention_scale(head_dim: int) -> float:
    """Softmax temperature ``1 / sqrt(head_dim)`` as a plain Python ``float``.

    **Call this from ``__init__`` or ``build``, NEVER from ``call()``.** The returned
    value is a Python scalar precisely so that it folds into the traced graph as a
    constant; computing it inside ``call()`` with ``keras.ops.sqrt`` would add a live
    op to every forward pass for a quantity that is fixed at construction time. This is
    the standing anchor ``plan_2026-06-14_33b77a7a/D-002``, already pinned in
    ``performer_attention.py`` and ``lighthouse_attention.py``.

    Note the direction: the scale **divides** the logits (it is ``1/sqrt(d)``, not
    ``sqrt(d)``). Layers storing ``self.scale = sqrt(head_dim)`` and later dividing by
    it are a different convention and must not adopt this helper without checking their
    ``call()``.

    :param head_dim: Per-head dimension (``dim // num_heads``). Expected positive;
        deliberately **not** re-validated here, so that adopting this helper at an
        existing call site cannot change which exception that site raises. Callers
        validate ``dim``/``num_heads`` via :func:`validate_head_divisibility` and their
        own positivity checks in ``__init__``.
    :type head_dim: int

    :return: ``1.0 / sqrt(head_dim)`` as a Python ``float``, never a tensor.
    :rtype: float
    """
    return 1.0 / math.sqrt(float(head_dim))

# ---------------------------------------------------------------------
