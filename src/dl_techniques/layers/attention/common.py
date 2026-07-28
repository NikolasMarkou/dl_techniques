"""
Shared primitives for the ``layers/attention`` package.

This module is the shared home for five small pieces that were previously re-derived
across the package: the additive attention-mask bias constant, the dtype in which a
masked softmax must be evaluated, the fp16-safe bias *application* itself, the
``dim % num_heads`` divisibility check, and the softmax temperature
(``1 / sqrt(head_dim)``).

Adoption is **deliberately partial, and unevenly so** — read this before assuming a
number you see in a sibling module came from here:

-   ``validate_head_divisibility`` (11 importers) and ``compute_attention_scale``
    (14 importers) ARE genuinely consolidated. Where a module declines them, it says
    so in a comment naming the measurement — e.g. ``progressive_focused_attention.py``
    and ``mmdit_joint_attention.py`` both keep ``head_dim ** -0.5``, which is NOT
    bit-identical to ``compute_attention_scale`` (it mismatches in the last ULP for
    218 of head_dim 1..1024, including 32 and 128).

-   ``apply_attention_mask`` (added by ``plan-2026-07-27T183600-b4ef45f0``) is the *behavioral*
    counterpart to the pair below: it performs the ``ops.where`` bias application that
    ``MASK_BIAS_VALUE`` / ``mask_dtype`` only describe. It exists because the same
    three-line incantation was hand-written correctly in exactly one module
    (``energy_attention.py``) and incorrectly — as the ``0 * -inf = NaN`` arithmetic
    form — in ten others. It takes the **keep predicate itself** as an argument and
    performs NO polarity inference; see its docstring for why that is not negotiable.
    Its ``rescue_axis`` argument addresses the second, separable hazard — a query row
    that keeps NOTHING, whose all-``MASK_BIAS_VALUE`` logits softmax to ``NaN`` in
    fp16 and to meaningless uniform garbage in float32. The convention the rescue
    installs is: **a slice that keeps nothing is treated as keeping everything**, so
    such a row returns ``softmax(unbiased logits)`` and is finite and identical in
    float16 / float32 / float64. ``rescue_axis`` names the axis the CALLER's softmax
    reduces over; it defaults to ``-1``, so the rescue is **ON by default**.
    ``rescue_axis=None`` is the explicit, documented opt-OUT (``ring_attention.py``
    is the one site that takes it — a per-tile rescue there would read every
    entirely-masked tile as degenerate and un-mask the FUTURE under a causal mask,
    measured at 24.14; it rescues once over the full key axis before its block loop
    instead). A site whose softmax reduces over some other axis must name that axis
    explicitly — the axis is never inferred, for the same reason polarity is never
    inferred. Seven of the ten adopters therefore DERIVE the axis from their own
    ``probability_config`` (whose ``axis`` key ``ProbabilityOutput`` honors) rather
    than inheriting the ``-1`` default; ``capsule_routing`` pins the axis in
    ``__init__`` instead, and ``ring`` opts out. A ``keep`` whose extent along the
    named axis is statically 1 is REJECTED — it cannot mask anything, because softmax
    is invariant to a constant shift of the axis it reduces over (D-017).

-   ``MASK_BIAS_VALUE`` / ``mask_dtype`` are **no longer single-use.** After
    ``plan-2026-07-27T183600-b4ef45f0`` they are reached — directly or through
    ``apply_attention_mask`` — by every masked layer in the package except four.
    ``energy_attention.py`` still imports the pair directly under the legacy private
    aliases ``_MASK_BIAS_VALUE`` / ``_mask_dtype`` (the D-007 contract; do not remove
    those aliases). Ten modules migrated to ``apply_attention_mask``:
    ``capsule_routing``, ``differential``, ``gated``, ``group_query``, ``hopfield``,
    ``multi_head_cross``, ``multi_head_latent``, ``ring``, ``rpc``, ``single_window``.

    **Exactly FOUR modules still carry a LOCAL ``-1e9``-family value**, counted
    mechanically with comments and docstrings stripped:

    * ``attention_routing_capsule.py:390`` — ``ops.where``-form and the ``-inf`` is
      provably harmless at that site; deliberately left byte-unchanged.
    * ``ideogram4_attention.py`` (``_MASK_NEG``) — ``ops.where``-form with an
      explicit cast; structurally safe.
    * ``lighthouse_attention.py`` (``_MASK_SENTINEL``, a per-dtype TABLE) — the most
      careful form in the package, but a second thing to keep correct; out of scope
      (prior D-009).
    * ``progressive_focused_attention.py:870`` — a dead ``'threshold'`` branch.

    *(This corrects two earlier figures in this same docstring: the "9 other modules"
    of D-011 and the "fourteen other modules" recorded at step 10 of the previous
    plan. The step-10 count was right at the time it was taken; ten of those fourteen
    have since migrated.)*

Architecture:
    Deliberately **flat**: five module-level names, no classes, no registry, no Keras
    serialization registration. Nothing here is a layer, and nothing here holds state.
    A Keras layer imports these the same way it imports ``math`` — they are helpers,
    not a base class. Keeping the module free of classes is what stops a shared-helper
    file from quietly growing into a framework that every layer must inherit from.

    1.  **Masking** — ``MASK_BIAS_VALUE`` together with ``mask_dtype()`` form a pair.
        The constant is only safe inside the dtype the helper returns; using one
        without the other reintroduces the fp16 overflow documented below.
        ``apply_attention_mask()`` bundles that pairing into a single call so a call
        site cannot use one half without the other.

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
from typing import Any, Optional

import keras
from keras import ops

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


# DECISION plan-2026-07-27T183600-b4ef45f0/D-002
# The `keep` argument is the KEEP PREDICATE, supplied verbatim by the call site. This
# function performs NO polarity inference, and a `mask_is_inverted=`-style flag must
# never be added.
#
# WHAT NOT TO DO, and why:
#   * Do NOT infer the polarity from the mask (e.g. always assume `mask > 0` means
#     "keep"). The ten call sites do NOT agree: `rpc_attention.py` spells its predicate
#     `mask == 0`, `capsule_routing_attention.py` passes a raw boolean, and the other
#     eight multiply a `1 = keep` float mask. Guessing wrong INVERTS masking — the layer
#     then attends only to padding and ignores real content — with no shape error, no
#     exception, and a finite output that a finiteness test happily accepts. Taking the
#     predicate as an argument puts the polarity in each site's own one-line diff.
#   * Do NOT make `out_dtype` default to the compute dtype ("cast back at the end").
#     Casting `-1e9` back into fp16 makes it `-inf` again. That is harmless where every
#     softmax row keeps >= 1 position, but it is EXACTLY the defect this helper was
#     written to remove at `rpc_attention.py` (the `-inf` reaches `ops.svd`, which
#     NaN-poisons the whole decomposition) and at `capsule_routing_attention.py` (a
#     fully-masked row becomes all-`-inf` -> NaN). Both are fixed BECAUSE of this
#     default, and both regress the moment it changes. The safe behavior is the
#     default; the unsafe one must be
#     spelled out at the call site where a reader can see it.
#   * Do NOT reintroduce the arithmetic form `logits + (1 - keep) * MASK_BIAS_VALUE`
#     (see the MASK_BIAS_VALUE anchor above) or a per-dtype sentinel table.
# See decisions.md D-002 (plan-2026-07-27T183600-b4ef45f0) and D-002 at MASK_BIAS_VALUE above.
#
# DECISION plan-2026-07-27T183600-b4ef45f0/D-008
# `rescue_axis` is the package-wide fully-masked-row rescue. It was hoisted here from
# `capsule_routing_attention.py` (D-006), which held the only copy, once four more
# sites needed exactly the same expression.
#
# WHAT NOT TO DO, and why:
#   * Do NOT add a SECOND helper for the rescue. That was an explicit STOP tripwire in
#     this plan's Complexity Budget: the rescue is one `logical_or` on the predicate
#     that only makes sense in the presence of the bias it guards, and six call sites
#     copying a two-line expression is the duplication this parameter exists to avoid.
#   * Do NOT rescue AFTER the softmax with `ops.where(row_keeps_something, w, 0)`. The
#     forward pass looks clean while the UNSELECTED branch still contributes `0 * NaN`
#     in the BACKWARD pass — a visible forward NaN traded for an invisible NaN
#     gradient. Rescuing in the predicate never forms the all-`-inf` row at all.
#   * Do NOT branch on the data (`if ops.any(...)`). The rescue must be graph-safe
#     under `@tf.function` / jit; the only Python-level branch here is on the static
#     `rescue_axis` argument, which is known at trace time.
#
# DECISION plan-2026-07-27T183600-b4ef45f0/D-009
# `rescue_axis` DEFAULTS TO `-1`: the rescue is ON, and `None` is the explicit
# opt-OUT. This REVERSES the D-008 default. D-008 shipped it as opt-in (`None`)
# purely so that un-migrated call sites stayed byte-identical; the user ruled on
# 2026-07-28 that backwards compatibility is not what this package is optimizing for
# ("I care about correctness"), so the correct behavior is now the one a site gets
# for free and the NaN-producing behavior is the one it must ask for.
#
# WHAT NOT TO DO, and why:
#   * Do NOT restore `rescue_axis=None` as the default "so nothing changes silently".
#     That is exactly the hedge that was removed. A row keeping nothing yielding
#     `softmax(all -inf) = 0/0 = NaN` is not a behavior worth preserving by default,
#     and every site in this package reduces its softmax over `-1`.
#   * Do NOT infer the axis from the rank or shape of `logits`. The DEFAULT is a
#     documented constant, which is a different thing from inference: a site whose
#     softmax reduces over some other axis MUST pass that axis explicitly, and a site
#     that wants no rescue MUST pass `None`. Silently guessing per-tensor would be
#     the same class of shape-error-free mistake as guessing polarity (I2).
#   * Do NOT read the default as "the caller no longer owns the axis". It does — the
#     default is only correct for a site whose softmax reduces over the LAST axis.
#     TEN modules now adopt this helper and NINE take the rescue (`ring_attention.py`
#     opts out per tile, D-011). Six of them build that softmax from a user-supplied
#     `probability_config`, whose `axis` key `ProbabilityOutput` honors
#     (`activations/probability_output.py:180`), so those sites DERIVE `rescue_axis`
#     from their own config instead of inheriting this default — see D-017 and each
#     site's adoption comment.
# ACCEPTED COST, stated plainly: a caller whose mask is wrong now gets finite garbage
# instead of a loud NaN, at every site, in every dtype.
# See decisions.md D-009 and D-008 (plan-2026-07-27T183600-b4ef45f0) and D-006 (the
# local original at capsule_routing_attention.py, superseded by this parameter).
#
# DECISION plan-2026-07-27T183600-b4ef45f0/D-017
# A `keep` that is BROADCAST across `rescue_axis` — statically size 1 there while
# `logits` is statically longer — is REJECTED with a `ValueError`. It is never
# silently accepted and never silently "rescued".
#
# WHY, measured: softmax is invariant to a constant shift of the row it reduces over,
# so an additive bias that is CONSTANT along the reduced axis cannot mask anything, in
# any implementation. A size-1 `rescue_axis` is exactly that case, so such a mask is
# always a caller mistake (a query-axis mask passed where a key-axis mask was
# expected, or a mask that was never broadcast). MEASURED at `GatedAttention` in
# float32 with a `(B, H, Q, 1)` mask blanking query rows 5+: the output was
# BIT-IDENTICAL (maxdiff 0.0) to a no-op all-ones mask, while the pre-plan arithmetic
# form differed by 1.849 — and that 1.849 was not masking either, it was the uniform
# distribution you get once `logits + -1e9` rounds every entry of the row to the same
# float (the ulp at 1e9 is 64). Both answers are wrong; only one of them is loud.
#
# WHAT NOT TO DO, and why:
#   * Do NOT "just skip the rescue when the axis is size 1". That RE-CREATES the
#     all-`MASK_BIAS_VALUE` row this helper exists to prevent: cast back to fp16 it is
#     all-`-inf`, and its softmax is the whole-batch NaN of D-002/D-008. Skipping
#     restores the NaN; rejecting reports the caller's actual error.
#   * Do NOT downgrade this to a warning or a log line. The mask provably has NO
#     effect, so every number the caller reads afterwards was computed as if no mask
#     had been passed at all.
#   * Do NOT extend the check to `rescue_axis=None`. With the opt-out the caller has
#     named no softmax axis, so the helper has no basis for deciding which axis must
#     carry information — `ring_attention.py` passes per-TILE slices there (D-011).
#   * Do NOT make the check dynamic or data-dependent. It reads the STATIC shape only;
#     a trace-time-unknown (`None`) extent is accepted and left to the caller, because
#     branching on tensor DATA would break D-008's graph-safety guarantee.
#   * Do NOT simplify the condition to "`keep` is size 1 along the axis". The test is
#     BROADCAST — size 1 in `keep` while `logits` is genuinely longer. A length-1 key
#     axis (both size 1) is an ordinary single-token sequence and must keep working:
#     rejecting it broke `test_text_decoder.py::test_single_token_input`, which was
#     caught by the consumer gate and is now pinned by
#     `test_common.py::test_a_genuinely_length_one_axis_is_not_rejected`.
# See decisions.md D-017 (plan-2026-07-27T183600-b4ef45f0).
def apply_attention_mask(
        logits: Any,
        keep: Any,
        *,
        out_dtype: Optional[str] = None,
        rescue_axis: Optional[int] = -1,
) -> Any:
    """Add the additive attention-mask bias to ``logits`` in a dtype where it is finite.

    Generalizes the one hand-written-correct instance of this pattern in the package
    (``energy_attention.py``, anchor ``plan_2026-07-13_57c9833e/D-009``): the whole
    logits -> bias -> softmax chain is evaluated in :func:`mask_dtype`, and the bias is
    built with ``ops.where`` rather than arithmetic, so ``0 * -inf = NaN`` is impossible
    **structurally** and not merely by virtue of the dtype.

    **Degenerate-row semantics (the default).** A slice of ``keep`` that keeps NOTHING
    is treated as keeping EVERYTHING: it receives no bias at all, and its softmax is a
    finite uniform-ish distribution over every key instead of ``softmax(all -inf) =
    0/0 = NaN``. This is ON by default (``rescue_axis=-1``); ``rescue_axis=None`` opts
    OUT and restores the un-rescued, NaN-capable behavior. The accepted cost of the
    default is stated plainly: a caller whose mask is wrong gets finite garbage rather
    than a loud NaN.

    This function is orthogonal to broadcasting: it does no ``expand_dims``, ``reshape``
    or ``repeat``. Each call site keeps its own (deliberately non-unified) broadcast and
    cast order and passes an already-broadcastable ``keep``.

    :param logits: Pre-softmax attention scores, any shape, any float dtype. Cast up to
        ``mask_dtype(...)`` internally; the caller's tensor is not modified.
    :type logits: keras.KerasTensor
    :param keep: The **keep predicate**, supplied by the call site: any tensor
        broadcastable against ``logits`` whose nonzero / ``True`` entries mean "attend
        to this position". Bool, integer and float masks are all accepted (compared as
        ``> 0`` after casting). NO polarity inference is performed — a site whose mask
        spells masking as ``mask == 0`` passes its own inverted expression, and a site
        holding a boolean keep-mask passes it straight through.

        **PRECONDITION: ``keep`` must be BINARY** (``{0, 1}`` / ``{False, True}``).
        The comparison is ``> 0``, so a graded value such as ``0.5`` is FULL KEEP —
        it receives no bias at all. This is a deliberate difference from the
        arithmetic form ``logits + (1 - m) * MASK_BIAS_VALUE`` that this helper
        replaced, which interpolated (``m = 0.5`` gave ``-5e8``, i.e. effectively
        masked). MEASURED at ``GatedAttention`` in float32 with a masked half of
        ``0.5``: this helper reproduces the ALL-ONES output exactly while the old
        form reproduced the HARD-mask output exactly, a difference of 1.81. Soft
        masking is **not** a supported mode and must not be added: an additive
        ``-inf``-scale bias has no meaningful partial value, and a per-element
        interpolation is the very ``0 * -inf`` arithmetic the D-002 anchor forbids.
        Pinned by ``test_common.py::TestApplyAttentionMaskAssumesABinaryKeep``.
    :type keep: keras.KerasTensor
    :param out_dtype: Dtype of the returned tensor. ``None`` (the default) returns
        ``mask_dtype(...)`` — the SAFE choice, which keeps the biased logits out of
        fp16 so that even a fully-masked row stays finite. Pass the compute dtype
        explicitly ONLY where the downstream consumer has been verified safe with
        ``-inf`` entries (i.e. every softmax row provably keeps at least one position
        and nothing but a softmax consumes the result).
    :type out_dtype: Optional[str]
    :param rescue_axis: Axis of ``keep`` along which the downstream softmax reduces
        (the KEY axis). Defaults to ``-1``, i.e. **the rescue is ON**: a slice of
        ``keep`` along the last axis that keeps NOTHING is treated as keeping
        EVERYTHING, so an all-``MASK_BIAS_VALUE`` row is never formed and
        ``softmax(all -inf) = 0/0 = NaN`` is structurally impossible.

        Pass ``rescue_axis=None`` to **opt OUT** — the un-rescued behavior, in which a
        fully-masked row keeps its full ``MASK_BIAS_VALUE`` bias (finite in
        :func:`mask_dtype`, ``NaN`` after a softmax once cast back to fp16). Opt out
        only where a wrong mask should be a loud failure rather than finite garbage.

        Pass an explicit axis where the downstream softmax does NOT reduce over the
        last axis. The default is a documented constant, not an inference: this
        function never looks at the rank or shape of ``logits`` to pick an axis.
        See the D-009 anchor above.

        Note the rescue is evaluated on ``keep``'s OWN shape, before any broadcast
        against ``logits``, so a rank-2 ``(B, N)`` key mask expanded to ``(B, 1, 1, N)``
        is rescued per batch element, which is what "this mask keeps nothing" means
        for that layout.
    :type rescue_axis: Optional[int]

    :raises ValueError: If ``keep`` is **broadcast across** ``rescue_axis`` — i.e. its
        static extent there is 1 while ``logits`` is statically longer. Such a
        predicate is constant along the axis the caller's softmax reduces over, and
        softmax is invariant to a constant shift of that axis, so the mask provably
        cannot mask anything whatever the implementation does. This is a hard error
        rather than a silent no-op because it is a shape-error-free caller mistake:
        a query-axis mask supplied where a key-axis mask was expected, or a mask
        that was never broadcast. NOT raised when ``rescue_axis`` is ``None`` (no
        softmax axis was named), when the extent is unknown at trace time, or when
        ``logits`` is itself size 1 along that axis (an ordinary single-token
        sequence). See the D-017 anchor above.

    :return: ``logits + bias``, broadcast to the common shape of ``logits`` and
        ``keep``, in ``out_dtype`` if given, else in ``mask_dtype(...)``.
    :rtype: keras.KerasTensor
    """
    md = mask_dtype(keras.backend.standardize_dtype(logits.dtype))
    x = ops.cast(logits, md)
    kept = ops.cast(keep, md) > 0.0
    if rescue_axis is not None:
        # D-017. STATIC shapes only, and deliberately inline: a second module-level
        # helper in this file is a named STOP tripwire of the plan that wrote it. The
        # condition is BROADCAST, not size: `keep` is rejected only when it is
        # size 1 along the reduced axis WHILE `logits` is genuinely longer there. A
        # length-1 key axis (`logits` size 1 too) is an ordinary single-token
        # sequence, not a mistake — measured: `TextDecoder` feeds exactly that, and
        # rejecting it broke `test_text_decoder.py::test_single_token_input`.
        keep_shape = tuple(getattr(keep, "shape", ()) or ())
        logits_shape = tuple(getattr(logits, "shape", ()) or ())
        if (
                -len(keep_shape) <= rescue_axis < len(keep_shape)
                and -len(logits_shape) <= rescue_axis < len(logits_shape)
                and keep_shape[rescue_axis] == 1
                and logits_shape[rescue_axis] not in (1, None)
        ):
            raise ValueError(
                f"apply_attention_mask: `keep` has shape {keep_shape}, whose extent "
                f"along rescue_axis={rescue_axis} is size 1, while `logits` has "
                f"shape {logits_shape} — so the predicate is BROADCAST across the "
                "axis the caller's softmax reduces over. Softmax is invariant to a "
                "constant shift of the axis it reduces over, so such a mask cannot "
                "mask anything and would be silently ignored. Broadcast the mask "
                "along the softmax axis, pass the axis this site's softmax actually "
                "reduces over as `rescue_axis=`, or pass `rescue_axis=None` to opt "
                "out of the fully-masked-slice rescue entirely."
            )
        # A row that keeps nothing keeps everything. Graph-safe: `rescue_axis` is a
        # Python argument fixed at trace time, and the tensor-level expression has no
        # data-dependent control flow.
        kept = ops.logical_or(
            kept,
            ops.logical_not(ops.any(kept, axis=rescue_axis, keepdims=True)),
        )
    # Both `where` branches are tensors IN `md`: a bare Python `0.0` / `-1e9` pair is
    # promoted to float32 and then collides with float64 logits under a float64 policy.
    bias = ops.where(
        kept,
        ops.zeros_like(x),
        ops.full_like(x, MASK_BIAS_VALUE),
    )
    biased = x + bias
    if out_dtype is not None:
        biased = ops.cast(biased, out_dtype)
    return biased


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
