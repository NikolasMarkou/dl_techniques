r"""
Shared primitives for the ``layers/attention`` package.

Five small things live here that sibling modules used to re-derive:

* the additive attention-mask bias constant;
* the dtype a masked softmax must be evaluated in;
* the fp16-safe application of that bias;
* the ``dim % num_heads`` divisibility check;
* the softmax temperature ``1 / sqrt(head_dim)``.

**Module map:**

.. code-block:: text

    MASK_BIAS_VALUE ──┐
    (the -1e9 bias)   │
                      ├─► apply_attention_mask(logits, keep, ...)
    mask_dtype(dt)  ──┘   the one place the pair is used together
    (>= float32)

    validate_head_divisibility(dim, num_heads) ─► None, or ValueError
    compute_attention_scale(head_dim)          ─► a Python float

Adoption is partial, and uneven. Read this before assuming a number you see in a
sibling module came from here.

-   ``validate_head_divisibility`` (12 importers) and ``compute_attention_scale``
    (15 importers) ARE genuinely consolidated. Where a module declines one, it
    says so in a comment naming the measurement. ``progressive_focused_attention.py``
    and ``mmdit_joint_attention.py`` both keep ``head_dim ** -0.5``, which is NOT
    bit-identical to ``compute_attention_scale`` — it mismatches in the last ULP
    for 218 of head_dim 1..1024, including 32 and 128.

    **Those two counts are MECHANICAL. Re-derive them before editing either**,
    from the repository root. Use an AST walk, never a text search: the two names
    also appear in prose, in this very docstring among other places, so a ``grep``
    counts discussion as adoption::

        .venv/bin/python - <<'PY'
        import ast, collections, pathlib
        c = collections.Counter()
        for p in pathlib.Path("src/dl_techniques/layers").rglob("*.py"):
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(n, ast.ImportFrom):
                    if (n.module or "").split(".")[-1] == "common":
                        c.update(a.name for a in n.names)
        print(c["validate_head_divisibility"], c["compute_attention_scale"])
        PY

    It prints ``12 15``. The last dotted component is what is matched, so a
    relative ``from .common import ...`` is counted. The walk is scoped to
    ``layers/`` on purpose: every importer of this module lives inside
    ``layers/attention/``, and the loose module match would otherwise count an
    unrelated ``some_package/common.py`` elsewhere in ``src/``. ``test_common.py``
    runs the SAME scope; widen or narrow the two together or the number gets a
    second home again. Both numbers were one short
    of the truth until 2026-08-28, because nothing re-derived them. The 11 / 8
    pair below has a test that re-derives it on every run.

-   ``apply_attention_mask`` (added by ``plan-2026-07-27T183600-b4ef45f0``) is the
    *behavioral* counterpart to the pair below. It performs the ``ops.where`` bias
    application that ``MASK_BIAS_VALUE`` / ``mask_dtype`` only describe. It exists
    because the same three-line incantation was hand-written correctly in exactly
    one module (``energy_attention.py``) and incorrectly — as the ``0 * -inf = NaN``
    arithmetic form — in ten others. It takes the **keep predicate itself** as an
    argument and performs NO polarity inference; see its docstring for why that is
    not negotiable.

    Its ``rescue_axis`` argument addresses a second, separable hazard: a query row
    that keeps NOTHING, whose all-``MASK_BIAS_VALUE`` logits softmax to ``NaN`` in
    fp16 and to meaningless uniform garbage in float32. The convention the rescue
    installs is: **a slice that keeps nothing is treated as keeping everything**.
    Such a row returns ``softmax(unbiased logits)`` and is finite and identical in
    float16 / float32 / float64. ``rescue_axis`` names the axis the CALLER's softmax
    reduces over. It defaults to ``-1``, so the rescue is **ON by default**.

    ``rescue_axis=None`` is the explicit, documented opt-OUT. ``ring_attention.py``
    is the one site that takes it: a per-tile rescue there would read every
    entirely-masked tile as degenerate and un-mask the FUTURE under a causal mask,
    measured at 24.14. It rescues once over the full key axis before its block loop
    instead. A site whose softmax reduces over some other axis must name that axis
    explicitly. The axis is never inferred, for the same reason polarity is never
    inferred. **EIGHT of the ELEVEN adopters** therefore DERIVE the axis from their
    own ``probability_config`` (whose ``axis`` key ``ProbabilityOutput`` honors)
    rather than inheriting the ``-1`` default. Of the remaining three,
    ``capsule_routing`` PINS the axis in ``__init__`` (its ``_site_config`` overrides
    any caller value, so there is nothing to derive), ``ring`` OPTS OUT per tile
    (``rescue_axis=None``, D-011), and ``beit`` passes ``rescue_axis=-1`` as a
    LITERAL because it has no ``probability_config`` at all — it calls
    ``ops.softmax(..., axis=-1)`` directly, so there is no config object to derive
    the axis from. That literal is a necessity of the layer's surface, not an
    oversight, and it must not be "fixed" by inventing a ``probability_config``
    parameter for it.

    A ``keep`` whose extent along every named axis is statically 1 while ``logits``
    is longer is REJECTED. It cannot mask anything, because softmax is invariant to
    a constant shift of the axis it reduces over (D-017). ``rescue_axis`` may also
    be a TUPLE of axes, in which case the rescue and that rejection are JOINT over
    them (D-018).

    **Both counts above are MECHANICAL — re-derive them in one command each, from
    the repository root, and never hand-edit one without the other**::

        # 11 adopters (files calling the module-level helper; excludes this file's
        # own definition and the private `self._apply_attention_mask` wrappers):
        grep -rlE '(^|[^._[:alnum:]])apply_attention_mask\(' \
            src/dl_techniques/layers/attention/*.py | grep -v '/common.py$' | wc -l

        # 8 of them derive the axis from their own probability_config.
        # The `grep -v '/common.py$'` is REQUIRED and was missing until
        # 2026-08-27: the search string appears in this very docstring as the
        # example being discussed, so without the exclusion the command
        # SELF-MATCHES and returns 9, not 8. The pytest guard below has always
        # excluded this file programmatically, so the enforced number was right
        # while the copy-pasteable command beside it was not -- a maintainer
        # following the docstring literally would have "fixed" a correct number.
        grep -rl 'rescue_axis=(self.probability_config' \
            src/dl_techniques/layers/attention/*.py | grep -v '/common.py$' | wc -l

    These two numbers previously drifted THREE ways at once across this docstring,
    the anchor below and the plan's own records. They are now stated ONCE
    (here) and pinned executably by
    ``test_common.py::TestTheAdopterCountsAreMechanical``, which runs exactly the
    greps above and fails if this paragraph disagrees with the source. Anything
    else that needs the number cites this paragraph rather than repeating it.

-   ``MASK_BIAS_VALUE`` / ``mask_dtype`` are **no longer single-use.** After
    ``plan-2026-07-27T183600-b4ef45f0`` they are reached — directly or through the
    helper above — by every masked layer in the package except four.
    ``energy_attention.py`` still imports the pair directly under the legacy private
    aliases ``_MASK_BIAS_VALUE`` / ``_mask_dtype`` (the D-007 contract; do not
    remove those aliases). Ten modules migrated to the helper:
    ``capsule_routing``, ``differential``, ``gated``, ``group_query``, ``hopfield``,
    ``multi_head_cross``, ``multi_head_latent``, ``ring``, ``rpc``,
    ``single_window``. (``beit`` ADOPTED the helper at birth rather than migrating
    to it, which is why that migration list is ten while the adopter count above is
    ELEVEN. The migration list is history and does not move; the adopter count is
    mechanical and does.)

    **Exactly FOUR modules still carry a LOCAL ``-1e9``-family value**, counted
    mechanically with comments and docstrings stripped:

    * ``attention_routing_capsule.py`` (``_apply_top_k_mask``) — ``ops.where``-form,
      and the ``-inf`` is provably harmless there. Left byte-unchanged on purpose.
    * ``ideogram4_attention.py`` (``_MASK_NEG``) — ``ops.where``-form with an
      explicit cast; structurally safe.
    * ``lighthouse_attention.py`` (``_MASK_SENTINEL``, a per-dtype TABLE) — the most
      careful form in the package, but a second thing to keep correct; out of scope
      (prior D-009).
    * ``progressive_focused_attention.py`` — the dead ``'threshold'`` branch of
      ``_apply_sparsity``.

Architecture:
    Flat on purpose: five module-level names, no classes, no registry, no Keras
    serialization registration. Nothing here is a layer and nothing here holds
    state. A Keras layer imports these the way it imports ``math`` — they are
    helpers, not a base class. Keeping the module free of classes is what stops a
    shared-helper file from growing into a framework every layer must inherit from.

    1.  **Masking** — ``MASK_BIAS_VALUE`` and ``mask_dtype()`` form a pair. The
        constant is only safe inside the dtype the helper returns; using one without
        the other reintroduces the fp16 overflow documented below.
        ``apply_attention_mask()`` bundles that pairing into a single call, so a call
        site cannot use one half without the other.

    2.  **Validation** — ``validate_head_divisibility()`` centralizes a check that
        was written 19 times with 6 slightly different message spellings. Its
        parameter names are configurable so that each call site keeps naming *its
        own* constructor arguments (``dim`` / ``hidden_size`` / ``embed_dim``).

    3.  **Scaling** — ``compute_attention_scale()`` returns a plain Python ``float``,
        so the softmax temperature is a graph constant folded at trace time rather
        than a per-call ``ops.sqrt`` node.

Foundational Mathematics:
    Masked scaled dot-product attention applies an additive bias ``b`` to the logits
    before the softmax::

        A = softmax( (Q K^T) * s + b ),   s = 1 / sqrt(d_head)

    with ``b = 0`` at kept positions and ``b = MASK_BIAS_VALUE`` at masked ones.
    Because ``exp(-1e9) == 0`` in float32, the masked positions receive exactly zero
    probability mass. That argument holds **only** while ``-1e9`` is representable —
    see the fp16 note on ``MASK_BIAS_VALUE``.

References:
    - Vaswani et al. (2017). "Attention Is All You Need". NeurIPS. (The
      ``1/sqrt(d_k)`` scaling and the additive mask bias.)
    - ``GUIDE.md`` in this package, section "What lives in ``common.py``".
"""

# ---------------------------------------------------------------------

import math
from typing import Any, Optional, Sequence, Union

import keras

# ---------------------------------------------------------------------

# DECISION plan-2026-07-27T130643-38c5646a/D-002 — promoted from `energy_attention.py`'s
# `plan_2026-07-13_57c9833e/D-009`, which was the package's only copy of this guard.
# `np.float16(-1e9)` is `-inf`, so use this constant only inside a `mask_dtype(...)` chain
# and cast back at the end. Do NOT bias in the compute dtype. Do NOT write
# `(1 - keep) * MASK_BIAS_VALUE` — `0 * -inf = NaN`. See decisions.md D-002.
MASK_BIAS_VALUE = -1e9


def mask_dtype(compute_dtype: str) -> str:
    """Dtype in which a masked softmax / logsumexp chain must be evaluated.

    Always AT LEAST float32, whatever the global mixed-precision policy. That is what
    keeps :data:`MASK_BIAS_VALUE` finite, and it keeps the ``logsumexp`` out of fp16. A
    ``float64`` policy is honored rather than silently downcast.

    :param compute_dtype: The layer's compute dtype (e.g. ``'float16'`` under
        ``mixed_float16``).
    :type compute_dtype: str

    :return: ``'float64'`` if the layer already computes in float64, else ``'float32'``.
    :rtype: str
    """
    return "float64" if compute_dtype == "float64" else "float32"


def apply_attention_mask(
        logits: Any,
        keep: Any,
        *,
        out_dtype: Optional[str] = None,
        rescue_axis: Optional[Union[int, Sequence[int]]] = -1,
) -> Any:
    """Add the additive attention-mask bias to ``logits`` in a dtype where it is finite.

    Generalizes the one hand-written-correct instance of this pattern in the package
    (``energy_attention.py``, anchor ``plan_2026-07-13_57c9833e/D-009``). The whole
    logits -> bias -> softmax chain runs in :func:`mask_dtype`, and the bias is built
    with ``keras.ops.where`` rather than arithmetic, so ``0 * -inf = NaN`` cannot happen
    at all. That is a structural guarantee, not a consequence of the dtype.

    **Architecture Overview:**

    .. code-block:: text

          logits [any shape]         keep [broadcastable]
                  │                           │
                  ▼                           ▼
        cast to mask_dtype(..)       kept = cast(keep) > 0
             (>= float32)                     │
                  │              ┌────────────┴────────────┐
                  │              ▼                         ▼
                  │      rescue_axis is None       an int, or a tuple
                  │         (rescue OFF)                   │
                  │              │              reject a keep broadcast
                  │              │              over EVERY named axis
                  │              │                         │
                  │              │              a slice keeping NOTHING
                  │              │              keeps EVERYTHING, joint
                  │              │              over the named axes
                  │              └────────────┬────────────┘
                  │                           ▼
                  │       ┌────────────────────────────────────────┐
                  │       │ bias = where(kept, 0, MASK_BIAS_VALUE) │
                  │       └───────────────────┬────────────────────┘
                  │                           │
                  └───────────────────────┬───┘
                                          ▼
                                   logits + bias
                                          │
                             ┌────────────┴────────────┐
                             ▼                         ▼
                     out_dtype is None          out_dtype given
                     -> mask_dtype(..)          -> cast to it
                             │                         │
                             └────────────┬────────────┘
                                          ▼
                              biased logits, for the
                              CALLER's own softmax

    A ``rescue_axis`` that is neither ``None``, an ``int``, nor a non-empty tuple of
    ``int`` raises before any of the above runs.

    No ``expand_dims``, ``reshape`` or ``repeat`` happens here. Each call site keeps its
    own broadcast and cast order and passes an already-broadcastable ``keep``.

    **Degenerate-row semantics (the default).** A slice of ``keep`` that keeps NOTHING
    is treated as keeping EVERYTHING: it receives no bias at all, and its softmax is a
    finite spread over every key instead of ``softmax(all -inf) = 0/0 = NaN``. This is
    ON by default (``rescue_axis=-1``). ``rescue_axis=None`` opts OUT and restores the
    un-rescued, NaN-capable behavior. The cost of the default is real: a caller whose
    mask is wrong gets finite garbage rather than a loud NaN.

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
        function never looks at the rank or shape of ``logits`` to pick an axis. That
        is the same rule as the polarity rule on ``keep`` — the caller states what it
        means, and this helper guesses nothing.

        Note the rescue is evaluated on ``keep``'s OWN shape, before any broadcast
        against ``logits``, so a rank-2 ``(B, N)`` key mask expanded to ``(B, 1, 1, N)``
        is rescued per batch element, which is what "this mask keeps nothing" means
        for that layout.

        A **tuple/list of axes** is also accepted, because ``keras.layers.Softmax``
        accepts one and the deriving sites forward it verbatim. The rescue is then
        JOINT over those axes: a whole reduced BLOCK that keeps nothing keeps
        everything. That is the exact generalization, not an approximation. See the
        D-018 comment in the body below.
    :type rescue_axis: Optional[Union[int, Sequence[int]]]

    :raises ValueError: If ``rescue_axis`` is neither ``None``, an ``int``, nor a
        non-empty tuple/list of ``int``. The message names the parameter; an internal
        ``TypeError`` from comparing an ``int`` against a tuple is never an acceptable
        outcome (D-018).
    :raises ValueError: If ``keep`` is **broadcast across** ``rescue_axis`` — i.e. its
        static extent there is 1 while ``logits`` is statically longer, along EVERY
        named axis. Such a predicate is constant over the block the caller's softmax
        reduces over. Softmax is invariant to a constant shift of that block, so the
        mask provably cannot mask anything, whatever the implementation does.

        It is a hard error rather than a silent no-op because it is a
        shape-error-free caller mistake: a query-axis mask supplied where a key-axis
        mask was expected, or a mask that was never broadcast.

        It is NOT raised in four cases: ``rescue_axis`` is ``None`` (no softmax axis
        was named); the extent is unknown at trace time; ``logits`` is itself size 1
        along that axis (an ordinary single-token sequence); or, for a multi-axis
        softmax, ``keep`` varies along at least one of the named axes. See the D-017 and D-018 comments in the body below.

    :return: ``logits + bias``, broadcast to the common shape of ``logits`` and
        ``keep``, in ``out_dtype`` if given, else in ``mask_dtype(...)``.
    :rtype: keras.KerasTensor
    """
    md = mask_dtype(keras.backend.standardize_dtype(logits.dtype))
    x = keras.ops.cast(logits, md)
    kept = keras.ops.cast(keep, md) > 0.0
    if rescue_axis is not None:
        # D-018. `keras.layers.Softmax` accepts a TUPLE axis and `ProbabilityOutput`
        # forwards `type_config` verbatim, so a deriving site (D-017 (b)) can hand us
        # `(1, 2)`. Normalize to a tuple of axes ONCE, here, so nothing below can
        # compare an int against a tuple — that comparison was a raw `TypeError`.
        axes = tuple(rescue_axis) if isinstance(rescue_axis, (tuple, list)) else (rescue_axis,)
        if not axes or not all(
                isinstance(a, int) and not isinstance(a, bool) for a in axes
        ):
            raise ValueError(
                f"apply_attention_mask: rescue_axis={rescue_axis!r} is not an axis. "
                "Pass an int, a non-empty tuple/list of ints (the axes the caller's "
                "softmax reduces over, as `keras.layers.Softmax` spells them), or "
                "None to opt out of the fully-masked-slice rescue."
            )
        # D-017. STATIC shapes only, and inline on purpose: a second module-level
        # helper in this file is a named STOP tripwire of the plan that wrote it. The
        # condition is BROADCAST, not size: `keep` is rejected only when it is
        # size 1 along the reduced axis WHILE `logits` is genuinely longer there. A
        # length-1 key axis (`logits` size 1 too) is an ordinary single-token
        # sequence, not a mistake — measured: `TextDecoder` feeds exactly that, and
        # rejecting it broke `test_text_decoder.py::test_single_token_input`.
        # For a MULTI-axis softmax the reduction is JOINT, so the mask is a no-op only
        # when it is broadcast along EVERY named axis: varying along even one of them
        # still perturbs the joint block. Hence `all(...)`, not `any(...)` (D-018).
        keep_shape = tuple(getattr(keep, "shape", ()) or ())
        logits_shape = tuple(getattr(logits, "shape", ()) or ())
        if all(
                -len(keep_shape) <= a < len(keep_shape)
                and -len(logits_shape) <= a < len(logits_shape)
                and keep_shape[a] == 1
                and logits_shape[a] not in (1, None)
                for a in axes
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
        # A row that keeps nothing keeps everything — and for a multi-axis softmax a
        # "row" is the JOINT block over `axes`, which is what a single `keras.ops.any` over
        # the whole tuple computes. Graph-safe: `rescue_axis` is a Python argument
        # fixed at trace time, and the tensor-level expression has no data-dependent
        # control flow. The bare int is passed through unchanged when there is one
        # axis, so the single-axis graph is byte-identical to the pre-D-018 one.
        kept = keras.ops.logical_or(
            kept,
            keras.ops.logical_not(
                keras.ops.any(
                    kept,
                    axis=axes[0] if len(axes) == 1 else axes,
                    keepdims=True,
                )
            ),
        )
    # Both `where` branches are tensors IN `md`: a bare Python `0.0` / `-1e9` pair is
    # promoted to float32 and then collides with float64 logits under a float64 policy.
    bias = keras.ops.where(
        kept,
        keras.ops.zeros_like(x),
        keras.ops.full_like(x, MASK_BIAS_VALUE),
    )
    biased = x + bias
    if out_dtype is not None:
        biased = keras.ops.cast(biased, out_dtype)
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
    ``lighthouse_attention.py``.

    Note the direction: the scale **divides** the logits (it is ``1/sqrt(d)``, not
    ``sqrt(d)``). Layers storing ``self.scale = sqrt(head_dim)`` and later dividing by
    it are a different convention and must not adopt this helper without checking their
    ``call()``.

    :param head_dim: Per-head dimension (``dim // num_heads``). Expected positive, and
        **not** re-validated here. That is on purpose: adopting this helper at an
        existing call site must not change which exception that site raises. Callers
        validate ``dim`` / ``num_heads`` via :func:`validate_head_divisibility` and their
        own positivity checks in ``__init__``.
    :type head_dim: int

    :return: ``1.0 / sqrt(head_dim)`` as a Python ``float``, never a tensor.
    :rtype: float
    """
    return 1.0 / math.sqrt(float(head_dim))

# ---------------------------------------------------------------------
