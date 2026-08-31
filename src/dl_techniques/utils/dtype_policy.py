"""Dtype-derived numeric policy: a mask sentinel, a stability floor, a promotion.

This module holds the numbers and the dtype choice that a mixed-precision forward
pass gets wrong when each author picks them independently, and it derives all of
them from the target dtype's own ``finfo`` rather than from a per-dtype table:

- :func:`mask_sentinel` -- the large-magnitude bias written into a masked
  position before a ``softmax`` / ``max`` / ``min``.
- :func:`stability_floor` -- the epsilon that CLAMPS a divisor, i.e. the value
  passed to ``ops.maximum(value, floor)``.
- :func:`accumulation_dtype` -- the dtype a precision-sensitive reduction should
  be computed in, for the ADDITIVE ``value + eps`` shape, where a coarsened
  epsilon biases every input rather than only the degenerate ones.

**Which of the last two a site needs** is decided by the SHAPE of the guard, and
getting it wrong is not a style question -- it was MEASURED to cost up to
``+6.21`` nats at one of this repository's own ``log`` sites:

============================  ==================  ==============================
guard shape                   correct tool        why
============================  ==================  ==============================
``ops.maximum(value, eps)``   :func:`stability_floor`  inert while ``value > eps``,
                                                  so a coarse float16 floor only
                                                  moves already-degenerate values
``value + eps`` in a ``log``  :func:`accumulation_dtype`  the epsilon is added to
or a divisor                                      EVERY input, so coarsening it
                                                  biases the whole regime
============================  ==================  ==============================

Both failures are silent, both are dtype-dependent, and both therefore survive a
green float32 suite: ``float16`` tops out at ``65504`` so a ``-1e9`` sentinel is
``-inf`` there, and ``float16(1e-8)`` is exactly ``0.0`` so such an epsilon reads
as protection and provides none. Each function's docstring carries the numbers
and the ground for its own value; ``tests/test_utils/test_dtype_policy.py`` opens
by proving both hazards reproduce.

Scope, deliberately narrow. This module is pure numeric policy ONLY. It defines
no Keras object, registers nothing, and imports neither ``dl_techniques.layers``
nor ``dl_techniques.models`` -- at module level or inside a function -- so
``dl_techniques.utils`` stays free of an import cycle and ``models/`` can adopt
the policy without depending on ``layers/attention`` internals. That is asserted
mechanically over the whole ``utils/`` tree, so it is a guarded invariant rather
than a convention.

The application contract for attention masking lives elsewhere and is not
duplicated here: ``dl_techniques.layers.attention.common.apply_attention_mask``
owns the ``keep``-predicate semantics, the ``mask_dtype`` floor and the
fully-masked-row rescue. This module owns only the number.
"""

import math
import keras
import numpy as np
from typing import Any, Tuple, Union

# Every spelling the repository passes around: a plain name, a ``mixed_*`` policy
# name, a ``keras.DTypePolicy``, a numpy dtype or scalar type, or ``None``.
_DTypeLike = Union[str, np.dtype, type, Any, None]

# The two constants below are UPPER BOUNDS on a magnitude that is otherwise
# derived from the dtype itself; :func:`mask_sentinel` states the ground for
# each and which one binds per dtype.

#: Nothing above this masks any harder -- ``exp`` underflows to exactly ``0.0``
#: below roughly ``-745`` even in float64 -- and it is the incumbent
#: ``layers.attention.common.MASK_BIAS_VALUE``, reproduced rather than moved.
_MAX_USEFUL_MASK_MAGNITUDE = 1e9

#: Factor of range left unused below ``finfo.max``, so a doubled sentinel plus
#: ordinary-scale logits stays finite. Still admits ``1e4`` in float16.
_OVERFLOW_HEADROOM = 4.0

# ---------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------


def _resolve_dtype_name(dtype: _DTypeLike) -> str:
    """Normalize any accepted dtype spelling to a plain dtype name.

    A ``keras.DTypePolicy`` names two dtypes; the one that matters is
    ``compute_dtype``, because that is where the arithmetic -- and so the
    overflow or the underflow -- happens. This is the ``str(dtype)`` sniffing at
    ``layers/moe/gating.py:23``, done by resolving the policy rather than by
    substring-matching its repr.

    :param dtype: any spelling :func:`mask_sentinel` accepts.
    :return: a plain dtype name such as ``"float16"``.
    :raises ValueError: if the argument names no dtype at all.
    """
    if dtype is None:
        return keras.config.floatx()

    compute_dtype = getattr(dtype, "compute_dtype", None)
    if isinstance(compute_dtype, str):
        return compute_dtype

    if isinstance(dtype, str):
        # `keras.DTypePolicy` is the authority on the `mixed_*` mapping, and it
        # is an identity on a plain name. Doing this by hand (stripping a
        # "mixed_" prefix) would be a second, drifting copy of that mapping.
        try:
            return keras.DTypePolicy(dtype).compute_dtype
        except ValueError as exc:
            raise ValueError(
                f"{dtype!r} is not a dtype or dtype-policy name."
            ) from exc

    try:
        return np.dtype(dtype).name
    except TypeError as exc:
        raise ValueError(
            f"{dtype!r} is not a dtype, a dtype-policy name, a "
            f"keras.DTypePolicy, or None."
        ) from exc


def _dtype_facts(name: str) -> Tuple[Any, Any]:
    """Return ``(numpy_dtype, finfo)`` for a floating-point dtype name.

    ``bfloat16`` is not a numpy dtype, so ``np.finfo("bfloat16")`` raises; the
    fallback goes through ``ml_dtypes``, a hard dependency of TensorFlow. That
    import is local so a bfloat16 caller, not every importer, would be the one
    to see it missing.

    :param name: a plain dtype name, as returned by :func:`_resolve_dtype_name`.
    :return: the numpy-compatible dtype object and its ``finfo``.
    :raises ValueError: if ``name`` is not a floating-point dtype.
    """
    try:
        return np.dtype(name), np.finfo(name)
    except (TypeError, ValueError):
        pass

    try:
        import ml_dtypes
    except ImportError as exc:  # pragma: no cover - ml_dtypes ships with TF
        raise ValueError(
            f"{name!r} is unknown to numpy and ml_dtypes is not installed."
        ) from exc

    extended_dtype = getattr(ml_dtypes, name, None)
    if extended_dtype is None:
        raise ValueError(
            f"{name!r} is not a floating-point dtype this policy can reason "
            f"about."
        )
    return np.dtype(extended_dtype), ml_dtypes.finfo(extended_dtype)


def _largest_power_of_ten_at_or_below(value: float) -> float:
    """Return the largest power of ten that does not exceed ``value``.

    Rounding down to a power of ten is what makes the sentinel a recognizable
    number (``1e4``, ``1e9``) rather than an opaque fraction of a ``finfo.max``,
    at the cost of up to one order of magnitude of headroom it has no use for.

    :param value: a strictly positive magnitude.
    :return: ``10 ** floor(log10(value))``.
    """
    return 10.0 ** math.floor(math.log10(value))


# ---------------------------------------------------------------------
# Public policy
# ---------------------------------------------------------------------


def mask_sentinel(dtype: _DTypeLike = None) -> float:
    """Return the additive mask bias for a masked position in ``dtype``.

    The returned value is negative, finite in ``dtype``, exactly representable
    in ``dtype`` (it is snapped to the dtype's own grid, so it round-trips), and
    large enough that ``exp`` of it is exactly ``0.0`` there. For a
    ``min``-reduction, negate it: IEEE-754 magnitudes are symmetric, so the
    positive counterpart is exactly representable too.

    Use it inside ``keras.ops.where(keep, logits, sentinel)``. Do NOT write
    ``logits + (1 - keep) * sentinel``: that form multiplies by the sentinel, and
    under float16 a sentinel that has overflowed to ``-inf`` turns every
    **unmasked** position into ``NaN`` via ``0 * -inf``.

    **Values, and the ground for each.** Two bounds are applied, both derived
    from ``finfo``; the smaller wins.

    ==========  ===============  ========================================
    dtype       returned value   binding ground
    ==========  ===============  ========================================
    float16     ``-10000.0``     overflow: ``finfo.max`` is ``65504``
    bfloat16    ``-9984.0``      narrowing (``1e4`` on the bfloat16 grid)
    float32     ``-1e9``         usefulness cap
    float64     ``-1e9``         usefulness cap
    ==========  ===============  ========================================

    - **The overflow bound** is the largest power of ten at or below
      ``finfo(dtype).max / 4``. In float16 that is ``1e4``; ``-1e8`` and ``-1e9``
      both become ``-inf`` there, which is the defect this function exists to
      remove.
    - **The narrowing bound** applies to any dtype with fewer mantissa bits than
      float32, i.e. to float16 and bfloat16. Such a value is capped at the
      *float16*-safe magnitude, so a sentinel produced under a bfloat16 policy
      survives being stored or cast to float16 without becoming ``-inf``.
    - **The usefulness cap** is ``1e9``. Nothing above it masks any harder, and
      it is the incumbent ``layers.attention.common.MASK_BIAS_VALUE``, which is
      therefore reproduced exactly rather than moved.

    **The bfloat16 disagreement in the tree, settled.** ``layers/moe/gating.py``
    returns ``-1e4`` for bfloat16 and ``layers/attention/lighthouse_attention.py``
    returns ``-1e9``. The contradiction is real. The claim that lighthouse's
    value is "the numerically correct one" is **UNSUPPORTED** -- nothing in this
    repository measures the two apart. bfloat16 carries float32's exponent range
    (max about ``3.39e38``), so ``-1e9`` neither overflows nor rounds to ``-inf``
    there, and ``exp(-1e4)`` is already exactly ``0.0``; both values produce
    identical masked softmax output. This policy sides with ``-1e4`` on the
    stated narrowing ground above -- a bfloat16 tensor is a reduced-precision
    tensor and is one cast away from float16 -- not on a correctness claim about
    bfloat16 arithmetic, which would be unfounded. Neither existing site is
    edited by this module; both are safe as written.

    :param dtype: the COMPUTE dtype the sentinel will be materialized in. A
        dtype name, a ``mixed_*`` policy name, a ``keras.DTypePolicy``, a numpy
        dtype, or ``None`` for ``keras.config.floatx()``.
    :return: a negative Python float, exactly representable in ``dtype``.
    :raises ValueError: if ``dtype`` is not a floating-point dtype.
    """
    name = _resolve_dtype_name(dtype)
    numpy_dtype, info = _dtype_facts(name)

    magnitude = _largest_power_of_ten_at_or_below(
        float(info.max) / _OVERFLOW_HEADROOM
    )

    is_reduced_precision = info.nmant < _dtype_facts("float32")[1].nmant
    if is_reduced_precision:
        float16_max = float(_dtype_facts("float16")[1].max)
        magnitude = min(
            magnitude,
            _largest_power_of_ten_at_or_below(float16_max / _OVERFLOW_HEADROOM),
        )

    magnitude = min(magnitude, _MAX_USEFUL_MASK_MAGNITUDE)

    # Snap to the dtype's own grid so the returned value round-trips unchanged.
    # bfloat16 has 7 mantissa bits and cannot hold 1e4 exactly; returning the
    # unsnapped literal would mean the documented value and the materialized
    # value differ, which is how a "policy" starts drifting from what runs.
    return -float(np.array(magnitude, dtype=numpy_dtype))


def accumulation_dtype(dtype: _DTypeLike = None) -> str:
    """Return the dtype a precision-sensitive reduction should be computed in.

    ``float16`` and ``bfloat16`` map to ``"float32"``; ``float32`` and
    ``float64`` map to themselves. So the returned dtype is never NARROWER than
    the one passed in -- promoting a float64 computation down to float32 would
    be the same defect in the opposite direction, and is the one bug in the
    house exemplar this function generalizes
    (``models/language/colbert/components.py``'s ``_safe_l2_normalize``, which
    casts to ``"float32"`` unconditionally).

    **Use this, not** :func:`stability_floor`, **whenever the epsilon is ADDED**
    -- ``ops.log(p + eps)``, ``x / (y + eps)``, ``ops.sqrt(v + eps)``. An added
    epsilon shifts every input, so raising it to the dtype's smallest normal
    magnitude (``6.10e-05`` in float16, which is what :func:`stability_floor`
    must do) buys finiteness by paying accuracy across the whole regime rather
    than only at the degenerate point. MEASURED in
    ``layers/time_series/xlstm_blocks.py`` under ``mixed_float16``, with the
    floor at ``6.10e-05`` and the same input in float32 as reference: the
    log-forget gate read ``-9.61`` where the correct value is ``-12.00``
    (``+2.39`` nats), and ``-9.70`` against ``-15.91`` (``+6.21`` nats); the
    sLSTM output divide lost ``90.9%`` of its magnitude and the mLSTM normalizer
    ``37.3%``, in regimes that were EXACT under the bare pre-fix literal.

    The idiom is three lines, and it is a NO-OP at ``float32`` and ``float64``
    by construction -- the casts are identities and ``stability_floor`` returns
    ``requested`` unchanged -- so it cannot move a number outside half
    precision::

        accum = accumulation_dtype(self.compute_dtype)
        y = keras.ops.log(
            keras.ops.cast(p, accum) + stability_floor(accum, 1e-8)
        )
        y = keras.ops.cast(y, self.compute_dtype)

    Keep ``stability_floor`` in that expression rather than inlining the
    literal: it is the identity at ``accum``, and it keeps the site countable by
    the same census that finds every other epsilon in the tree.

    Promotion is not free -- the intermediate is twice the width, and the result
    is cast back, so a value that overflows the compute dtype becomes ``inf``
    where a coarse floor would have clamped it to something finite-but-wrong.
    Prefer being wrong loudly.

    :param dtype: the COMPUTE dtype of the surrounding layer, in any spelling
        :func:`mask_sentinel` accepts.
    :return: a plain dtype name, never narrower than ``dtype``.
    :raises ValueError: if ``dtype`` is not a floating-point dtype.
    """
    name = _resolve_dtype_name(dtype)
    _, info = _dtype_facts(name)

    if info.nmant < _dtype_facts("float32")[1].nmant:
        return "float32"
    return name


def stability_floor(dtype: _DTypeLike, requested: float) -> float:
    """Return an epsilon that is strictly positive once materialized in ``dtype``.

    ``requested`` is the value the call site wants -- almost always a float32-era
    literal such as ``1e-8`` or ``1e-10``. The returned value is the larger of
    ``requested`` and the smallest NORMAL magnitude of ``dtype``, so it can
    neither round to ``0.0`` nor land in the subnormal range where precision is
    already degraded. In float16 that lifts every floor below ``6.10e-05`` up to
    ``6.10e-05``; in float32 and float64 the requested value is almost always
    returned unchanged.

    **Scope.** That lift is a real loss of resolution, so this function is the
    right tool ONLY where the epsilon is inert against a healthy value -- the
    ``ops.maximum(value, floor)`` shape, where nothing moves while
    ``value > floor``. Where the epsilon is ADDED (``log(p + eps)``,
    ``x / (y + eps)``), it shifts every input and the lift becomes a measured
    accuracy regression; use :func:`accumulation_dtype` and apply this floor in
    the promoted dtype, where ``requested`` is returned unchanged.

    This follows the house pattern at
    ``models/language/colbert/components.py:99-113`` (``_safe_l2_normalize``) and
    its ``max(literal, finfo(compute_dtype).tiny)`` siblings in
    ``models/neural_computer/nam/cell.py`` and
    ``models/vision/keypoints/superpoint/model.py`` -- the only exemplars in the
    tree carrying a MEASURED before/after NaN regression. Do not "simplify" a
    call to this function back to a bare literal, and do not write
    ``keras.ops.cast(1e-10, x.dtype)``: that casts the LITERAL, producing ``0.0``
    in float16, so the cast that looks protective is the mechanism of the defect
    (``layers/time_series/ema_layer.py:198``).

    The floor is a magnitude only. Where the whole reduction can afford to run
    in float32 -- as ``_safe_l2_normalize`` does -- promoting is stronger than
    flooring; :func:`accumulation_dtype` names that dtype, and this function is
    then called with it (returning ``requested`` unchanged).

    :param dtype: the COMPUTE dtype the epsilon will be materialized in,
        in any spelling :func:`mask_sentinel` accepts.
    :param requested: the epsilon the call site asks for. Must be finite and
        strictly positive.
    :return: a Python float, ``>= requested``, that is normal (never zero, never
        subnormal) in ``dtype``.
    :raises ValueError: if ``dtype`` is not a floating-point dtype, if
        ``requested`` is not finite and strictly positive, or if ``requested``
        exceeds what ``dtype`` can represent at all.
    """
    name = _resolve_dtype_name(dtype)
    _, info = _dtype_facts(name)

    requested = float(requested)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError(
            f"requested must be a finite, strictly positive epsilon; got "
            f"{requested!r}."
        )
    if requested > float(info.max):
        raise ValueError(
            f"requested epsilon {requested!r} exceeds the maximum of {name} "
            f"({float(info.max)!r}); it would materialize as inf."
        )

    return max(requested, float(info.tiny))
