r"""Shared primitives for the ``layers/attention`` package: the attention-mask
bias constant and its dtype rule, ``apply_attention_mask`` (the fp16-safe way
to apply that bias), ``validate_head_divisibility``, and
``compute_attention_scale``.

Masking is a pair: ``MASK_BIAS_VALUE`` is only safe inside the dtype
``mask_dtype()`` returns, since ``np.float16(-1e9)`` is ``-inf`` and
``0 * -inf = NaN``. ``apply_attention_mask`` bundles the pair into one call so
a site cannot use half of it, takes the keep predicate itself with no
polarity inference, and by default treats a fully-masked row as keeping
everything rather than producing a NaN softmax (``rescue_axis=None`` opts
out). Adoption of these helpers across sibling modules is partial; a module
that declines one says so in a comment naming what it measured instead.

Flat: no classes, no registry, no Keras serialization. A Keras
layer imports these the way it imports ``math``.

References:
    - Vaswani et al., 2017. Attention Is All You Need. (https://arxiv.org/abs/1706.03762)
    - ``GUIDE.md`` in this package, section "What lives in ``common.py``".
"""

import math
from typing import Any, Optional, Sequence, Union

import keras

from dl_techniques.utils.dtype_policy import mask_sentinel

# DECISION plan-2026-07-27T130643-38c5646a/D-002: only bias inside a
# mask_dtype(...) chain, never the compute dtype -- np.float16(-1e9) is -inf.
# Never write (1 - keep) * MASK_BIAS_VALUE: 0 * -inf = NaN. See decisions.md.
MASK_BIAS_VALUE = mask_sentinel("float32")


def mask_dtype(compute_dtype: str) -> str:
    """Dtype in which a masked softmax / logsumexp chain must be evaluated.

    Always at least float32, whatever the global mixed-precision policy. That is what
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

    Architecture:

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
                  │              │              over every named axis
                  │              │                         │
                  │              │              a slice keeping nothing
                  │              │              keeps everything, joint
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
                              caller's own softmax

    A ``rescue_axis`` that is neither ``None``, an ``int``, nor a non-empty tuple of
    ``int`` raises before any of the above runs.

    No ``expand_dims``, ``reshape`` or ``repeat`` happens here. Each call site keeps its
    own broadcast and cast order and passes an already-broadcastable ``keep``.

    Degenerate-row semantics (the default). A slice of ``keep`` that keeps nothing
    is treated as keeping everything: it receives no bias at all, and its softmax is a
    finite spread over every key instead of ``softmax(all -inf) = 0/0 = NaN``. This is
    on by default (``rescue_axis=-1``). ``rescue_axis=None`` opts out and restores the
    un-rescued, NaN-capable behavior. The cost of the default is real: a caller whose
    mask is wrong gets finite garbage rather than a loud NaN.

    :param logits: Pre-softmax attention scores, any shape, any float dtype. Cast up to
        ``mask_dtype(...)`` internally; the caller's tensor is not modified.
    :type logits: keras.KerasTensor
    :param keep: The keep predicate, supplied by the call site: any tensor
        broadcastable against ``logits`` whose nonzero / ``True`` entries mean "attend
        to this position". Bool, integer and float masks are all accepted (compared as
        ``> 0`` after casting). No polarity inference is performed — a site whose mask
        spells masking as ``mask == 0`` passes its own inverted expression, and a site
        holding a boolean keep-mask passes it straight through.

        ``keep`` must be binary (``{0, 1}`` / ``{False, True}``).
        The comparison is ``> 0``, so a graded value such as ``0.5`` is a full keep —
        it receives no bias at all. This differs from the
        arithmetic form ``logits + (1 - m) * MASK_BIAS_VALUE`` that this helper
        replaced, which interpolated (``m = 0.5`` gave ``-5e8``, i.e. effectively
        masked). Measured at ``GatedAttention`` in float32 with a masked half of
        ``0.5``: this helper reproduces the all-ones output exactly while the old
        form reproduced the hard-mask output exactly, a difference of 1.81. Soft
        masking is not a supported mode and must not be added: an additive
        ``-inf``-scale bias has no meaningful partial value, and a per-element
        interpolation is the very ``0 * -inf`` arithmetic the D-002 anchor forbids.
        Pinned by ``test_common.py::TestApplyAttentionMaskAssumesABinaryKeep``.
    :type keep: keras.KerasTensor
    :param out_dtype: Dtype of the returned tensor. ``None`` (the default) returns
        ``mask_dtype(...)`` — the safe choice, which keeps the biased logits out of
        fp16 so that even a fully-masked row stays finite. Pass the compute dtype
        explicitly only where the downstream consumer has been verified safe with
        ``-inf`` entries (i.e. every softmax row provably keeps at least one position
        and nothing but a softmax consumes the result).
    :type out_dtype: Optional[str]
    :param rescue_axis: Axis of ``keep`` along which the downstream softmax reduces
        (the key axis). Defaults to ``-1``, i.e. the rescue is on: a slice of
        ``keep`` along the last axis that keeps nothing is treated as keeping
        everything, so an all-``MASK_BIAS_VALUE`` row is never formed and
        ``softmax(all -inf) = 0/0 = NaN`` is structurally impossible.

        Pass ``rescue_axis=None`` to opt out — the un-rescued behavior, in which a
        fully-masked row keeps its full ``MASK_BIAS_VALUE`` bias (finite in
        :func:`mask_dtype`, ``NaN`` after a softmax once cast back to fp16). Opt out
        only where a wrong mask should be a loud failure rather than finite garbage.

        Pass an explicit axis where the downstream softmax does not reduce over the
        last axis. The default is a documented constant, not an inference: this
        function never looks at the rank or shape of ``logits`` to pick an axis. That
        is the same rule as the polarity rule on ``keep`` — the caller states what it
        means, and this helper guesses nothing.

        Note the rescue is evaluated on ``keep``'s own shape, before any broadcast
        against ``logits``, so a rank-2 ``(B, N)`` key mask expanded to ``(B, 1, 1, N)``
        is rescued per batch element, which is what "this mask keeps nothing" means
        for that layout.

        A tuple/list of axes is also accepted, because ``keras.layers.Softmax``
        accepts one and the deriving sites forward it verbatim. The rescue is then
        joint over those axes: a whole reduced block that keeps nothing keeps
        everything. That is the exact generalization, not an approximation. See the
        D-018 comment in the body below.
    :type rescue_axis: Optional[Union[int, Sequence[int]]]

    :raises ValueError: If ``rescue_axis`` is neither ``None``, an ``int``, nor a
        non-empty tuple/list of ``int``. The message names the parameter; an internal
        ``TypeError`` from comparing an ``int`` against a tuple is never an acceptable
        outcome (D-018).
    :raises ValueError: If ``keep`` is broadcast across ``rescue_axis`` — i.e. its
        static extent there is 1 while ``logits`` is statically longer, along every
        named axis. Such a predicate is constant over the block the caller's softmax
        reduces over. Softmax is invariant to a constant shift of that block, so the
        mask provably cannot mask anything, whatever the implementation does.

        It is a hard error rather than a silent no-op because it is a
        shape-error-free caller mistake: a query-axis mask supplied where a key-axis
        mask was expected, or a mask that was never broadcast.

        It is not raised in four cases: ``rescue_axis`` is ``None`` (no softmax axis
        was named); the extent is unknown at trace time; ``logits`` is itself size 1
        along that axis (an ordinary single-token sequence); or, for a multi-axis
        softmax, ``keep`` varies along at least one of the named axes.
        See the D-017 and D-018 comments in the body below.

    :return: ``logits + bias``, broadcast to the common shape of ``logits`` and
        ``keep``, in ``out_dtype`` if given, else in ``mask_dtype(...)``.
    :rtype: keras.KerasTensor
    """
    # DECISION plan-2026-09-03T033750-9bdf25f4/D-007: use getattr(d, "name", None)
    # or str(d), not keras.backend.standardize_dtype (a banned Keras-2 residue) or a bare str(d) (wrong for a tf.DType). See decisions.md.
    md = mask_dtype(getattr(logits.dtype, "name", None) or str(logits.dtype))
    x = keras.ops.cast(logits, md)
    kept = keras.ops.cast(keep, md) > 0.0
    if rescue_axis is not None:
        # D-018: normalize to a tuple of axes once, since keras.layers.Softmax
        # accepts a tuple axis and a deriving site can hand us one.
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
        # D-017: static shapes only. Rejected only when broadcast (size 1) while
        # logits is genuinely longer -- a length-1 key axis with logits also size 1 is an ordinary single-token sequence, not a mistake.
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
                f"shape {logits_shape} — so the predicate is broadcast across the "
                "axis the caller's softmax reduces over. Softmax is invariant to a "
                "constant shift of the axis it reduces over, so such a mask cannot "
                "mask anything and would be silently ignored. Broadcast the mask "
                "along the softmax axis, pass the axis this site's softmax actually "
                "reduces over as `rescue_axis=`, or pass `rescue_axis=None` to opt "
                "out of the fully-masked-slice rescue entirely."
            )
        # A row that keeps nothing keeps everything; for a multi-axis softmax a
        # "row" is the joint block over axes, computed by one keras.ops.any call.
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

    The ``*_name`` keyword arguments exist so each call site can name its own
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

    Call this from ``__init__`` or ``build``, never from ``call()``. The returned
    value is a Python scalar precisely so that it folds into the traced graph as a
    constant; computing it inside ``call()`` with ``keras.ops.sqrt`` would add a live
    op to every forward pass for a quantity that is fixed at construction time. This is
    the standing anchor ``plan_2026-06-14_33b77a7a/D-002``, already pinned in
    ``lighthouse_attention.py``.

    Note the direction: the scale divides the logits (it is ``1/sqrt(d)``, not
    ``sqrt(d)``). Layers storing ``self.scale = sqrt(head_dim)`` and later dividing by
    it are a different convention and must not adopt this helper without checking their
    ``call()``.

    :param head_dim: Per-head dimension (``dim // num_heads``). Expected positive, and
        not re-validated here: adopting this helper at an
        existing call site must not change which exception that site raises. Callers
        validate ``dim`` / ``num_heads`` via :func:`validate_head_divisibility` and their
        own positivity checks in ``__init__``.
    :type head_dim: int

    :return: ``1.0 / sqrt(head_dim)`` as a Python ``float``, never a tensor.
    :rtype: float
    """
    return 1.0 / math.sqrt(float(head_dim))
