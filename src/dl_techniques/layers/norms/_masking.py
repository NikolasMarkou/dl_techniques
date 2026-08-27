"""Shared honesty rule for ``supports_masking`` across ``layers/norms/``.

``supports_masking = True`` is a promise. It says every ``(sample, position)``
slot of the output depends only on the SAME slot of the input. A Keras mask that
was valid on the input is then still valid on the output.

For a normalization layer that promise belongs to the normalized AXIS, not to the
class. ``RMSNorm`` keeps it at the default ``axis=-1`` and breaks it at
``axis=1``. Normalizing the token axis puts one token's magnitude into every
other token's statistic. Measured cross-token leak on a ``(3, 5, 8)`` input at
``axis=1``: ``2.063`` for ``RMSNorm``, and ``0.914`` to ``23.068`` across the
seven classes that carry the flag.

**Every leak figure quoted in this package is measured at batch size 3**, on a
``(3, 5, 8)`` input built with ``numpy.random.default_rng(0)`` and perturbed at
one ``(sample, token)`` slot. The guard test that pins the same behaviour,
``tests/test_layers/test_norms/test_the_norms_propagate_masks.py``, runs at
``BATCH = 4`` on purpose, so a reader who reruns it gets a NEARBY but different
number -- ``max_logit_norm`` measures ``0.922736`` at batch 3 and ``0.910089``
at batch 4. The test asserts the leak is large, never a specific digit.

So the rule lives here once. All seven flag-carrying classes call it instead of
re-deriving the test per file. Measured: exactly seven modules in this package
import it, and each one calls it twice, in ``__init__`` and again in ``build()``.

See ``decisions.md`` D-012 of plan ``plan-2026-08-25T195813-d5a035ab``.
"""

from typing import Optional, Sequence, Union

__all__ = ["normalizes_only_the_feature_axis"]

# DECISION plan-2026-08-25T195813-d5a035ab/D-012
# Decide supports_masking in build(), from the axis resolved against the rank.
# Do NOT set it True in __init__: at axis=1 the leak is 0.914-23.068 on (3, 5, 8).
# Do NOT test the spelling `axis == -1` instead: axis=2 at rank 3 IS the feature
# axis and leaks 0.0, so test_the_norms_propagate_masks.py::
# test_the_feature_axis_configurations_still_claim_masking fails. See decisions.md D-012.


def normalizes_only_the_feature_axis(
    axis: Union[int, Sequence[int]],
    rank: Optional[int] = None,
) -> bool:
    """Decide whether an ``axis`` spec names the trailing (feature) axis and nothing else.

    The function answers at two different moments, and the two answers differ on
    purpose.

    **Decision path:**

    .. code-block:: text

              axis spec (int or sequence)
                          │
                          ▼
              ┌───────────────────────┐
              │ empty sequence?       │──── yes ──► False
              └───────────┬───────────┘
                          │ no
                          ▼
              ┌───────────────────────┐
              │ rank is None?         │──── yes ──► every entry == -1
              │ (the __init__ moment) │
              └───────────┬───────────┘
                          │ no (the build() moment)
                          ▼
              ┌───────────────────────┐
              │ rank < 2?             │──── yes ──► False
              └───────────┬───────────┘
                          │ no
                          ▼
              ┌───────────────────────────────────────┐
              │ resolve negatives against rank, then  │
              │ answer resolved == {rank - 1}         │
              └───────────────────────────────────────┘

    In the ``__init__`` moment the input rank is unknown, so only the spelling
    ``-1`` is accepted. ``-1`` names the trailing axis at every rank. A
    non-negative spelling such as ``2`` cannot be decided yet and returns
    ``False``. ``build()`` then upgrades it if the rank makes it the feature
    axis. Measured on a rank-3 input: ``axis=2`` returns ``False`` before
    ``build()`` and ``True`` after it.

    A rank below 2 returns ``False``. On a rank-1 input the trailing axis IS the
    batch axis, so normalizing it mixes samples.

    :param axis: A single axis index, or a sequence of them.
    :type axis: Union[int, Sequence[int]]
    :param rank: Rank of the input tensor, or ``None`` if not yet known.
    :type rank: Optional[int]

    :return: ``True`` only when every normalized axis is the trailing axis.
    :rtype: bool

    Example:

    .. code-block:: python

        normalizes_only_the_feature_axis(-1)          # True
        normalizes_only_the_feature_axis(2)           # False, rank unknown
        normalizes_only_the_feature_axis(2, rank=3)   # True
        normalizes_only_the_feature_axis(1, rank=3)   # False
        normalizes_only_the_feature_axis(-1, rank=1)  # False
    """
    axes = list(axis) if isinstance(axis, (list, tuple)) else [axis]
    if not axes:
        return False

    if rank is None:
        return all(ax == -1 for ax in axes)

    if rank < 2:
        return False

    resolved = {ax + rank if ax < 0 else ax for ax in axes}
    return resolved == {rank - 1}
