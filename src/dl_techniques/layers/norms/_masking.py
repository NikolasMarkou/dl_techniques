"""Shared honesty rule for ``supports_masking`` across ``layers/norms/``.

``supports_masking = True`` is a PROMISE that each ``(sample, position)`` slot of
the output depends only on the SAME slot of the input, so a Keras mask that was
valid on the input is still valid on the output. For a normalization layer that
promise is a property of the *normalized axis*, not of the class: ``RMSNorm``
honours it at the default ``axis=-1`` and breaks it at ``axis=1``, where one
token's magnitude enters every other token's statistic (measured cross-token leak
on a ``(3, 5, 8)`` input: ``2.063``).

The rule therefore lives here once and is consulted by all seven flag-carrying
classes rather than being re-derived per file.

See ``decisions.md`` D-012 of plan ``plan-2026-08-25T195813-d5a035ab``.
"""

from typing import Optional, Sequence, Union

__all__ = ["normalizes_only_the_feature_axis"]

# DECISION plan-2026-08-25T195813-d5a035ab/D-012
# Do NOT go back to `self.supports_masking = True` in __init__. That is what shipped
# first, and it advertised token-independence for `axis=1`, where the measured
# cross-token leak is 0.914-23.068 on a (3, 5, 8) input. Do NOT replace this with a
# spelling-only test (`axis == -1`) either: `axis=2` on a rank-3 input IS the feature
# axis and measures a 0.0 leak, so refusing the flag there would be timid rather than
# honest, and `tests/test_layers/test_norms/test_the_norms_propagate_masks.py::
# test_the_feature_axis_configurations_still_claim_masking` fails if you do.
# The rank-aware answer is only available in build(), which is why every caller
# refines the flag there. See decisions.md D-012.


def normalizes_only_the_feature_axis(
    axis: Union[int, Sequence[int]],
    rank: Optional[int] = None,
) -> bool:
    """Decide whether an ``axis`` spec names the trailing (feature) axis and nothing else.

    Two modes, matching the two moments at which the question can be asked:

    * ``rank is None`` - the ``__init__`` moment, where the input rank is unknown.
      Only the spelling ``-1`` (or a one-element container holding it) is accepted,
      because ``-1`` names the trailing axis at every rank. A non-negative spelling
      such as ``2`` is undecidable here and returns ``False``; ``build()`` then
      upgrades it if the rank turns out to make it the feature axis.
    * ``rank`` given - the ``build()`` moment. Every axis is resolved against the
      rank and the answer is exact.

    A rank below 2 returns ``False``: on a rank-1 input the trailing axis IS the
    batch axis, so normalizing it mixes samples.

    :param axis: A single axis index, or a sequence of them.
    :type axis: Union[int, Sequence[int]]
    :param rank: Rank of the input tensor, or ``None`` if not yet known.
    :type rank: Optional[int]
    :return: ``True`` only when every normalized axis is the trailing axis.
    :rtype: bool
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
