"""Helpers the schedule and layer-discovery callbacks share.

Two things lived as copies: the linear / cosine / exp interpolation (in the
temperature, harmonic-exponent and noise-sigma callbacks) and the walk over a model's
layer tree (in the two annealing callbacks and the spatial-loss logger). Each now has
one home here. ``models/vision/depth_anything/teacher_ema.py`` carries a fourth copy of
the cosine and linear algebra on a different variable layout; it is deliberately NOT
moved (plan-2026-11-11T131349-13b28b37 deferred it).
"""

import math
from typing import List

import keras

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------

SCHEDULES = frozenset({"linear", "cosine", "exp"})


def interpolate_schedule(
    schedule: str, start: float, end: float, epoch: int, total_epochs: int
) -> float:
    """Value of a ``start`` -> ``end`` schedule at ``epoch``.

    Contract: pure float64 Python math, no state. ``epoch`` is clamped to
    ``[0, total_epochs - 1]``, so any out-of-range epoch returns an endpoint value.
    ``total_epochs == 1`` returns ``end``. ``exp`` interpolates geometrically in
    log-space and so needs ``start > 0`` and ``end > 0``; a non-positive endpoint
    raises the ``ValueError`` of ``math.log``. Callers validate their own endpoints
    at construction; this function only validates ``schedule``.

    :param schedule: ``'linear'``, ``'cosine'`` or ``'exp'``.
    :type schedule: str
    :param start: Value at epoch 0.
    :type start: float
    :param end: Value at epoch ``total_epochs - 1``.
    :type end: float
    :param epoch: Zero-based epoch index.
    :type epoch: int
    :param total_epochs: Schedule length, at least 1.
    :type total_epochs: int
    :return: The interpolated value.
    :rtype: float
    :raises ValueError: If ``schedule`` is not one of the three names.
    """
    if schedule not in SCHEDULES:
        raise ValueError(
            f"schedule must be one of {sorted(SCHEDULES)}; got {schedule!r}."
        )
    if total_epochs == 1:
        return end
    frac = min(max(epoch, 0), total_epochs - 1) / (total_epochs - 1)
    if schedule == "linear":
        return start + (end - start) * frac
    if schedule == "cosine":
        # DECISION plan-2026-11-11T131349-13b28b37/D-032: ONE cosine form,
        # ``end + (start - end) * 0.5 * (1 + cos)``. The noise-sigma curriculum used
        # to write it as ``start + (end - start) * 0.5 * (1 - cos)``; the two agree in
        # exact arithmetic and differ by up to ~7e-15 in float64 (measured over 1.9M
        # points). Do NOT keep a second form to stay bit-identical in float64: the
        # float32 value each callback assigns is what the model reads, and it was equal
        # on every measured point. Pinned by
        # tests/test_callbacks/test_callback_shared_helpers.py (frozen formulas,
        # np.array_equal on the float32 values).
        cos = 0.5 * (1.0 + math.cos(math.pi * frac))
        return end + (start - end) * cos
    log_start = math.log(start)
    log_end = math.log(end)
    return math.exp(log_start + (log_end - log_start) * frac)


def walk_layers(model: keras.Model) -> List[keras.layers.Layer]:
    """Every layer below ``model``, found through lists, dicts and sub-models.

    Contract: returns the model's descendants, each exactly once, ``model`` itself
    EXCLUDED (a callback's target is a layer inside the model, never the model).
    ``Sequential`` models may contribute their ``InputLayer``; no target predicate
    used by the callbacks accepts one. Order is not specified. Never raises for a
    model whose private ``_flatten_layers`` has moved: the fallback is a depth-first
    walk over each layer's attributes that are layers or lists / tuples of layers.

    ``Layer._flatten_layers`` is private but stable across the Keras 3 series;
    ``Layer.submodules`` does not exist on Keras 3.8 and must not be used.

    :param model: The model a callback is attached to.
    :type model: keras.Model
    :return: Descendant layers, root excluded.
    :rtype: List[keras.layers.Layer]
    """
    flatten = getattr(model, "_flatten_layers", None)
    if callable(flatten):
        try:
            return list(flatten(include_self=False))
        except Exception:  # noqa: BLE001 - fall through to the manual walk
            logger.debug(
                "_flatten_layers failed; falling back to a manual sub-layer walk",
                exc_info=True,
            )

    found: List[keras.layers.Layer] = []
    seen = set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        found.append(layer)
        for attribute in vars(layer).values():
            if isinstance(attribute, keras.layers.Layer):
                visit(attribute)
            elif isinstance(attribute, (list, tuple)):
                for entry in attribute:
                    if isinstance(entry, keras.layers.Layer):
                        visit(entry)

    visit(model)
    return found[1:]  # found[0] is ``model`` itself
