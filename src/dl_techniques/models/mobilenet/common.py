"""Shared helpers for the four MobileNet architectures.

Interface contract for :func:`materialize_for_summary` — the only thing here:

    materialize_for_summary(model, input_shape) -> None

    model:       a SUBCLASSED ``keras.Model`` whose sub-layers are created in
                 ``__init__`` and connected in ``call``.
    input_shape: the UNBATCHED input shape, e.g. ``(224, 224, 3)``. ``None`` or
                 a shape containing ``None`` is a no-op.
    returns:     nothing; the model is left built, with weights materialized.
    failure:     never raises for an already-built model. A genuine forward-pass
                 failure (an input shape the architecture cannot process)
                 propagates, which is the intended behaviour — a summary of a
                 model that cannot run is worse than an error.

Why a forward pass and not ``build()`` (MEASURED 2026-08-15): for a subclassed
model, ``keras.Model.build(batch_shape)`` only marks the model built; it does not
walk ``call`` and therefore materializes NO sub-layer weights. All four MobileNet
``summary()`` overrides used a ``build(...)`` variant — V1's batch-shaped
``build((None, *shape))``, V2/V3's ``keras.Input`` route, V4's outright wrong
3-tuple — and all four printed a summary whose ``count_params()`` was exactly
**0**. Only invoking the model materializes the weights.
"""

from typing import Optional, Sequence

import keras

# ---------------------------------------------------------------------


def materialize_for_summary(
        model: keras.Model,
        input_shape: Optional[Sequence[Optional[int]]],
) -> None:
    """Ensure ``model`` is really built before ``summary()`` prints it."""
    if model.built:
        return
    if not input_shape:
        return
    if any(dim is None for dim in input_shape):
        return
    model(keras.ops.zeros((1, *tuple(int(d) for d in input_shape))))

# ---------------------------------------------------------------------
