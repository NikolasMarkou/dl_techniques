"""Shared BatchNorm constants and the ``materialize_for_summary`` helper, used by all four MobileNet architectures.

``REFERENCE_BN_MOMENTUM`` and ``REFERENCE_BN_EPSILON`` hold the one
BatchNorm momentum/epsilon pair all four generations use; see the
``D-111`` anchor below for where the values come from.
``reference_bn_kwargs()`` returns that pair as a fresh keyword dict, so a
model spells its BatchNorm configuration once instead of in every layer.

``materialize_for_summary(model, input_shape)`` runs a real forward pass
on a subclassed ``keras.Model`` so its weights are materialized before
``summary()`` prints it. Calling ``build(batch_shape)`` on a subclassed
model only marks it built without walking ``call``, so it does not create
any sub-layer weights; a forward pass does. It is a no-op when the model
is already built or ``input_shape`` contains ``None``, and it does not
catch a genuine forward-pass failure.
"""

from typing import Dict, Optional, Sequence

import keras

# ---------------------------------------------------------------------

# DECISION plan-2026-08-22T035419-a11304c8/D-111: momentum=0.99, epsilon=1e-3, traced to TF Model Garden's MobileNet backbone (V1-V4), not TF-Slim.
# All four models must forward epsilon=REFERENCE_BN_EPSILON into every BatchNorm, including inside the shared block layers (create_normalization_layer defaults to 1e-6). See decisions.md D-111, D-203.
REFERENCE_BN_MOMENTUM: float = 0.99
REFERENCE_BN_EPSILON: float = 1e-3

# ---------------------------------------------------------------------


def reference_bn_kwargs() -> Dict[str, float]:
    """Return the family's BatchNorm keywords as a new dict.

    Interface contract: takes no argument and cannot fail. The result is
    ``{"momentum": REFERENCE_BN_MOMENTUM, "epsilon": REFERENCE_BN_EPSILON}`` and
    is a different dict on every call, because a layer that stores its keyword
    dict (``ConvStem.normalization_kwargs``, the block's ``normalization_args``)
    must not see another layer's later edit. Callers: the stem, block and head
    construction of ``MobileNetV1`` to ``MobileNetV4``; unpack it into a
    ``BatchNormalization`` or pass it as a layer's normalization keywords.
    """
    return {"momentum": REFERENCE_BN_MOMENTUM, "epsilon": REFERENCE_BN_EPSILON}


# ---------------------------------------------------------------------


def materialize_for_summary(
        model: keras.Model,
        input_shape: Optional[Sequence[Optional[int]]],
) -> None:
    """Run a forward pass so a subclassed model's weights are materialized before `summary()`.

    :param model: A subclassed `keras.Model` built in `__init__`/`call`.
    :param input_shape: Unbatched input shape, e.g. `(224, 224, 3)`. A no-op
        when falsy or containing `None`.
    """
    if model.built:
        return
    if not input_shape:
        return
    if any(dim is None for dim in input_shape):
        return
    model(keras.ops.zeros((1, *tuple(int(d) for d in input_shape))))

# ---------------------------------------------------------------------
