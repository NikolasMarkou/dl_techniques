"""Shared helpers for the four MobileNet architectures.

Two things live here: the four architectures' shared BatchNorm constants
(``REFERENCE_BN_MOMENTUM``, ``REFERENCE_BN_EPSILON`` — see the ``D-111`` anchor
below for their external provenance) and :func:`materialize_for_summary`.

Interface contract for :func:`materialize_for_summary`:

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

# DECISION plan-2026-08-22T035419-a11304c8/D-111
# The BatchNorm moving-average momentum and epsilon for ALL FOUR MobileNet
# generations, stated ONCE here because four files construct BatchNorm and an
# untraced numeric repeated four times is the defect R-083 names.
#
# Traced to a named external reference, FETCHED 2026-08-22:
# TensorFlow Model Garden, `official/vision/modeling/backbones/mobilenet.py`
# (https://raw.githubusercontent.com/tensorflow/models/master/official/vision/
#  modeling/backbones/mobilenet.py) -- the ONE `MobileNet` backbone class whose
# `MODEL_ID_TO_STRUCTURE` hosts `MobileNetV1`, `MobileNetV2`, `MobileNetV3Large`
# and `MobileNetV4ConvSmall/Medium/Large` -- declares
# `norm_momentum: float = 0.99, norm_epsilon: float = 0.001` at lines 150-151
# (`Conv2DBNBlock`) and 1228-1229 (`MobileNet`). This is the only reference that
# spans V1..V4, and it agrees exactly with the values below.
#
# Do NOT change these to TF-Slim's legacy `batch_norm_decay`. That is a REAL and
# DIFFERENT named reference (`research/slim/nets/mobilenet_v1.py:433` -> 0.9997;
# `research/slim/nets/mobilenet/mobilenet.py:439` -> 0.997, shared by V2 and V3),
# it has no V4 entry at all, and adopting it would give three generations three
# different momenta traced to a codebase this package is not a port of. The two
# Google references genuinely disagree with each other; that conflict is the
# finding, and this constant records which side was taken and why.
#
# Do NOT route these sites through `layers/norms/create_normalization_layer`
# "like resnet does". MEASURED 2026-08-22: that factory defaults `epsilon=1e-6`
# while `keras.layers.BatchNormalization` defaults `epsilon=1e-3`, so the reroute
# silently divides epsilon by 1000 and CHANGES THE INFERENCE FORWARD PASS -- the
# opposite of the training-only edit intended here.
#
# Momentum itself is inference-inert, MEASURED: two BatchNorm layers with
# identical weights and momentum 0.99 vs 0.9997 give `max|delta| == 0.0` exactly
# at `training=False`, while their `moving_mean` diverges by 1.75e-2 after a
# single `training=True` step.
#
# KNOWN, MEASURED DEVIATION -- these constants govern only the SIX BatchNorm
# layers this package constructs by hand (V1 `conv1_bn`; V2 `conv1_bn`,
# `conv_last_bn`; V3 `stem_bn`, `last_bn`; V4 `stem_bn`). Every OTHER BatchNorm
# in these models is created inside the shared depthwise-separable / inverted-
# residual block layers, which route through `create_normalization_layer` and
# therefore inherit its `epsilon=1e-6`. Counted 2026-08-22 at
# `input_shape=(64, 64, 3)`: V1 1 site at 1e-3 vs 26 at 1e-6; V2 2 vs 51; V3 2
# vs 45; V4 1 vs 61. Momentum is a uniform 0.99 everywhere and matches the
# reference; epsilon does NOT, at 183 of 189 layers. Aligning those 183 to the
# reference's 1e-3 is a real outstanding item, but it CHANGES THE INFERENCE
# FORWARD PASS for the whole family and is deliberately not done here -- it needs
# its own ruling, not a drive-by. Do not "make the stem consistent" by moving
# these six DOWN to 1e-6: that walks away from the fetched reference instead of
# toward it, and it would change inference too.
# See decisions.md D-111.
REFERENCE_BN_MOMENTUM: float = 0.99
REFERENCE_BN_EPSILON: float = 1e-3

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
