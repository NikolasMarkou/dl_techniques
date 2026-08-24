"""N-9: every BatchNorm in every MobileNet generation must carry the reference epsilon.

The defect this pins
====================

``REFERENCE_BN_EPSILON = 1e-3`` is traced to a fetched external reference (TF
Model Garden ``official/vision/modeling/backbones/mobilenet.py``, D-111). But
before D-203 only **6 of 189** BatchNorm layers actually used it: the six each
model constructs by hand. Every other norm is built inside the shared
depthwise-separable / inverted-bottleneck block layers, which route through
``create_normalization_layer`` -- a factory whose OWN epsilon default is ``1e-6``,
a thousand times smaller (D-202). Passing ``normalization_type='batch_norm'``
selects the *type* and says nothing about the *epsilon*.

Counted at ``input_shape=(64, 64, 3)`` before the fix: V1 1 vs 26, V2 2 vs 51,
V3 2 vs 45, V4 1 vs 61.

Why the obvious probe is worthless here
=======================================

Comparing the models' softmax output before and after the change reads
``max|delta| = 0.0`` for all four -- **not because nothing changed, but because a
freshly built classifier emits a uniform ``0.1`` over 10 classes** (measured
output std: exactly ``0.0`` for V1/V3/V4). Any epsilon experiment run on an
untrained network's head is measuring its degeneracy, not its arithmetic.
``test_the_epsilon_change_is_visible_on_a_live_network`` therefore overwrites
every weight from one seeded stream first, and asserts the two epsilons give
DIFFERENT outputs -- so it fails both if the fix is reverted and if the probe
goes dead again.
"""

from collections import Counter
from typing import List

import keras
import numpy as np
import pytest

from dl_techniques.models.mobilenet.common import (
    REFERENCE_BN_EPSILON,
    REFERENCE_BN_MOMENTUM,
)
from dl_techniques.models.mobilenet.mobilenet_v1 import MobileNetV1
from dl_techniques.models.mobilenet.mobilenet_v2 import MobileNetV2
from dl_techniques.models.mobilenet.mobilenet_v3 import MobileNetV3
from dl_techniques.models.mobilenet.mobilenet_v4 import MobileNetV4

_INPUT_SHAPE = (64, 64, 3)

#: BatchNorm layer count per generation at ``_INPUT_SHAPE``, MEASURED 2026-08-23.
#: Pinned so that "all norms are at 1e-3" cannot pass by finding zero norms.
_EXPECTED_BN_COUNT = {"V1": 27, "V2": 53, "V3": 47, "V4": 62}

_MODELS = {
    "V1": MobileNetV1,
    "V2": MobileNetV2,
    "V3": MobileNetV3,
    "V4": MobileNetV4,
}


def _build(cls) -> keras.Model:
    keras.utils.set_random_seed(7)
    model = cls(num_classes=10, input_shape=_INPUT_SHAPE)
    model(keras.ops.zeros((1, *_INPUT_SHAPE)))
    return model


def _batch_norms(model: keras.Model) -> List[keras.layers.BatchNormalization]:
    found, stack, seen = [], [model], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for sub in (getattr(node, "_layers", None) or []):
            stack.append(sub)
        for sub in (getattr(node, "layers", None) or []):
            stack.append(sub)
        if isinstance(node, keras.layers.BatchNormalization):
            found.append(node)
    return sorted(found, key=lambda layer: layer.path)


def _randomize(model: keras.Model, seed: int = 1234) -> None:
    """Overwrite every weight from one seeded stream.

    Two models randomized with the same seed hold bit-identical weights, so any
    output difference between them is attributable to epsilon alone.
    """
    rng = np.random.RandomState(seed)
    for weight in model.weights:
        shape = tuple(weight.shape)
        if "moving_variance" in weight.path:
            value = rng.uniform(0.05, 4.0, size=shape)
        elif weight.path.endswith("gamma") or weight.path.endswith("kernel"):
            value = rng.normal(0.0, 0.25, size=shape)
        else:
            value = rng.normal(0.0, 0.1, size=shape)
        weight.assign(value.astype("float32"))


@pytest.mark.parametrize("generation", sorted(_MODELS))
class TestEveryBatchNormUsesTheReferenceEpsilon:

    def test_the_layer_count_is_what_the_epsilon_claim_was_measured_against(
        self, generation
    ):
        norms = _batch_norms(_build(_MODELS[generation]))
        assert len(norms) == _EXPECTED_BN_COUNT[generation], (
            f"MobileNet{generation} now has {len(norms)} BatchNorm layers, not "
            f"{_EXPECTED_BN_COUNT[generation]}. The epsilon assertion below "
            "would pass vacuously against zero layers, so this count is pinned "
            "first."
        )

    def test_no_layer_is_left_on_the_factory_default(self, generation):
        norms = _batch_norms(_build(_MODELS[generation]))
        histogram = Counter(float(layer.epsilon) for layer in norms)
        offenders = [
            layer.path
            for layer in norms
            if float(layer.epsilon) != pytest.approx(REFERENCE_BN_EPSILON)
        ]
        assert not offenders, (
            f"MobileNet{generation}: {len(offenders)} of {len(norms)} BatchNorm "
            f"layers do not use REFERENCE_BN_EPSILON={REFERENCE_BN_EPSILON}. "
            f"Observed epsilon histogram: {dict(histogram)}. Almost certainly a "
            "block layer lost its `normalization_args`/`normalization_kwargs` "
            "and fell back to create_normalization_layer's own 1e-6 -- which is "
            "1000x smaller and is NOT the fetched reference. See D-203. "
            f"First offenders: {offenders[:6]}"
        )

    def test_momentum_is_still_uniform_and_matches_the_reference(self, generation):
        norms = _batch_norms(_build(_MODELS[generation]))
        observed = {float(layer.momentum) for layer in norms}
        assert observed == {REFERENCE_BN_MOMENTUM}, (
            f"MobileNet{generation} momenta are {sorted(observed)}, expected a "
            f"uniform {REFERENCE_BN_MOMENTUM}. D-111 fetched this value; unlike "
            "epsilon it was already correct everywhere."
        )


@pytest.mark.parametrize("generation", sorted(_MODELS))
def test_the_epsilon_change_is_visible_on_a_live_network(generation):
    """The change is real, AND the instrument that shows it is not dead."""
    inputs = np.random.RandomState(0).rand(2, *_INPUT_SHAPE).astype("float32")

    shipped = _build(_MODELS[generation])
    _randomize(shipped)
    y_shipped = np.array(shipped(inputs, training=False))

    assert y_shipped.std() > 1e-4, (
        "the probe is DEAD: this network's output is (near) constant, so it "
        "cannot distinguish any epsilon from any other. A freshly built "
        "MobileNet classifier emits a uniform softmax and reads max|delta| = "
        "0.0 for a change that demonstrably moves the arithmetic. Re-randomize."
    )

    reverted = _build(_MODELS[generation])
    _randomize(reverted)
    for layer in _batch_norms(reverted):
        layer.epsilon = 1e-6  # the pre-D-203 factory default
    y_reverted = np.array(reverted(inputs, training=False))

    delta = float(np.max(np.abs(y_shipped - y_reverted)))
    assert delta > 0.0, (
        f"MobileNet{generation}: forcing every BatchNorm back to the factory's "
        "1e-6 produced a bit-identical forward pass. Either the epsilon is not "
        "reaching the norms at all, or this probe went degenerate again. "
        "MEASURED 2026-08-23 with a live network: V1 1.1e-3, V2 9.8e-4, "
        "V3 4.6e-4, V4 3.7e-3."
    )
