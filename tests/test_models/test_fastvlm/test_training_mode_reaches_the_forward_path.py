"""
``training`` must reach FastVLM's stochastic layers.

``FastVLM.__init__`` calls ``_build_model`` -- which hard-codes ``training=None``
on every sublayer -- and then ``super().__init__(inputs=, outputs=)``, making the
class a ``Functional`` model with an overriding ``call()`` that passes
``training=training``. Read as source text that is a disagreement between two
forward paths, and it was routed forward twice as a defect ("dropout/BatchNorm
unreachable in training mode"). It is REFUTED: Keras dispatches to the
most-derived ``call()``, so ``_build_model``'s ``training=None`` calls run
exactly once, at construction, to trace shapes.

This file pins the refutation so nobody re-derives the false alarm and "fixes"
it. MEASURED at the geometry below (CPU, seed 3, ``dropout_rate=0.9``): two
``training=True`` calls diverge by 400.48, ``training=True`` vs ``False`` by
224.41, two ``training=False`` calls are bit-identical at 0.0, and one
``training=True`` forward moves all 10 ``moving_mean`` tensors.

Deleting the ``call()`` override does NOT make these assertions fail (measured:
``max|train1-train2|`` stays at exactly 400.475). Keras 3's ``Functional.call``
injects the caller's ``training`` into every operation in the traced graph, so
the value recorded at trace time is overridden at runtime -- a second,
independent reason the routed claim is false, and the reason the structural
assertion below reads ``"call" in FastVLM.__dict__`` rather than
``type(model).call is FastVLM.call`` (with the override gone, the latter is
still True, resolving to ``Functional.call`` -- it cannot fail and proves
nothing).

RED proof: hard-code ``training=False`` inside ``call()``. Then
``test_two_training_true_forwards_differ`` fails at ``max|delta| = 0.0`` and
``test_every_batchnorm_moving_mean_updates_under_training_true`` names all 10
stalled indices.
"""

import numpy as np
import keras

from dl_techniques.models.fastvlm.model import FastVLM

GEOMETRY = dict(input_shape=(32, 32, 3), embed_dims=[16, 32, 64], depths=[1, 1, 1])


def _build(dropout_rate):
    keras.utils.set_random_seed(3)
    return FastVLM(dropout_rate=dropout_rate, **GEOMETRY)


def _inputs():
    return keras.ops.convert_to_tensor(
        np.random.RandomState(0).randn(2, 32, 32, 3).astype("float32")
    )


def _forward(model, x, training):
    return np.asarray(keras.ops.convert_to_numpy(model(x, training=training)))


class TestTrainingModeReachesTheForwardPath:
    def test_the_class_defines_its_own_call(self):
        # `type(model).call is FastVLM.call` would be True even with the
        # override deleted (it resolves to Functional.call), so the structural
        # claim is made against __dict__, which can actually fail.
        assert "call" in FastVLM.__dict__
        assert keras.Model in FastVLM.__mro__

    def test_two_training_true_forwards_differ(self):
        model = _build(0.9)
        x = _inputs()
        delta = float(
            np.max(np.abs(_forward(model, x, True) - _forward(model, x, True)))
        )
        # Measured 400.48. The defect signal is exactly 0.0 (dropout never
        # firing), so the bar only has to separate "some" from "none".
        assert delta > 1e-3, (
            "two training=True forwards are identical "
            f"(max|delta| = {delta:.6e}); dropout is not reaching training mode"
        )

    def test_training_true_differs_from_training_false(self):
        model = _build(0.9)
        x = _inputs()
        delta = float(
            np.max(np.abs(_forward(model, x, True) - _forward(model, x, False)))
        )
        assert delta > 1e-3, (
            f"training=True and training=False agree (max|delta| = {delta:.6e})"
        )

    def test_two_training_false_forwards_are_bit_identical(self):
        # Control: the divergence above is the stochastic path, not nondeterminism.
        model = _build(0.9)
        x = _inputs()
        delta = float(
            np.max(np.abs(_forward(model, x, False) - _forward(model, x, False)))
        )
        assert delta == 0.0, f"inference is nondeterministic: max|delta| = {delta:.6e}"

    def test_every_batchnorm_moving_mean_updates_under_training_true(self):
        model = _build(0.0)
        bns = [
            layer
            for layer in model._flatten_layers()
            if isinstance(layer, keras.layers.BatchNormalization)
        ]
        assert len(bns) == 10, f"expected 10 BatchNormalization layers, found {len(bns)}"
        before = [np.array(keras.ops.convert_to_numpy(b.moving_mean)) for b in bns]
        model(_inputs(), training=True)
        after = [np.array(keras.ops.convert_to_numpy(b.moving_mean)) for b in bns]
        stalled = [
            i
            for i, (a, b) in enumerate(zip(after, before))
            if float(np.max(np.abs(a - b))) == 0.0
        ]
        assert not stalled, f"BatchNorm moving_mean did not move at indices {stalled}"
