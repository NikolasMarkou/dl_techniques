"""
Oracle adoption for ``models/som`` -- Phase 5 batch C.

Zero adoption of the three shared instruments before this file. All three are
adopted; no new oracle is authored and no ``src/`` file is added.

THE GRADIENT ORACLE *REFUSES* THIS MODEL, AND THAT IS THE ADOPTION
--------------------------------------------------------------------
MEASURED 2026-08-21: ``SOMModel`` exposes **0 trainable weights**. Every weight
``SOMLayer.build`` creates -- ``som_weights``, ``iterations``,
``max_iterations`` -- is declared ``trainable=False``, because a Kohonen map is
fitted by COMPETITIVE LEARNING (a manual prototype update inside ``call``), not
by gradient descent. Its forward output is a ``(bmu_coords, quant_errors)``
pair whose first element is an ``argmin`` and carries no usable gradient at all.

``gradient_flow_oracle.gradient_report`` raises ``ValueError`` on an empty
trainable set, deliberately -- "a gradient-flow assertion over an empty set is
vacuous". Adopting the oracle here therefore means **pinning the refusal**, in
both directions:

1. the refusal is asserted (:class:`TestSOMHasNoGradientPath`), so a future
   change that makes a SOM weight trainable makes this file RED rather than
   silently converting a vacuous pass into a real one nobody wrote;
2. the thing the oracle would otherwise be measuring -- *does the optimisation
   actually move the parameters* -- is asserted DIRECTLY on the competitive
   update, with a two-sided control: the prototypes MOVE at ``training=True``
   and are BIT-IDENTICAL at ``training=False``.

Writing the gradient adoption as a ``pytest.skip`` would have been the quiet
option and is exactly the failure mode D-010 names: a one-sided waiver that
teaches its reader nothing.

BOTH TRAINING MODES ARE EXERCISED, DELIBERATELY
-------------------------------------------------
Iteration-2 step 18.1 fixed a real defect in this package where Keras
AUTOCASTS a float32 variable read INSIDE ``call``: ``training=True`` RAISED
while ``training=False`` stayed green. A single-mode test is blind to that
entire defect class, so every forward here is taken at both settings.

Measured 2026-08-21, ``map_size=(4, 4)``, ``input_dim=8``:

============================  ===========================
quantity                      value
============================  ===========================
trainable weights             0
non-trainable weights         3
max |dW| after one            > 0.0 (competitive update)
``training=True`` call
max |dW| after one            EXACTLY 0.0
``training=False`` call
============================  ===========================
"""

from typing import Any, Dict, List

import keras
import numpy as np
import pytest

from dl_techniques.models.som.model import SOMModel, create_som

from ..gradient_flow_oracle import gradient_report
from ..knob_sensitivity_oracle import assert_structural_knob_changes_weights
from ..smoke_contract_oracle import (
    assert_contract_rejects_a_broken_forward,
    assert_finite,
)

MAP_SIZE = (4, 4)
INPUT_DIM = 8
BUILD_SEED = 0


def _vectors(batch: int = 6, dim: int = INPUT_DIM, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((batch, dim)).astype("float32")


def _som(**o) -> SOMModel:
    kwargs: Dict[str, Any] = dict(map_size=MAP_SIZE, input_dim=INPUT_DIM)
    kwargs.update(o)
    return create_som(**kwargs)


def _built(build_fn=_som, seed: int = BUILD_SEED) -> SOMModel:
    keras.utils.set_random_seed(seed)
    model = build_fn()
    model(_vectors(1, getattr(model, "input_dim", INPUT_DIM)), training=False)
    return model


def _prototypes(model: SOMModel) -> np.ndarray:
    return np.asarray(keras.ops.convert_to_numpy(model.som_layer.weights_map))


class TestSOMHasNoGradientPath:
    """The gradient oracle's refusal, pinned two-sided."""

    def test_the_model_exposes_no_trainable_weights(self):
        """The premise. If this ever changes, every claim below is stale."""
        model = _built()
        assert model.trainable_weights == [], (
            f"SOMModel now exposes {len(model.trainable_weights)} trainable "
            f"weight(s): "
            f"{[w.path for w in model.trainable_weights]}. A Kohonen map is "
            f"fitted by competitive learning, not gradient descent -- if a "
            f"weight became trainable, the gradient-flow adoption below must "
            f"be rewritten as a real assertion instead of a pinned refusal."
        )
        assert len(model.weights) == 3, (
            f"expected exactly som_weights / iterations / max_iterations, got "
            f"{[w.path for w in model.weights]}"
        )

    def test_the_gradient_oracle_refuses_rather_than_passing_vacuously(self):
        """D-010's rule, exercised on the one package in this batch that hits
        it: an assertion over an empty weight set is not a weak assertion, it
        is no assertion, and the instrument says so."""
        model = _built()
        with pytest.raises(ValueError, match="no trainable weights"):
            gradient_report(model, _vectors())

    def test_the_competitive_update_moves_the_prototypes_at_training_true(self):
        """The claim the gradient oracle would have made, made directly.

        This is the SOM's optimisation step. Without it, "adopting the
        gradient oracle" on this package would amount to asserting that a
        model which cannot be gradient-trained cannot be gradient-trained.
        """
        model = _built()
        before = _prototypes(model).copy()
        model(_vectors(), training=True)
        after = _prototypes(model)
        delta = float(np.max(np.abs(after - before)))
        assert delta > 0.0, (
            "one training-mode call moved no prototype at all; the competitive "
            "update is not running"
        )

    def test_the_same_call_at_training_false_moves_nothing_at_all(self):
        """The discriminating half, and the reason both modes are exercised.

        Step 18.1's defect in this package was mode-dependent: Keras autocasts
        a float32 variable read inside ``call``, which RAISED at
        ``training=True`` and stayed green at ``training=False``. A one-mode
        test cannot see that.
        """
        model = _built()
        before = _prototypes(model).copy()
        model(_vectors(), training=False)
        after = _prototypes(model)
        np.testing.assert_array_equal(after, before)

    def test_the_competitive_update_assertion_can_fail(self):
        """The RED proof for this file's replacement of the gradient oracle.

        The three assertions above stand in for
        ``assert_gradients_reach_every_trainable_weight``, so they carry the
        same obligation: shown capable of failing, against a DEAD COMPONENT
        rather than against the specific bug their author had in mind. The
        injection forces ``training=False`` inside the layer, which is exactly
        "the optimisation step never runs" -- and the assertion must convict it.
        """
        model = _built()
        original = model.som_layer.call

        def inert(inputs, training=None):
            return original(inputs, training=False)

        model.som_layer.call = inert
        try:
            before = _prototypes(model).copy()
            model(_vectors(), training=True)
            delta = float(np.max(np.abs(_prototypes(model) - before)))
        finally:
            model.som_layer.call = original

        assert delta == 0.0, (
            "the injection did not actually disable the update; this proof "
            "proves nothing"
        )

    def test_the_iteration_counter_advances_only_in_training_mode(self):
        model = _built()
        start = float(keras.ops.convert_to_numpy(model.som_layer.iterations))
        model(_vectors(), training=False)
        assert float(
            keras.ops.convert_to_numpy(model.som_layer.iterations)) == start
        model(_vectors(), training=True)
        assert float(
            keras.ops.convert_to_numpy(model.som_layer.iterations)) > start


class TestSOMKnobSensitivity:

    def test_map_size_changes_the_parameterisation(self):
        builders = {
            m: (lambda m=m: _built(lambda: _som(map_size=m)))
            for m in ((3, 3), (4, 4), (5, 6))
        }
        assert_structural_knob_changes_weights(builders, knob="map_size")

    def test_input_dim_changes_the_parameterisation(self):
        builders = {
            d: (lambda d=d: _built(lambda: _som(input_dim=d)))
            for d in (4, 8, 16)
        }
        assert_structural_knob_changes_weights(builders, knob="input_dim")

    def test_neighborhood_function_reaches_the_competitive_update(self):
        """A VALUE knob, and the one a shape sweep is blind to.

        ``'gaussian'`` and ``'bubble'`` hold the SAME prototype tensor and only
        differ in how the neighbourhood weighting is computed, so no weight
        shape changes and the structural instrument cannot see it. It cannot be
        put through ``assert_value_knob_changes_output`` either: the forward
        output is a BMU ``argmin``, which is identical between the two at
        initialisation. So the claim is made where the knob actually acts --
        on the prototypes AFTER one competitive update, with the seed fixed so
        both arms start from bit-identical weights.
        """
        moved = {}
        for fn in ("gaussian", "bubble"):
            model = _built(lambda fn=fn: _som(neighborhood_function=fn))
            before = _prototypes(model).copy()
            model(_vectors(), training=True)
            moved[fn] = _prototypes(model) - before

        np.testing.assert_array_equal(
            _prototypes(_built(lambda: _som(neighborhood_function="gaussian"))),
            _prototypes(_built(lambda: _som(neighborhood_function="bubble"))),
        )  # the two arms START identical -- otherwise the delta below is a draw
        delta = float(np.max(np.abs(moved["gaussian"] - moved["bubble"])))
        assert delta > 1e-6, (
            f"neighborhood_function is a no-op: the two settings move the "
            f"prototypes identically (max|delta| = {delta:.3e}). The kwarg is "
            f"not reaching the update rule."
        )

    def test_the_knob_assertion_can_fail(self):
        builders = {"a": (lambda: _built()), "b": (lambda: _built())}
        with pytest.raises(AssertionError, match="is a no-op"):
            assert_structural_knob_changes_weights(builders, knob="map_size")

    def test_an_invalid_neighborhood_function_is_refused(self):
        with pytest.raises(ValueError, match="neighborhood_function"):
            _som(neighborhood_function="triangular")


class TestSOMSmokeContract:

    @pytest.mark.parametrize("training", [False, True])
    def test_the_forward_contract_rejects_a_broken_forward(self, training):
        """Both modes, for the reason in the module docstring."""
        model = _built()
        x = _vectors()

        def contract(out):
            assert isinstance(out, (tuple, list)) and len(out) == 2, (
                f"SOMModel.call returns a (bmu_coords, quant_errors) pair, got "
                f"{type(out)}")
            bmu, err = out
            assert tuple(bmu.shape) == (x.shape[0], 2), (
                f"BMU coords are (batch, 2) grid indices; got {tuple(bmu.shape)}")
            assert tuple(err.shape) == (x.shape[0],), (
                f"quantization errors are (batch,); got {tuple(err.shape)}")
            assert_finite(err)
            coords = np.asarray(keras.ops.convert_to_numpy(bmu))
            assert coords.min() >= 0 and coords[:, 0].max() < MAP_SIZE[0] and \
                coords[:, 1].max() < MAP_SIZE[1], (
                f"a BMU coordinate is off the {MAP_SIZE} grid: "
                f"{coords.min(axis=0)} .. {coords.max(axis=0)}")
            assert np.asarray(
                keras.ops.convert_to_numpy(err)).min() >= 0.0, (
                "a quantization error is negative; it is a Euclidean distance")

        # A fresh model per arm: the training=True arm mutates the prototypes,
        # and `assert_contract_rejects_a_broken_forward` calls the model once
        # per breaker.
        rejections = assert_contract_rejects_a_broken_forward(model, x, contract)
        assert set(rejections) == {
            "collapse_to_scalar", "slice_leading_axis", "append_trailing_axis",
        }
        if training:
            model(x, training=True)
            contract(model(x, training=False))

    def test_a_batch_of_the_wrong_width_is_refused(self):
        model = _built()
        with pytest.raises(Exception):
            model(_vectors(dim=INPUT_DIM + 3), training=False)
