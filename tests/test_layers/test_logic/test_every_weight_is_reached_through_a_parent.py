"""Every weight is created, and reached by a gradient, through a
parent's `call()`.

Three §16.3 items:

* §13.2.1 build through a parent's `call()` -- the only probe that
  sees the `StatelessScope` trap. 61 direct `layer.build(...)` calls
  in this directory bypass that path entirely.
* §13.2.2 per-variable gradient flow: non-`None` AND non-zero, named
  by `var.path`, with the `len(trainable_variables) > 0` anti-vacuity
  assertion. Grep for a non-zero gradient assertion in this directory
  returned zero hits before 2026-08-29.
* §16.3  `ops.all(ops.isfinite(y))` in every forward test -- the
  meta-tests for the directory-wide observer that implements it.

Constants pinned here, all closed forms, all at `atol=0.0, rtol=0`:

  `temperature` == log(expm1(1.0)) == 0.5413248546129181
      the pre-softplus raw value that makes softplus(raw) == the
      default `temperature_init=1.0`.
  `scaling_factor` == 1.0
      the default `scaling_init`.

`operation_weights` is deliberately NOT pinned: its default
initializer is `'zeros'`, so a live weight and an all-zero one read
identically there. That is the §13.2.1 caveat -- pin a DISCRIMINATING
entry, not merely a present one.
"""

import math

import keras
import numpy as np
import pytest
import tensorflow as tf

from .logic_subject_oracle import (
    SUBJECTS,
    SUBJECT_NAMES,
    build_through_parent,
    weight_paths_below,
)
from dl_techniques.layers.logic.logic_operators import LearnableLogicOperator
from dl_techniques.layers.logic.neural_circuit import CircuitDepthLayer

#: softplus(RAW_TEMPERATURE) == 1.0, the default temperature_init.
RAW_TEMPERATURE = float(math.log(math.expm1(1.0)))


def _as_numpy(weight):
    return keras.ops.convert_to_numpy(weight)


class TestTheWeightsSurviveABuildThroughAParent:
    """§13.2.1. A child built only by another layer's `call()`."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_parent_build_creates_the_same_weight_layout(self, name):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        through_parent = build_through_parent(subject.make(), subject)

        keras.utils.set_random_seed(7)
        functional = subject.make()
        subject.model(functional)

        assert through_parent.built, (
            f"{name} was not built by its parent's call()"
        )
        assert weight_paths_below(through_parent), (
            f"{name} created no weights under a parent build"
        )
        assert weight_paths_below(through_parent) == (
            weight_paths_below(functional)
        ), f"{name}: parent build and functional build disagree"

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_the_scalar_initializers_ran_under_the_parent_build(
            self, name
    ):
        """Pin the two DISCRIMINATING closed forms. Under a
        `StatelessScope` the `assign()`-in-`build()` shape leaves a
        weight at its `add_weight` fill value, so a table that reads
        its closed form is a table that was really initialized.
        """
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        child = build_through_parent(subject.make(), subject)

        checked = 0
        for weight in child.weights:
            actual = _as_numpy(weight)
            if weight.path.endswith("temperature"):
                # The reference is built from the ROUND-TRIPPED bits:
                # RAW_TEMPERATURE is a Python float64 and the weight is
                # float32, so comparing the two directly leaves a
                # 7.16e-10 float32 rounding residue that has nothing to
                # do with the claim being made (measured 2026-08-29).
                np.testing.assert_allclose(
                    actual,
                    np.asarray(RAW_TEMPERATURE).astype(actual.dtype),
                    atol=0.0, rtol=0,
                    err_msg=f"{weight.path} is not softplus^-1(1.0)",
                )
                checked += 1
            elif weight.path.endswith("scaling_factor"):
                np.testing.assert_allclose(
                    actual, np.asarray(1.0).astype(actual.dtype),
                    atol=0.0, rtol=0,
                    err_msg=f"{weight.path} is not the default 1.0",
                )
                checked += 1
        assert checked > 0, (
            f"{name} exposed no temperature or scaling_factor weight, "
            f"so this guard asserted nothing"
        )

    def test_an_all_zero_table_would_fail_the_pin(self):
        """The twin. `RAW_TEMPERATURE` is 0.5413..., not 0.0, so the
        pin above discriminates a live weight from a zeroed one; the
        assertion this test makes is that the two values are not the
        same number.
        """
        assert abs(RAW_TEMPERATURE) > 0.5, (
            "the pinned closed form collapsed to a value an all-zero "
            "weight would also satisfy"
        )


class TestGradientsReachEveryTrainableWeight:
    """§13.2.2. Non-`None` and non-zero, per variable, by `var.path`."""

    @pytest.mark.parametrize("name", SUBJECT_NAMES)
    def test_every_trainable_weight_receives_a_non_zero_gradient(
            self, name
    ):
        subject = SUBJECTS[name]
        keras.utils.set_random_seed(7)
        layer = build_through_parent(subject.make_live(), subject)
        sample = subject.inputs()
        tensors = (
            [keras.ops.convert_to_tensor(v) for v in sample]
            if subject.arity > 1
            else keras.ops.convert_to_tensor(sample)
        )

        with tf.GradientTape() as tape:
            loss = keras.ops.mean(
                keras.ops.square(layer(tensors, training=True))
            )
        gradients = tape.gradient(loss, layer.trainable_variables)

        assert len(layer.trainable_variables) > 0, (
            f"{name} has no trainable variables"
        )
        for variable, gradient in zip(
                layer.trainable_variables, gradients
        ):
            assert gradient is not None, (
                f"no gradient for {variable.path}"
            )
            assert np.any(_as_numpy(gradient) != 0.0), (
                f"all-zero gradient for {variable.path}"
            )

    def test_the_default_zeros_initializer_starves_the_temperature(self):
        """The configuration twin, and the reason `make_live` exists.

        MEASURED 2026-08-29 (RTX 4070, policy float32,
        LearnableLogicOperator(operation_types=['and','or','xor']),
        batch 4 on an 8x16 grid): with the default
        `operation_initializer='zeros'` the gate softmax is exactly
        uniform, so d(loss)/d(temperature) is identically 0.0 and the
        guard above would be structurally unable to observe the
        temperature at all. This is not a defect in the layer; it is a
        defect in any test that runs the gradient arm at the default.
        """
        keras.utils.set_random_seed(7)
        subject = SUBJECTS["LearnableLogicOperator"]
        layer = build_through_parent(subject.make(), subject)
        tensors = [
            keras.ops.convert_to_tensor(v) for v in subject.inputs()
        ]

        with tf.GradientTape() as tape:
            loss = keras.ops.mean(
                keras.ops.square(layer(tensors, training=True))
            )
        gradients = tape.gradient(loss, layer.trainable_variables)
        by_name = {
            v.path.split("/")[-1]: g
            for v, g in zip(layer.trainable_variables, gradients)
        }
        temperature = by_name["temperature"]

        assert temperature is not None
        assert float(np.max(np.abs(_as_numpy(temperature)))) == 0.0, (
            "the zeros-initializer starvation this test documents no "
            "longer reproduces; re-derive make_live"
        )

    def test_output_only_routing_leaves_the_routing_weights_dead(self):
        """The routing weights are created in both routing modes so a
        checkpoint from either loads into either, but only 'classic'
        reads them (source comment, `CircuitDepthLayer.build`).

        MEASURED 2026-08-29: under the default
        `circuit_routing='output_only'` the gradient for
        `routing_weights` is `None`, not merely small. Under 'classic'
        it is not `None` -- that is the twin, and it is what makes this
        a documented design consequence rather than a silent dead
        weight.
        """
        keras.utils.set_random_seed(7)
        subject = SUBJECTS["CircuitDepthLayer"]
        sample = keras.ops.convert_to_tensor(subject.inputs())

        default = build_through_parent(subject.make(), subject)
        with tf.GradientTape() as tape:
            loss = keras.ops.mean(
                keras.ops.square(default(sample, training=True))
            )
        dead = {
            v.path.split("/")[-1]: g
            for v, g in zip(
                default.trainable_variables,
                tape.gradient(loss, default.trainable_variables),
            )
        }["routing_weights"]
        assert dead is None, (
            "output_only routing now reaches routing_weights; the "
            "gradient arm can drop its 'classic' requirement"
        )

        keras.utils.set_random_seed(7)
        classic = build_through_parent(subject.make_live(), subject)
        with tf.GradientTape() as tape:
            loss = keras.ops.mean(
                keras.ops.square(classic(sample, training=True))
            )
        live = {
            v.path.split("/")[-1]: g
            for v, g in zip(
                classic.trainable_variables,
                tape.gradient(loss, classic.trainable_variables),
            )
        }["routing_weights"]
        assert live is not None, (
            "classic routing did not reach routing_weights"
        )
        assert np.any(_as_numpy(live) != 0.0), (
            "classic routing left routing_weights with a zero gradient"
        )


class TestTheFinitenessObserverIsNotVacuous:
    """Meta-tests for the directory-wide `ops.all(ops.isfinite(y))`
    instrument installed by `conftest.py`.

    The observer skips outputs that carry no values (a `KerasTensor`,
    or a graph tensor inside a `tf.function` trace). Without these two
    tests, an observer that skipped EVERYTHING would look identical to
    one that checked everything.
    """

    def test_the_observer_checked_at_least_one_concrete_forward_pass(
            self, finite_forward_observer
    ):
        subject = SUBJECTS["LearnableLogicOperator"]
        before = finite_forward_observer.concrete
        model = subject.model()
        model(subject.inputs(), training=False)
        assert finite_forward_observer.concrete > before, (
            "the finiteness observer skipped a real forward pass; it "
            "is asserting nothing for the whole directory"
        )

    def test_the_observer_skips_a_graph_traced_forward_pass(
            self, finite_forward_observer
    ):
        """The skip branch is reached by a `tf.function` trace, not by
        the functional build: all four classes implement
        `compute_output_shape`, so Keras never traces their `call` to
        infer a symbolic output (measured 2026-08-29 -- an earlier
        version of this test asserted the functional build and read
        `symbolic == 0`).
        """
        subject = SUBJECTS["LearnableLogicOperator"]
        keras.utils.set_random_seed(7)
        layer = build_through_parent(subject.make(), subject)
        tensors = [
            keras.ops.convert_to_tensor(v) for v in subject.inputs()
        ]

        @tf.function
        def traced(operands):
            return layer(operands, training=False)

        before = finite_forward_observer.symbolic
        traced(tensors)
        assert finite_forward_observer.symbolic > before, (
            "a tf.function trace no longer produces a value-less "
            "tensor; the observer's skip branch is unreachable"
        )

    def test_the_observer_fails_on_a_non_finite_forward_output(self):
        """The RED proof, in the committed record (§13.1 rule 2).

        A NaN input reaches the sigmoid the gates apply, so the output
        is NaN. The observer must turn that into a failure.
        """
        subject = SUBJECTS["LearnableLogicOperator"]
        keras.utils.set_random_seed(7)
        layer = build_through_parent(subject.make(), subject)
        poisoned = [v.copy() for v in subject.inputs()]
        poisoned[0][0, 0, 0] = np.nan

        with pytest.raises(AssertionError, match="non-finite output"):
            layer(
                [keras.ops.convert_to_tensor(v) for v in poisoned],
                training=False,
            )

    def test_the_observer_wraps_the_composite_classes_too(self):
        """A NaN produced INSIDE a CircuitDepthLayer stage is caught at
        the stage, not only at the outermost class.
        """
        subject = SUBJECTS["CircuitDepthLayer"]
        keras.utils.set_random_seed(7)
        layer = build_through_parent(subject.make(), subject)
        poisoned = subject.inputs().copy()
        poisoned[0, 0, 0] = np.nan

        with pytest.raises(AssertionError, match="non-finite output"):
            layer(keras.ops.convert_to_tensor(poisoned), training=False)


class TestTheObservedClassesAreTheRealOnes:
    """A repo-wide parametrized guard asserts a non-empty subject set
    (§16.3). This one asserts the observer's subject list is the four
    classes the package exports, not an empty tuple.
    """

    def test_the_subject_registry_covers_the_four_classes(self):
        assert len(SUBJECT_NAMES) == 4, SUBJECT_NAMES
        assert set(SUBJECT_NAMES) == {
            "LearnableLogicOperator",
            "LearnableArithmeticOperator",
            "CircuitDepthLayer",
            "LearnableNeuralCircuit",
        }
        assert LearnableLogicOperator.__name__ in SUBJECT_NAMES
        assert CircuitDepthLayer.__name__ in SUBJECT_NAMES
