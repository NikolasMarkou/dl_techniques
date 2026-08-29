"""One Initializer INSTANCE must not make two weights identical.

The property under test, stated once: a resolved
``keras.initializers.Initializer`` OBJECT reused for two weights whose
shapes coincide draws bit-identical values, because the instance has
already materialized its seed. Passing the same object twice happens three
ways in ``layers/logic``: a parent hands its own resolved object to every
child stage, a caller puts an instance into ``inner_logic_kwargs`` /
``inner_arithmetic_kwargs``, and a caller aliases one object across two
constructor parameters of one layer. All three were measured at
``max|delta| = 0.0`` before ``clone_initializer`` was applied at every
``add_weight`` site.

Two traps these guards are built around, both paid for on an earlier plan:

* a SEEDED initializer reads ``0.0`` WITH the clone as well as without it,
  because keeping an explicit seed is what the helper promises. Every
  guard here is therefore UNSEEDED, and the one seeded case is labelled a
  CONTROL and asserts ``0.0``.
* a default initializer that is constant (``"zeros"``, ``"ones"``, the
  softplus-temperature ``Constant``) is indistinguishable shared or
  cloned. The scalar weights are therefore probed with an unseeded
  ``RandomNormal()`` instance and ``softplus_temperature=False``, which is
  the only configuration in which ``temperature_initializer`` reaches a
  weight at all.
"""
import numpy as np
import pytest
import keras

from dl_techniques.layers.logic import (
    CircuitDepthLayer,
    LearnableArithmeticOperator,
    LearnableNeuralCircuit,
)


def max_abs_delta(a, b):
    """Largest elementwise gap between two weights of equal shape.

    :param a: First weight (a Keras variable or anything ``np.asarray``
        accepts).
    :type a: Any
    :param b: Second weight, same shape as ``a``.
    :type b: Any
    :return: ``max(|a - b|)`` as a float.
    :rtype: float
    :raises AssertionError: If the two shapes differ, which would make the
        comparison meaningless rather than merely false.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shapes differ: {a.shape} vs {b.shape}"
    return float(np.max(np.abs(a - b)))


class TestSharedInitializerInstanceDoesNotShareValues:
    """Each measured pair, unseeded, at a coinciding-shape configuration."""

    def test_neural_circuit_stages_draw_independently(self):
        """Every stage gets the parent's ONE resolved object (measured 0.0).

        ``num_*_ops_per_depth`` is held constant so ``total_operators`` is
        the same at every depth and the shapes really do coincide.
        """
        circuit = LearnableNeuralCircuit(
            circuit_depth=3,
            num_logic_ops_per_depth=2,
            num_arithmetic_ops_per_depth=2,
            routing_initializer=keras.initializers.RandomNormal(),
            combination_initializer=keras.initializers.RandomNormal(),
        )
        circuit.build((4, 8))
        stages = circuit.circuit_layers
        assert stages[0].routing_weights.shape == stages[1].routing_weights.shape
        for i in (1, 2):
            assert max_abs_delta(
                stages[0].routing_weights, stages[i].routing_weights
            ) > 0.0
            assert max_abs_delta(
                stages[0].combination_weights, stages[i].combination_weights
            ) > 0.0

    def test_inner_kwargs_instance_does_not_reach_every_child(self):
        """An instance in ``inner_*_kwargs`` is splatted into every child."""
        layer = CircuitDepthLayer(
            num_logic_ops=3,
            num_arithmetic_ops=3,
            inner_logic_kwargs={
                "operation_initializer": keras.initializers.RandomNormal()
            },
            inner_arithmetic_kwargs={
                "operation_initializer": keras.initializers.RandomNormal()
            },
        )
        layer.build((4, 8))
        for i in (1, 2):
            assert max_abs_delta(
                layer.logic_operators[0].operation_weights,
                layer.logic_operators[i].operation_weights,
            ) > 0.0
            assert max_abs_delta(
                layer.arithmetic_operators[0].operation_weights,
                layer.arithmetic_operators[i].operation_weights,
            ) > 0.0

    def test_scalar_weights_of_sibling_children_draw_independently(self):
        """The scalars, not just the selection vectors.

        ``temperature`` and ``scaling_factor`` are shape ``()``, so they
        coincide across every sibling and across each other. They are the
        half of this defect a kernels-only guard cannot see.
        """
        layer = CircuitDepthLayer(
            num_logic_ops=2,
            num_arithmetic_ops=2,
            inner_logic_kwargs={
                "softplus_temperature": False,
                "temperature_initializer": keras.initializers.RandomNormal(),
            },
            inner_arithmetic_kwargs={
                "softplus_temperature": False,
                "temperature_initializer": keras.initializers.RandomNormal(),
                "scaling_initializer": keras.initializers.RandomNormal(),
            },
        )
        layer.build((4, 8))
        assert layer.logic_operators[0].temperature.shape == ()
        assert max_abs_delta(
            layer.logic_operators[0].temperature,
            layer.logic_operators[1].temperature,
        ) > 0.0
        assert max_abs_delta(
            layer.arithmetic_operators[0].temperature,
            layer.arithmetic_operators[1].temperature,
        ) > 0.0
        assert max_abs_delta(
            layer.arithmetic_operators[0].scaling_factor,
            layer.arithmetic_operators[1].scaling_factor,
        ) > 0.0

    def test_one_object_aliased_across_two_roles_of_one_layer(self):
        """``selection_mode='global'`` makes the two roles coincide at (N,)."""
        shared = keras.initializers.RandomNormal()
        layer = CircuitDepthLayer(
            num_logic_ops=2,
            num_arithmetic_ops=2,
            selection_mode="global",
            routing_initializer=shared,
            combination_initializer=shared,
        )
        layer.build((4, 8))
        assert layer.routing_weights.shape == layer.combination_weights.shape
        assert max_abs_delta(
            layer.routing_weights, layer.combination_weights
        ) > 0.0

    def test_one_object_aliased_across_temperature_and_scaling(self):
        """Both are shape ``()``, so one object gives one number twice."""
        shared = keras.initializers.RandomNormal()
        layer = LearnableArithmeticOperator(
            softplus_temperature=False,
            use_scaling=True,
            temperature_initializer=shared,
            scaling_initializer=shared,
        )
        layer.build((4, 8))
        assert layer.temperature.shape == layer.scaling_factor.shape == ()
        assert max_abs_delta(layer.temperature, layer.scaling_factor) > 0.0


class TestControlsAndCleanCases:
    """What must NOT change, each with the twin that says the probe works."""

    def test_control_seeded_initializer_still_draws_identically(self):
        """CONTROL. Reads 0.0 with the clone AND without it.

        A seeded instance is an explicit request for reproducibility and
        ``clone_initializer`` keeps it. A guard built on a seeded
        initializer therefore passes either way and proves nothing; this
        one exists to state that, and its twin above is
        ``test_neural_circuit_stages_draw_independently``, the same
        construction unseeded.
        """
        circuit = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=2,
            num_arithmetic_ops_per_depth=2,
            routing_initializer=keras.initializers.RandomNormal(seed=42),
            combination_initializer=keras.initializers.RandomNormal(seed=42),
        )
        circuit.build((4, 8))
        stages = circuit.circuit_layers
        assert max_abs_delta(
            stages[0].routing_weights, stages[1].routing_weights
        ) == 0.0
        assert max_abs_delta(
            stages[0].combination_weights, stages[1].combination_weights
        ) == 0.0

    def test_a_string_was_never_the_defect(self):
        """A string is resolved once per consumer, so it was always safe.

        This is the precondition the fix rests on: only a resolved OBJECT
        shared between two weights is the defect.
        """
        layer = CircuitDepthLayer(
            num_logic_ops=2,
            inner_logic_kwargs={"operation_initializer": "random_normal"},
        )
        layer.build((4, 8))
        assert (
            layer.logic_operators[0].operation_initializer
            is not layer.logic_operators[1].operation_initializer
        )
        assert max_abs_delta(
            layer.logic_operators[0].operation_weights,
            layer.logic_operators[1].operation_weights,
        ) > 0.0

    def test_constant_initializers_are_unaffected(self):
        """Cloning a constant initializer must not perturb its value.

        ``"zeros"`` is the package default for every selection weight, and
        the softplus temperature is a ``Constant``. The twin that says this
        assertion can fail is ``test_channel_mix_dense_kernels_differ``,
        which runs the same comparison on a random default and gets a
        non-zero answer.
        """
        circuit = LearnableNeuralCircuit(
            circuit_depth=2,
            num_logic_ops_per_depth=2,
            num_arithmetic_ops_per_depth=2,
            use_layer_norm=True,
        )
        circuit.build((4, 8))
        stages = circuit.circuit_layers
        assert float(np.max(np.abs(np.asarray(stages[0].routing_weights)))) == 0.0
        assert max_abs_delta(
            stages[0].combination_weights, stages[1].combination_weights
        ) == 0.0
        assert max_abs_delta(
            circuit.layer_norms[0].gamma, circuit.layer_norms[1].gamma
        ) == 0.0
        assert max_abs_delta(
            stages[0].logic_operators[0].temperature,
            stages[1].logic_operators[0].temperature,
        ) == 0.0

    def test_channel_mix_dense_kernels_differ(self):
        """The twin of the constant case, on the one random default here."""
        circuit = LearnableNeuralCircuit(circuit_depth=2, channel_mix="dense")
        circuit.build((4, 8))
        dense0 = circuit.circuit_layers[0]._channel_mix_layer
        dense1 = circuit.circuit_layers[1]._channel_mix_layer
        assert max_abs_delta(dense0.kernel, dense1.kernel) > 0.0
        # The only bias weight in the package, and it is 'zeros'.
        assert max_abs_delta(dense0.bias, dense1.bias) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"routing_initializer": "zeros"},
        {"routing_initializer": keras.initializers.RandomNormal(seed=3)},
        {"combination_initializer": "glorot_uniform"},
    ],
)
def test_config_round_trip_survives_the_clone(kwargs):
    """Cloning happens at the weight, so ``get_config`` still reports the
    initializer the caller passed.

    :param kwargs: One initializer argument, as a string or an instance.
    :type kwargs: dict
    """
    layer = CircuitDepthLayer(**kwargs)
    config = layer.get_config()
    rebuilt = CircuitDepthLayer.from_config(config)
    assert rebuilt.get_config() == config
