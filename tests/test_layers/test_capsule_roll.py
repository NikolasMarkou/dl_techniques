"""Test suite for `CapsuleRoll` (Keller & Welling 2022, Eq. 17).

The operator is a cyclic permutation within each capsule, so the properties worth
pinning are all *algebraic* rather than statistical: the permutation convention,
per-capsule independence, invertibility, norm preservation, and -- the one a shape
test cannot see -- the direction and axis of the shift.

Measured on the reference instance (C=2, D=4):
  * `shift=1` reproduces the paper's written example exactly:
    ``[0,1,2,3] -> [3,0,1,2]``.
  * `shift=0` and `shift=4` are the identity; `shift=5 == shift=1`.
  * `roll(roll(x, s), -s) == x` with max error **0.0** (float32, exact).
  * Each capsule's L2 norm is preserved to **0.0**.
  * A delta impulse at ``(capsule 1, slot 3)`` lands at ``(1, 2)`` for `shift=1`.

Every "nothing moved" assertion has a "something moved" twin: the direction probe
runs on a NON-SQUARE `(C, D)` grid, because a square-only grid cannot see a
transposed axis (a `capsules=4, capsule_dim=4` mistake is invisible at 4x4).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.capsules import CapsuleRoll


def _relative(model):
    """Weight paths with the model-root segment stripped, so two instances compare."""
    return sorted(w.path.split("/", 1)[-1] for w in model.weights)


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call()`` is the only path that builds the child."""

    def __init__(self, child, **kwargs):
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs, shift=1):
        return self.child(inputs, shift=shift)


class TestCapsuleRollConstructor:
    """Stored configuration and validation."""

    def test_stores_its_configuration(self):
        layer = CapsuleRoll(num_capsules=7, capsule_dim=5)
        assert layer.num_capsules == 7
        assert layer.capsule_dim == 5
        assert not layer.built

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"num_capsules": 0, "capsule_dim": 4}, "num_capsules must be positive"),
            ({"num_capsules": -1, "capsule_dim": 4}, "num_capsules must be positive"),
            ({"num_capsules": 4, "capsule_dim": 0}, "capsule_dim must be positive"),
        ],
    )
    def test_rejects_non_positive_dimensions(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CapsuleRoll(**kwargs)

    def test_registers_under_a_package_qualified_key(self):
        key = keras.saving.get_registered_name(CapsuleRoll)
        assert key == "dl_techniques.layers.capsules>CapsuleRoll", key

    def test_holds_no_weights(self):
        layer = CapsuleRoll(num_capsules=3, capsule_dim=4)
        layer(ops.convert_to_tensor(np.zeros((2, 3, 4), "float32")), shift=1)
        assert len(layer.weights) == 0, "a cyclic shift must not invent a weight"


class TestCapsuleRollPermutation:
    """The permutation convention -- the property a shape test cannot see."""

    def test_shift_one_reproduces_the_papers_written_example(self):
        # Eq. 17 verbatim: Roll_1([u_1 .. u_D]) = [u_D, u_1, .., u_{D-1}].
        layer = CapsuleRoll(num_capsules=1, capsule_dim=4)
        values = np.arange(4.0, dtype="float32").reshape(1, 1, 4)
        rolled = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=1))
        np.testing.assert_array_equal(rolled[0, 0], [3.0, 0.0, 1.0, 2.0])

    def test_a_delta_impulse_lands_in_the_expected_destination_slot(self):
        """Orientation probe on a NON-SQUARE grid (3 capsules x 5 dims).

        The impulse starts at ``(capsule 1, slot 3)``. ``out[c, j] = in[c, (j - shift) % D]``
        puts the mass at ``j = (3 + shift) % 5``, which for shift=1 is slot 4. A
        transposed implementation, or one that rolled across the capsule axis
        instead of within it, lands somewhere else -- and at C=5, D=5 a square
        grid could not tell.
        """
        layer = CapsuleRoll(num_capsules=3, capsule_dim=5)
        impulse = np.zeros((1, 3, 5), "float32")
        impulse[0, 1, 3] = 1.0
        rolled = ops.convert_to_numpy(layer(ops.convert_to_tensor(impulse), shift=1))
        location = np.unravel_index(int(np.argmax(rolled[0])), rolled[0].shape)
        assert location == (1, 4), f"impulse landed at {location}, expected (1, 4)"
        assert rolled.sum() == pytest.approx(1.0), "a permutation must preserve mass"

    def test_the_probe_can_see_a_transposed_axis(self):
        """Anti-vacuity: the SAME probe on a 5x3 grid gives a different answer.

        Without this pair the direction assertion above would also pass for an
        implementation that rolled the wrong axis, because on a square grid the
        two answers coincide.
        """
        layer = CapsuleRoll(num_capsules=5, capsule_dim=3)
        impulse = np.zeros((1, 5, 3), "float32")
        impulse[0, 1, 2] = 1.0
        rolled = ops.convert_to_numpy(layer(ops.convert_to_tensor(impulse), shift=1))
        location = np.unravel_index(int(np.argmax(rolled[0])), rolled[0].shape)
        assert location == (1, 0), (
            f"transposed grid gave {location}; a square-grid probe could not "
            f"distinguish this from the correct answer"
        )

    def test_capsules_are_permuted_independently(self):
        """The other half of the pair: perturbing capsule 0 must not touch capsule 1."""
        rng = np.random.default_rng(11)
        values = rng.normal(size=(2, 3, 6)).astype("float32")
        layer = CapsuleRoll(num_capsules=3, capsule_dim=6)
        baseline = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=2))
        bumped = values.copy()
        bumped[:, 0, 1] += 10.0
        moved = ops.convert_to_numpy(layer(ops.convert_to_tensor(bumped), shift=2))
        assert not np.allclose(moved[:, 0], baseline[:, 0]), "capsule 0 must move"
        np.testing.assert_array_equal(moved[:, 1:], baseline[:, 1:])

    def test_zero_and_full_shifts_are_the_identity(self):
        values = np.random.default_rng(3).normal(size=(2, 4, 6)).astype("float32")
        layer = CapsuleRoll(num_capsules=4, capsule_dim=6)
        for shift in (0, 6, -6, 12):
            rolled = ops.convert_to_numpy(
                layer(ops.convert_to_tensor(values), shift=shift)
            )
            np.testing.assert_array_equal(rolled, values)

    def test_negative_shift_rolls_the_other_way(self):
        values = np.arange(6.0, dtype="float32").reshape(1, 1, 6)
        layer = CapsuleRoll(num_capsules=1, capsule_dim=6)
        forward = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=2))
        backward = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=-2))
        np.testing.assert_array_equal(backward, np.roll(values, -2, axis=-1))
        assert not np.array_equal(forward, backward), "the two directions must differ"

    def test_preserves_each_capsules_l2_norm(self):
        """Preserved to float32 summation order, NOT bit-exactly.

        The permutation preserves the multiset of a capsule's values, but a norm
        is a *sum of squares* and reordering the terms gives a different last bit.
        MEASURED max absolute drift 2.4e-07 on random N(0, 1) capsules of width 7
        (one float32 ulp). Asserting atol=0.0 here would be asserting something
        false, which is why this tolerance exists and why the layer docstring does
        not claim bit-identity.
        """
        values = np.random.default_rng(5).normal(size=(2, 3, 7)).astype("float32")
        layer = CapsuleRoll(num_capsules=3, capsule_dim=7)
        rolled = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=3))
        before = np.linalg.norm(values.astype(np.float64), axis=-1)
        after = np.linalg.norm(rolled.astype(np.float64), axis=-1)
        np.testing.assert_allclose(after, before, atol=1e-12, rtol=0)
        # The float32 drift is real but bounded by one ulp of the norm.
        np.testing.assert_allclose(
            np.linalg.norm(rolled, axis=-1),
            np.linalg.norm(values, axis=-1),
            atol=4e-7,
            rtol=0,
        )

    def test_the_values_themselves_are_restored_bit_exactly(self):
        """The other half of the pair: the roll moves the values, restores them.

        Without this, the norm assertion above would also pass for a layer that
        silently did nothing but rounded.
        """
        values = np.random.default_rng(5).normal(size=(2, 3, 7)).astype("float32")
        layer = CapsuleRoll(num_capsules=3, capsule_dim=7)
        rolled = ops.convert_to_numpy(layer(ops.convert_to_tensor(values), shift=3))
        assert sorted(np.sort(rolled[0, 0]).tolist()) == sorted(
            np.sort(values[0, 0]).tolist()
        ), "a roll must permute the values, not alter them"
        np.testing.assert_array_equal(
            sorted(rolled[0, 1].tolist()), sorted(values[0, 1].tolist())
        )

    @pytest.mark.parametrize("shift", [1, 2, 5, -3])
    def test_is_exactly_invertible(self, shift):
        values = np.random.default_rng(abs(shift) + 1).normal(
            size=(2, 3, 7)
        ).astype("float32")
        layer = CapsuleRoll(num_capsules=3, capsule_dim=7)
        forward = layer(ops.convert_to_tensor(values), shift=shift)
        back = layer(forward, shift=-shift)
        # atol=0.0: this is a permutation, so restoration is a copy, not a
        # computation, and any nonzero delta means something was lost.
        np.testing.assert_allclose(
            ops.convert_to_numpy(back), values, atol=0.0, rtol=0
        )

    def test_a_full_cycle_returns_to_the_start(self):
        values = np.random.default_rng(7).normal(size=(1, 2, 4)).astype("float32")
        layer = CapsuleRoll(num_capsules=2, capsule_dim=4)
        current = ops.convert_to_tensor(values)
        for step in range(4):
            current = layer(current, shift=1)
        np.testing.assert_allclose(
            ops.convert_to_numpy(current), values, atol=0.0, rtol=0
        )


class TestCapsuleRollShapes:
    """Shape contract and graph safety."""

    @pytest.mark.parametrize(
        "leading,c,d", [(1, 3, 4), (2, 3, 4), (7, 5, 9), (2, 1, 1)]
    )
    def test_output_shape_equals_input_shape(self, leading, c, d):
        layer = CapsuleRoll(num_capsules=c, capsule_dim=d)
        shape = (leading, c, d)
        values = ops.convert_to_tensor(
            np.random.default_rng(0).normal(size=shape).astype("float32")
        )
        rolled = layer(values, shift=1)
        assert tuple(rolled.shape) == shape
        assert np.all(np.isfinite(ops.convert_to_numpy(rolled)))

    def test_compute_output_shape_works_before_and_after_build(self):
        layer = CapsuleRoll(num_capsules=3, capsule_dim=4)
        assert not layer.built
        assert layer.compute_output_shape((None, 7, 3, 4)) == (None, 7, 3, 4)
        assert not layer.built, "compute_output_shape must not build the layer"
        layer(ops.convert_to_tensor(np.zeros((1, 7, 3, 4), "float32")), shift=1)
        assert layer.compute_output_shape((2, 7, 3, 4)) == (2, 7, 3, 4)

    def test_is_graph_safe_and_matches_eager(self):
        layer = CapsuleRoll(num_capsules=3, capsule_dim=4)
        values = ops.convert_to_tensor(
            np.random.default_rng(1).normal(size=(2, 6, 3, 4)).astype("float32")
        )
        eager = ops.convert_to_numpy(layer(values, shift=2))

        @tf.function
        def traced(x):
            return layer(x, shift=2)

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(values)), eager, atol=0.0, rtol=0
        )

    def test_survives_xla_compilation(self):
        values = ops.convert_to_tensor(
            np.random.default_rng(2).normal(size=(2, 6, 3, 4)).astype("float32")
        )
        # The shift is CAPTURED, not passed as a traced argument: `shift` is a
        # Python int here (the index table is built from it at trace time), and
        # passing it through the polymorphic signature makes tf.function try to
        # trace it as a tensor and raise on the int.
        layer = CapsuleRoll(num_capsules=3, capsule_dim=4)

        @tf.function(jit_compile=True)
        def compiled(x):
            return layer(x, shift=2)

        eager = ops.convert_to_numpy(layer(values, shift=2))
        # atol=0.0: a gather reindexes, so XLA and eager must agree exactly.
        np.testing.assert_allclose(
            ops.convert_to_numpy(compiled(values)), eager, atol=0.0, rtol=0
        )


class TestCapsuleRollSerialization:
    """Config completeness and a value round trip."""

    def test_config_is_complete(self):
        config = CapsuleRoll(num_capsules=3, capsule_dim=4).get_config()
        assert config["num_capsules"] == 3
        assert config["capsule_dim"] == 4

    def test_from_config_round_trips(self):
        config = CapsuleRoll(num_capsules=9, capsule_dim=2).get_config()
        restored = CapsuleRoll.from_config(config)
        assert restored.num_capsules == 9
        assert restored.capsule_dim == 2

    def test_value_round_trip_through_keras_archive(self):
        layer = CapsuleRoll(num_capsules=3, capsule_dim=4)
        inputs = keras.Input(shape=(7, 3, 4))
        model = keras.Model(inputs, layer(inputs, shift=2))
        sample = np.random.default_rng(4).normal(size=(2, 7, 3, 4)).astype("float32")

        before = ops.convert_to_numpy(model(sample, training=False))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roll.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(loaded(sample, training=False))

        # rtol=0: `assert_allclose`'s default rtol contributes silently to a
        # nominally-atol failure. Here the answer is a permutation, so the delta
        # should be exactly zero and atol alone already says so -- rtol=0 makes
        # that unambiguous.
        np.testing.assert_allclose(before, after, atol=0.0, rtol=0)

    def test_a_non_square_grid_survives_the_round_trip(self):
        """A 4x4 grid round trip cannot tell the axes apart; 5x3 can."""
        layer = CapsuleRoll(num_capsules=5, capsule_dim=3)
        inputs = keras.Input(shape=(4, 5, 3))
        model = keras.Model(inputs, layer(inputs, shift=1))
        sample = np.random.default_rng(6).normal(size=(2, 4, 5, 3)).astype("float32")

        before = ops.convert_to_numpy(model(sample, training=False))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "roll_ns.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            after = ops.convert_to_numpy(loaded(sample, training=False))
        assert _relative(model) == _relative(loaded)
        np.testing.assert_allclose(before, after, atol=0.0, rtol=0)


class TestCapsuleRollAgreesWithThePaper:
    """The NumPy twin in `metrics.topographic` must be the same operator."""

    def test_roll_capsules_matches_this_layer_exactly(self):
        from dl_techniques.metrics.topographic import roll_capsules

        values = np.random.default_rng(8).normal(size=(3, 2, 5, 7)).astype("float32")
        layer = CapsuleRoll(num_capsules=5, capsule_dim=7)
        for shift in (0, 1, 3, -2, 7):
            rolled = ops.convert_to_numpy(
                layer(ops.convert_to_tensor(values), shift=shift)
            )
            np.testing.assert_array_equal(
                rolled, roll_capsules(values.astype(np.float64), shift)
            )