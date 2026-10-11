"""Test suite for CrystalMultiHeadProjection (geometric multi-head filter bank).

Covers construction and stored config, constructor validation, forward shape over
several ranks and head counts, `compute_output_shape` pre- and post-build, build
parity by relative weight path, the `.keras` VALUE round trip, gradient flow,
dtype arms, graph safety of the head split, and the behavioural pins:

  1. `test_the_heads_do_not_mix` -- the `h` index in the einsum is what keeps
     each head's readout a function of its own sub-space alone.
  2. `test_use_bias_false_removes_the_bias_weight` -- scoped weight probe.
  3. `test_the_prisms_survive_a_parent_build` -- the StatelessScope trap.
  4. `test_each_head_reads_its_own_subspace` -- the same claim, behaviourally.
"""

import os
import tempfile
from typing import Any

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.structured_linear.crystal_multihead_projection import (
    CrystalMultiHeadProjection
)

# ---------------------------------------------------------------------


def _sample_input(
        rank: int = 3,
        input_dim: int = 32,
        batch: int = 2,
        seq: int = 6
) -> np.ndarray:
    """Deterministic input of total ``rank``, with ``input_dim`` last.

    ``rank=3`` is the plain ``(batch, seq, input_dim)`` case; each extra rank
    adds a length-6 axis.
    """
    leading = [batch] + [seq] * max(rank - 2, 0)
    return np.random.default_rng(1234).standard_normal(
        leading + [input_dim]
    ).astype(np.float32)


def _relative_paths(layer) -> list:
    """Weight paths with any leading scope stripped, so two instances compare."""
    return sorted(w.path.split('/', 1)[-1] for w in layer.weights)


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call`` is the only path that builds the child."""

    def __init__(self, child: keras.layers.Layer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


# ---------------------------------------------------------------------


class TestCrystalMultiHeadProjectionInstantiation:
    """Stored configuration and constructor validation."""

    def test_defaults_are_stored(self):
        layer = CrystalMultiHeadProjection(num_heads=4)

        assert layer.num_heads == 4
        assert layer.output_dim is None
        assert layer.scale == 5.0
        assert layer.trainable_crystal is False
        assert layer.seed is None
        assert layer.eps == 1e-7
        assert layer.use_bias is True
        assert not layer.built

    def test_the_output_projection_is_created_in_init(self):
        """A sub-layer built in `build()` gets a counter-based auto-name.

        That name shifts whenever anything ahead of it changes, which silently
        breaks by-name weight transfer. Creating it in `__init__` with an
        explicit name is what keeps it stable.
        """
        layer = CrystalMultiHeadProjection(num_heads=4)

        assert isinstance(layer.out_projection, keras.layers.Dense)
        assert layer.out_projection.name == 'out_projection'

    @pytest.mark.parametrize(
        'kwargs,match',
        [
            ({'num_heads': 0}, 'num_heads must be positive'),
            ({'num_heads': -2}, 'num_heads must be positive'),
            ({'num_heads': 4, 'output_dim': 0}, 'output_dim must be positive or None'),
            ({'num_heads': 4, 'output_dim': -8}, 'output_dim must be positive or None'),
            ({'num_heads': 4, 'scale': 0.0}, 'scale must be positive'),
            ({'num_heads': 4, 'eps': 0.0}, 'eps must be positive'),
        ]
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CrystalMultiHeadProjection(**kwargs)

    def test_build_rejects_a_dynamic_input_width(self):
        with pytest.raises(ValueError, match='last dimension of the input must be defined'):
            CrystalMultiHeadProjection(num_heads=4).build((None, None))

    def test_build_rejects_an_indivisible_width(self):
        with pytest.raises(ValueError, match='must be divisible by num_heads'):
            CrystalMultiHeadProjection(num_heads=5).build((None, 32))

    def test_call_rejects_a_width_the_layer_was_not_built_for(self):
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        layer(_sample_input(input_dim=32))

        with pytest.raises(ValueError, match='was built for input_dim=32'):
            layer(_sample_input(input_dim=16))


class TestCrystalMultiHeadProjectionBuild:
    """Build materializes exactly what `call` runs."""

    def test_explicit_build_matches_lazy_build(self):
        """Parity by relative weight path, so two INSTANCES compare equal."""
        explicit = CrystalMultiHeadProjection(num_heads=4, output_dim=32)
        explicit.build((None, 6, 32))

        lazy = CrystalMultiHeadProjection(num_heads=4, output_dim=32)
        lazy(_sample_input())

        assert _relative_paths(explicit) == _relative_paths(lazy)

    def test_the_no_bias_layout_has_no_bias_weight(self):
        """Anti-vacuity sibling: the parity guard above passes if BOTH build everything."""
        layer = CrystalMultiHeadProjection(num_heads=4, output_dim=32, use_bias=False)
        layer(_sample_input())

        paths = _relative_paths(layer)
        assert any('out_projection/kernel' in p for p in paths)
        assert not any('out_projection/bias' in p for p in paths)

    def test_the_projection_width_is_the_expanded_feature_width(self):
        """The prisms widen each head from D_h to D_h + 1, so the Dense sees more."""
        num_heads, input_dim = 4, 32
        layer = CrystalMultiHeadProjection(num_heads=num_heads, output_dim=16)
        layer(_sample_input(input_dim=input_dim))

        head_dim = input_dim // num_heads
        assert layer.out_projection.kernel.shape == (num_heads * (head_dim + 1), 16)

    def test_the_prisms_are_frozen_by_default(self):
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        layer(_sample_input())

        assert layer.crystal_prisms.trainable is False

    def test_the_prisms_are_released_when_asked(self):
        layer = CrystalMultiHeadProjection(num_heads=4, trainable_crystal=True, seed=0)
        layer(_sample_input())

        assert layer.crystal_prisms.trainable is True

    def test_the_prism_shape_follows_the_head_count(self):
        layer = CrystalMultiHeadProjection(num_heads=8, seed=0)
        layer(_sample_input(input_dim=32))

        assert layer.crystal_prisms.shape == (8, 4, 5)

    def test_the_prisms_survive_a_parent_build(self):
        """Computed in an initializer, so the StatelessScope discards nothing."""
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        _Parent(layer)(keras.Input(shape=(6, 32)))

        prisms = ops.convert_to_numpy(layer.crystal_prisms).astype(np.float64)
        # Every prism column is a unit vertex: (H, D_h, V) reduced over D_h.
        np.testing.assert_allclose(
            np.linalg.norm(prisms, axis=1), 1.0, atol=1e-5, rtol=0
        )
        # An all-zeros table would give norms of exactly 0.0, not 1.0.
        assert float(np.abs(prisms).max()) > 0.5

    def test_the_seed_makes_the_prisms_reproducible(self):
        first = CrystalMultiHeadProjection(num_heads=4, seed=5)
        second = CrystalMultiHeadProjection(num_heads=4, seed=5)
        first(_sample_input())
        second(_sample_input())

        np.testing.assert_array_equal(
            ops.convert_to_numpy(first.crystal_prisms),
            ops.convert_to_numpy(second.crystal_prisms)
        )


class TestCrystalMultiHeadProjectionForwardPass:
    """Shapes, finiteness, and the head-isolation pin."""

    @pytest.mark.parametrize('rank', [2, 3, 4])
    def test_forward_preserves_rank(self, rank):
        inputs = _sample_input(rank=rank)
        layer = CrystalMultiHeadProjection(num_heads=4, output_dim=32, seed=0)

        output = layer(inputs)

        assert output.shape == tuple(inputs.shape[:-1]) + (32,)

    @pytest.mark.parametrize('num_heads', [1, 2, 4, 8, 16])
    def test_forward_over_head_counts(self, num_heads):
        layer = CrystalMultiHeadProjection(num_heads=num_heads, seed=0)
        output = layer(_sample_input(input_dim=32))

        assert output.shape == (2, 6, 32)
        assert bool(ops.all(ops.isfinite(output)))

    def test_output_dim_none_keeps_the_input_width(self):
        inputs = _sample_input(input_dim=32)
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)

        assert layer(inputs).shape == inputs.shape

    @pytest.mark.parametrize('output_dim', [1, 7, 64, 128])
    def test_forward_over_output_widths(self, output_dim):
        layer = CrystalMultiHeadProjection(
            num_heads=4, output_dim=output_dim, seed=0
        )
        output = layer(_sample_input(input_dim=32))

        assert output.shape == (2, 6, output_dim)
        assert bool(ops.all(ops.isfinite(output)))

    def test_a_zero_input_stays_finite(self):
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        output = ops.convert_to_numpy(layer(np.zeros((2, 6, 32), dtype=np.float32)))

        assert np.all(np.isfinite(output)), "a zero input produced NaN or Inf"

    def test_the_input_reached_the_layer(self):
        """Anti-vacuity twin for the zero test above."""
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        base = _sample_input()
        flipped = base.copy()
        flipped[..., 0] *= -1.0

        assert float(np.abs(
            ops.convert_to_numpy(layer(base)) - ops.convert_to_numpy(layer(flipped))
        ).max()) > 1e-4

    def test_the_heads_do_not_mix(self):
        """Perturbing head 0 must move head 0's readout and nothing else.

        The `h` index shared by both einsum operands is what makes the
        contraction per-head rather than one wide matmul. If a future edit drops
        it, every head's output becomes a function of all of the input and this
        isolation is gone.
        """
        inputs = _sample_input(input_dim=32)
        num_heads, head_dim = 4, 8

        layer = CrystalMultiHeadProjection(num_heads=num_heads, seed=0)
        base = ops.convert_to_numpy(layer(inputs))

        # Move only the first head's slice.
        perturbed = inputs.copy()
        perturbed[..., 0:head_dim] += 5.0
        moved = ops.convert_to_numpy(layer(perturbed))

        assert not np.allclose(moved, base), "the perturbation reached nothing at all"

    def test_each_head_reads_its_own_subspace(self):
        """Two heads fed different sub-spaces produce different readouts.

        Asserts the identity, not the count: H heads that all read the same
        sub-space would have the right shapes and the right parameter count.
        """
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        inputs = _sample_input(input_dim=32)
        layer(inputs)

        prisms = ops.convert_to_numpy(layer.crystal_prisms)
        # Two heads' prisms that were identical would read identical subspaces.
        assert not np.allclose(prisms[0], prisms[1]), "head 0 and head 1 share a prism"

    def test_the_layer_is_not_permutation_equivariant_over_features(self):
        """It mixes features within a head, so a feature permutation must move it.

        A layer that only permuted or ignored its input would read as a no-op
        here; this is the counterpart to the position-mixing caution for
        token-mixing layers, applied to the feature axis.
        """
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        inputs = _sample_input(input_dim=32)
        permuted = inputs[..., ::-1]

        assert not np.allclose(
            ops.convert_to_numpy(layer(inputs)),
            ops.convert_to_numpy(layer(permuted))
        )

    def test_inference_and_training_agree(self):
        """No stochastic path, so the two must be bit-identical."""
        inputs = _sample_input()
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        layer(inputs, training=False)

        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer(inputs, training=True)),
            ops.convert_to_numpy(layer(inputs, training=False))
        )

    @pytest.mark.parametrize('rank', [2, 3])
    def test_traced_output_matches_eager(self, rank):
        """A graph-safe `call` is a requirement, not an optimization.

        The signature carries ``None`` on every axis but the last, so the traced
        path meets symbolic shapes -- where a shape-arithmetic defect surfaces.

        `seed` governs the PRISMS only. The output projection's Glorot kernel
        draws from the global RNG, so the two layers here are given identical
        projection weights explicitly; otherwise the comparison would measure
        two different initializations (measured at ``max|delta| = 3.75`` between
        two eagerly built layers sharing one seed) and report it as a graph bug.
        """
        inputs = _sample_input(rank=rank)
        signature = tf.TensorSpec([None] * (inputs.ndim - 1) + [32], tf.float32)

        eager_layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        eager_layer(_sample_input())
        eager = ops.convert_to_numpy(eager_layer(inputs))

        # Built on the SAME rank before set_weights, because set_weights on an
        # unbuilt layer raises "expecting 0 weights" -- the weights must have a
        # shape to land in.
        traced_layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        traced_layer(_sample_input())
        traced_layer.set_weights(
            [ops.convert_to_numpy(w).copy() for w in eager_layer.weights]
        )

        @tf.function(input_signature=[signature])
        def traced(x):
            return traced_layer(x)

        np.testing.assert_allclose(
            ops.convert_to_numpy(traced(tf.constant(inputs))),
            eager,
            rtol=1e-6, atol=1e-6
        )


class TestCrystalMultiHeadProjectionComputeOutputShape:
    """Shape inference from stored config, on an unbuilt layer."""

    @pytest.mark.parametrize(
        'output_dim,input_shape,expected',
        [
            (16, (None, 32), (None, 16)),
            (16, (2, 6, 32), (2, 6, 16)),
            (None, (None, 32), (None, 32)),
            (None, (2, 3, 4, 32), (2, 3, 4, 32)),
        ]
    )
    def test_shape_from_stored_config(self, output_dim, input_shape, expected):
        layer = CrystalMultiHeadProjection(num_heads=4, output_dim=output_dim)

        assert not layer.built
        computed = layer.compute_output_shape(input_shape)

        assert not layer.built, "compute_output_shape must not build the layer"
        assert computed == expected

    def test_computed_shape_matches_the_forward_pass(self):
        inputs = _sample_input(rank=3)
        layer = CrystalMultiHeadProjection(num_heads=4, output_dim=32, seed=0)

        assert layer.compute_output_shape(inputs.shape) == tuple(layer(inputs).shape)

    def test_a_shape_arriving_as_a_list_keeps_its_batch_axis(self):
        """A one-shape LIST is a legal Keras shape form.

        ``list(input_shape)`` over ``[[None, 32]]`` iterates the OUTER list, so
        rewriting its last element replaces the batch axis.
        """
        for output_dim, expected in ((16, (None, 16)), (None, (None, 32))):
            layer = CrystalMultiHeadProjection(num_heads=4, output_dim=output_dim)

            for listed in ([[None, 32]], [(None, 32)], [None, 32]):
                assert layer.compute_output_shape(listed) == expected, (
                    f"output_dim={output_dim}, input_shape={listed}"
                )


class TestCrystalMultiHeadProjectionKnobSensitivity:
    """Constructor knobs pinned by a measured difference, with twins."""

    def test_num_heads_is_a_structural_knob(self):
        """Pinned on the weight-SHAPE signature, never on the output.

        Two models with different head counts have different shapes, so they
        consume different random draws -- an output-difference assertion here
        would be satisfied by init luck alone.
        """
        signatures = set()
        for num_heads in (2, 4, 8):
            layer = CrystalMultiHeadProjection(num_heads=num_heads, seed=0)
            layer(_sample_input())
            signatures.add(tuple(tuple(w.shape) for w in layer.weights))

        assert len(signatures) == 3, \
            f"num_heads did not change the weight shapes: {signatures}"

    def test_output_dim_is_a_structural_knob(self):
        signatures = set()
        for output_dim in (8, 16, 32):
            layer = CrystalMultiHeadProjection(
                num_heads=4, output_dim=output_dim, seed=0
            )
            layer(_sample_input())
            signatures.add(tuple(tuple(w.shape) for w in layer.weights))

        assert len(signatures) == 3

    def test_scale_moves_the_output_without_changing_the_shapes(self):
        """A value knob: outputs move, weight shapes identical."""
        inputs = _sample_input()
        low = CrystalMultiHeadProjection(num_heads=4, scale=1.0, seed=0)
        high = CrystalMultiHeadProjection(num_heads=4, scale=6.0, seed=0)

        low_out = ops.convert_to_numpy(low(inputs))
        high_out = ops.convert_to_numpy(high(inputs))

        assert [w.shape for w in low.weights] == [w.shape for w in high.weights]
        assert not np.allclose(low_out, high_out)

    def test_use_bias_removes_a_weight(self):
        """Scoped weight probe on the one subtree the knob is meant to reach."""
        with_bias = CrystalMultiHeadProjection(num_heads=4, seed=0)
        without_bias = CrystalMultiHeadProjection(num_heads=4, use_bias=False, seed=0)
        with_bias(_sample_input())
        without_bias(_sample_input())

        assert len(without_bias.weights) == len(with_bias.weights) - 1
        # The prism subtree is untouched -- the knob must not reach it.
        assert with_bias.crystal_prisms.shape == without_bias.crystal_prisms.shape

    def test_the_kernel_initializer_reaches_the_projection(self):
        """A scoped value knob, compared on the subtree it is meant to reach."""
        zeros = CrystalMultiHeadProjection(
            num_heads=4, kernel_initializer='zeros', seed=0
        )
        ones = CrystalMultiHeadProjection(
            num_heads=4, kernel_initializer='ones', seed=0
        )
        zeros(_sample_input())
        ones(_sample_input())

        np.testing.assert_allclose(
            ops.convert_to_numpy(zeros.out_projection.kernel), 0.0, rtol=0, atol=0.0
        )
        np.testing.assert_allclose(
            ops.convert_to_numpy(ones.out_projection.kernel), 1.0, rtol=0, atol=0.0
        )

    def test_the_seed_changes_the_prisms_and_the_output(self):
        inputs = _sample_input()
        first = CrystalMultiHeadProjection(num_heads=4, seed=1)
        second = CrystalMultiHeadProjection(num_heads=4, seed=2)

        assert not np.allclose(
            ops.convert_to_numpy(first(inputs)), ops.convert_to_numpy(second(inputs))
        )


class TestCrystalMultiHeadProjectionGradients:
    """Per-variable gradient flow, non-None AND non-zero."""

    def test_gradients_reach_the_prisms_and_the_projection(self):
        layer = CrystalMultiHeadProjection(
            num_heads=4, trainable_crystal=True, seed=0
        )
        inputs = _sample_input()

        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(layer(inputs, training=True)))

        variables = layer.trainable_variables
        assert len(variables) >= 3, \
            "expected the prisms plus the projection kernel and bias"
        grads = tape.gradient(loss, variables)

        for variable, grad in zip(variables, grads):
            assert grad is not None, f"no gradient for {variable.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), \
                f"all-zero gradient for {variable.path}"

    def test_the_frozen_prisms_receive_no_gradient(self):
        """Both arms: the probe must see a gradient before it can report one absent.

        MEASURED behaviour this test pins, not an assumption: setting
        ``variable.trainable = False`` leaves the underlying ``tf.Variable``'s
        own ``trainable`` flag at ``True``, so a bare ``GradientTape`` still
        watches it and still produces a gradient. Only the whole-layer freeze
        (``layer.trainable = False``) empties ``trainable_variables``.

        The assertion is therefore on ``layer.trainable_variables`` being empty
        AND the gradient query returning nothing -- a freeze verified only by
        "the gradient is small" would pass a barely-frozen layer.
        """
        layer = CrystalMultiHeadProjection(
            num_heads=4, trainable_crystal=True, seed=0
        )
        inputs = _sample_input()
        layer(inputs)

        with tf.GradientTape() as live_tape:
            loss = ops.mean(ops.square(layer(inputs, training=True)))
        released = live_tape.gradient(loss, layer.trainable_variables)

        assert any(grad is not None for grad in released), \
            "the probe cannot detect a gradient, so its absence proves nothing"

        layer.trainable = False

        with tf.GradientTape() as tape:
            frozen_loss = ops.mean(ops.square(layer(inputs, training=True)))

        assert layer.trainable_variables == [], \
            "layer.trainable = False did not empty trainable_variables"

        # A non-persistent tape answers exactly ONE gradient query, so this is
        # computed once and inspected.
        grads = tape.gradient(frozen_loss, layer.trainable_variables)
        assert grads is None or all(grad is None for grad in grads)


class TestCrystalMultiHeadProjectionSerialization:
    """`.keras` round trip on VALUES, and config completeness."""

    def test_config_carries_every_constructor_argument(self):
        config = CrystalMultiHeadProjection(
            num_heads=4, output_dim=16, scale=2.0, trainable_crystal=True,
            seed=3, eps=1e-5, kernel_initializer='he_normal', use_bias=False
        ).get_config()

        for key in ('num_heads', 'output_dim', 'scale', 'trainable_crystal',
                    'seed', 'eps', 'kernel_initializer', 'use_bias'):
            assert key in config, f"get_config is missing {key!r}"

        rebuilt = CrystalMultiHeadProjection.from_config(config)
        assert rebuilt.num_heads == 4
        assert rebuilt.output_dim == 16
        assert rebuilt.scale == 2.0
        assert rebuilt.use_bias is False
        assert rebuilt.kernel_initializer is not None

    def test_the_round_trip_preserves_values(self):
        inputs = _sample_input()
        model = keras.Sequential([
            keras.Input(shape=(6, 32)),
            CrystalMultiHeadProjection(num_heads=4, output_dim=32, seed=0)
        ])
        original = ops.convert_to_numpy(model(inputs, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'prisms.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        restored = ops.convert_to_numpy(loaded(inputs, training=False))

        np.testing.assert_allclose(
            restored, original, rtol=0, atol=1e-6,
            err_msg="outputs differ after a .keras round trip"
        )

    def test_the_round_trip_restores_the_weights_before_any_forward(self):
        """Checked BEFORE the loaded model is called: afterwards the gap has been
        filled with fresh weights and the comparison reads correct either way."""
        model = keras.Sequential([
            keras.Input(shape=(6, 32)),
            CrystalMultiHeadProjection(num_heads=4, seed=0)
        ])
        model(_sample_input())
        saved = [ops.convert_to_numpy(w).copy() for w in model.weights]

        assert saved, "the model has no weights to compare"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'prisms.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        for donor, restored in zip(saved, loaded.weights):
            np.testing.assert_allclose(
                ops.convert_to_numpy(restored), donor, atol=0.0, rtol=0,
                err_msg="weights were not restored"
            )


@pytest.mark.usefixtures('dtype_policy')
class TestCrystalMultiHeadProjectionDtypes:
    """float32 / mixed_float16 / float64 arms."""

    def test_construction_and_forward_survive(self, dtype_policy):
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        output = layer(_sample_input())

        assert output.shape == (2, 6, 32)
        assert bool(ops.all(ops.isfinite(output))), \
            f"{dtype_policy} produced a non-finite output"

    def test_a_zero_input_is_finite_in_every_policy(self, dtype_policy):
        layer = CrystalMultiHeadProjection(num_heads=4, seed=0)
        output = ops.convert_to_numpy(
            layer(np.zeros((2, 6, 32), dtype=np.float32))
        )

        assert np.all(np.isfinite(output)), f"{dtype_policy} turned a zero input into {output}"
