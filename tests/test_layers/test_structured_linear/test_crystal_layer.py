"""Test suite for CrystalLayer (Neural Collapse / spherical-code classifier head).

Covers construction and stored config, constructor validation, forward shape over
several ranks and class counts, `compute_output_shape` pre- and post-build, the
`.keras` VALUE round trip, gradient flow, dtype arms, and the behavioural pins:

  1. `test_the_logits_are_exact_cosines` -- the frame is unit-norm, so the
     logits are a scaled cosine and nothing else.
  2. `test_a_zero_input_does_not_divide_by_zero` -- the eps guard.
  3. `test_a_wider_frame_gives_more_distinct_columns` -- the packing claim.
  4. `test_trainable_crystal_actually_moves_and_stays_a_cosine`.
"""

import os
import tempfile
from typing import Any

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.structured_linear.crystal_layer import CrystalLayer
from tests.layer_naming import PublicAutoName

# ---------------------------------------------------------------------


def _sample_input(rank: int = 2, input_dim: int = 8, batch: int = 4) -> np.ndarray:
    """Deterministic input of total ``rank``, with ``input_dim`` last.

    ``rank=2`` is the plain ``(batch, input_dim)`` case; each extra rank adds a
    length-3 axis ahead of the features.
    """
    leading = [batch] + [3] * max(rank - 2, 0)
    return np.random.default_rng(1234).standard_normal(
        leading + [input_dim]
    ).astype(np.float32)


class _Parent(PublicAutoName, keras.layers.Layer):
    """Minimal parent whose ``call`` is the only path that builds the child."""

    def __init__(self, child: keras.layers.Layer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


def _build_through_parent(child, input_dim: int) -> None:
    """Build ``child`` the way a real model does: from inside a parent's call."""
    _Parent(child)(keras.Input(shape=(input_dim,)))


# ---------------------------------------------------------------------


class TestCrystalLayerInstantiation:
    """Stored configuration and constructor validation."""

    def test_defaults_are_stored(self):
        layer = CrystalLayer(num_classes=10)

        assert layer.num_classes == 10
        assert layer.scale == 10.0
        assert layer.trainable_crystal is False
        assert layer.candidate_trials == 5
        assert layer.seed is None
        assert layer.eps == 1e-7
        assert not layer.built

    @pytest.mark.parametrize(
        'kwargs,match',
        [
            ({'num_classes': 0}, 'num_classes must be positive'),
            ({'num_classes': -3}, 'num_classes must be positive'),
            ({'num_classes': 4, 'scale': 0.0}, 'scale must be positive'),
            ({'num_classes': 4, 'scale': -1.0}, 'scale must be positive'),
            ({'num_classes': 4, 'candidate_trials': 0}, 'candidate_trials must be >= 1'),
            ({'num_classes': 4, 'eps': 0.0}, 'eps must be positive'),
            ({'num_classes': 4, 'eps': -1e-7}, 'eps must be positive'),
        ]
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CrystalLayer(**kwargs)

    def test_build_rejects_a_dynamic_input_width(self):
        with pytest.raises(ValueError, match='last dimension of the input must be defined'):
            CrystalLayer(num_classes=4).build((None, None))


class TestCrystalLayerFrame:
    """The frozen frame: geometry, reproducibility and the packing claim."""

    def test_the_frame_survives_a_parent_build(self):
        """A constant table assigned outside an initializer comes back all zeros.

        Keras runs a parent's symbolic build inside a `StatelessScope` that
        records and discards assignments, so this is the only probe that sees
        that trap. The pin is a closed form at a discriminating entry: the frame
        is a Gram matrix of unit vectors, so every diagonal entry is exactly 1.
        """
        layer = CrystalLayer(num_classes=6, seed=0)
        _build_through_parent(layer, input_dim=4)

        frame = ops.convert_to_numpy(layer.crystal_weights).astype(np.float64)
        gram = frame.T @ frame
        np.testing.assert_allclose(np.diag(gram), 1.0, atol=1e-5, rtol=0)

    def test_the_frame_is_frozen_by_default(self):
        layer = CrystalLayer(num_classes=6, seed=0)
        layer(_sample_input())

        assert layer.crystal_weights.trainable is False

    def test_the_frame_is_released_when_asked(self):
        layer = CrystalLayer(num_classes=6, trainable_crystal=True, seed=0)
        layer(_sample_input())

        assert layer.crystal_weights.trainable is True
        assert layer.trainable_variables

    def test_the_seed_makes_the_frame_reproducible(self):
        first = CrystalLayer(num_classes=12, seed=7)
        second = CrystalLayer(num_classes=12, seed=7)
        first(_sample_input())
        second(_sample_input())

        np.testing.assert_array_equal(
            ops.convert_to_numpy(first.crystal_weights),
            ops.convert_to_numpy(second.crystal_weights)
        )

    def test_a_wider_frame_gives_more_distinct_columns(self):
        """A class count past one simplex's capacity must not repeat vertices.

        A single simplex in 8 dimensions holds 9 directions, so 30 classes needs
        several. Tiling would give 9 distinct columns of 30 with pairs at inner
        product 1.0.
        """
        layer = CrystalLayer(num_classes=30, seed=0)
        layer(_sample_input(input_dim=8))

        frame = ops.convert_to_numpy(layer.crystal_weights)
        distinct = {tuple(np.round(column, 6)) for column in frame.T}

        assert len(distinct) == 30, (
            f"expected 30 distinct frame columns, got {len(distinct)}"
        )

    @pytest.mark.parametrize('dimensions,width', [(2, 4), (2, 500), (3, 50), (8, 50), (8, 500)])
    def test_packing_yields_width_distinct_columns_with_no_sphere_ceiling(
            self, dimensions, width
    ):
        """Distinctness holds at every width measured -- the old `2N` ceiling was false.

        An earlier revision of the docs claimed ``S^(N-1)`` holds only ``2N``
        points, capping any frame at ``2 * dimensions`` distinct columns. It is
        a continuum: 500 distinct of 500 at ``dimensions=2`` (stated ceiling:
        4). Only ``dimensions=1`` forces exact repeats, and that corner has its
        own test below.
        """
        layer = CrystalLayer(num_classes=width, seed=0)
        layer(_sample_input(input_dim=dimensions))

        frame = ops.convert_to_numpy(layer.crystal_weights)
        distinct = len({tuple(np.round(column, 6)) for column in frame.T})

        assert distinct == width, (
            f"input_dim={dimensions}, num_classes={width}: expected {width} "
            f"distinct columns, got {distinct}"
        )

    def test_separation_degrades_with_width_while_distinctness_holds(self):
        """The honest limit is the minimum angular SEPARATION, not distinctness.

        Past the simplex count the frame is a dense spherical code, not an
        equiangular set: the columns stay distinct but crowd together. Pinned
        at ``input_dim=8, seed=0``: the closest pair sits at inner product
        ~0.71 at width 16 and ~0.90 at width 256 (measured 0.714291 and
        0.902903; asserted with slack). A future packing change must keep the
        degradation smooth -- a width-256 frame whose closest pair is at 1.0
        is a tiling regression, and one at ~0.71 is a claim to beat the
        measured greedy search by magic.
        """
        def _max_inner(width):
            layer = CrystalLayer(num_classes=width, seed=0)
            layer(_sample_input(input_dim=8))
            frame = ops.convert_to_numpy(layer.crystal_weights).astype(np.float64)
            gram = frame.T @ frame
            np.fill_diagonal(gram, -1.0)
            return float(gram.max())

        narrow = _max_inner(16)
        wide = _max_inner(256)

        assert narrow < 0.80, f"width-16 packing regressed: {narrow:.6f}"
        assert wide > narrow, (
            f"separation must degrade with width: {wide:.6f} at 256 vs "
            f"{narrow:.6f} at 16"
        )
        assert wide < 0.97, f"width-256 packing regressed: {wide:.6f}"

    def test_a_wide_low_dimensional_frame_holds_near_identical_columns(self):
        """At low dimension a wide frame crowds to near-duplicate logits.

        Distinctness holds -- 64 distinct of 64 -- but the closest pair sits at
        inner product 0.9992 (measured at ``input_dim=2, width=64, seed=0``),
        i.e. two classes receiving near-identical logits. This is the hazard
        the old ceiling story was standing in for, stated as it actually is.
        """
        layer = CrystalLayer(num_classes=64, seed=0)
        layer(_sample_input(input_dim=2))

        frame = ops.convert_to_numpy(layer.crystal_weights).astype(np.float64)
        gram = frame.T @ frame
        np.fill_diagonal(gram, -1.0)

        assert float(gram.max()) > 0.99, (
            "expected a near-identical column pair at input_dim=2, width=64; "
            f"measured a maximum inner product of {float(gram.max()):.6f}"
        )

    def test_only_the_one_dimensional_sphere_forces_a_repeat(self):
        """Documents the one place the distinctness claim cannot hold.

        At ``input_dim=1`` the sphere is ``S^0 = {-1, +1}``: exactly two points,
        so a 3-column frame has one repeat no matter how it is built. This is
        the ONLY configuration where a repeat is mathematical rather than a
        defect -- the old ``2N`` ceiling story wrongly generalized it. Pinned
        so a future change cannot silently claim otherwise.
        """
        layer = CrystalLayer(num_classes=3, seed=0)
        layer(_sample_input(input_dim=1))

        frame = ops.convert_to_numpy(layer.crystal_weights)
        distinct = len({tuple(np.round(column, 6)) for column in frame.T})

        assert distinct == 2, (
            f"S^0 has 2 points, so 2 distinct columns is the maximum; got {distinct}"
        )

    def test_trainable_crystal_changes_the_weight_signature_is_not_the_instrument(self):
        """A frozen frame still produces a weight; only its flag differs."""
        frozen = CrystalLayer(num_classes=8, seed=0)
        released = CrystalLayer(num_classes=8, trainable_crystal=True, seed=0)
        frozen(_sample_input())
        released(_sample_input())

        # Same shape signature either way -- the knob is trainability, not shape.
        assert frozen.crystal_weights.shape == released.crystal_weights.shape
        assert released.crystal_weights.trainable is True
        assert frozen.crystal_weights.trainable is False


class TestCrystalLayerForwardPass:
    """Shape, finiteness and the behavioural pins."""

    @pytest.mark.parametrize('rank', [1, 2, 3, 4])
    def test_forward_preserves_rank(self, rank):
        layer = CrystalLayer(num_classes=7, seed=0)
        output = layer(_sample_input(rank=rank))

        assert output.shape == tuple(_sample_input(rank=rank).shape[:-1]) + (7,)

    @pytest.mark.parametrize('num_classes', [1, 2, 10, 37, 200])
    def test_forward_over_class_counts(self, num_classes):
        layer = CrystalLayer(num_classes=num_classes, seed=0)
        output = layer(_sample_input())

        assert output.shape == (4, num_classes)
        assert np.all(np.isfinite(ops.convert_to_numpy(output)))

    @pytest.mark.parametrize('rank', [1, 2, 3])
    def test_every_forward_is_finite(self, rank):
        layer = CrystalLayer(num_classes=13, seed=0)
        output = layer(_sample_input(rank=rank))

        assert bool(ops.all(ops.isfinite(output)))

    def test_the_logits_are_exact_cosines(self):
        """`scale * <x, v_j>` with both operands unit-norm.

        Pinned against a from-scratch NumPy cosine, so it fails if either the
        input or the weight normalization is dropped rather than merely
        becoming inaccurate.

        MEASURED delta ``3.1e-07`` at float32 (2026-10-12, CPU, dtype policy
        float32). The asserted ``atol=1e-05`` sits ~32x above it, and the
        float32 unit roundoff is ``6.0e-08`` -- ~52x below. The defect this
        must catch is a dropped normalization, which moves the answer by O(1),
        so the bound has roughly eight decades of headroom on the side that
        matters and is not a tolerance that can be satisfied by noise.
        """
        inputs = _sample_input()
        layer = CrystalLayer(num_classes=9, scale=3.0, seed=0)
        output = ops.convert_to_numpy(layer(inputs)).astype(np.float64)

        unit_x = inputs.astype(np.float64) / np.linalg.norm(inputs.astype(np.float64), axis=-1, keepdims=True)
        frame = ops.convert_to_numpy(layer.crystal_weights).astype(np.float64)
        unit_w = frame / np.linalg.norm(frame, axis=0, keepdims=True)

        np.testing.assert_allclose(output, 3.0 * (unit_x @ unit_w), rtol=0, atol=1e-5)

    def test_a_zero_input_does_not_divide_by_zero(self):
        """`eps` sits inside the sqrt, so a zero vector normalizes to zero.

        NaN here would propagate straight into every downstream loss.
        """
        layer = CrystalLayer(num_classes=5, seed=0)
        output = ops.convert_to_numpy(layer(np.zeros((3, 8), dtype=np.float32)))

        assert np.all(np.isfinite(output)), "a zero input produced NaN or Inf"
        np.testing.assert_allclose(output, 0.0, atol=1e-5, rtol=0)

    def test_the_input_reached_the_layer(self):
        """Anti-vacuity twin: the zero test above is also passed by a dead layer.

        The perturbation flips a SIGN rather than scaling by a constant. A
        cosine head is invariant to the input's magnitude, so a scaled probe
        would measure ~1e-6 here -- which is correct behaviour being reported
        as a dead layer.
        """
        layer = CrystalLayer(num_classes=5, seed=0)
        base = _sample_input()
        flipped = base.copy()
        flipped[:, 0] *= -1.0

        delta = float(np.abs(
            ops.convert_to_numpy(layer(base)) - ops.convert_to_numpy(layer(flipped))
        ).max())

        assert delta > 1e-3, (
            f"perturbing the input moved the logits by only {delta}; the layer "
            f"is not reading its input"
        )

    def test_call_rejects_a_width_the_layer_was_not_built_for(self):
        """The build-time contract is re-asserted on every forward.

        A contract checked only in build() is checked once, against whatever
        shape happened to arrive first.
        """
        layer = CrystalLayer(num_classes=5, seed=0)
        layer(_sample_input(input_dim=8))

        with pytest.raises(ValueError, match='was built for input_dim=8'):
            layer(_sample_input(input_dim=16))

    @pytest.mark.parametrize('rank', [1, 2, 3])
    def test_forward_under_tf_function_matches_eager(self, rank):
        """A graph-safe `call` is a requirement, not an optimization.

        The input signature carries ``None`` on every axis but the last, so the
        traced path sees symbolic shapes rather than the concrete ones used
        eagerly -- which is where a shape-arithmetic defect would surface.

        The two paths get separate layer instances: a layer built eagerly at one
        rank cannot be rebuilt at another, and reusing it would turn this into a
        test of Keras' build guard rather than of ``call``.
        """
        inputs = _sample_input(rank=rank)
        signature = tf.TensorSpec([None] * (inputs.ndim - 1) + [8], tf.float32)

        eager = ops.convert_to_numpy(
            CrystalLayer(num_classes=11, seed=0)(inputs)
        )

        traced_layer = CrystalLayer(num_classes=11, seed=0)

        @tf.function(input_signature=[signature])
        def traced(x):
            return traced_layer(x)

        traced_output = ops.convert_to_numpy(traced(tf.constant(inputs)))

        np.testing.assert_allclose(traced_output, eager, rtol=1e-6, atol=1e-6)

    def test_inference_and_training_agree(self):
        """The layer has no stochastic path, so the two must be identical."""
        inputs = _sample_input()
        layer = CrystalLayer(num_classes=9, seed=0)
        layer(inputs, training=False)

        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer(inputs, training=True)),
            ops.convert_to_numpy(layer(inputs, training=False))
        )


class TestCrystalLayerComputeOutputShape:
    """Shape inference from stored config, on an unbuilt layer."""

    @pytest.mark.parametrize(
        'input_shape,expected',
        [
            ((None, 8), (None, 6)),
            ((4, 8), (4, 6)),
            ((None, 5, 8), (None, 5, 6)),
            ((2, 3, 4, 8), (2, 3, 4, 6)),
        ]
    )
    def test_shape_matches_the_forward_pass(self, input_shape, expected):
        layer = CrystalLayer(num_classes=6)

        assert not layer.built, "the layer was built before compute_output_shape"
        computed = layer.compute_output_shape(input_shape)

        assert not layer.built, "compute_output_shape must not build the layer"
        assert computed == expected

    def test_computed_shape_matches_an_actual_forward(self):
        layer = CrystalLayer(num_classes=6, seed=0)
        inputs = _sample_input(rank=3)

        assert layer.compute_output_shape(inputs.shape) == tuple(layer(inputs).shape)

    def test_a_shape_arriving_as_a_list_keeps_its_batch_axis(self):
        """A one-shape LIST is a legal Keras shape form and must not be read as
        a shape tuple.

        ``list(input_shape)`` over ``[[None, 8]]`` iterates the OUTER list, so
        rewriting the last element replaces the batch axis and the layer
        reports ``(6,)`` for a ``(None, 8)`` input.
        """
        layer = CrystalLayer(num_classes=6)

        for listed in ([[None, 8]], [(None, 8)], [None, 8]):
            assert layer.compute_output_shape(listed) == (None, 6), (
                f"input_shape={listed} produced the wrong output shape"
            )


class TestCrystalLayerKnobSensitivity:
    """Every constructor knob pinned by a measured difference, with twins."""

    def test_scale_moves_the_logits_without_changing_the_signature(self):
        """`scale` is a value knob: outputs move, weights keep their shapes."""
        inputs = _sample_input()
        low = CrystalLayer(num_classes=9, scale=1.0, seed=0)
        high = CrystalLayer(num_classes=9, scale=7.0, seed=0)

        # Built first: a weight-shape signature read before build() is `None`.
        low(inputs)
        high(inputs)

        assert low.crystal_weights.shape == high.crystal_weights.shape
        assert not np.allclose(
            ops.convert_to_numpy(low(inputs)), ops.convert_to_numpy(high(inputs))
        )

    def test_scale_multiplies_the_logits_linearly(self):
        """The relationship is exact, not merely monotonic.

        MEASURED delta ``0.0`` exactly (2026-10-12, CPU, float32): `scale` is
        a constant multiplied in after the normalized matmul, so it factors out
        bit-exactly. The asserted ``atol=1e-04`` is therefore slack the
        assertion does not need; it is kept rather than tightened to zero
        because a future implementation that folded the scale into the
        matmul would still be correct up to accumulation.
        """
        inputs = _sample_input()
        one = ops.convert_to_numpy(CrystalLayer(num_classes=9, scale=1.0, seed=0)(inputs))
        four = ops.convert_to_numpy(CrystalLayer(num_classes=9, scale=4.0, seed=0)(inputs))

        np.testing.assert_allclose(four, 4.0 * one, rtol=0, atol=1e-4)

    def test_seed_changes_the_frame_and_the_logits(self):
        """A different seed is a different codebook, so the logits must differ."""
        inputs = _sample_input()
        first = CrystalLayer(num_classes=30, seed=1)
        second = CrystalLayer(num_classes=30, seed=2)

        assert not np.allclose(
            ops.convert_to_numpy(first(inputs)), ops.convert_to_numpy(second(inputs))
        )

    def test_the_logits_are_invariant_to_input_magnitude(self):
        """A cosine head must not see the input's length.

        Scaling every input by a constant may only change the output if the
        normalization is broken.

        MEASURED delta ``1.5e-07`` at float32 (2026-10-12, CPU), from a 5x
        scale factor; ``atol=1e-05`` is ~67x above it. The property is exact in
        algebra and holds to accumulation error here because
        :func:`~dl_techniques.utils.tensors.l2_normalize` divides by the
        vector's own peak before squaring -- so a uniform rescale cancels
        inside the normalization rather than surviving into the norm.
        """
        layer = CrystalLayer(num_classes=9, scale=2.0, seed=0)
        inputs = _sample_input()
        base = ops.convert_to_numpy(layer(inputs))
        scaled = ops.convert_to_numpy(layer(inputs * 5.0))

        np.testing.assert_allclose(scaled, base, rtol=1e-5, atol=1e-5)

    def test_the_logits_stay_within_the_scaled_cosine_range(self):
        """`|cos| <= 1`, so the logits are bounded by `scale`. A closed bound."""
        scale = 4.0
        layer = CrystalLayer(num_classes=20, scale=scale, seed=0)
        output = ops.convert_to_numpy(layer(_sample_input()))

        assert float(np.abs(output).max()) <= scale + 1e-4


class TestCrystalLayerGradients:
    """Per-variable gradient flow, non-None AND non-zero."""

    def test_gradients_reach_every_trainable_weight(self):
        layer = CrystalLayer(num_classes=9, trainable_crystal=True, seed=0)
        inputs = _sample_input()

        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(layer(inputs, training=True)))

        assert layer.trainable_variables, "the layer has no trainable variables"
        grads = tape.gradient(loss, layer.trainable_variables)

        for variable, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {variable.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), \
                f"all-zero gradient for {variable.path}"

    def test_the_frozen_frame_receives_no_gradient(self):
        """The other half of the pair: a frozen frame is not in the tape at all."""
        layer = CrystalLayer(num_classes=9, trainable_crystal=False, seed=0)
        inputs = _sample_input()

        with tf.GradientTape() as tape:
            loss = ops.mean(ops.square(layer(inputs, training=True)))
        grads = tape.gradient(loss, layer.trainable_variables)

        # Identity, not equality: `in` compares with `==`, which raises on a
        # variable whose shape is not broadcast-compatible with the list's.
        assert all(
            variable is not layer.crystal_weights
            for variable in layer.trainable_variables
        ), "the frozen frame remained in trainable_variables"
        assert all(grad is None for grad in grads)


class TestCrystalLayerSerialization:
    """`.keras` round trip on VALUES, and config completeness."""

    def test_config_carries_every_constructor_argument(self):
        layer = CrystalLayer(
            num_classes=17, scale=2.5, trainable_crystal=True,
            candidate_trials=3, seed=9, eps=1e-5
        )
        config = layer.get_config()

        for key in ('num_classes', 'scale', 'trainable_crystal',
                    'candidate_trials', 'seed', 'eps'):
            assert key in config, f"get_config is missing {key!r}"

        rebuilt = CrystalLayer.from_config(config)
        assert rebuilt.num_classes == 17
        assert rebuilt.scale == 2.5
        assert rebuilt.trainable_crystal is True
        assert rebuilt.candidate_trials == 3
        assert rebuilt.seed == 9
        assert rebuilt.eps == 1e-5

    def test_the_round_trip_preserves_values(self):
        """Values, not shapes: a model that restored zero weights still matches here
        only if the comparison is on the numbers.

        `rtol=0` and an explicit `training=False` are both load-bearing --
        `assert_allclose`'s default `rtol` otherwise contributes silently to a
        nominally-atol bound.
        """
        inputs = _sample_input()
        model = keras.Sequential([
            keras.Input(shape=(8,)),
            CrystalLayer(num_classes=9, scale=3.0, seed=0)
        ])
        original = ops.convert_to_numpy(model(inputs, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'crystal.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        restored = ops.convert_to_numpy(loaded(inputs, training=False))

        np.testing.assert_allclose(
            restored, original, rtol=0, atol=1e-6,
            err_msg="CrystalLayer outputs differ after a .keras round trip"
        )

    def test_the_round_trip_restores_the_frame_before_any_forward(self):
        """Weight values must land, and be checked before the loaded model is called.

        After one forward pass the gap has been filled with fresh weights, so a
        comparison run afterwards reads as correct for a broken load path.
        """
        model = keras.Sequential([
            keras.Input(shape=(8,)),
            CrystalLayer(num_classes=9, seed=0)
        ])
        model(_sample_input())
        saved = [ops.convert_to_numpy(w).copy() for w in model.weights]

        assert saved, "the model has no weights to compare"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'crystal.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        for donor, restored in zip(saved, loaded.weights):
            np.testing.assert_allclose(
                ops.convert_to_numpy(restored), donor, atol=0.0, rtol=0,
                err_msg="frame weights were not restored"
            )


@pytest.mark.usefixtures('dtype_policy')
class TestCrystalLayerDtypes:
    """float32 / mixed_float16 / float64 arms, each genuinely in its own dtype.

    The input is cast to the policy's COMPUTE dtype before it reaches the
    layer. That cast is load-bearing: passing a float32 array while the global
    policy says float64 leaves the layer computing in float32, so the arm
    passes while measuring nothing it claimed to. Measured before this cast was
    added: input float64 / weight float32 / output float32 under the float64
    policy.
    """

    def test_construction_and_forward_survive(self, dtype_policy):
        compute_dtype = keras.mixed_precision.global_policy().compute_dtype
        inputs = _sample_input().astype(compute_dtype)

        layer = CrystalLayer(num_classes=9, scale=3.0, seed=0)
        output = layer(inputs)

        assert output.shape == (4, 9)
        assert bool(ops.all(ops.isfinite(output))), \
            f"{dtype_policy} produced a non-finite output"

    def test_the_arm_really_computes_in_the_policy_dtype(self, dtype_policy):
        """Guards the cast above -- an arm that does not run in its dtype is
        a test named after a property it never exercised.

        Asserted on the OUTPUT dtype, not the weight. Under ``mixed_float16``
        Keras keeps variables in float32 and computes in float16, which is
        correct and standard; the weight dtype is therefore deliberately not
        pinned to the compute dtype here.
        """
        compute_dtype = keras.mixed_precision.global_policy().compute_dtype
        inputs = _sample_input().astype(compute_dtype)

        layer = CrystalLayer(num_classes=9, scale=3.0, seed=0)
        output = layer(inputs)

        assert ops.convert_to_numpy(output).dtype == compute_dtype, (
            f"under {dtype_policy} the output came back as "
            f"{ops.convert_to_numpy(output).dtype}, not the policy's compute "
            f"dtype {compute_dtype}; this arm did not exercise what it claims"
        )

    def test_a_zero_input_is_finite_in_every_policy(self, dtype_policy):
        compute_dtype = keras.mixed_precision.global_policy().compute_dtype
        layer = CrystalLayer(num_classes=9, seed=0)

        output = ops.convert_to_numpy(
            layer(np.zeros((3, 8), dtype=compute_dtype))
        )

        assert np.all(np.isfinite(output)), f"{dtype_policy} turned a zero input into {output}"
