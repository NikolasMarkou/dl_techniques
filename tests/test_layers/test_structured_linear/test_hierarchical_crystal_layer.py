"""Test suite for HierarchicalCrystalLayer (two-tier spherical-code classifier).

Covers construction and stored config, constructor validation, forward shape and
finiteness over several ranks and macro/micro splits, **both** branches of
``compute_output_shape``, the `.keras` VALUE round trip, gradient flow, dtype
arms, graph safety of the (..., M, K) regrouping, and the behavioural pins:

  1. `test_the_joint_probabilities_sum_to_one` -- the Bayesian identity.
  2. `test_the_joint_probabilities_are_the_product_of_the_two_softmaxes` --
     checked against a from-scratch NumPy reference, so a joint that happened
     to be renormalized would still fail.
  3. `test_the_micro_frame_is_packed_not_tiled` -- the wider-than-one-simplex
     case, which is the edge the layer exists to survive.
  4. `test_cluster_spread_collapses_and_fans_the_leaves`.
"""

import os
import tempfile
from typing import Any

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import ops

from dl_techniques.layers.structured_linear.hierarchical_crystal_layer import (
    HierarchicalCrystalLayer
)

# ---------------------------------------------------------------------

#: Measured maximum deviation of the joint probabilities from 1.0, over ranks
#: 2 through 4 and a float32 accumulation of the product of two softmaxes. The
#: identity is exact in exact arithmetic; this is the float32 floor.
BAYES_ATOL = 1e-5


def _sample_input(rank: int = 2, input_dim: int = 4, batch: int = 4) -> np.ndarray:
    """Deterministic input of total ``rank``, with ``input_dim`` last."""
    leading = [batch] + [3] * max(rank - 2, 0)
    return np.random.default_rng(1234).standard_normal(
        leading + [input_dim]
    ).astype(np.float32)


def _layer(macro: int = 3, micro: int = 12, **kwargs) -> HierarchicalCrystalLayer:
    return HierarchicalCrystalLayer(
        num_macro_classes=macro, num_micro_per_macro=micro, **kwargs
    )


class _Parent(keras.layers.Layer):
    """Minimal parent whose ``call`` is the only path that builds the child."""

    def __init__(self, child: keras.layers.Layer, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.child = child

    def call(self, inputs):
        return self.child(inputs)


# ---------------------------------------------------------------------


class TestHierarchicalCrystalLayerInstantiation:
    """Stored configuration and constructor validation."""

    def test_defaults_are_stored_and_total_is_derived(self):
        layer = _layer(3, 12)

        assert layer.num_macro == 3
        assert layer.num_micro == 12
        assert layer.total_classes == 36
        assert layer.cluster_spread == 0.25
        assert layer.scale == 10.0
        assert layer.return_hierarchy is False
        assert layer.trainable_crystal is False
        assert layer.seed is None
        assert layer.eps == 1e-7
        assert not layer.built

    @pytest.mark.parametrize(
        'kwargs,match',
        [
            ({'num_macro_classes': 0, 'num_micro_per_macro': 4},
             'num_macro_classes must be positive'),
            ({'num_macro_classes': 4, 'num_micro_per_macro': 0},
             'num_micro_per_macro must be positive'),
            ({'num_macro_classes': 4, 'num_micro_per_macro': -2},
             'num_micro_per_macro must be positive'),
            ({'num_macro_classes': 4, 'num_micro_per_macro': 4, 'cluster_spread': -0.1},
             'cluster_spread must be non-negative'),
            ({'num_macro_classes': 4, 'num_micro_per_macro': 4, 'scale': 0.0},
             'scale must be positive'),
            ({'num_macro_classes': 4, 'num_micro_per_macro': 4, 'eps': 0.0},
             'eps must be positive'),
        ]
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            HierarchicalCrystalLayer(**kwargs)

    def test_a_zero_spread_is_allowed(self):
        assert _layer(3, 4, cluster_spread=0.0).cluster_spread == 0.0

    def test_build_rejects_a_dynamic_input_width(self):
        with pytest.raises(ValueError, match='last dimension of the input must be defined'):
            _layer(3, 4).build((None, None))

    def test_call_rejects_a_width_the_layer_was_not_built_for(self):
        layer = _layer(3, 4, seed=0)
        layer(_sample_input(input_dim=4))

        with pytest.raises(ValueError, match='was built for input_dim=4'):
            layer(_sample_input(input_dim=8))


class TestHierarchicalCrystalLayerFrames:
    """Both frames survive a parent build and are unit-norm."""

    def test_both_frames_survive_a_parent_build(self):
        """Computed inside an initializer, so nothing is discarded by the scope."""
        layer = _layer(3, 6, seed=0)
        _Parent(layer)(keras.Input(shape=(4,)))

        for variable in (layer.macro_weights, layer.micro_weights):
            values = ops.convert_to_numpy(variable).astype(np.float64)
            np.testing.assert_allclose(
                np.linalg.norm(values, axis=0), 1.0, atol=1e-5, rtol=0,
                err_msg=f"{variable.path} is not unit-norm after a parent build"
            )

    def test_weight_shapes_match_the_configuration(self):
        layer = _layer(3, 12, seed=0)
        layer(_sample_input(input_dim=4))

        assert layer.macro_weights.shape == (4, 3)
        assert layer.micro_weights.shape == (4, 36)

    def test_both_frames_are_frozen_by_default(self):
        layer = _layer(3, 6, seed=0)
        layer(_sample_input(input_dim=4))

        assert layer.macro_weights.trainable is False
        assert layer.micro_weights.trainable is False

    def test_both_frames_are_released_when_asked(self):
        layer = _layer(3, 6, trainable_crystal=True, seed=0)
        layer(_sample_input(input_dim=4))

        assert layer.macro_weights.trainable is True
        assert layer.micro_weights.trainable is True

    def test_the_micro_frame_is_packed_not_tiled(self):
        """12 leaves in 4 dimensions exceeds the 5 one simplex can supply.

        A simplex in 4 dimensions holds 5 directions, so 12 leaves per group
        needs packing. Tiling would give 5 distinct columns with pairs at inner
        product 1.0 -- every leaf in a group receiving an identical logit.
        """
        layer = _layer(3, 12, seed=0)
        layer(_sample_input(input_dim=4))

        micro = ops.convert_to_numpy(layer.micro_weights)
        distinct = {tuple(np.round(column, 6)) for column in micro.T}

        assert len(distinct) == 36, (
            f"expected 36 distinct leaf columns across 3 groups, got {len(distinct)}"
        )

    def test_groups_do_not_share_one_displacement_frame(self):
        """Two groups must not receive identical bundles.

        With a shared frame the micro columns would repeat with a period equal to
        the group size -- the same defect at one axis removed.
        """
        layer = _layer(3, 6, seed=0)
        layer(_sample_input(input_dim=4))

        micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)
        first_group = micro[:, 0:6]
        second_group = micro[:, 6:12]

        assert not np.allclose(first_group, second_group)

    def test_the_seed_makes_both_frames_reproducible(self):
        first, second = _layer(3, 12, seed=4), _layer(3, 12, seed=4)
        first(_sample_input(input_dim=4))
        second(_sample_input(input_dim=4))

        np.testing.assert_array_equal(
            ops.convert_to_numpy(first.macro_weights),
            ops.convert_to_numpy(second.macro_weights)
        )
        np.testing.assert_array_equal(
            ops.convert_to_numpy(first.micro_weights),
            ops.convert_to_numpy(second.micro_weights)
        )

    @pytest.mark.parametrize('seed', [None, 0, 7])
    def test_the_leaf_bundles_sit_on_the_STORED_centroids(self, seed):
        """The leaf geometry must describe the classifier that actually runs.

        This is the guard for a defect that was present and entirely invisible
        at an explicit seed. ``_micro_initializer`` rebuilt the macro frame
        instead of reusing the one ``macro_weights`` received, so the leaf
        bundles were centred on a *different* set of centroids whenever the two
        draws disagreed -- and with ``seed=None`` the draws are independent by
        construction, so they always did.

        Measured before the fix: leaf-to-stored-centroid cosine of 0.31 at
        ``seed=None`` against 0.98 at ``seed=0``, where the two independent
        draws happened to coincide. A test that only used an explicit seed would
        have passed throughout.

        The bound is per group, leaf block against ITS OWN centroid: the bundles
        are tight around their centre by construction (``cluster_spread=0.25``
        puts the worst leaf cosine near 0.99), so a loose block means the two
        frames disagree, not that the aperture is wide.
        """
        layer = _layer(3, 5, cluster_spread=0.25, seed=seed)
        layer(_sample_input(input_dim=4))

        macro = ops.convert_to_numpy(layer.macro_weights).astype(np.float64)
        micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)

        for group in range(3):
            centroid = macro[:, group]
            leaves = micro[:, group * 5:(group + 1) * 5]
            worst = float(np.abs(leaves.T @ centroid).max())

            assert worst > 0.9, (
                f"seed={seed!r}: group {group}'s leaves sit at cosine {worst:.4f} "
                f"from the centroid actually stored in macro_weights, so the "
                f"leaf frame was built around a different centroid"
            )

    def test_a_rebuilt_layer_reuses_its_own_seed_root(self):
        """`SeedSequence.spawn` is stateful; a second spawn must not shift it.

        Every initializer asking the root for its own children would advance the
        counter and hand back a different stream each time.
        """
        layer = _layer(3, 5, cluster_spread=0.25, seed=0)
        layer(_sample_input(input_dim=4))
        first = ops.convert_to_numpy(layer.micro_weights).copy()

        # Force a second build of the same instance.
        layer = _layer(3, 5, cluster_spread=0.25, seed=0)
        layer(_sample_input(input_dim=4))
        layer(_sample_input(input_dim=4))

        np.testing.assert_array_equal(
            ops.convert_to_numpy(layer.micro_weights), first,
            err_msg="a second forward pass changed the leaf frame's seed root"
        )


class TestHierarchicalCrystalLayerForwardPass:
    """Shapes, finiteness, and the two output branches."""

    @pytest.mark.parametrize('rank', [1, 2, 3, 4])
    def test_flat_mode_preserves_rank(self, rank):
        inputs = _sample_input(rank=rank)
        layer = _layer(3, 5, seed=0)

        output = layer(inputs)

        assert output.shape == tuple(inputs.shape[:-1]) + (15,)

    @pytest.mark.parametrize('macro,micro', [(1, 1), (2, 3), (3, 12), (8, 7)])
    def test_flat_mode_over_class_trees(self, macro, micro):
        layer = _layer(macro, micro, seed=0)
        output = layer(_sample_input(input_dim=4))

        assert output.shape == (4, macro * micro)
        assert bool(ops.all(ops.isfinite(output)))

    @pytest.mark.parametrize('rank', [1, 2, 3, 4])
    def test_hierarchy_mode_reports_all_three_outputs(self, rank):
        inputs = _sample_input(rank=rank)
        layer = _layer(3, 5, return_hierarchy=True, seed=0)

        outputs = layer(inputs)
        base = tuple(inputs.shape[:-1])

        assert set(outputs) == {'macro_logits', 'micro_logits', 'joint_probabilities'}
        assert outputs['macro_logits'].shape == base + (3,)
        assert outputs['micro_logits'].shape == base + (15,)
        assert outputs['joint_probabilities'].shape == base + (15,)

    def test_the_joint_probabilities_sum_to_one(self):
        """The Bayesian identity the layer is built to satisfy."""
        layer = _layer(3, 12, return_hierarchy=True, seed=0)
        joint = ops.convert_to_numpy(layer(_sample_input(input_dim=4))['joint_probabilities'])

        np.testing.assert_allclose(
            joint.sum(axis=-1), 1.0, rtol=0, atol=BAYES_ATOL,
            err_msg="joint leaf probabilities do not sum to 1"
        )

    @pytest.mark.parametrize('rank', [1, 2, 3, 4])
    def test_the_bayes_identity_holds_at_every_rank(self, rank):
        """The regrouping is the fragile part, so it is checked per rank."""
        layer = _layer(3, 12, return_hierarchy=True, seed=0)
        joint = ops.convert_to_numpy(
            layer(_sample_input(rank=rank, input_dim=4))['joint_probabilities']
        )

        np.testing.assert_allclose(
            joint.sum(axis=-1), 1.0, rtol=0, atol=BAYES_ATOL
        )

    def test_the_joint_probabilities_are_the_product_of_the_two_softmaxes(self):
        """Checked against a from-scratch NumPy reference.

        A joint that was merely renormalized to sum to one would pass the sum
        test above; only the product law distinguishes it.
        """
        inputs = _sample_input(input_dim=4).astype(np.float64)
        layer = _layer(3, 5, scale=7.0, return_hierarchy=True, seed=0)
        outputs = {
            key: ops.convert_to_numpy(value).astype(np.float64)
            for key, value in layer(inputs.astype(np.float32)).items()
        }

        macro = ops.convert_to_numpy(layer.macro_weights).astype(np.float64)
        micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)
        unit_x = inputs / np.linalg.norm(inputs, axis=-1, keepdims=True)
        unit_macro = macro / np.linalg.norm(macro, axis=0, keepdims=True)
        unit_micro = micro / np.linalg.norm(micro, axis=0, keepdims=True)

        def _softmax(z, axis=-1):
            shifted = z - z.max(axis=axis, keepdims=True)
            exponentiated = np.exp(shifted)
            return exponentiated / exponentiated.sum(axis=axis, keepdims=True)

        macro_probs = _softmax(7.0 * (unit_x @ unit_macro))
        micro_logits = 7.0 * (unit_x @ unit_micro)
        grouped = micro_logits.reshape(*micro_logits.shape[:-1], 3, 5)
        conditional = _softmax(grouped, axis=-1)

        np.testing.assert_allclose(
            outputs['macro_logits'], 7.0 * (unit_x @ unit_macro), rtol=0, atol=1e-4
        )
        np.testing.assert_allclose(
            outputs['joint_probabilities'],
            (macro_probs[..., :, None] * conditional).reshape(*micro_logits.shape),
            rtol=0, atol=1e-5
        )

    def test_the_joint_is_a_probability_distribution(self):
        """Non-negative and bounded by one, which a renormalized logit would not be."""
        layer = _layer(3, 12, return_hierarchy=True, seed=0)
        joint = ops.convert_to_numpy(layer(_sample_input(input_dim=4))['joint_probabilities'])

        assert joint.min() >= 0.0
        assert joint.max() <= 1.0

    def test_a_zero_input_stays_finite(self):
        layer = _layer(3, 5, return_hierarchy=True, seed=0)
        outputs = layer(np.zeros((2, 4), dtype=np.float32))

        for key, value in outputs.items():
            values = ops.convert_to_numpy(value)
            assert np.all(np.isfinite(values)), f"{key} went non-finite on a zero input"

    def test_inference_and_training_agree(self):
        inputs = _sample_input(input_dim=4)
        layer = _layer(3, 5, return_hierarchy=True, seed=0)
        layer(inputs, training=False)

        for key in ('macro_logits', 'micro_logits', 'joint_probabilities'):
            np.testing.assert_array_equal(
                ops.convert_to_numpy(layer(inputs, training=True)[key]),
                ops.convert_to_numpy(layer(inputs, training=False)[key])
            )

    def test_the_leaf_logits_do_not_depend_on_the_macro_mode(self):
        """`return_hierarchy` selects the output type, not the leaf computation."""
        inputs = _sample_input(input_dim=4)
        flat = _layer(3, 5, seed=0)
        nested = _layer(3, 5, return_hierarchy=True, seed=0)

        np.testing.assert_allclose(
            ops.convert_to_numpy(nested(inputs)['micro_logits']),
            ops.convert_to_numpy(flat(inputs)),
            rtol=0, atol=1e-6
        )


class TestClusterSpread:
    """The aperture knob: 0 collapses the bundle, larger values fan it out."""

    def test_zero_spread_collapses_every_leaf_onto_its_centroid(self):
        layer = _layer(3, 5, cluster_spread=0.0, seed=0)
        layer(_sample_input(input_dim=4))

        macro = ops.convert_to_numpy(layer.macro_weights)
        micro = ops.convert_to_numpy(layer.micro_weights)

        for group in range(3):
            centroid = macro[:, group]
            group_leaves = micro[:, group * 5:(group + 1) * 5]
            for leaf in group_leaves.T:
                np.testing.assert_allclose(
                    leaf, centroid, atol=1e-5, rtol=0,
                    err_msg="cluster_spread=0 must place every leaf on its centroid"
                )

    def test_a_positive_spread_pushes_the_leaves_off_the_centroid(self):
        """The other half of the pair: a collapsed bundle is also what 0 gives.

        Measured per group, leaf-block against ITS OWN centroid. A block is
        matched to group ``g`` at columns ``[g * micro, (g + 1) * micro)``;
        comparing every leaf against every centroid would let an unrelated pair
        stand in for the one under test.
        """
        layer = _layer(3, 5, cluster_spread=0.3, seed=0)
        layer(_sample_input(input_dim=4))

        macro = ops.convert_to_numpy(layer.macro_weights).astype(np.float64)
        micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)

        for group in range(3):
            leaves = micro[:, group * 5:(group + 1) * 5]
            worst = float(np.abs(leaves.T @ macro[:, group]).max())
            assert worst < 0.999, (
                f"group {group} still has a leaf sitting on its centroid "
                f"(cosine {worst:.6f}) at cluster_spread=0.3"
            )

    def test_a_wider_spread_weakens_the_leaf_centroid_cosine(self):
        """The aperture is monotonic in the angular spread it produces."""
        cosines = []
        for spread in (0.0, 0.15, 0.5):
            layer = _layer(3, 5, cluster_spread=spread, seed=0)
            layer(_sample_input(input_dim=4))
            macro = ops.convert_to_numpy(layer.macro_weights).astype(np.float64)
            micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)
            worst = max(
                float(np.abs(micro[:, g * 5:(g + 1) * 5].T @ macro[:, g]).max())
                for g in range(3)
            )
            cosines.append(worst)

        assert cosines[0] > cosines[1] > cosines[2], (
            f"cluster_spread did not monotonically open the bundles: {cosines}"
        )

    def test_no_group_reproduces_the_macro_frame(self):
        """The defect this package found: a group whose seed coincides with the
        macro seed reproduces the macro simplex exactly, and its first leaf then
        sits at cosine 1.0 with its own centroid at EVERY spread.

        Pinned at the largest aperture, where a genuine bundle is furthest from
        its centroid and a coincident one is still exactly on it.
        """
        layer = _layer(3, 5, cluster_spread=0.9, seed=42)
        layer(_sample_input(input_dim=4))

        macro = ops.convert_to_numpy(layer.macro_weights).astype(np.float64)
        micro = ops.convert_to_numpy(layer.micro_weights).astype(np.float64)

        for group in range(3):
            leaves = micro[:, group * 5:(group + 1) * 5]
            worst = float(np.abs(leaves.T @ macro[:, group]).max())
            assert worst < 0.9999, (
                f"group {group} shares its displacement frame with the macro "
                f"frame (cosine {worst:.8f}); its seed collided"
            )


class TestHierarchicalCrystalLayerComputeOutputShape:
    """Both branches of the mode flag, from stored config, unbuilt."""

    def test_flat_branch(self):
        layer = _layer(3, 5)

        assert not layer.built
        assert layer.compute_output_shape((None, 4)) == (None, 15)
        assert layer.compute_output_shape((2, 3, 4)) == (2, 3, 15)
        assert not layer.built, "compute_output_shape must not build the layer"

    def test_hierarchy_branch_states_every_key(self):
        layer = _layer(3, 5, return_hierarchy=True)

        computed = layer.compute_output_shape((None, 4))

        assert computed == {
            'macro_logits': (None, 3),
            'micro_logits': (None, 15),
            'joint_probabilities': (None, 15),
        }

    def test_the_computed_shape_matches_the_forward_pass(self):
        inputs = _sample_input(rank=3, input_dim=4)
        flat = _layer(3, 5, seed=0)
        nested = _layer(3, 5, return_hierarchy=True, seed=0)

        assert flat.compute_output_shape(inputs.shape) == tuple(flat(inputs).shape)

        computed = nested.compute_output_shape(inputs.shape)
        for key, shape in computed.items():
            assert shape == tuple(nested(inputs)[key].shape)


class TestHierarchicalCrystalLayerKnobSensitivity:
    """Constructor knobs pinned by measured differences, with twins."""

    def test_the_seed_changes_both_frames_and_the_logits(self):
        inputs = _sample_input(input_dim=4)
        first, second = _layer(3, 12, seed=1), _layer(3, 12, seed=2)

        assert not np.allclose(
            ops.convert_to_numpy(first(inputs)), ops.convert_to_numpy(second(inputs))
        )

    def test_scale_multiplies_the_logits_linearly(self):
        inputs = _sample_input(input_dim=4)
        one = ops.convert_to_numpy(_layer(3, 5, scale=1.0, seed=0)(inputs))
        four = ops.convert_to_numpy(_layer(3, 5, scale=4.0, seed=0)(inputs))

        np.testing.assert_allclose(four, 4.0 * one, rtol=0, atol=1e-4)

    def test_the_logits_are_invariant_to_input_magnitude(self):
        layer = _layer(3, 5, seed=0)
        inputs = _sample_input(input_dim=4)

        np.testing.assert_allclose(
            ops.convert_to_numpy(layer(inputs * 4.0)),
            ops.convert_to_numpy(layer(inputs)),
            rtol=1e-5, atol=1e-5
        )

    def test_the_input_reached_the_layer(self):
        """Anti-vacuity: the invariance test above is also passed by a dead layer.

        The perturbation flips a sign, since a cosine head ignores magnitude.
        """
        layer = _layer(3, 5, seed=0)
        base = _sample_input(input_dim=4)
        flipped = base.copy()
        flipped[:, 0] *= -1.0

        assert float(np.abs(
            ops.convert_to_numpy(layer(base)) - ops.convert_to_numpy(layer(flipped))
        ).max()) > 1e-3


class TestHierarchicalCrystalLayerGraphSafety:
    """The (..., M, K) regrouping must survive tracing with dynamic shapes."""

    @pytest.mark.parametrize('rank', [2, 3, 4])
    def test_traced_output_matches_eager(self, rank):
        inputs = _sample_input(rank=rank, input_dim=4)
        signature = tf.TensorSpec([None] * (inputs.ndim - 1) + [4], tf.float32)

        eager = {
            key: ops.convert_to_numpy(value)
            for key, value in _layer(3, 12, return_hierarchy=True, seed=0)(inputs).items()
        }

        traced_layer = _layer(3, 12, return_hierarchy=True, seed=0)

        @tf.function(input_signature=[signature])
        def traced(x):
            return traced_layer(x)

        traced_output = traced(tf.constant(inputs))

        # Indexed BY KEY. The traced dict flattens in a different order from its
        # insertion order (the micro logits come first), so a positional read of
        # `keras.tree.flatten` silently compares unrelated tensors.
        np.testing.assert_allclose(
            ops.convert_to_numpy(traced_output['joint_probabilities']),
            eager['joint_probabilities'],
            rtol=1e-6, atol=1e-6,
            err_msg="traced joint probabilities differ from the eager ones"
        )
        np.testing.assert_allclose(
            ops.convert_to_numpy(traced_output['macro_logits']),
            eager['macro_logits'],
            rtol=1e-6, atol=1e-6,
            err_msg="traced macro logits differ from the eager ones"
        )

    def test_the_traced_joint_still_sums_to_one(self):
        """Normalization under tracing, not only eagerly."""
        layer = _layer(3, 12, return_hierarchy=True, seed=0)

        @tf.function(input_signature=[tf.TensorSpec([None, None, 4], tf.float32)])
        def traced(x):
            return layer(x)

        joint = traced(np.zeros((4, 3, 4), dtype=np.float32))['joint_probabilities']

        np.testing.assert_allclose(
            ops.convert_to_numpy(joint).sum(axis=-1), 1.0, rtol=0, atol=BAYES_ATOL
        )


class TestHierarchicalCrystalLayerGradients:
    """Per-variable gradient flow, non-None AND non-zero."""

    def test_gradients_reach_both_frames(self):
        layer = _layer(3, 6, return_hierarchy=True, trainable_crystal=True, seed=0)
        inputs = _sample_input(input_dim=4)

        with tf.GradientTape() as tape:
            outputs = layer(inputs, training=True)
            loss = ops.sum(ops.square(outputs['joint_probabilities']))

        assert len(layer.trainable_variables) == 2, "expected macro and micro frames"
        grads = tape.gradient(loss, layer.trainable_variables)

        for variable, grad in zip(layer.trainable_variables, grads):
            assert grad is not None, f"no gradient for {variable.path}"
            assert np.any(ops.convert_to_numpy(grad) != 0.0), \
                f"all-zero gradient for {variable.path}"

    def test_the_frozen_frames_receive_no_gradient(self):
        """A frozen frame must be absent from `trainable_variables` AND ungraduated.

        The other half of the pair with the test above. It is run on a
        `trainable_crystal=True` layer whose frames are then frozen, so the
        gradient query is non-empty -- querying an empty list returns `[]`, over
        which `all(...)` is vacuously true and would pass a layer that was never
        frozen at all.
        """
        layer = _layer(3, 6, return_hierarchy=True, trainable_crystal=True, seed=0)
        inputs = _sample_input(input_dim=4)
        layer(inputs)

        # Confirm the released frames DO produce gradients before freezing, so
        # the absence below is attributable to the freeze and not to a probe
        # that could never have seen a gradient.
        with tf.GradientTape() as live_tape:
            loss = ops.sum(ops.square(
                layer(inputs, training=True)['joint_probabilities']
            ))
        assert any(
            grad is not None and np.any(ops.convert_to_numpy(grad) != 0.0)
            for grad in live_tape.gradient(loss, layer.trainable_variables)
        ), "the probe cannot detect a gradient, so its absence proves nothing"

        layer.macro_weights.trainable = False
        layer.micro_weights.trainable = False

        with tf.GradientTape() as tape:
            frozen_loss = ops.sum(ops.square(
                layer(inputs, training=True)['joint_probabilities']
            ))
        grads = tape.gradient(frozen_loss, layer.trainable_variables)

        assert layer.trainable_variables == [], "frozen frames remained trainable"
        assert grads == [], "a frozen frame produced a gradient"


class TestHierarchicalCrystalLayerSerialization:
    """`.keras` round trip on VALUES, and config completeness."""

    def test_config_carries_every_constructor_argument(self):
        config = _layer(
            4, 9, cluster_spread=0.4, scale=3.0, return_hierarchy=True,
            trainable_crystal=True, seed=5, eps=1e-5
        ).get_config()

        for key in ('num_macro_classes', 'num_micro_per_macro', 'cluster_spread',
                    'scale', 'return_hierarchy', 'trainable_crystal', 'seed', 'eps'):
            assert key in config, f"get_config is missing {key!r}"

        rebuilt = HierarchicalCrystalLayer.from_config(config)
        assert rebuilt.num_macro == 4
        assert rebuilt.num_micro == 9
        assert rebuilt.cluster_spread == 0.4
        assert rebuilt.return_hierarchy is True
        assert rebuilt.seed == 5

    def test_total_classes_is_not_serialized(self):
        """It is derived from two constructor arguments; storing it invites drift."""
        assert 'total_classes' not in _layer(3, 5).get_config()

    def test_the_flat_round_trip_preserves_values(self):
        inputs = _sample_input(input_dim=4)
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            _layer(3, 5, scale=3.0, seed=0)
        ])
        original = ops.convert_to_numpy(model(inputs, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hierarchical.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        restored = ops.convert_to_numpy(loaded(inputs, training=False))

        np.testing.assert_allclose(
            restored, original, rtol=0, atol=1e-6,
            err_msg="flat-mode outputs differ after a .keras round trip"
        )

    def test_the_hierarchy_round_trip_preserves_values(self):
        """A dict output is the case a bare tensor comparison cannot cover."""
        inputs = _sample_input(input_dim=4)
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            _layer(3, 5, scale=3.0, return_hierarchy=True, seed=0)
        ])
        original = {
            key: ops.convert_to_numpy(value)
            for key, value in model(inputs, training=False).items()
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hierarchical_dict.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        restored = {
            key: ops.convert_to_numpy(value)
            for key, value in loaded(inputs, training=False).items()
        }

        assert set(restored) == set(original)
        for key in original:
            np.testing.assert_allclose(
                restored[key], original[key], rtol=0, atol=1e-6,
                err_msg=f"{key} differs after a .keras round trip"
            )

    def test_the_round_trip_restores_the_frames_before_any_forward(self):
        model = keras.Sequential([
            keras.Input(shape=(4,)),
            _layer(3, 5, seed=0)
        ])
        model(_sample_input(input_dim=4))
        saved = [ops.convert_to_numpy(w).copy() for w in model.weights]

        assert saved, "the model has no weights to compare"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hierarchical.keras')
            model.save(path)
            loaded = keras.models.load_model(path)

        for donor, restored in zip(saved, loaded.weights):
            np.testing.assert_allclose(
                ops.convert_to_numpy(restored), donor, atol=0.0, rtol=0
            )


@pytest.mark.usefixtures('dtype_policy')
class TestHierarchicalCrystalLayerDtypes:
    """float32 / mixed_float16 / float64 arms."""

    def test_construction_and_forward_survive(self, dtype_policy):
        layer = _layer(3, 5, seed=0)
        output = layer(_sample_input(input_dim=4))

        assert output.shape == (4, 15)
        assert bool(ops.all(ops.isfinite(output))), \
            f"{dtype_policy} produced a non-finite leaf logit"

    def test_the_bayes_identity_survives_every_policy(self, dtype_policy):
        """A mixed-precision softmax must not break the normalization."""
        layer = _layer(3, 5, return_hierarchy=True, seed=0)
        joint = ops.convert_to_numpy(
            layer(_sample_input(input_dim=4))['joint_probabilities']
        )

        # float16 accumulates coarser than float32, so the bound is widened
        # to the dtype's own resolution rather than left at the float32 value.
        atol = 1e-2 if dtype_policy == 'mixed_float16' else BAYES_ATOL
        np.testing.assert_allclose(joint.sum(axis=-1), 1.0, rtol=0, atol=atol)
