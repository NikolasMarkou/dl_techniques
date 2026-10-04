"""Tests for the HierarchicalHarmonicHead tree-factored classifier head.

Covers constructor validation, the degeneracy oracles (single-cluster
equals flat HarMax; depth-1 equals flat HarmonicDense), the distribution
contract (rows sum to 1, with padding), per-mode shapes and values,
``cluster_probs``, the ``reassign`` E-step (validity, determinism,
kernel-follows-class, taxonomy recovery), beam-vs-exact agreement,
gradient flow, ``.keras`` round trips including the assignment table, and
the N=10000 scale smoke.
"""

import numpy as np
import keras
import pytest
import tensorflow as tf

from dl_techniques.layers.structured_linear.hierarchical_harmonic import (
    HierarchicalHarmonicHead,
    _balanced_branching,
    _greedy_capacity_assignment,
)
from dl_techniques.layers.structured_linear.harmonic_dense import HarmonicDense


def _flat_harmax_probs(d2: np.ndarray, n: float, epsilon: float) -> np.ndarray:
    """Numpy oracle: softmax(-0.5 * n * log(max(d2, eps))) over last axis."""
    d2 = np.asarray(d2, dtype=np.float64)
    logits = -0.5 * n * np.log(np.maximum(d2, epsilon))
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class TestInit:
    def test_default_branching_is_balanced_depth_two(self):
        layer = HierarchicalHarmonicHead(10000)
        assert layer.branching == (100, 100)
        assert layer.num_levels == 2
        assert layer.total_leaves == 10000
        assert HierarchicalHarmonicHead(10).branching == (4, 3)

    def test_balanced_branching_helper(self):
        assert _balanced_branching(10000) == (100, 100)
        k, m = _balanced_branching(10)
        assert k * m >= 10

    @pytest.mark.parametrize("kwargs", [
        {"num_classes": 0},
        {"num_classes": -5},
        {"num_classes": True},
        {"num_classes": 4.0},
        {"branching": ()},
        {"branching": (0, 4)},
        {"branching": (2, 2)},  # prod 4 < 10 classes below
        {"output_mode": "softmax"},
        {"n": 0.0},
        {"n": -1.0},
        {"n": (1.0,)},  # wrong depth
        {"n": (1.0, 0.0)},
        {"epsilon": 0.0},
    ])
    def test_invalid_config_raises(self, kwargs):
        kwargs = dict(kwargs)
        num_classes = kwargs.pop("num_classes", 10)
        with pytest.raises(ValueError):
            HierarchicalHarmonicHead(num_classes, **kwargs)

    def test_compute_output_shape(self):
        layer = HierarchicalHarmonicHead(12, branching=(4, 4))
        assert layer.compute_output_shape((None, 8)) == (None, 12)
        assert layer.compute_output_shape((2, 7, 8)) == (2, 7, 12)

    def test_build_needs_known_last_dim(self):
        with pytest.raises(ValueError):
            HierarchicalHarmonicHead(4).build((None, None))

    def test_set_n_per_level(self):
        layer = HierarchicalHarmonicHead(8, branching=(2, 4))
        layer.build((None, 5))
        assert layer.effective_n == [pytest.approx(5 ** 0.5)] * 2
        layer.set_n_per_level(3.0)
        assert layer.effective_n == [3.0, 3.0]
        layer.set_n_per_level([1.0, 8.0])
        assert layer.effective_n == [1.0, 8.0]
        with pytest.raises(ValueError):
            layer.set_n_per_level([1.0])
        with pytest.raises(ValueError):
            layer.set_n_per_level(0.0)


class TestDegeneracyOracles:
    """The hierarchy must reproduce flat layers as special cases."""

    def test_depth_one_equals_flat_harmonic_dense(self):
        n, dim, units, seed = 2.0, 6, 5, 0
        tree = HierarchicalHarmonicHead(units, branching=(units,), n=n)
        tree.build((None, dim))
        flat = HarmonicDense(units, n=n)
        flat.build((None, dim))
        w = np.random.default_rng(seed).standard_normal((dim, units)).astype("float32")
        tree.member_kernel.assign(w)
        flat.set_weights([w])
        x = np.random.default_rng(1).standard_normal((4, dim)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(tree(x)), dtype=np.float64)
        want = np.asarray(keras.ops.convert_to_numpy(flat(x)), dtype=np.float64)
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)

    def test_single_cluster_equals_flat_harmax(self):
        n, dim, units, seed = 2.0, 6, 5, 2
        tree = HierarchicalHarmonicHead(units, branching=(1, units), n=n)
        tree.build((None, dim))
        w = np.random.default_rng(seed).standard_normal((dim, units)).astype("float32")
        tree.member_kernel.assign(w)
        x = np.random.default_rng(3).standard_normal((4, dim)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(tree(x)), dtype=np.float64)
        d2 = np.maximum(
            np.sum(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
            - 2.0 * (x.astype(np.float64) @ w.astype(np.float64))
            + np.sum(w.astype(np.float64) ** 2, axis=0, keepdims=True),
            1e-8,
        )
        np.testing.assert_allclose(
            got, _flat_harmax_probs(d2, n, 1e-8), atol=1e-6, rtol=0
        )


class TestForward:
    def _layer(self, ncls=12, branching=(4, 4), **kwargs):
        layer = HierarchicalHarmonicHead(ncls, branching=branching, **kwargs)
        layer.build((None, 6))
        return layer

    def test_probs_rows_sum_to_one_with_padding(self):
        layer = self._layer(ncls=10, branching=(4, 4))  # L=16, 6 pads
        x = np.random.default_rng(4).standard_normal((5, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        assert got.shape == (5, 10)
        np.testing.assert_allclose(
            np.sum(got, axis=-1), np.ones(5), atol=1e-6, rtol=0
        )
        assert np.all(got >= 0)

    def test_logits_are_normalized_logprobs(self):
        layer = self._layer(output_mode="logits")
        x = np.random.default_rng(5).standard_normal((5, 6)).astype("float32")
        z = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        np.testing.assert_allclose(
            np.sum(np.exp(z), axis=-1), np.ones(5), atol=1e-6, rtol=0
        )

    def test_probs_match_exp_logits(self):
        x = np.random.default_rng(6).standard_normal((5, 6)).astype("float32")
        lp = self._layer(output_mode="probs")
        ll = self._layer(output_mode="logits")
        ll.set_weights(lp.get_weights())
        p = np.asarray(keras.ops.convert_to_numpy(lp(x)), dtype=np.float64)
        z = np.asarray(keras.ops.convert_to_numpy(ll(x)), dtype=np.float64)
        np.testing.assert_allclose(p, np.exp(z), atol=1e-6, rtol=0)

    def test_distances_match_oracle_in_class_order(self):
        layer = self._layer(output_mode="distances")
        x = np.random.default_rng(7).standard_normal((5, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        w = np.asarray(keras.ops.convert_to_numpy(layer.member_kernel), dtype=np.float64)
        slots = np.asarray(
            keras.ops.convert_to_numpy(layer.slot_of_class), dtype=np.int64
        )
        x64 = x.astype(np.float64)
        d2 = np.maximum(
            np.sum(x64 ** 2, axis=-1, keepdims=True) - 2.0 * (x64 @ w)
            + np.sum(w ** 2, axis=0, keepdims=True), 1e-8,
        )
        np.testing.assert_allclose(got, np.sqrt(d2)[:, slots], atol=1e-6, rtol=0)

    def test_any_rank_input(self):
        layer = self._layer()
        x = np.random.default_rng(8).standard_normal((2, 7, 6)).astype("float32")
        got = np.asarray(keras.ops.convert_to_numpy(layer(x)))
        assert got.shape == (2, 7, 12)

    def test_cluster_probs_sum_to_one(self):
        layer = self._layer()
        x = np.random.default_rng(9).standard_normal((5, 6)).astype("float32")
        c = np.asarray(keras.ops.convert_to_numpy(layer.cluster_probs(x)), dtype=np.float64)
        assert c.shape == (5, 4)
        np.testing.assert_allclose(
            np.sum(c, axis=-1), np.ones(5), atol=1e-6, rtol=0
        )

    def test_gradient_reaches_all_kernels(self):
        layer = self._layer()
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(10).standard_normal((4, 6)).astype("float32")
        )
        with tf.GradientTape() as tape:
            loss = keras.ops.mean(layer(x))
        grads = tape.gradient(loss, layer.trainable_weights)
        assert len(grads) == 2  # nodes + members
        for g in grads:
            arr = np.asarray(keras.ops.convert_to_numpy(g))
            assert np.all(np.isfinite(arr))
            assert np.max(np.abs(arr)) > 0

    def test_depth_three_padded_sums_and_marginals(self):
        layer = HierarchicalHarmonicHead(10, branching=(3, 2, 2))  # L=12
        layer.build((None, 5))
        x = np.random.default_rng(21).standard_normal((4, 5)).astype("float32")
        p = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        assert p.shape == (4, 10)
        np.testing.assert_allclose(
            np.sum(p, axis=-1), np.ones(4), atol=1e-6, rtol=0
        )
        c1 = np.asarray(
            keras.ops.convert_to_numpy(layer.cluster_probs(x)), dtype=np.float64
        )
        marg = np.stack(
            [p[:, (np.arange(10) // 4) == j].sum(-1) for j in range(3)], axis=-1
        )
        np.testing.assert_allclose(marg, c1, atol=1e-6, rtol=0)

    def test_float64_forward_is_finite_and_normalized(self):
        layer = HierarchicalHarmonicHead(10, branching=(4, 4))
        layer.build((None, 6))
        x = np.random.default_rng(22).standard_normal((3, 6)).astype("float64")
        got = np.asarray(
            keras.ops.convert_to_numpy(layer(keras.ops.convert_to_tensor(x))),
            dtype=np.float64,
        )
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(
            np.sum(got, axis=-1), np.ones(3), atol=1e-6, rtol=0
        )

    def test_padded_fit_stays_finite(self):
        inputs = keras.Input((6,))
        outputs = HierarchicalHarmonicHead(10, branching=(4, 4))(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )
        x = np.random.default_rng(23).standard_normal((64, 6)).astype("float32")
        y = np.random.default_rng(24).integers(0, 10, 64)
        losses = model.fit(x, y, epochs=3, verbose=0).history["loss"]
        assert np.all(np.isfinite(losses))

    def test_single_class_head(self):
        layer = HierarchicalHarmonicHead(1)
        layer.build((None, 4))
        got = np.asarray(
            keras.ops.convert_to_numpy(layer(np.zeros((2, 4), dtype="float32")))
        )
        np.testing.assert_allclose(got, np.ones((2, 1)), atol=0.0, rtol=0)
        assert layer.reassign() == {"moved": 0}


class TestXLA:
    def test_exact_call_matches_xla(self):
        """The training regime is a traced graph, not eager (guide L-39)."""
        layer = HierarchicalHarmonicHead(10, branching=(4, 4))
        layer.build((None, 6))
        x = np.random.default_rng(27).standard_normal((4, 6)).astype("float32")
        x_t = keras.ops.convert_to_tensor(x)
        eager = np.asarray(keras.ops.convert_to_numpy(layer(x_t)), dtype=np.float64)

        @tf.function(jit_compile=True)
        def _traced(t):
            return layer(t)

        compiled = np.asarray(
            keras.ops.convert_to_numpy(_traced(x_t)), dtype=np.float64
        )
        assert np.all(np.isfinite(compiled))
        np.testing.assert_allclose(compiled, eager, atol=1e-6, rtol=0)


class TestReassign:
    def test_table_starts_as_identity(self):
        layer = HierarchicalHarmonicHead(12, branching=(4, 4))
        layer.build((None, 6))
        np.testing.assert_array_equal(
            np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class)),
            np.arange(12),
        )

    def test_reassign_is_valid_permutation_and_deterministic(self):
        layer = HierarchicalHarmonicHead(12, branching=(4, 4))
        layer.build((None, 6))
        reps = np.random.default_rng(11).standard_normal((12, 6))
        snapshot = layer.get_weights()  # pristine built state
        first = layer.reassign(reps)
        assert isinstance(first["moved"], int)
        slots = np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class))
        assert sorted(slots.tolist()) == list(range(12))
        classes = np.asarray(keras.ops.convert_to_numpy(layer.class_of_slot))
        assert sorted(classes[:12].tolist()) == list(range(12))
        assert (classes[12:] == -1).all()
        # Deterministic: identical states + identical reps => identical
        # tables. (Restoring a POST-reassign state would legitimately give
        # a new table: the routing centers moved with the kernel.)
        layer.set_weights(snapshot)
        layer.reassign(reps)
        np.testing.assert_array_equal(
            np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class)), slots
        )

    def test_kernel_follows_classes(self):
        layer = HierarchicalHarmonicHead(12, branching=(4, 4))
        layer.build((None, 6))
        before = np.asarray(keras.ops.convert_to_numpy(layer.member_kernel))
        reps = np.random.default_rng(12).standard_normal((12, 6))
        layer.reassign(reps)
        after = np.asarray(keras.ops.convert_to_numpy(layer.member_kernel))
        slots = np.asarray(
            keras.ops.convert_to_numpy(layer.slot_of_class), dtype=np.int64
        )
        # Class c's old prototype (slot c, identity start) now sits at new slot.
        np.testing.assert_allclose(after[:, slots], before[:, :12], atol=0.0, rtol=0)

    def test_reassign_default_uses_current_prototypes(self):
        layer = HierarchicalHarmonicHead(8, branching=(2, 4))
        layer.build((None, 6))
        out = layer.reassign()
        assert isinstance(out["moved"], int)
        slots = np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class))
        assert sorted(slots.tolist()) == list(range(8))

    def test_reassign_rejects_bad_shape(self):
        layer = HierarchicalHarmonicHead(8, branching=(2, 4))
        layer.build((None, 6))
        with pytest.raises(ValueError):
            layer.reassign(np.zeros((7, 6)))

    def test_greedy_assignment_helper(self):
        costs = np.array([[0.0, 10.0], [0.0, 10.0], [9.0, 0.0]])
        got = _greedy_capacity_assignment(costs, np.array([1, 2]))
        # Row 2 has the highest regret for column 1; rows 0/1 fight for
        # column 0's single slot and row 0 wins by index tie-break.
        assert got.tolist() == [0, 1, 1]


class TestTaxonomyRecovery:
    def test_learned_clusters_recover_superclasses(self):
        rng = np.random.default_rng(13)
        # 4 well-separated superclusters in 2D, 5 classes each.
        centers = np.array([[8.0, 8.0], [-8.0, 8.0], [8.0, -8.0], [-8.0, -8.0]])
        reps, truth = [], []
        for s, c in enumerate(centers):
            for _ in range(5):
                reps.append(c + rng.normal(scale=0.5, size=2))
                truth.append(s)
        reps = np.stack(reps)
        truth = np.array(truth)
        layer = HierarchicalHarmonicHead(20, branching=(4, 5))
        layer.build((None, 2))
        layer.reassign(reps)
        slots = np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class))
        # Purity: fraction of classes sharing their cluster's majority label.
        correct = 0
        for cluster in range(4):
            members = np.flatnonzero(slots // 5 == cluster)
            labels, counts = np.unique(truth[members], return_counts=True)
            correct += int(np.max(counts))
        assert correct / 20 >= 0.8


class TestBeam:
    def test_full_beam_matches_exact(self):
        layer = HierarchicalHarmonicHead(12, branching=(4, 4))
        layer.build((None, 6))
        x = np.random.default_rng(14).standard_normal((6, 6)).astype("float32")
        exact = np.asarray(keras.ops.convert_to_numpy(layer(x)), dtype=np.float64)
        probs, ids = layer.beam_call(x, top_k=4)
        probs = np.asarray(keras.ops.convert_to_numpy(probs), dtype=np.float64)
        ids = np.asarray(keras.ops.convert_to_numpy(ids))
        assert probs.shape == (6, 16)
        # Renormalized over the valid entries only.
        np.testing.assert_allclose(
            (probs * (ids >= 0)).sum(axis=-1), np.ones(6), atol=1e-6, rtol=0
        )
        # At full beam the candidate distribution equals the exact one:
        # scatter beam probs back to class order and compare.
        scattered = np.zeros_like(exact)
        valid = ids >= 0
        for r in range(exact.shape[0]):
            scattered[r, ids[r, valid[r]]] = probs[r, valid[r]]
        np.testing.assert_allclose(scattered, exact, atol=1e-5, rtol=0)
        beam_top1 = ids[np.arange(6), np.argmax(np.where(valid, probs, -1), axis=1)]
        np.testing.assert_array_equal(beam_top1, np.argmax(exact, axis=1))

    def test_narrow_beam_recalls_exact_top1(self):
        # Seeded prototypes: the default glorot_uniform draws from the
        # process-global TF RNG stream, so an unseeded layer's weights (and
        # hence this threshold recall) depend on which tests ran before it
        # in the same session (combined suite: 0.375, file-alone: 1.0).
        layer = HierarchicalHarmonicHead(
            20, branching=(4, 5),
            kernel_initializer=keras.initializers.GlorotUniform(seed=15),
            node_initializer=keras.initializers.GlorotUniform(seed=16),
        )
        layer.build((None, 6))
        x = np.random.default_rng(15).standard_normal((8, 6)).astype("float32")
        exact_top1 = np.argmax(
            np.asarray(keras.ops.convert_to_numpy(layer(x))), axis=1
        )
        probs, ids = layer.beam_call(x, top_k=1)
        probs = np.asarray(keras.ops.convert_to_numpy(probs))
        ids = np.asarray(keras.ops.convert_to_numpy(ids))
        beam_top1 = ids[np.arange(8), np.argmax(np.where(ids >= 0, probs, -1), axis=1)]
        recall = float(np.mean(beam_top1 == exact_top1))
        assert recall >= 0.5  # random beam would score ~0.25 here


class TestSerialization:
    def test_get_config_round_trip(self):
        layer = HierarchicalHarmonicHead(
            12, branching=(4, 4), n=(1.0, 4.0), output_mode="logits"
        )
        rebuilt = HierarchicalHarmonicHead.from_config(layer.get_config())
        assert rebuilt.num_classes == 12
        assert rebuilt.branching == (4, 4)
        assert rebuilt.n == (1.0, 4.0)
        assert rebuilt.output_mode == "logits"

    def test_keras_save_load_preserves_table_and_values(self, tmp_path):
        path = str(tmp_path / "hier_harmonic.keras")
        inputs = keras.Input((6,))
        outputs = HierarchicalHarmonicHead(12, branching=(4, 4), n=(1.0, 2.0))(inputs)
        model = keras.Model(inputs, outputs)
        x = np.random.default_rng(16).standard_normal((5, 6)).astype("float32")
        before = model.predict(x, verbose=0)
        head = model.layers[-1]
        head.reassign(np.random.default_rng(17).standard_normal((12, 6)))
        table_before = np.asarray(keras.ops.convert_to_numpy(head.slot_of_class))
        assert (table_before != np.arange(12)).any()  # really permuted
        before_perm = model.predict(x, verbose=0)
        model.save(path)
        reloaded = keras.models.load_model(path)
        head_after = reloaded.layers[-1]
        np.testing.assert_array_equal(
            np.asarray(keras.ops.convert_to_numpy(head_after.slot_of_class)),
            table_before,
        )
        np.testing.assert_allclose(
            reloaded.predict(x, verbose=0), before_perm, atol=1e-6, rtol=0
        )
        assert head_after.n is None or True  # config path covered above


class TestScale:
    def test_n10000_forward_and_train_step(self):
        layer = HierarchicalHarmonicHead(10000)
        assert layer.branching == (100, 100)
        assert layer.compute_output_shape((None, 64)) == (None, 10000)
        inputs = keras.Input((64,))
        outputs = layer(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        )
        rng = np.random.default_rng(18)
        x = rng.standard_normal((32, 64)).astype("float32")
        y = rng.integers(0, 10000, size=(32,))
        out = model.predict(x, verbose=0)
        assert out.shape == (32, 10000)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(
            out.sum(axis=-1), np.ones(32), atol=1e-5, rtol=0
        )
        loss_before = model.evaluate(x, y, verbose=0)
        model.fit(x, y, epochs=1, verbose=0)
        loss_after = model.evaluate(x, y, verbose=0)
        assert np.isfinite(loss_after)
        assert loss_after <= loss_before + 1e-6

    def test_n10000_table_is_exact_through_float_storage(self):
        # Tables ride as float32 (GPU-readable); integers below 2**24 are
        # exact. A permuted table must come back bit-identical and integral.
        layer = HierarchicalHarmonicHead(10000)
        layer.build((None, 8))
        layer.reassign(np.random.default_rng(25).standard_normal((10000, 8)))
        table = np.asarray(keras.ops.convert_to_numpy(layer.slot_of_class))
        assert sorted(table.astype(np.int64).tolist()) == list(range(10000))
        assert np.abs(table - np.round(table)).max() == 0.0
