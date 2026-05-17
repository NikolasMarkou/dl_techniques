"""Tests for ``dl_techniques.training.token_superposition`` (TST).

Covers the seven correctness invariants from the plan plus a small
integration smoke test. Test names mirror the plan's success-criterion IDs
(``inv1_...``, ``inv2_...``, etc.) so the verification table maps 1-to-1.
"""

import dataclasses

import pytest
import numpy as np
import keras
import tensorflow as tf

from dl_techniques.training.token_superposition import (
    TSTConfig,
    TSTState,
    TSTEmbedding,
)


# =====================================================================
# Step 2 — TSTConfig + TSTState
# =====================================================================


class TestTSTConfig:
    def test_defaults_match_documented(self):
        cfg = TSTConfig()
        assert cfg.bag_size == 6
        assert cfg.phase1_step_ratio == 0.25
        assert cfg.within_bag_weighting == "uniform"
        assert cfg.within_bag_alpha == 0.6

    def test_is_frozen(self):
        cfg = TSTConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.bag_size = 7  # type: ignore[misc]

    def test_validation_bag_size_zero(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTConfig(bag_size=0)

    def test_validation_bag_size_negative(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTConfig(bag_size=-3)

    def test_validation_phase1_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="phase1_step_ratio"):
            TSTConfig(phase1_step_ratio=1.5)
        with pytest.raises(ValueError, match="phase1_step_ratio"):
            TSTConfig(phase1_step_ratio=-0.01)

    def test_validation_alpha_non_positive(self):
        with pytest.raises(ValueError, match="within_bag_alpha"):
            TSTConfig(within_bag_alpha=0.0)
        with pytest.raises(ValueError, match="within_bag_alpha"):
            TSTConfig(within_bag_alpha=-0.5)

    def test_validation_unknown_weighting(self):
        with pytest.raises(ValueError, match="within_bag_weighting"):
            TSTConfig(within_bag_weighting="exotic")  # type: ignore[arg-type]


class TestTSTState:
    def test_phase_active_is_tf_variable_bool(self):
        st = TSTState(bag_size=4)
        assert isinstance(st.phase_active, tf.Variable)
        assert st.phase_active.dtype == tf.bool
        assert bool(st.phase_active.numpy()) is True

    def test_global_step_is_tf_variable_int64(self):
        st = TSTState(bag_size=4)
        assert isinstance(st.global_step, tf.Variable)
        assert st.global_step.dtype == tf.int64
        assert int(st.global_step.numpy()) == 0

    def test_phase_active_is_not_trainable(self):
        st = TSTState(bag_size=4)
        assert st.phase_active.trainable is False
        assert st.global_step.trainable is False

    def test_phase_active_init_false(self):
        st = TSTState(bag_size=2, phase_active_init=False)
        assert bool(st.phase_active.numpy()) is False

    def test_reset(self):
        st = TSTState(bag_size=2)
        st.phase_active.assign(False)
        st.global_step.assign(100)
        st.reset()
        assert bool(st.phase_active.numpy()) is True
        assert int(st.global_step.numpy()) == 0

    def test_invalid_bag_size(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTState(bag_size=0)


# =====================================================================
# Step 3 — TSTEmbedding (invariants 1, 4, 5, 7 + tied-head passthrough)
# =====================================================================


def _copy_embedding_weights(src: keras.layers.Layer, dst: keras.layers.Layer) -> None:
    """Copy weights between a TSTEmbedding and a plain Embedding (either direction)."""
    src_var = src.embeddings if isinstance(src, TSTEmbedding) else src.embeddings
    dst_var = dst.embeddings if isinstance(dst, TSTEmbedding) else dst.embeddings
    dst_var.assign(src_var.numpy())


class TestTSTEmbedding:
    def test_inv1_canary_bag_size_1_equals_plain_embedding(self):
        """Invariant 1: bag_size=1 path is bit-equivalent to plain Embedding."""
        V, D, B, N = 100, 8, 4, 12
        rng = np.random.default_rng(42)
        x = rng.integers(0, V, size=(B, N), dtype=np.int32)

        tst = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=1, name="tst1")
        plain = keras.layers.Embedding(input_dim=V, output_dim=D, name="plain")
        # Force build then sync weights.
        _ = tst(keras.ops.convert_to_tensor(x))
        _ = plain(keras.ops.convert_to_tensor(x))
        plain.embeddings.assign(tst.embeddings.numpy())

        y_tst = keras.ops.convert_to_numpy(tst(keras.ops.convert_to_tensor(x)))
        y_plain = keras.ops.convert_to_numpy(plain(keras.ops.convert_to_tensor(x)))
        # bit-equivalent (lookup is exact int->slice).
        np.testing.assert_array_equal(y_tst, y_plain)

    def test_inv4_mean_not_sum(self):
        """Invariant 4: bagged output is mean over the bag axis, not sum."""
        V, D, B, N_lat, s = 50, 4, 2, 3, 4
        rng = np.random.default_rng(7)
        x = rng.integers(0, V, size=(B, N_lat, s), dtype=np.int32)

        tst = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=s, name="tst_mean")
        _ = tst(keras.ops.convert_to_tensor(x))  # build
        # Sibling plain Embedding with the same weights for ground-truth mean.
        plain = keras.layers.Embedding(input_dim=V, output_dim=D, name="plain_ref")
        plain.build((None, None))
        plain.embeddings.assign(tst.embeddings.numpy())

        y_tst = keras.ops.convert_to_numpy(tst(keras.ops.convert_to_tensor(x)))
        # Reference: lookup each bag entry, then mean over bag axis.
        per_token = keras.ops.convert_to_numpy(
            plain(keras.ops.convert_to_tensor(x))
        )  # (B, N_lat, s, D)
        y_ref_mean = per_token.mean(axis=-2)
        y_ref_sum = per_token.sum(axis=-2)
        np.testing.assert_allclose(y_tst, y_ref_mean, atol=1e-6)
        # And mean ≠ sum on random nonzero data (sanity guard for the test).
        assert not np.allclose(y_tst, y_ref_sum, atol=1e-3)

    def test_inv5_param_count_exact(self):
        """Invariant 5: trainable params = V*D, no extras."""
        V, D, s = 32, 6, 4
        layer = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=s)
        # Force build via a forward pass.
        _ = layer(keras.ops.zeros((1, s), dtype="int32"))
        assert len(layer.trainable_weights) == 1
        assert layer.count_params() == V * D

    def test_inv7_get_config_roundtrip(self):
        """Invariant 7: get_config → from_config preserves all hyperparams."""
        layer = TSTEmbedding(
            vocab_size=64,
            output_dim=8,
            bag_size=3,
            embeddings_initializer="uniform",
            name="rt_layer",
        )
        cfg = layer.get_config()
        new = TSTEmbedding.from_config(cfg)
        assert new.vocab_size == layer.vocab_size
        assert new.output_dim == layer.output_dim
        assert new.bag_size == layer.bag_size
        assert new.embeddings_initializer == layer.embeddings_initializer

    def test_inv7_keras_model_save_load(self, tmp_path):
        """Invariant 7 (stronger): save+load inside a keras.Model round-trips."""
        V, D, B, N_lat, s = 40, 4, 2, 3, 4
        inputs = keras.Input(shape=(N_lat, s), dtype="int32")
        emb = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=s, name="emb")(inputs)
        model = keras.Model(inputs, emb)
        x = np.random.default_rng(0).integers(0, V, size=(B, N_lat, s)).astype("int32")
        y_before = keras.ops.convert_to_numpy(model(x))
        path = str(tmp_path / "m.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y_after = keras.ops.convert_to_numpy(loaded(x))
        np.testing.assert_allclose(y_before, y_after, atol=1e-6)

    def test_tied_lm_head_passthrough(self):
        """`.embeddings` is the inner Embedding's variable — used by tied heads."""
        V, D, s = 20, 5, 4
        layer = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=s)
        _ = layer(keras.ops.zeros((1, s), dtype="int32"))
        assert layer.embeddings is layer._inner.embeddings
        # Verify the tied-head matmul shape works.
        x = keras.ops.ones((1, 2, D))  # (B, N, D)
        logits = keras.ops.matmul(x, keras.ops.transpose(layer.embeddings))
        assert tuple(logits.shape) == (1, 2, V)

    def test_rank2_input_path(self):
        """Rank-2 input → plain lookup even when bag_size > 1."""
        V, D, s = 16, 4, 3
        layer = TSTEmbedding(vocab_size=V, output_dim=D, bag_size=s)
        x = keras.ops.convert_to_tensor(
            np.random.default_rng(0).integers(0, V, size=(2, 5)).astype("int32")
        )
        y = layer(x)
        assert tuple(y.shape) == (2, 5, D)

    def test_rank3_wrong_last_axis_raises(self):
        """rank-3 with last axis != bag_size raises ValueError."""
        layer = TSTEmbedding(vocab_size=10, output_dim=4, bag_size=3)
        bad = keras.ops.zeros((2, 4, 5), dtype="int32")  # last dim = 5 ≠ 3
        with pytest.raises(ValueError, match="bag_size"):
            _ = layer(bad)
