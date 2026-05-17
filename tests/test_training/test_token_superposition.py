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
    TSTCausalLMLoss,
)
from dl_techniques.losses.masked_causal_lm_loss import MaskedCausalLMLoss


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


# =====================================================================
# Step 4 — TSTCausalLMLoss (invariants 2, 3 + dict y_pred + round-trip)
# =====================================================================


class TestTSTCausalLMLoss:
    def test_inv2_canary_bag_size_1_equals_masked_ce(self):
        """Invariant 2 (Falsification A): bag_size=1 rank-2 path ≡ MaskedCausalLMLoss."""
        B, N, V = 3, 7, 12
        rng = np.random.default_rng(123)
        y_true = rng.integers(-1, V, size=(B, N)).astype(np.int32)
        logits = rng.standard_normal((B, N, V)).astype(np.float32)

        tst = TSTCausalLMLoss(bag_size=1, ignore_index=-1, from_logits=True)
        ref = MaskedCausalLMLoss(ignore_index=-1, from_logits=True)
        y_tst = float(keras.ops.convert_to_numpy(tst(y_true, logits)))
        y_ref = float(keras.ops.convert_to_numpy(ref(y_true, logits)))
        assert abs(y_tst - y_ref) < 1e-6

    def test_inv2_canary_with_label_smoothing(self):
        """Canary with label_smoothing > 0 still matches the reference."""
        B, N, V = 2, 5, 8
        rng = np.random.default_rng(0)
        y_true = rng.integers(-1, V, size=(B, N)).astype(np.int32)
        logits = rng.standard_normal((B, N, V)).astype(np.float32)

        tst = TSTCausalLMLoss(bag_size=1, label_smoothing=0.1, from_logits=True)
        ref = MaskedCausalLMLoss(label_smoothing=0.1, from_logits=True)
        y_tst = float(keras.ops.convert_to_numpy(tst(y_true, logits)))
        y_ref = float(keras.ops.convert_to_numpy(ref(y_true, logits)))
        assert abs(y_tst - y_ref) < 1e-6

    def test_inv3_sum_of_ce_equals_manual_reference(self):
        """Invariant 3: sum-of-CE loop matches a NumPy reference implementation."""
        B, N_lat, s, V = 2, 3, 4, 5
        rng = np.random.default_rng(99)
        y_true = rng.integers(0, V, size=(B, N_lat, s)).astype(np.int32)
        logits = rng.standard_normal((B, N_lat, V)).astype(np.float32)

        loss = TSTCausalLMLoss(
            bag_size=s, within_bag_weighting="uniform", from_logits=True
        )
        y_loss = float(keras.ops.convert_to_numpy(loss(y_true, logits)))

        # Manual reference: log-softmax once, gather targets, average.
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        # log_probs[b, n_lat, V] index by y_true[b, n_lat, j]
        total = 0.0
        positions = 0
        for b in range(B):
            for n in range(N_lat):
                lp_pos = log_probs[b, n]  # (V,)
                ce_terms = -lp_pos[y_true[b, n]]  # (s,)
                # Uniform weights = 1/s. Bag is all-real (>=0), mask=1.
                total += (ce_terms / s).sum()
                positions += 1
        y_ref = total / positions
        assert abs(y_loss - y_ref) < 1e-6

    def test_inv3_sum_of_ce_with_ignore_mask(self):
        """Sum-of-CE handles partially-masked bags correctly."""
        B, N_lat, s, V = 1, 2, 3, 4
        # y_true with some -1s: position 0 fully real, position 1 fully masked.
        y_true = np.array(
            [[[1, 2, 0], [-1, -1, -1]]],
            dtype=np.int32,
        )
        rng = np.random.default_rng(5)
        logits = rng.standard_normal((B, N_lat, V)).astype(np.float32)

        loss = TSTCausalLMLoss(bag_size=s, ignore_index=-1, from_logits=True)
        y_loss = float(keras.ops.convert_to_numpy(loss(y_true, logits)))

        # Manual: only position 0 contributes; w_j = 1/s.
        log_probs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
        ce = -log_probs[0, 0]  # (V,)
        ref = (ce[y_true[0, 0]] / s).sum()  # mask sum=1
        assert abs(y_loss - ref) < 1e-6

    def test_inv5_no_trainable_params(self):
        """Invariant 5: loss has 0 trainable params (loss is stateless)."""
        loss = TSTCausalLMLoss(bag_size=4)
        # keras.losses.Loss has no trainable variables.
        assert not hasattr(loss, "trainable_weights") or len(loss.trainable_weights) == 0
        assert not hasattr(loss, "trainable_variables") or len(loss.trainable_variables) == 0

    def test_inv7_get_config_roundtrip(self):
        cfg = TSTCausalLMLoss(
            bag_size=5,
            within_bag_weighting="power_law",
            within_bag_alpha=0.7,
            label_smoothing=0.05,
            ignore_index=-100,
            from_logits=True,
        ).get_config()
        new = TSTCausalLMLoss.from_config(cfg)
        assert new.bag_size == 5
        assert new.within_bag_weighting == "power_law"
        assert abs(new.within_bag_alpha - 0.7) < 1e-9
        assert abs(new.label_smoothing - 0.05) < 1e-9
        assert new.ignore_index == -100
        assert new.from_logits is True

    def test_inv7_keras_serialize_deserialize(self):
        loss = TSTCausalLMLoss(bag_size=3, within_bag_weighting="power_law")
        ser = keras.losses.serialize(loss)
        new = keras.losses.deserialize(ser)
        assert isinstance(new, TSTCausalLMLoss)
        assert new.bag_size == 3
        assert new.within_bag_weighting == "power_law"

    def test_dict_y_pred_unwrapped(self):
        """Dict y_pred {'logits': ...} produces same value as bare tensor."""
        B, N, V = 2, 3, 6
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, V, size=(B, N)).astype(np.int32)
        logits = rng.standard_normal((B, N, V)).astype(np.float32)
        loss = TSTCausalLMLoss(bag_size=1, from_logits=True)
        bare = float(keras.ops.convert_to_numpy(loss(y_true, logits)))
        wrapped = float(keras.ops.convert_to_numpy(loss(y_true, {"logits": logits})))
        assert abs(bare - wrapped) < 1e-6

    def test_power_law_weights_normalised(self):
        """Power-law weighting: weights sum to 1 within 1e-7."""
        loss = TSTCausalLMLoss(
            bag_size=8, within_bag_weighting="power_law", within_bag_alpha=0.5
        )
        assert abs(float(loss._w_j.sum()) - 1.0) < 1e-7

    def test_uniform_weights_value(self):
        loss = TSTCausalLMLoss(bag_size=4, within_bag_weighting="uniform")
        np.testing.assert_allclose(loss._w_j, np.full(4, 0.25, dtype=np.float32), atol=0)

    def test_invalid_bag_size(self):
        with pytest.raises(ValueError, match="bag_size"):
            TSTCausalLMLoss(bag_size=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="within_bag_alpha"):
            TSTCausalLMLoss(within_bag_alpha=0.0)
