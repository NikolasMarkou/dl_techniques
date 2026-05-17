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
    TSTPhaseCallback,
    tst_phase1_transform,
    tst_phase2_transform,
    apply_tst,
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


# =====================================================================
# Step 5 — TSTPhaseCallback + two phase transforms + apply_tst (D-007)
#
# Regression canaries: the phase1/phase2 transforms produce
# statically-determined ranks INDEPENDENT of state.phase_active. The
# previously-failing mid-iteration toggle test (Falsification C soft
# trigger) is replaced by these invariants — D-006 / D-007.
# =====================================================================


def _make_synthetic_ntp_ds(
    batch_size: int = 4,
    seq_len: int = 24,
    vocab: int = 64,
    num_batches: int = 3,
    seed: int = 0,
) -> tf.data.Dataset:
    """Synthetic ``(input_ids, labels)`` dataset matching
    ``preprocess_clm_packed_dataset`` shape contract."""
    rng = np.random.default_rng(seed)
    inputs = rng.integers(0, vocab, size=(num_batches, batch_size, seq_len)).astype(np.int32)
    labels = rng.integers(0, vocab, size=(num_batches, batch_size, seq_len)).astype(np.int32)
    return tf.data.Dataset.from_tensor_slices((inputs, labels))


class TestTSTPhase1Transform:
    """Regression canary — phase1_transform ALWAYS yields rank-3 / rank-3."""

    def test_yields_rank3_inputs_and_labels(self):
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=24)
        bag = 6
        out = tst_phase1_transform(ds, bag_size=bag)
        for inp, lab in out.take(3):
            assert inp.shape.rank == 3, f"Phase1 inputs must be rank-3, got {inp.shape}"
            assert lab.shape.rank == 3, f"Phase1 labels must be rank-3, got {lab.shape}"
            assert int(inp.shape[-1]) == bag
            assert int(lab.shape[-1]) == bag
            assert int(inp.shape[-2]) == 24 // bag
            assert int(lab.shape[-2]) == 24 // bag

    def test_rank_is_independent_of_state_phase_active(self):
        """Canary for D-007: even if a TSTState exists and is toggled, the
        phase1 transform's output rank is determined entirely by which
        function the user called — NOT by any tf.Variable read."""
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=24)
        state = TSTState(bag_size=6, phase_active_init=True)
        out = tst_phase1_transform(ds, bag_size=6)
        # Toggle the state mid-iteration — must not affect output rank.
        it = iter(out)
        first_inp, first_lab = next(it)
        assert first_inp.shape.rank == 3
        state.phase_active.assign(False)
        second_inp, second_lab = next(it)
        assert second_inp.shape.rank == 3
        assert second_lab.shape.rank == 3
        state.phase_active.assign(True)
        third_inp, third_lab = next(it)
        assert third_inp.shape.rank == 3

    def test_validator_n_not_divisible_by_bag(self):
        """User Rule R1: raise loudly when N % bag_size != 0."""
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=25)  # 25 % 6 != 0
        with pytest.raises(ValueError, match="bag_size"):
            tst_phase1_transform(ds, bag_size=6)

    def test_validator_bag_size_zero(self):
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=24)
        with pytest.raises(ValueError, match="bag_size"):
            tst_phase1_transform(ds, bag_size=0)

    def test_rejects_non_pair_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(np.zeros((3, 4, 24), dtype=np.int32))
        with pytest.raises(ValueError, match="input_ids, labels"):
            tst_phase1_transform(ds, bag_size=6)


class TestTSTPhase2Transform:
    """Regression canary — phase2_transform ALWAYS yields rank-2 / rank-2."""

    def test_yields_rank2_inputs_and_labels(self):
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=24)
        out = tst_phase2_transform(ds)
        for inp, lab in out.take(3):
            assert inp.shape.rank == 2, f"Phase2 inputs must be rank-2, got {inp.shape}"
            assert lab.shape.rank == 2, f"Phase2 labels must be rank-2, got {lab.shape}"

    def test_rank_is_independent_of_state_phase_active(self):
        """Canary for D-007 (mirror of TestTSTPhase1Transform's rank canary)."""
        ds = _make_synthetic_ntp_ds(batch_size=4, seq_len=24)
        state = TSTState(bag_size=6, phase_active_init=False)
        out = tst_phase2_transform(ds)
        it = iter(out)
        a_inp, a_lab = next(it)
        assert a_inp.shape.rank == 2 and a_lab.shape.rank == 2
        state.phase_active.assign(True)
        b_inp, b_lab = next(it)
        assert b_inp.shape.rank == 2 and b_lab.shape.rank == 2

    def test_rejects_non_pair_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(np.zeros((3, 4, 24), dtype=np.int32))
        with pytest.raises(ValueError, match="input_ids, labels"):
            tst_phase2_transform(ds)


class TestTSTPhaseCallback:
    def test_flip_step_computation(self):
        state = TSTState(bag_size=6)
        cb = TSTPhaseCallback(state, total_steps=100, phase1_step_ratio=0.25)
        assert cb.flip_step == 25
        assert cb.already_flipped is False

    def test_flips_at_boundary(self):
        state = TSTState(bag_size=6, phase_active_init=True)
        cb = TSTPhaseCallback(state, total_steps=10, phase1_step_ratio=0.5)
        assert cb.flip_step == 5
        # Simulate 5 batches: at end of batch 5, global_step becomes 5 → flips.
        for i in range(5):
            cb.on_train_batch_end(batch=i)
        assert int(state.global_step.numpy()) == 5
        assert bool(state.phase_active.numpy()) is False
        assert cb.already_flipped is True

    def test_does_not_flip_before_boundary(self):
        state = TSTState(bag_size=6, phase_active_init=True)
        cb = TSTPhaseCallback(state, total_steps=10, phase1_step_ratio=0.5)
        for i in range(4):
            cb.on_train_batch_end(batch=i)
        assert bool(state.phase_active.numpy()) is True
        assert cb.already_flipped is False

    def test_flip_is_idempotent_across_many_batches(self):
        """Safe to attach to BOTH phase fits (D-007)."""
        state = TSTState(bag_size=6, phase_active_init=True)
        cb = TSTPhaseCallback(state, total_steps=10, phase1_step_ratio=0.5)
        for i in range(20):
            cb.on_train_batch_end(batch=i)
        assert cb.already_flipped is True
        assert bool(state.phase_active.numpy()) is False
        assert int(state.global_step.numpy()) == 20

    def test_validates_inputs(self):
        state = TSTState(bag_size=6)
        with pytest.raises(ValueError, match="TSTState"):
            TSTPhaseCallback(None, total_steps=10, phase1_step_ratio=0.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="total_steps"):
            TSTPhaseCallback(state, total_steps=0, phase1_step_ratio=0.5)
        with pytest.raises(ValueError, match="phase1_step_ratio"):
            TSTPhaseCallback(state, total_steps=10, phase1_step_ratio=1.5)


class TestApplyTST:
    def test_returns_four_tuple(self):
        """D-007: apply_tst returns (state, callbacks, phase1_fn, phase2_fn)."""
        cfg = TSTConfig(bag_size=4, phase1_step_ratio=0.5)
        result = apply_tst(cfg, total_steps=20)
        assert len(result) == 4
        state, callbacks, phase1_fn, phase2_fn = result
        assert isinstance(state, TSTState)
        assert isinstance(callbacks, list) and len(callbacks) == 1
        assert isinstance(callbacks[0], TSTPhaseCallback)
        assert callable(phase1_fn)
        assert callable(phase2_fn)

    def test_phase_fns_produce_correct_ranks(self):
        cfg = TSTConfig(bag_size=4, phase1_step_ratio=0.5)
        _state, _cbs, phase1_fn, phase2_fn = apply_tst(cfg, total_steps=20)
        ds = _make_synthetic_ntp_ds(batch_size=2, seq_len=16)
        p1 = phase1_fn(ds)
        p2 = phase2_fn(ds)
        for inp, lab in p1.take(1):
            assert inp.shape.rank == 3 and lab.shape.rank == 3
            assert int(inp.shape[-1]) == 4
        for inp, lab in p2.take(1):
            assert inp.shape.rank == 2 and lab.shape.rank == 2

    def test_callback_flip_step_matches_config(self):
        cfg = TSTConfig(bag_size=4, phase1_step_ratio=0.3)
        _state, callbacks, _p1, _p2 = apply_tst(cfg, total_steps=100)
        assert callbacks[0].flip_step == 30

    def test_invalid_total_steps(self):
        cfg = TSTConfig(bag_size=4)
        with pytest.raises(ValueError, match="total_steps"):
            apply_tst(cfg, total_steps=0)


class TestTwoPhaseRankComposesWithLayerAndLoss:
    """End-to-end rank wiring: phase1 ds → TSTEmbedding rank-3 → loss rank-3;
    phase2 ds → TSTEmbedding rank-2 → loss rank-2. No mid-graph flip.
    """

    def test_phase1_rank_flows_through_layer_and_loss(self):
        ds = _make_synthetic_ntp_ds(batch_size=2, seq_len=12, vocab=32, num_batches=1, seed=1)
        p1 = tst_phase1_transform(ds, bag_size=4)
        emb = TSTEmbedding(vocab_size=32, output_dim=8, bag_size=4)
        loss_fn = TSTCausalLMLoss(bag_size=4, ignore_index=-1, from_logits=True)
        for inp, lab in p1.take(1):
            assert inp.shape.rank == 3
            e = emb(inp)
            assert e.shape.rank == 3  # (B, N_lat, d)
            # Fake logits at the latent positions.
            logits = keras.random.normal((int(e.shape[0]), int(e.shape[1]), 32))
            value = float(loss_fn(lab, logits))
            assert np.isfinite(value)

    def test_phase2_rank_flows_through_layer_and_loss(self):
        ds = _make_synthetic_ntp_ds(batch_size=2, seq_len=12, vocab=32, num_batches=1, seed=2)
        p2 = tst_phase2_transform(ds)
        emb = TSTEmbedding(vocab_size=32, output_dim=8, bag_size=4)
        loss_fn = TSTCausalLMLoss(bag_size=4, ignore_index=-1, from_logits=True)
        for inp, lab in p2.take(1):
            assert inp.shape.rank == 2
            e = emb(inp)
            assert e.shape.rank == 3  # (B, N, d) — Embedding adds the d axis
            logits = keras.random.normal((int(e.shape[0]), int(e.shape[1]), 32))
            value = float(loss_fn(lab, logits))
            assert np.isfinite(value)
