"""Tests for WaveFieldMemoryLLM (model class, custom train_step, warmup, save/load)."""

import os
import tempfile

import numpy as np
import pytest
import keras
import tensorflow as tf
from keras import ops

from dl_techniques.losses import MaskedCausalLMLoss
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
    split_trainable_by_prefix,
    memory_llm_custom_objects,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _tiny_kwargs():
    return dict(
        vocab_size=128,
        embed_dim=32,
        depth=4,
        num_heads=4,
        max_seq_len=16,
        field_size=32,
        d_k=8,
        d_v=16,
        s_lt=32,
        top_k=4,
        diversity_subsample=8,
        infonce_negatives=8,
    )


def _build_tiny() -> WaveFieldMemoryLLM:
    m = WaveFieldMemoryLLM(**_tiny_kwargs())
    dummy = np.random.randint(0, 128, size=(1, 16)).astype(np.int32)
    m(dummy, training=False)
    return m


# ---------------------------------------------------------------------
# Construction & forward
# ---------------------------------------------------------------------


class TestConstruction:

    def test_factory_tiny(self):
        m = WaveFieldMemoryLLM.from_variant(
            "tiny", vocab_size=128, max_seq_len=16,
            d_k=8, d_v=16, s_lt=32, top_k=4,
            diversity_subsample=8, infonce_negatives=8,
        )
        assert m.depth == 4
        assert m.L_write == 1
        assert m.L_read == 2
        assert m.L_write < m.L_read < m.depth

    def test_invalid_topology(self):
        # depth=2 -> L_write=max(1,0)=1, L_read=max(2,1)=2 == depth -> fails.
        with pytest.raises(ValueError, match="tap topology"):
            WaveFieldMemoryLLM(
                vocab_size=64, embed_dim=16, depth=2, num_heads=2,
                max_seq_len=8, d_k=4, d_v=8, s_lt=16,
            )

    def test_d_v_constraint(self):
        with pytest.raises(ValueError, match="d_v"):
            WaveFieldMemoryLLM(
                vocab_size=64, embed_dim=16, depth=4, num_heads=2,
                max_seq_len=8, d_k=4, d_v=16, s_lt=16,
            )


class TestForwardShape:

    def test_forward_shape(self):
        m = _build_tiny()
        x = np.random.randint(0, 128, size=(2, 16)).astype(np.int32)
        out = m(x, training=False)
        assert "logits" in out
        assert "last_hidden_state" in out
        assert tuple(out["logits"].shape) == (2, 16, 128)
        assert tuple(out["last_hidden_state"].shape) == (2, 16, 32)
        assert np.all(np.isfinite(np.asarray(out["logits"])))


# ---------------------------------------------------------------------
# Variable split (memory vs backbone)
# ---------------------------------------------------------------------


class TestVariableSplit:

    def test_split_partitions_all(self):
        m = _build_tiny()
        memory_vars, backbone_vars = split_trainable_by_prefix(m.trainable_variables)
        assert len(memory_vars) > 0
        assert len(backbone_vars) > 0
        # Every variable accounted for.
        assert (
            len(memory_vars) + len(backbone_vars)
            == len(m.trainable_variables)
        )
        # No overlap.
        m_ids = {id(v) for v in memory_vars}
        b_ids = {id(v) for v in backbone_vars}
        assert m_ids.isdisjoint(b_ids)

    def test_memory_vars_have_memory_or_gate_prefix(self):
        m = _build_tiny()
        memory_vars, _ = split_trainable_by_prefix(m.trainable_variables)
        for v in memory_vars:
            assert "memory_" in v.name or "gate_" in v.name


# ---------------------------------------------------------------------
# Custom train_step
# ---------------------------------------------------------------------


class TestCustomTrainStep:

    def test_custom_train_step_runs(self):
        m = _build_tiny()
        loss_fn = MaskedCausalLMLoss()
        m.compile(
            backbone_optimizer=keras.optimizers.AdamW(
                learning_rate=1e-5, weight_decay=0.01, clipnorm=1.0,
            ),
            memory_optimizer=keras.optimizers.AdamW(
                learning_rate=3e-4, weight_decay=0.01, clipnorm=1.0,
            ),
            loss={"logits": loss_fn},
        )
        # Wire output_names so dict-keyed loss/metrics work on the
        # subclassed model (LESSONS — required for dict-keyed compile).
        m.output_names = ["logits"]

        x = np.random.randint(0, 128, size=(2, 16)).astype(np.int32)
        y = np.random.randint(0, 128, size=(2, 16)).astype(np.int32)

        ds = tf.data.Dataset.from_tensor_slices(
            (x, {"logits": y}),
        ).batch(2)

        before = int(m._global_step.numpy())
        history = m.fit(ds, epochs=1, verbose=0)
        after = int(m._global_step.numpy())

        # Global step incremented at least once.
        assert after > before
        # Loss recorded.
        assert "loss" in history.history


# ---------------------------------------------------------------------
# warmup_memory_keys
# ---------------------------------------------------------------------


class TestWarmupMemoryKeys:

    def test_warmup_updates_K_lt(self):
        m = _build_tiny()
        before = np.asarray(m.lt_memory.K_lt).copy()

        # Build a tiny dataset of (input_ids, _label).
        x = np.random.randint(0, 128, size=(8, 16)).astype(np.int32)
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2)

        m.warmup_memory_keys(ds, num_batches=4)
        after = np.asarray(m.lt_memory.K_lt)

        assert not np.allclose(before, after)
        # current_phase restored to its prior value (1 at init).
        assert int(m.current_phase.numpy()) == 1


# ---------------------------------------------------------------------
# Save/Load round-trip
# ---------------------------------------------------------------------


class TestSaveLoadRoundTrip:

    def test_save_load_preserves_forward(self, tmp_path):
        m = _build_tiny()
        # Exercise gate / phase: bump current_phase to 2 and re-save.
        m.current_phase.assign(2)
        m._global_step.assign(7)

        x = np.random.randint(0, 128, size=(2, 16)).astype(np.int32)
        before = np.asarray(m(x, training=False)["logits"])

        path = str(tmp_path / "m.keras")
        m.save(path)

        loaded = keras.models.load_model(
            path, custom_objects=memory_llm_custom_objects(),
        )
        after = np.asarray(loaded(x, training=False)["logits"])
        np.testing.assert_allclose(before, after, atol=1e-5, rtol=1e-5)

        # Phase + global_step preserved.
        assert int(loaded.current_phase.numpy()) == 2
        assert int(loaded._global_step.numpy()) == 7
