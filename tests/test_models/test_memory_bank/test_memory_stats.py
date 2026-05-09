"""Tests for MemoryStats callback (O2)."""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.models.memory_bank.memory_stats import MemoryStats
from dl_techniques.models.memory_bank.wave_field_memory_llm import (
    WaveFieldMemoryLLM,
)
from dl_techniques.losses import MaskedCausalLMLoss


def _build_tiny() -> WaveFieldMemoryLLM:
    m = WaveFieldMemoryLLM(
        vocab_size=128, embed_dim=32, depth=4, num_heads=4,
        max_seq_len=16, field_size=32, d_k=8, d_v=16, s_lt=32, top_k=4,
        diversity_subsample=8, infonce_negatives=8,
    )
    dummy = np.random.randint(0, 128, size=(1, 16)).astype(np.int32)
    m(dummy, training=False)
    return m


class TestMemoryStatsConstruction:

    def test_construction_defaults(self):
        cb = MemoryStats(log_every=10)
        assert cb.log_every == 10
        assert cb.probe_dataset is None
        assert cb.probe_batches == 4

    def test_invalid_log_every_raises(self):
        with pytest.raises(ValueError, match="log_every"):
            MemoryStats(log_every=0)


class TestMemoryStatsRunsWithoutError:
    """The callback must not raise during normal training."""

    def _make_dataset(self):
        x = np.random.randint(0, 128, size=(8, 16)).astype(np.int32)
        y = np.random.randint(0, 128, size=(8, 16)).astype(np.int32)
        return tf.data.Dataset.from_tensor_slices(
            (x, {"logits": y}),
        ).batch(2)

    def _make_probe(self):
        x = np.random.randint(0, 128, size=(4, 16)).astype(np.int32)
        return tf.data.Dataset.from_tensor_slices(x).batch(2)

    def test_runs_during_fit_with_probe(self):
        m = _build_tiny()
        m.compile(
            backbone_optimizer=keras.optimizers.AdamW(1e-5),
            memory_optimizer=keras.optimizers.AdamW(3e-4),
            loss={"logits": MaskedCausalLMLoss()},
        )
        m.output_names = ["logits"]
        cb = MemoryStats(
            log_every=1,  # log every batch
            probe_dataset=self._make_probe(),
            probe_batches=2,
            svd_subsample=16,
        )
        m.fit(
            self._make_dataset(),
            epochs=1, verbose=0, callbacks=[cb],
        )
        # Callback successfully observed at least one batch (no
        # exceptions raised; fit completes).

    def test_runs_during_fit_without_probe(self):
        """Even without a probe_dataset, the structural stats path must
        not raise — only key utilization is empty (no hit accumulation
        happened) and effective rank is computable from V_lt alone."""
        m = _build_tiny()
        m.compile(
            backbone_optimizer=keras.optimizers.AdamW(1e-5),
            memory_optimizer=keras.optimizers.AdamW(3e-4),
            loss={"logits": MaskedCausalLMLoss()},
        )
        m.output_names = ["logits"]
        cb = MemoryStats(
            log_every=1, probe_dataset=None, svd_subsample=16,
        )
        m.fit(
            self._make_dataset(),
            epochs=1, verbose=0, callbacks=[cb],
        )


class TestMemoryStatsReexport:
    """O2/O9: MemoryStats is re-exported from the package __init__."""

    def test_top_level_import(self):
        from dl_techniques.models.memory_bank import MemoryStats as _MS
        assert _MS is MemoryStats
