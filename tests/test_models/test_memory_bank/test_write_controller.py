"""Tests for MemoryWriteController."""

import numpy as np
import pytest
import keras
import tensorflow as tf
from keras import ops

from dl_techniques.models.memory_bank.write_controller import (
    MemoryWriteController,
)


class TestMemoryWriteController:

    def test_forward_shapes_full_length(self):
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=12,
        )
        x = np.random.randn(2, 12, 32).astype(np.float32)
        k, v, mask = ctrl(x)
        assert tuple(k.shape) == (2, 12, 8)
        assert tuple(v.shape) == (2, 12, 16)
        assert tuple(mask.shape) == (2, 12)
        # All positions valid.
        np.testing.assert_allclose(np.asarray(mask), np.ones((2, 12)))

    def test_forward_shapes_with_padding(self):
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=16,
        )
        x = np.random.randn(2, 5, 32).astype(np.float32)
        k, v, mask = ctrl(x)
        assert tuple(k.shape) == (2, 16, 8)
        assert tuple(v.shape) == (2, 16, 16)
        assert tuple(mask.shape) == (2, 16)
        m = np.asarray(mask)
        # First 5 positions valid, remainder padding.
        np.testing.assert_allclose(m[:, :5], np.ones((2, 5)))
        np.testing.assert_allclose(m[:, 5:], np.zeros((2, 11)))

    def test_padded_keys_are_zero(self):
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=10,
        )
        x = np.random.randn(1, 4, 32).astype(np.float32)
        k, v, _ = ctrl(x)
        k = np.asarray(k)
        v = np.asarray(v)
        np.testing.assert_allclose(k[:, 4:, :], 0.0)
        np.testing.assert_allclose(v[:, 4:, :], 0.0)

    def test_get_config_round_trip(self):
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=20,
            initializer_range=0.05,
        )
        cfg = ctrl.get_config()
        clone = MemoryWriteController.from_config(cfg)
        assert clone.d_k == 8
        assert clone.d_v == 16
        assert clone.embed_dim == 32
        assert clone.max_seq_len == 20
        assert clone.initializer_range == 0.05

    def test_compute_output_shape(self):
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=20,
        )
        out = ctrl.compute_output_shape((None, 5, 32))
        assert out == ((None, 20, 8), (None, 20, 16), (None, 20))

    def test_assert_t_too_large_raises(self):
        """B6: T > max_seq_len must raise (was: silent zero-shape pad)."""
        ctrl = MemoryWriteController(
            d_k=8, d_v=16, embed_dim=32, max_seq_len=10,
        )
        x = np.random.randn(1, 12, 32).astype(np.float32)
        with pytest.raises((tf.errors.InvalidArgumentError, ValueError)):
            ctrl(x)
