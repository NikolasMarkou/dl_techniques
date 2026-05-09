"""Tests for LongTermMemoryBank and WorkingMemoryBank."""

import numpy as np
import pytest
import keras

from dl_techniques.models.memory_bank.memory_banks import (
    LongTermMemoryBank,
    WorkingMemoryBank,
)


# ---------------------------------------------------------------------
# LongTermMemoryBank
# ---------------------------------------------------------------------


class TestLongTermMemoryBank:

    def test_initialization_and_build(self):
        bank = LongTermMemoryBank(s_lt=128, d_k=64, d_v=128)
        bank.build()
        assert bank.K_lt.shape == (128, 64)
        assert bank.V_lt.shape == (128, 128)
        # Names carry `memory_` prefix for optimizer split.
        assert "memory_" in bank.K_lt.name
        assert "memory_" in bank.V_lt.name

    def test_invalid_s_lt(self):
        with pytest.raises(ValueError, match="s_lt"):
            LongTermMemoryBank(s_lt=0, d_k=64, d_v=128)

    def test_invalid_d_k(self):
        with pytest.raises(ValueError, match="d_k"):
            LongTermMemoryBank(s_lt=64, d_k=0, d_v=128)

    def test_d_k_equals_d_v_rejected(self):
        with pytest.raises(ValueError, match="d_k must differ from d_v"):
            LongTermMemoryBank(s_lt=64, d_k=64, d_v=64)

    def test_call_returns_keys_and_values(self):
        bank = LongTermMemoryBank(s_lt=32, d_k=16, d_v=32)
        bank.build()
        k, v = bank(None)
        assert k.shape == (32, 16)
        assert v.shape == (32, 32)

    def test_assign_keys_from_kmeans(self):
        bank = LongTermMemoryBank(s_lt=8, d_k=4, d_v=16)
        bank.build()
        before = np.asarray(bank.K_lt).copy()
        centroids = np.random.RandomState(0).randn(8, 4).astype(np.float32)
        bank.assign_keys_from_kmeans(centroids)
        after = np.asarray(bank.K_lt)
        assert not np.allclose(before, after)
        np.testing.assert_allclose(after, centroids, atol=1e-6)

    def test_assign_keys_shape_mismatch(self):
        bank = LongTermMemoryBank(s_lt=8, d_k=4, d_v=16)
        bank.build()
        with pytest.raises(ValueError, match="shape"):
            bank.assign_keys_from_kmeans(np.zeros((9, 4), dtype=np.float32))

    def test_get_config_round_trip(self):
        bank = LongTermMemoryBank(
            s_lt=8, d_k=4, d_v=16, initializer_range=0.05,
        )
        cfg = bank.get_config()
        clone = LongTermMemoryBank.from_config(cfg)
        assert clone.s_lt == 8
        assert clone.d_k == 4
        assert clone.d_v == 16
        assert clone.initializer_range == 0.05


# ---------------------------------------------------------------------
# WorkingMemoryBank
# ---------------------------------------------------------------------


class TestWorkingMemoryBank:

    def test_initialization_and_forward(self):
        wm = WorkingMemoryBank(d_k=8, d_v=16, embed_dim=32)
        x = keras.ops.convert_to_tensor(np.random.randn(2, 5, 32).astype(np.float32))
        k, v = wm(x)
        assert k.shape == (2, 5, 8)
        assert v.shape == (2, 5, 16)

    def test_no_bias_on_K(self):
        wm = WorkingMemoryBank(d_k=8, d_v=16, embed_dim=32)
        wm(np.zeros((1, 1, 32), dtype=np.float32))
        # W_K Dense has use_bias=False, so only one weight (the kernel).
        assert len(wm.W_K.weights) == 1
        # W_V has bias.
        assert len(wm.W_V.weights) == 2

    def test_d_v_must_be_less_than_embed_dim(self):
        with pytest.raises(ValueError, match="d_v"):
            WorkingMemoryBank(d_k=8, d_v=32, embed_dim=32)

    def test_d_k_equals_d_v_rejected(self):
        with pytest.raises(ValueError, match="d_k"):
            WorkingMemoryBank(d_k=16, d_v=16, embed_dim=64)

    def test_get_config_round_trip(self):
        wm = WorkingMemoryBank(
            d_k=8, d_v=16, embed_dim=64, initializer_range=0.03,
        )
        cfg = wm.get_config()
        clone = WorkingMemoryBank.from_config(cfg)
        assert clone.d_k == 8
        assert clone.d_v == 16
        assert clone.embed_dim == 64
        assert clone.initializer_range == 0.03

    def test_compute_output_shape(self):
        wm = WorkingMemoryBank(d_k=8, d_v=16, embed_dim=32)
        out = wm.compute_output_shape((None, 7, 32))
        assert out == ((None, 7, 8), (None, 7, 16))


# ---------------------------------------------------------------------
# O4: multi_head_keys variant (per-head K/V)
# ---------------------------------------------------------------------


class TestMultiHeadKeysVariant:
    def test_lt_bank_mha_shapes(self):
        from dl_techniques.models.memory_bank.memory_banks import (
            LongTermMemoryBank,
        )
        bank = LongTermMemoryBank(
            s_lt=8, d_k=4, d_v=16, num_heads=4, multi_head_keys=True,
        )
        bank.build()
        assert tuple(bank.K_lt.shape) == (8, 4, 4)
        assert tuple(bank.V_lt.shape) == (8, 4, 16)

    def test_lt_bank_assign_keys_from_kmeans_mha(self):
        import numpy as np
        from dl_techniques.models.memory_bank.memory_banks import (
            LongTermMemoryBank,
        )
        bank = LongTermMemoryBank(
            s_lt=8, d_k=4, d_v=16, num_heads=4, multi_head_keys=True,
        )
        bank.build()
        # Pass MQA-shape centroids; bank should tile across heads.
        centroids = np.random.randn(8, 4).astype(np.float32)
        bank.assign_keys_from_kmeans(centroids)
        klt = np.asarray(bank.K_lt)
        for h in range(4):
            np.testing.assert_allclose(klt[:, h, :], centroids)

    def test_wm_bank_mha_shapes(self):
        import numpy as np
        from dl_techniques.models.memory_bank.memory_banks import (
            WorkingMemoryBank,
        )
        bank = WorkingMemoryBank(
            d_k=4, d_v=16, embed_dim=32, num_heads=4, multi_head_keys=True,
        )
        x = np.random.randn(2, 7, 32).astype(np.float32)
        k, v = bank(x)
        assert tuple(k.shape) == (2, 7, 4, 4)
        assert tuple(v.shape) == (2, 7, 4, 16)

    def test_lt_bank_round_trip_config_mha(self):
        from dl_techniques.models.memory_bank.memory_banks import (
            LongTermMemoryBank,
        )
        bank = LongTermMemoryBank(
            s_lt=8, d_k=4, d_v=16, num_heads=3, multi_head_keys=True,
        )
        cfg = bank.get_config()
        clone = LongTermMemoryBank.from_config(cfg)
        assert clone.num_heads == 3
        assert clone.multi_head_keys is True
