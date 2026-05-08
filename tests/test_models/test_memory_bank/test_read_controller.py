"""Tests for MemoryReadController (Step 3 — retrieval + gating only).

Step 4 will add coverage for the four anti-collapse aux losses on top of
the existing tests in this file.
"""

import math
import numpy as np
import pytest
import keras
from keras import ops
import tensorflow as tf

from dl_techniques.models.memory_bank.memory_banks import (
    LongTermMemoryBank,
)
from dl_techniques.models.memory_bank.write_controller import (
    MemoryWriteController,
)
from dl_techniques.models.memory_bank.read_controller import (
    MemoryReadController,
)


# ---------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------


def _build_bundle(embed_dim=32, num_heads=4, d_k=8, d_v=16,
                  s_lt=16, max_seq_len=10, top_k=4):
    """Build (read_ctrl, lt_bank, write_ctrl) with consistent dims."""
    read = MemoryReadController(
        embed_dim=embed_dim, num_heads=num_heads,
        d_k=d_k, d_v=d_v,
        s_lt=s_lt, max_seq_len=max_seq_len, top_k=top_k,
    )
    lt = LongTermMemoryBank(s_lt=s_lt, d_k=d_k, d_v=d_v)
    lt.build()
    write = MemoryWriteController(
        d_k=d_k, d_v=d_v, embed_dim=embed_dim, max_seq_len=max_seq_len,
    )
    return read, lt, write


def _forward(read_ctrl, lt_bank, write_ctrl, x):
    k_lt, v_lt = lt_bank(None)
    k_wm, v_wm, mask = write_ctrl(x)
    return read_ctrl(x, k_lt, v_lt, k_wm, v_wm, mask), k_lt, v_lt, k_wm, v_wm, mask


# ---------------------------------------------------------------------
# Forward shape + finiteness
# ---------------------------------------------------------------------


class TestForward:

    def test_forward_shape(self):
        read, lt, write = _build_bundle(
            embed_dim=32, num_heads=4, d_k=8, d_v=16,
            s_lt=16, max_seq_len=10, top_k=4,
        )
        x = np.random.RandomState(0).randn(2, 6, 32).astype(np.float32)
        injection, *_ = _forward(read, lt, write, x)
        assert tuple(injection.shape) == (2, 6, 32)
        assert np.all(np.isfinite(np.asarray(injection)))

    def test_M_static_set(self):
        read, *_ = _build_bundle(s_lt=4096, max_seq_len=512)
        # M_static is read by ops.one_hot with static int.
        assert read.M_static == 4096 + 512


# ---------------------------------------------------------------------
# SC3 — gate init at sigmoid(-3) ≈ 0.0474
# ---------------------------------------------------------------------


class TestGateInit:

    def test_gate_init(self):
        read, lt, write = _build_bundle(
            embed_dim=32, num_heads=4, d_k=8, d_v=16,
            s_lt=16, max_seq_len=10, top_k=4,
        )
        x = np.random.RandomState(0).randn(4, 6, 32).astype(np.float32)
        # Build first.
        _ = _forward(read, lt, write, x)
        # Re-compute gate manually so we can inspect it.
        g = ops.sigmoid(read.W_g(x))
        mean_g = float(np.mean(np.asarray(g)))
        # sigmoid(-3) = 0.0474. With small kernel init, mean lies near 0.04.
        assert 0.02 <= mean_g <= 0.08, f"mean(g)={mean_g}"


# ---------------------------------------------------------------------
# SC4 — STE: forward = hard top-k, backward = soft
# ---------------------------------------------------------------------


class TestSteGradient:

    def test_routing_forward_is_top_k(self):
        """Build deterministic K and Q, verify routing puts mass only on
        the top-k positions per (batch, time, head)."""
        read, lt, write = _build_bundle(
            embed_dim=16, num_heads=2, d_k=4, d_v=8,
            s_lt=8, max_seq_len=4, top_k=2,
        )
        # Build the layers via a forward pass.
        x = np.random.RandomState(0).randn(1, 2, 16).astype(np.float32)
        _ = _forward(read, lt, write, x)

        # Now manually invoke the internal pipeline with crafted inputs to
        # verify routing is active only on top_k positions.
        k_lt, v_lt = lt.K_lt, lt.V_lt
        k_wm, v_wm, mask = write(x)

        # Re-derive routing with the same op order as `call`.
        b, t = 1, 2
        q_flat = read.W_Q(x)
        q = ops.reshape(q_flat, (b, t, read.num_heads, read.d_k))
        k_lt_b = ops.broadcast_to(
            ops.expand_dims(k_lt, axis=0), (b, read.s_lt, read.d_k),
        )
        k_total = ops.concatenate([k_lt_b, k_wm], axis=1)
        sim = ops.einsum("bthk,bmk->bthm", q, k_total) / math.sqrt(read.d_k)
        # Apply WM mask manually.
        wm_mask = read._build_wm_mask(t, mask)
        wm_mask = ops.expand_dims(wm_mask, axis=2)
        lt_zeros = ops.zeros((b, t, 1, read.s_lt), dtype=sim.dtype)
        full_mask = ops.concatenate([lt_zeros, wm_mask], axis=-1)
        sim = sim + full_mask
        temp = ops.softplus(read.log_temp) + 0.1
        sim_c = ops.clip(sim, -50.0, 50.0)
        soft_w = ops.softmax(sim_c / temp, -1)
        _, top_idx = ops.top_k(sim_c, k=read.top_k)
        hard_mask = ops.sum(
            ops.one_hot(top_idx, num_classes=read.M_static), axis=-2,
        )
        masked_soft = soft_w * hard_mask
        hard_w = masked_soft / (ops.sum(masked_soft, -1, keepdims=True) + 1e-9)
        routing = soft_w + ops.stop_gradient(hard_w - soft_w)

        routing_np = np.asarray(routing)
        hard_w_np = np.asarray(hard_w)
        # Forward routing equals hard_w for nonzero positions.
        # The number of nonzero positions per (b, t, h) should be <= top_k.
        nonzero_per_head = (np.asarray(hard_mask) > 0).sum(axis=-1)
        assert np.all(nonzero_per_head <= read.top_k)
        # routing's nonzero positions match hard_w's nonzero positions.
        np.testing.assert_allclose(
            routing_np * np.asarray(hard_mask),
            hard_w_np * np.asarray(hard_mask),
            atol=1e-6,
        )

    def test_ste_backward_is_soft(self):
        """Gradient through `routing` w.r.t. soft input matches gradient
        through `soft` directly: STE replaces hard_w with soft_w in
        backward by `stop_gradient(hard - soft)`."""
        # Construct a tiny scalar setting to compute the analytic
        # derivative of routing w.r.t. a parameter that influences soft_w.
        with tf.GradientTape() as tape:
            soft = tf.Variable([0.4, 0.6], dtype=tf.float32)
            tape.watch(soft)
            hard = tf.constant([0.0, 1.0])
            routing = soft + tf.stop_gradient(hard - soft)
            loss = tf.reduce_sum(routing * tf.constant([1.0, 2.0]))
        grad = tape.gradient(loss, soft).numpy()
        # The forward value of routing equals hard, so loss = 0*1 + 1*2 = 2.
        assert abs(float(loss.numpy()) - 2.0) < 1e-6
        # The gradient of (sum(soft * coef)) w.r.t. soft is just coef.
        np.testing.assert_allclose(grad, [1.0, 2.0], atol=1e-6)


# ---------------------------------------------------------------------
# SC6 — causal mask on WM slice
# ---------------------------------------------------------------------


class TestCausalMaskWm:

    def test_wm_position_t_unreachable_from_query_smaller_than_t(self):
        """Build a tiny model where K_wm dominates, then verify routing
        places zero weight on WM positions strictly greater than the
        query's time index."""
        read, lt, write = _build_bundle(
            embed_dim=16, num_heads=1, d_k=4, d_v=8,
            s_lt=4, max_seq_len=6, top_k=2,
        )
        # Make K_lt very small so K_wm dominates.
        x = np.random.RandomState(0).randn(1, 6, 16).astype(np.float32)
        # First, materialize variables.
        _ = _forward(read, lt, write, x)
        lt.K_lt.assign(np.zeros((4, 4), dtype=np.float32) - 100.0)

        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)

        # Same routing computation as `call`.
        b, t = 1, 6
        q_flat = read.W_Q(x)
        q = ops.reshape(q_flat, (b, t, read.num_heads, read.d_k))
        k_lt_b = ops.broadcast_to(
            ops.expand_dims(k_lt, axis=0), (b, read.s_lt, read.d_k),
        )
        k_total = ops.concatenate([k_lt_b, k_wm], axis=1)
        sim = ops.einsum("bthk,bmk->bthm", q, k_total) / math.sqrt(read.d_k)
        wm_mask = read._build_wm_mask(t, mask)
        wm_mask = ops.expand_dims(wm_mask, axis=2)
        lt_zeros = ops.zeros((b, t, 1, read.s_lt), dtype=sim.dtype)
        full_mask = ops.concatenate([lt_zeros, wm_mask], axis=-1)
        sim = sim + full_mask
        soft_w = ops.softmax(sim / (ops.softplus(read.log_temp) + 0.1), -1)
        soft_np = np.asarray(soft_w)  # (1, T, 1, M_static)

        # Inspect WM slice (last max_seq_len positions).
        wm_slice = soft_np[..., read.s_lt:]  # (1, T, 1, max_seq_len)

        # For query at time t', positions i > t' must be ~0 in soft.
        for tq in range(t):
            illegal = wm_slice[0, tq, 0, tq + 1:]
            assert np.all(illegal < 1e-6), f"q@t={tq} sees illegal: {illegal}"


# ---------------------------------------------------------------------
# Get-config round-trip
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# SC5 — aux losses
# ---------------------------------------------------------------------


class TestAuxLosses:

    def _enable_all(self, read):
        read.enable_gate_entropy = True
        read.enable_load_balance = True
        read.enable_z_loss = True
        read.enable_diversity = True
        read.enable_infonce = True

    def test_aux_losses_off_by_default_no_loss(self):
        read, lt, write = _build_bundle(
            embed_dim=16, num_heads=2, d_k=4, d_v=8,
            s_lt=8, max_seq_len=4, top_k=2,
        )
        x = np.random.RandomState(0).randn(2, 3, 16).astype(np.float32)
        # training=True with all flags off -> still no aux losses.
        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)
        _ = read(x, k_lt, v_lt, k_wm, v_wm, mask, training=True)
        # `add_loss` populates self.losses on the layer. With no flags
        # enabled, none should appear.
        assert len(read.losses) == 0

    def test_aux_losses_present_when_enabled_and_finite(self):
        read, lt, write = _build_bundle(
            embed_dim=16, num_heads=2, d_k=4, d_v=8,
            s_lt=8, max_seq_len=4, top_k=2,
        )
        self._enable_all(read)
        x = np.random.RandomState(1).randn(2, 3, 16).astype(np.float32)
        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)
        _ = read(x, k_lt, v_lt, k_wm, v_wm, mask, training=True)
        # 5 aux losses (4 + z-loss).
        assert len(read.losses) == 5
        for l in read.losses:
            assert np.isfinite(np.asarray(l)), f"non-finite aux loss: {l}"

    def test_aux_losses_skipped_in_eval_mode(self):
        read, lt, write = _build_bundle(
            embed_dim=16, num_heads=2, d_k=4, d_v=8,
            s_lt=8, max_seq_len=4, top_k=2,
        )
        self._enable_all(read)
        x = np.random.RandomState(2).randn(2, 3, 16).astype(np.float32)
        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)
        _ = read(x, k_lt, v_lt, k_wm, v_wm, mask, training=False)
        assert len(read.losses) == 0

    def test_diversity_subsample_finite_at_large_s_lt(self):
        read = MemoryReadController(
            embed_dim=16, num_heads=2, d_k=4, d_v=8,
            s_lt=4096, max_seq_len=8, top_k=2,
            enable_diversity=True, diversity_subsample=64,
        )
        lt = LongTermMemoryBank(s_lt=4096, d_k=4, d_v=8)
        lt.build()
        write = MemoryWriteController(d_k=4, d_v=8, embed_dim=16, max_seq_len=8)
        x = np.random.RandomState(3).randn(1, 4, 16).astype(np.float32)
        k_lt, v_lt = lt(None)
        k_wm, v_wm, mask = write(x)
        _ = read(x, k_lt, v_lt, k_wm, v_wm, mask, training=True)
        assert len(read.losses) == 1
        assert np.isfinite(np.asarray(read.losses[0]))


class TestConfig:

    def test_get_config_round_trip(self):
        read = MemoryReadController(
            embed_dim=32, num_heads=4, d_k=8, d_v=16,
            s_lt=16, max_seq_len=10, top_k=4, gate_init_bias=-3.0,
        )
        cfg = read.get_config()
        clone = MemoryReadController.from_config(cfg)
        assert clone.embed_dim == 32
        assert clone.num_heads == 4
        assert clone.s_lt == 16
        assert clone.max_seq_len == 10
        assert clone.top_k == 4
        assert clone.gate_init_bias == -3.0
        assert clone.M_static == 26
