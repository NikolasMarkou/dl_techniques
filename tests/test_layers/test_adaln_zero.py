"""Tests for AdaLNZeroConditionalBlock.

Covers:
- Forward pass with correct shapes.
- Identity-at-init property (zero-initialized AdaLN final linear → block is identity).
- Serialization round-trip via keras.saving save/load.
- Causal mask enforcement (token at position t must not depend on tokens > t).
"""

import os
import tempfile

import keras
import numpy as np
import pytest
from keras import ops

from dl_techniques.layers.adaln_zero import AdaLNZeroConditionalBlock


DIM = 16
HEADS = 2
DIM_HEAD = 8
MLP_DIM = 32
T = 4
B = 2


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def xc(rng):
    x = rng.standard_normal((B, T, DIM)).astype("float32")
    c = rng.standard_normal((B, T, DIM)).astype("float32")
    return x, c


class TestAdaLNZero:
    def test_forward_shape(self, xc):
        x, c = xc
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM
        )
        y = block([x, c], training=False)
        assert tuple(y.shape) == (B, T, DIM)

    def test_identity_at_init(self, xc):
        """At init, AdaLN-zero is identity: y == x (within fp32 tol)."""
        x, c = xc
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM,
            dropout=0.0,
        )
        y = block([x, c], training=False)
        y_np = ops.convert_to_numpy(y)
        diff = np.max(np.abs(y_np - x))
        # Must be near-identity (tight tolerance: the whole point of AdaLN-zero).
        assert diff < 1e-5, (
            f"AdaLN-zero block is not identity at init: max|y - x| = {diff}"
        )

    def test_serialization_round_trip(self, xc, tmp_path):
        """Save/load round-trip produces identical outputs."""
        x, c = xc

        # Use a functional model (no subclassing) for reliable serialization.
        x_in = keras.Input(shape=(T, DIM), name="x")
        c_in = keras.Input(shape=(T, DIM), name="c")
        y_out = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM,
            name="block",
        )([x_in, c_in])
        model = keras.Model([x_in, c_in], y_out)

        y1 = ops.convert_to_numpy(model([x, c], training=False))

        path = str(tmp_path / "adaln_block.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        y2 = ops.convert_to_numpy(loaded([x, c], training=False))

        max_diff = np.max(np.abs(y1 - y2))
        assert max_diff < 1e-5, (
            f"Serialization round-trip mismatch: max|y1 - y2| = {max_diff}"
        )

    def test_causal_mask_does_not_leak_future(self, rng):
        """Changing x at position t=T-1 must not affect y at earlier positions."""
        # Use a fresh block and non-zero AdaLN weights so attention is active.
        block = AdaLNZeroConditionalBlock(
            dim=DIM, num_heads=HEADS, dim_head=DIM_HEAD, mlp_dim=MLP_DIM,
        )
        # Warm up the block so it's built.
        x = rng.standard_normal((B, T, DIM)).astype("float32")
        c = rng.standard_normal((B, T, DIM)).astype("float32")
        _ = block([x, c], training=False)

        # Manually break zero-init on the AdaLN final linear so attention
        # has a non-trivial effect — otherwise the test is vacuous.
        w, b = block.adaLN_linear.get_weights()
        w = rng.standard_normal(w.shape).astype("float32") * 0.1
        b = rng.standard_normal(b.shape).astype("float32") * 0.1
        block.adaLN_linear.set_weights([w, b])

        y_ref = ops.convert_to_numpy(block([x, c], training=False))

        # Perturb x only at t = T-1.
        x2 = x.copy()
        x2[:, -1] += rng.standard_normal((B, DIM)).astype("float32")
        y_perturbed = ops.convert_to_numpy(block([x2, c], training=False))

        # Positions 0 .. T-2 must be unchanged.
        diff = np.max(np.abs(y_ref[:, :-1] - y_perturbed[:, :-1]))
        assert diff < 1e-5, (
            f"Causal mask violation: earlier tokens changed by {diff} when "
            f"last token was perturbed."
        )
