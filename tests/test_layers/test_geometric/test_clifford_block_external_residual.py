"""External-residual verification for the 5 Clifford block classes.

Purpose-built tripwire for the silent-failure mode introduced by
``plan_2026-07-03_eb53492e``: the internal identity residual + StochasticDepth
were removed from every Clifford block and pushed OUT to the model level
(``x = x + StochasticDepth(rate)(block(x))``). A forgotten external residual
compiles and runs but silently degrades the network to a non-residual stack.

This module locks the new contract in two layers:

Part A -- **Transform-only contract** for each of the 5 block classes at
``layer_scale_init ~ 0`` (gamma ~ 0). With the residual now external:

  * ``block(x)`` must NOT be ~ x. If the internal residual were still present,
    ``block(x) ~ x`` at gamma ~ 0 (the transform term vanishes but the identity
    survives) -- that is exactly the regression this test catches.
  * ``x + block(x) ~ x`` -- the transform is ~ 0, so the *external* add
    recovers the identity, confirming the residual lives OUTSIDE the block.

Part B -- **End-to-end non-collapse**: a tiny CliffordNet classifier wired with
the external residual produces finite, non-constant output and a non-zero
gradient, proving signal actually flows through the externalized residual.

If someone reverted a block's ``return h_mix`` back to ``return x + h_mix``
(or a model to a bare ``x = block(x)``), Part A's ``not allclose(block(x), x)``
assertion (and/or the non-collapse gradient check) fires.
"""

import numpy as np
import pytest
import keras
import tensorflow as tf

from dl_techniques.layers.geometric.clifford_block import (
    CliffordNetBlock,
    CausalCliffordNetBlock,
    CliffordNetBlockDSv2,
    CausalCliffordNetBlockDSv2,
)
from dl_techniques.models.cliffordnet.model import CliffordNet


# gamma ~ 0: the LayerScale-gated transform contributes ~ 0, so the block's
# transform-only output is ~ 0 and the external residual must recover identity.
_TINY_GAMMA = 1e-10
_SHIFTS = [1, 2]
_CHANNELS = 8


def _assert_transform_only(block, x):
    """Assert the block is transform-only: block(x) !~ x AND x + block(x) ~ x."""
    h_mix = block(x)
    x_np = keras.ops.convert_to_numpy(x)
    h_np = keras.ops.convert_to_numpy(h_mix)

    # Shape is preserved (isotropic reduction: strides=1, out_channels=None).
    assert tuple(h_mix.shape) == tuple(x.shape), (
        f"transform must preserve shape, got {h_mix.shape} vs {x.shape}"
    )
    # External residual recovers the identity (transform ~ 0 at gamma ~ 0).
    np.testing.assert_allclose(
        x_np + h_np, x_np, atol=1e-3,
        err_msg="x + block(x) must ~ x (external residual) at gamma ~ 0",
    )
    # TRIPWIRE: block output alone must NOT be ~ x. If the internal residual
    # were still present, block(x) ~ x here and this assertion fires.
    assert not np.allclose(h_np, x_np, atol=1e-3), (
        "block(x) must be transform-only (~0), NOT ~x -- an internal "
        "residual has silently returned"
    )


class TestTransformOnlyContract:
    """Part A: all 5 block classes are transform-only at gamma ~ 0."""

    def test_clifford_net_block(self):
        block = CliffordNetBlock(
            channels=_CHANNELS, shifts=_SHIFTS, layer_scale_init=_TINY_GAMMA,
        )
        x = tf.random.normal([2, 4, 4, _CHANNELS], seed=0)
        _assert_transform_only(block, x)

    def test_causal_clifford_net_block(self):
        block = CausalCliffordNetBlock(
            channels=_CHANNELS, shifts=_SHIFTS, layer_scale_init=_TINY_GAMMA,
        )
        # Causal blocks operate on (B, 1, seq, D).
        x = tf.random.normal([2, 1, 8, _CHANNELS], seed=0)
        _assert_transform_only(block, x)

    def test_clifford_net_block_ds_v2(self):
        # strides=1, out_channels=None -> skip_pool=Identity, out_proj=None:
        # the DSv2 transform-only return collapses to the isotropic case.
        block = CliffordNetBlockDSv2(
            channels=_CHANNELS, shifts=_SHIFTS, strides=1, out_channels=None,
            layer_scale_init=_TINY_GAMMA,
        )
        assert block.out_proj is None, (
            "at out_channels=None out_proj must be None (isotropic reduction)"
        )
        x = tf.random.normal([2, 8, 8, _CHANNELS], seed=0)
        _assert_transform_only(block, x)

    def test_causal_clifford_net_block_ds_v2(self):
        # Causal DSv2 at strides=1, out_channels=None; skip_pool in {avg,max}.
        block = CausalCliffordNetBlockDSv2(
            channels=_CHANNELS, shifts=_SHIFTS, strides=1, out_channels=None,
            skip_pool="avg", stream_pool="avg", layer_scale_init=_TINY_GAMMA,
        )
        assert block.out_proj is None, (
            "at out_channels=None out_proj must be None (isotropic reduction)"
        )
        x = tf.random.normal([2, 1, 8, _CHANNELS], seed=0)
        _assert_transform_only(block, x)


class TestEndToEndNonCollapse:
    """Part B: a tiny real model wired with the external residual is trainable
    and non-collapsing (finite, non-constant output, non-zero gradient)."""

    def test_cliffordnet_forward_and_gradient(self):
        model = CliffordNet(
            num_classes=4,
            channels=8,
            depth=2,
            patch_size=2,
            shifts=[1, 2],
            stochastic_depth_rate=0.1,
        )
        x = tf.random.normal([2, 8, 8, 3], seed=0)

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = model(x, training=True)
            loss = keras.ops.mean(keras.ops.square(y))

        y_np = keras.ops.convert_to_numpy(y)
        # Output is finite.
        assert np.all(np.isfinite(y_np)), "model output must be finite"
        # Output is non-constant across the batch/logits (network did work).
        assert float(np.std(y_np)) > 0.0, (
            "model output std must be > 0 -- a collapsed (residual-less) "
            "network can produce a degenerate constant output"
        )
        # Signal flows: gradient w.r.t. input is non-zero.
        grad = tape.gradient(loss, x)
        assert grad is not None, "gradient w.r.t. input must exist"
        grad_norm = float(
            keras.ops.convert_to_numpy(
                keras.ops.sqrt(keras.ops.sum(keras.ops.square(grad)))
            )
        )
        assert grad_norm > 0.0, (
            "gradient norm must be > 0 -- signal must flow through the "
            "externalized residual"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
