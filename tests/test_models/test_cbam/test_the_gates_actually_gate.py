"""CBAM's central claim: two sigmoid gates, applied as a rank-one product.

Why this file exists
--------------------
`CBAM.call -> return inputs` -- the whole attention module deleted -- passes all
16 tests in `test_model.py`. Nothing there compares a gated value to an ungated
one; the module is checked by shape, by `isinstance`, by config round trip and
by "the model trains".

The claim, from Woo et al. (2018) eq. 1-2 and this layer's own docstring:

    F'  = M_c(F) (x) F        M_c has shape (B, 1, 1, C)
    F'' = M_s(F') (x) F'      M_s has shape (B, H, W, 1)

Both maps come out of a sigmoid, and each broadcasts along the axis the other
varies over. Two consequences that no shape test can see:

1. The elementwise ratio ``CBAM(x) / x`` is exactly ``M_c[c] * M_s[h, w]`` --
   a RANK-ONE matrix in (space x channel). MEASURED 2026-08-18 on a
   16-channel 8x8 input: the second singular value of that (64, 16) matrix is
   2e-08 of the first.
2. Every entry of that ratio lies strictly inside (0, 1), because it is a
   product of two sigmoids. MEASURED: 0.00409 to 0.88486.

...and the gates must actually VARY, or they are a constant rescale wearing an
attention module's name. MEASURED: channel gate max/min = 37.85, spatial gate
max/min = 3.22.

The identity injection (`CBAM.call -> return inputs`) makes the ratio identically
1.0, which fails claims 1's non-vacuity control and both halves of the variation
claim -- proven below by running the SAME contract under
`dead_component_oracle.layer_returns_its_input`.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.attention.convolutional_block_attention import CBAM
from dl_techniques.models.cbam.model import CBAMNet

from ..test_sam.dead_component_oracle import (
    component_response,
    layer_returns_its_input,
    no_op_kill,
)


CHANNELS = 16
SPATIAL = 8
#: Strictly positive, so the elementwise ratio `out / x` is well conditioned.
INPUT = (
    np.abs(
        np.random.default_rng(0)
        .normal(size=(2, SPATIAL, SPATIAL, CHANNELS))
        .astype("float32")
    )
    + 0.5
)


def _built_layer() -> CBAM:
    keras.utils.set_random_seed(3)
    layer = CBAM(channels=CHANNELS, ratio=4, kernel_size=7)
    layer(keras.ops.convert_to_tensor(INPUT), training=False)
    return layer


def _gate_ratio(layer: CBAM) -> np.ndarray:
    """`CBAM(x) / x` -- the pure gating factor, with `x` divided back out."""
    out = np.asarray(
        keras.ops.convert_to_numpy(
            layer(keras.ops.convert_to_tensor(INPUT), training=False)
        )
    )
    return out / INPUT


def _assert_cbam_really_gates(layer: CBAM) -> None:
    """The contract. Raises `AssertionError` when the gates are not gating."""
    ratio = _gate_ratio(layer)

    # (a) inside (0, 1): a product of two sigmoids, never an amplification
    assert float(ratio.min()) > 0.0, "a gate value is non-positive"
    assert float(ratio.max()) < 1.0, (
        f"the gate amplifies (max ratio {float(ratio.max()):.5f} >= 1): a "
        "product of two sigmoids cannot exceed 1, and the identity gives "
        "exactly 1.0 everywhere"
    )

    # (b) rank one: M_c varies only over channels, M_s only over space
    matrix = ratio[0].reshape(-1, CHANNELS)
    singular = np.linalg.svd(matrix, compute_uv=False)
    assert singular[1] <= 1e-5 * singular[0], (
        f"the gate is not a channel-times-spatial product: singular value "
        f"ratio {singular[1] / singular[0]:.3e} (measured 2e-08 for the real "
        f"layer). Either stage is leaking into the other's axis."
    )

    # (c) the two gates actually discriminate
    channel_gate = ratio[0].mean(axis=(0, 1))
    spatial_gate = ratio[0].mean(axis=-1)
    channel_spread = float(channel_gate.max() / channel_gate.min())
    spatial_spread = float(spatial_gate.max() / spatial_gate.min())
    assert channel_spread > 5.0, (
        f"the channel gate is nearly constant (max/min {channel_spread:.4f}); "
        f"measured 37.85 for the real layer, exactly 1.0 for the identity"
    )
    assert spatial_spread > 1.5, (
        f"the spatial gate is nearly constant (max/min {spatial_spread:.4f}); "
        f"measured 3.22 for the real layer, exactly 1.0 for the identity"
    )


class TestCBAMGatesAreRealGates:
    def test_the_gate_is_a_bounded_rank_one_product(self):
        _assert_cbam_really_gates(_built_layer())

    def test_the_contract_rejects_a_dead_cbam(self):
        """RED proof: the exact substitution that passes all 16 existing tests."""
        layer = _built_layer()
        with pytest.raises(AssertionError) as excinfo:
            with layer_returns_its_input(layer, name="CBAM"):
                _assert_cbam_really_gates(layer)
        # Name the assertion that fired, so a later edit cannot silently leave
        # one of the three claims proven zero times.
        assert "amplifies" in str(excinfo.value), (
            f"expected the (0, 1) bound to fire first, got: {excinfo.value}"
        )

    def test_the_contract_rejects_a_dead_channel_stage(self):
        """A second, independent injection: only the CHANNEL stage removed."""
        layer = _built_layer()
        ratio = _gate_ratio(layer)
        channel_gate = ratio[0].mean(axis=(0, 1))
        live_spread = float(channel_gate.max() / channel_gate.min())

        # The channel stage emits a (B, 1, 1, C) map; forcing it to 1.0 leaves
        # the spatial stage untouched, so the ratio stays rank one and inside
        # (0, 1) -- only claim (c)'s channel half can catch this.
        ones = keras.ops.ones((INPUT.shape[0], 1, 1, CHANNELS))
        original_call = layer.channel_attention.call
        layer.channel_attention.call = lambda *a, **k: ones
        try:
            killed = _gate_ratio(layer)
        finally:
            layer.channel_attention.call = original_call

        killed_gate = killed[0].mean(axis=(0, 1))
        killed_spread = float(killed_gate.max() / killed_gate.min())
        assert killed_spread == pytest.approx(1.0, abs=1e-4), (
            f"the channel-stage injection did not flatten the channel gate "
            f"({killed_spread:.5f}); the probe is not measuring what it claims"
        )
        assert live_spread > 5.0 * killed_spread, (
            f"the live channel gate ({live_spread:.4f}) is not distinguishable "
            f"from the flattened one ({killed_spread:.4f})"
        )


class TestCBAMReachesTheModelOutput:
    """`CBAMNet` must actually depend on its CBAM blocks."""

    @staticmethod
    def _model() -> CBAMNet:
        keras.utils.set_random_seed(11)
        model = CBAMNet(
            num_classes=4, dims=[8, 16], input_shape=(16, 16, 3), attention_ratio=4
        )
        model(np.zeros((1, 16, 16, 3), dtype="float32"), training=False)
        return model

    @staticmethod
    def _cbam_layers(model: CBAMNet):
        blocks = [
            layer
            for stage in model.stages
            for layer in stage
            if isinstance(layer, CBAM)
        ]
        assert blocks, "no CBAM block found in the model"
        return blocks

    def test_killing_every_cbam_block_moves_the_logits(self):
        import contextlib

        model = self._model()
        blocks = self._cbam_layers(model)
        images = np.random.default_rng(5).random((2, 16, 16, 3)).astype("float32")

        def metric() -> float:
            out = keras.ops.convert_to_numpy(model(images, training=False))
            return float(np.asarray(out).sum(axis=0)[0])

        def kill():
            stack = contextlib.ExitStack()
            for block in blocks:
                stack.enter_context(layer_returns_its_input(block, name=block.name))
            return stack

        # The instrument's own control first: killing nothing must move nothing.
        control = component_response(metric, no_op_kill, name="no-op", atol=0.0)
        assert not control.moved, f"the metric is nondeterministic: {control.summary()}"

        response = component_response(
            metric, kill, name=f"{len(blocks)} CBAM blocks", atol=1e-6
        )
        assert response.moved, (
            f"CBAMNet is INDIFFERENT to its attention blocks -- {response.summary()}"
        )
