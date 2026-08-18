"""The AdaLN conditioning is SiLU'd exactly ONCE, on the way to every modulation.

F-09 of the 2026-08-18 deep review, fixed under
``plan-2026-08-18T140459-7991552f/D-040``.

`Ideogram4Transformer.call` computes ``adaln_input = silu(adaln_proj(t_cond))``
and hands the SAME tensor to every block and to the head.
`Ideogram4TransformerBlock.call` consumed it raw, but
`Ideogram4FinalLayer.call` used to do ``adaln_modulation(silu(c))`` -- a SECOND
SiLU on an already-SiLU'd vector, so the velocity head alone was conditioned on
``silu(silu(t))``. `silu` is not idempotent on negatives (-2.0 -> -0.2384 ->
-0.1052), so this is a real flattening of the head's dependence on `t`, not a
reparameterization.

MEASURED at the TINY preset (seed 0): the tensor arriving at the head's
`adaln_modulation` differed from `adaln_input` by max|delta| = **2.71e-01**
pre-fix (mean |adaln_input| = 1.92e-01) and is now bit-identical (0.0). The
model's velocity OUTPUT moved by max|delta| = **2.83e-01**, i.e. **7.15%** of
its own peak magnitude -- the defect was live in the shipped forward pass, not
a rounding-level curiosity.

The two-instrument split matters: the equality assertion is what pins the fix;
the `silu(silu(x)) != silu(x)` assertion is what proves the equality assertion
would have been RED.
"""

import keras
import numpy as np
import pytest

from dl_techniques.layers.transformers.ideogram4_block import (
    Ideogram4FinalLayer,
    Ideogram4TransformerBlock,
)
from dl_techniques.models.ideogram4.config import get_ideogram4_config
from dl_techniques.models.ideogram4.transformer import Ideogram4Transformer

from .test_transformer import _make_batch, BATCH, SEQ_LEN


@pytest.fixture(scope="module")
def tiny_config():
    config, _ = get_ideogram4_config("tiny")
    return config


class TestSiluIsNotIdempotent:
    """The premise: a second SiLU is not a no-op."""

    def test_double_silu_differs_on_negatives(self):
        x = keras.ops.convert_to_tensor(
            np.array([-4.0, -2.0, -0.5, 0.0, 0.5, 2.0], dtype="float32")
        )
        once = keras.ops.convert_to_numpy(keras.ops.silu(x))
        twice = keras.ops.convert_to_numpy(keras.ops.silu(keras.ops.silu(x)))
        delta = np.abs(once - twice).max()
        assert delta > 1e-2, f"silu(silu(x)) == silu(x)?! max|delta| = {delta}"
        # The review's worked example, pinned.
        np.testing.assert_allclose(once[1], -0.238406, atol=1e-5)
        np.testing.assert_allclose(twice[1], -0.105061, atol=1e-5)


class TestFinalLayerAppliesNoSilu:
    """`Ideogram4FinalLayer` consumes `c` raw, exactly like the block does."""

    def _final(self, adaln_dim=8, hidden=16, out_channels=4):
        keras.utils.set_random_seed(0)
        return Ideogram4FinalLayer(
            hidden_size=hidden, out_channels=out_channels, adaln_dim=adaln_dim
        )

    def test_modulation_dense_sees_c_unchanged(self):
        layer = self._final()
        x = keras.random.normal((2, 5, 16), seed=1)
        c = keras.random.normal((2, 1, 8), seed=2)
        layer(x, c)  # build

        seen = {}
        inner = layer.adaln_modulation.call

        def spy(inputs, *args, **kwargs):
            seen["c"] = keras.ops.convert_to_numpy(inputs)
            return inner(inputs, *args, **kwargs)

        layer.adaln_modulation.call = spy
        layer(x, c)

        np.testing.assert_array_equal(seen["c"], keras.ops.convert_to_numpy(c))

    def test_head_and_block_agree_on_the_conditioning_convention(self):
        """Whatever the head does to `c`, the block must do the same.

        This is the invariant F-09 broke: the two sites disagreed.
        """
        keras.utils.set_random_seed(0)
        head = Ideogram4FinalLayer(hidden_size=16, out_channels=4, adaln_dim=8)
        block = Ideogram4TransformerBlock(
            hidden_size=16, intermediate_size=32, num_heads=2, adaln_dim=8
        )
        c = keras.random.normal((2, 1, 8), seed=2)
        x = keras.random.normal((2, 5, 16), seed=1)

        cos = keras.ops.ones((2, 5, 8))
        sin = keras.ops.zeros((2, 5, 8))
        seg = keras.ops.zeros((2, 5), dtype="int32")

        # Build BOTH fully before attaching spies: Keras forbids new state on a
        # partially-built layer, and `call` patching counts as attribute set.
        head(x, c)
        block(x, seg, cos, sin, c)

        seen = {}
        for tag, dense in (("head", head.adaln_modulation),
                           ("block", block.adaln_modulation)):
            inner = dense.call

            def spy(inputs, *a, _tag=tag, _inner=inner, **k):
                seen[_tag] = keras.ops.convert_to_numpy(inputs)
                return _inner(inputs, *a, **k)

            dense.call = spy

        head(x, c)
        block(x, seg, cos, sin, c)

        np.testing.assert_array_equal(seen["head"], seen["block"])


class TestConditioningInsideTheAssembledModel:
    """End-to-end: the head's modulation sees exactly `adaln_input`."""

    def test_head_modulation_input_equals_adaln_input(self, tiny_config):
        keras.utils.set_random_seed(0)
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        model(batch)  # build

        seen = {}
        head_inner = model.final_layer.adaln_modulation.call
        block_inner = model.blocks[0].adaln_modulation.call

        def head_spy(inputs, *a, **k):
            seen["head"] = keras.ops.convert_to_numpy(inputs)
            return head_inner(inputs, *a, **k)

        def block_spy(inputs, *a, **k):
            seen["block"] = keras.ops.convert_to_numpy(inputs)
            return block_inner(inputs, *a, **k)

        model.final_layer.adaln_modulation.call = head_spy
        model.blocks[0].adaln_modulation.call = block_spy
        model(batch)

        assert "head" in seen and "block" in seen
        delta = np.abs(seen["head"] - seen["block"]).max()
        assert delta == 0.0, (
            f"the head's conditioning differs from the trunk's by {delta:.3e} "
            f"-- a second SiLU (or some other transform) has come back"
        )


def _legacy_final_layer_call(self, x, c, training=None):
    """The pre-fix `Ideogram4FinalLayer.call` body, verbatim, for RED-proofs."""
    scale = 1.0 + self.adaln_modulation(keras.ops.silu(c))
    normed = self.norm_final(x, training=training)
    return self.linear(normed * scale)


class TestGuardIsRedAgainstTheLegacyBody:
    """Without the fix, the assertions above FAIL and the numbers move."""

    def test_modulation_input_assertion_would_be_red(self, monkeypatch):
        monkeypatch.setattr(
            Ideogram4FinalLayer, "call", _legacy_final_layer_call, raising=True
        )
        keras.utils.set_random_seed(0)
        layer = Ideogram4FinalLayer(hidden_size=16, out_channels=4, adaln_dim=8)
        x = keras.random.normal((2, 5, 16), seed=1)
        c = keras.random.normal((2, 1, 8), seed=2)
        layer(x, c)

        seen = {}
        inner = layer.adaln_modulation.call

        def spy(inputs, *a, **k):
            seen["c"] = keras.ops.convert_to_numpy(inputs)
            return inner(inputs, *a, **k)

        layer.adaln_modulation.call = spy
        layer(x, c)

        delta = np.abs(seen["c"] - keras.ops.convert_to_numpy(c)).max()
        assert delta > 1e-2, (
            "the legacy body no longer double-applies SiLU -- this RED-proof "
            "has stopped being an instrument"
        )

    def test_the_defect_moved_the_shipped_velocity(self, tiny_config):
        """The fix is a behaviour change, and this records its size."""
        keras.utils.set_random_seed(0)
        model = Ideogram4Transformer(config=tiny_config)
        batch = _make_batch(tiny_config)
        fixed = keras.ops.convert_to_numpy(model(batch))

        original = Ideogram4FinalLayer.call
        try:
            Ideogram4FinalLayer.call = _legacy_final_layer_call
            legacy = keras.ops.convert_to_numpy(model(batch))
        finally:
            Ideogram4FinalLayer.call = original

        delta = np.abs(fixed - legacy).max()
        assert delta > 1e-2, f"fix is a no-op on the velocity (max|delta|={delta})"
