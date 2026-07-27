"""Test Suite for SpatialAttention Layer (standalone).

First-ever STANDALONE regression coverage for SpatialAttention. Prior to this
file the layer was only exercised as a CBAM sub-layer (via
``test_convolutional_block_attention.py``). This suite tests it directly.

SpatialAttention is the CBAM spatial module: it pools over the channel axis
(avg + max), concatenates the two ``[B, H, W, 1]`` maps, runs a single Conv2D
with a sigmoid gate, and emits a per-location attention map of shape
``[B, H, W, 1]`` with values in ``[0, 1]``. Output is the MAP, not the gated
input.

C3 contract (documented in spatial_attention.py:209-214): the ``attention_mask``
parameter is ACCEPTED BUT IGNORED -- this is a vision layer with no token-mask
semantics. ``TestIgnoredMask`` PINS that documented contract: a masked and an
unmasked forward must produce IDENTICAL output.

Coverage:
1. Initialization & Configuration
2. Input Validation
3. Forward Pass (output map shape [B, H, W, 1], values in [0, 1])
4. Ignored-mask contract (C3 lock)
5. Serialization (get_config / from_config + full .keras model round-trip)
"""

import pytest
import numpy as np
import keras
import tempfile
import os

from dl_techniques.layers.attention.spatial_attention import SpatialAttention


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self):
        layer = SpatialAttention()
        assert layer.kernel_size == 7
        assert layer.use_bias is True
        assert layer.gate_activation_type == "sigmoid"
        # Conv sub-layer created in __init__ (modern Keras 3 pattern).
        assert layer.conv is not None
        assert layer.conv.filters == 1

    def test_custom_config(self):
        layer = SpatialAttention(kernel_size=3, use_bias=False)
        assert layer.kernel_size == 3
        assert layer.use_bias is False
        assert layer.conv.kernel_size == (3, 3)


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:
    """Tests for __init__ validation."""

    def test_invalid_kernel_size_nonpositive(self):
        with pytest.raises(ValueError, match="kernel_size must be positive"):
            SpatialAttention(kernel_size=0)

    def test_invalid_kernel_size_even(self):
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            SpatialAttention(kernel_size=4)


class TestBuildRankValidation:
    """``build`` must reject non-4D input with THIS layer's named ``ValueError``.

    Defect 8 (findings/non-mask-defects.md; plan step 6b). ``SpatialAttention``
    is the only CBAM half without a rank check: its sibling
    ``ChannelAttention.build`` raises
    ``"Expected 4D input shape (batch, height, width, channels), got ND: ..."``
    before touching Keras, while this one forced ``conv_input_shape[-1] = 2``
    onto a shape of the wrong length and let the failure surface from inside
    ``Conv2D.build`` — a Keras-INTERNAL message that names neither this layer
    nor the real problem.

    The assertions below are therefore on the TYPE **and the message**, not
    merely on "something raised": raising the wrong error is exactly the bug.
    """

    _MSG = "Expected 4D input shape"

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 32),            # 2D
            (2, 32, 3),         # 3D — the case named in the finding
            (2, 8, 8, 16, 3),   # 5D
        ],
    )
    def test_non_4d_input_raises_this_layers_named_error(self, shape):
        layer = SpatialAttention(kernel_size=3)
        with pytest.raises(ValueError, match=self._MSG):
            layer(keras.random.normal(shape))

    @pytest.mark.parametrize("shape", [(2, 32), (2, 32, 3), (2, 8, 8, 16, 3)])
    def test_the_message_names_the_offending_rank(self, shape):
        """The error must be actionable: it reports the rank it actually got."""
        layer = SpatialAttention(kernel_size=3)
        with pytest.raises(ValueError) as excinfo:
            layer.build(shape)
        assert f"got {len(shape)}D" in str(excinfo.value)

    def test_direct_build_call_raises_too(self):
        """Not only ``__call__``: an explicit ``build`` must reject it as well.

        Keras drives ``build(input_shape)`` directly on ``.keras`` reload, so a
        check that only fires through ``__call__`` would miss the reload path.
        """
        layer = SpatialAttention(kernel_size=3)
        with pytest.raises(ValueError, match=self._MSG):
            layer.build((2, 32, 3))

    def test_4d_input_is_still_accepted(self):
        """Anti-vacuity: the check must not have narrowed the SUPPORTED case."""
        layer = SpatialAttention(kernel_size=3)
        out = layer(keras.random.normal((2, 8, 8, 16)))
        assert out.shape == (2, 8, 8, 1)

    def test_cbam_the_main_consumer_still_builds(self):
        """CBAM feeds ``SpatialAttention`` its own 4D input — it must be unaffected."""
        from dl_techniques.layers.attention.convolutional_block_attention import CBAM

        block = CBAM(channels=16, ratio=4, kernel_size=3)
        out = block(keras.random.normal((2, 8, 8, 16)))
        assert out.shape == (2, 8, 8, 16)
        assert not np.any(np.isnan(np.array(out)))


# ==============================================================================
# 3. Forward Pass (output is the attention MAP [B, H, W, 1])
# ==============================================================================

class TestForward:
    """Forward-pass tests at minimal configs."""

    def test_output_shape_is_map(self):
        """Output is the spatial attention map [B, H, W, 1], NOT the gated input."""
        x = keras.random.normal((2, 8, 8, 16))
        layer = SpatialAttention(kernel_size=7)
        out = layer(x)
        assert out.shape == (2, 8, 8, 1)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_range_sigmoid(self):
        x = keras.random.normal((2, 8, 8, 16))
        layer = SpatialAttention(kernel_size=3)
        out = np.array(layer(x))
        assert np.all(out >= 0.0), "sigmoid gate output must be >= 0"
        assert np.all(out <= 1.0), "sigmoid gate output must be <= 1"

    def test_small_kernel(self):
        x = keras.random.normal((1, 4, 4, 8))
        layer = SpatialAttention(kernel_size=1)
        out = layer(x)
        assert out.shape == (1, 4, 4, 1)

    def test_deterministic_inference(self):
        x = keras.random.normal((2, 8, 8, 16))
        layer = SpatialAttention(kernel_size=7)
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)


# ==============================================================================
# 4. Ignored-mask contract (C3 LOCK)
# ==============================================================================

class TestIgnoredMask:
    """Pin the documented C3 contract: attention_mask is accepted but IGNORED."""

    def test_mask_is_ignored(self):
        """Calling with a non-None attention_mask of valid shape must produce
        output IDENTICAL to the no-mask call (the param is a no-op by design)."""
        x = keras.random.normal((2, 8, 8, 16))
        layer = SpatialAttention(kernel_size=7)

        out_no_mask = np.array(layer(x, attention_mask=None, training=False))

        # A non-trivial mask of a valid spatial shape.
        mask = keras.ops.convert_to_tensor(
            np.zeros((2, 8, 8, 1), dtype="float32")
        )
        out_with_mask = np.array(
            layer(x, attention_mask=mask, training=False)
        )

        np.testing.assert_array_equal(
            out_no_mask,
            out_with_mask,
            err_msg="attention_mask must be IGNORED (C3 contract) but changed output",
        )


# ==============================================================================
# 5. Serialization
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = SpatialAttention(kernel_size=3, use_bias=False)
        config = layer.get_config()
        assert config["kernel_size"] == 3
        assert config["use_bias"] is False
        assert config["gate_activation_type"] == "sigmoid"

    def test_from_config(self):
        layer = SpatialAttention(kernel_size=5)
        config = layer.get_config()
        rebuilt = SpatialAttention.from_config(config)
        assert rebuilt.kernel_size == 5
        assert rebuilt.use_bias == layer.use_bias

    def test_model_save_load_loop(self):
        """Full .keras save/load round-trip; deterministic -> assert exact output."""
        inputs = keras.Input(shape=(8, 8, 16))
        x = SpatialAttention(kernel_size=7)(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 8, 8, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spatial_attention.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded.predict(x_in, verbose=0)

        assert pred_orig.shape == pred_load.shape
        np.testing.assert_allclose(pred_orig, pred_load, atol=1e-6)


# ==============================================================================
# 6. Edge Cases
# ==============================================================================

class TestEdgeCases:

    def test_compute_output_shape(self):
        layer = SpatialAttention(kernel_size=7)
        assert layer.compute_output_shape((2, 8, 8, 16)) == (2, 8, 8, 1)

    def test_kwargs_passthrough(self):
        layer = SpatialAttention(kernel_size=7, name="spatial_special")
        assert layer.name == "spatial_special"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
