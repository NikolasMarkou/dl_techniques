"""Test Suite for SingleWindowAttention Layer.

First-ever STANDALONE regression coverage for SingleWindowAttention. Prior to
this file the layer was only ``isinstance``-checked as a sub-layer inside
``test_window_attention.py``. This suite exercises it directly.

SingleWindowAttention is multi-head self-attention restricted to a single square
window of ``window_size ** 2`` tokens. Input is 3D ``[B, N, dim]`` (per-window
tokens); the layer pads internally up to ``window_size ** 2`` and strips the
padding off the output, so output shape == input shape ``[B, N, dim]``. It
supports an optional ``attention_mask`` of shape ``[B, N]`` (1 = valid, 0 =
pad) combined with the internal padding mask.

Coverage:
1. Initialization & Configuration
2. Input Validation
3. Forward Pass (output shape == input shape)
4. Masked Forward (attention_mask is honored)
5. Serialization (get_config / from_config + full .keras model round-trip)
"""

import pytest
import numpy as np
import keras
import tempfile
import os

from dl_techniques.layers.attention.single_window_attention import (
    SingleWindowAttention,
)


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self):
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        assert layer.dim == 16
        assert layer.window_size == 4
        assert layer.num_heads == 2
        assert layer.head_dim == 8
        assert layer.attention_mode == "linear"
        assert layer.probability_type == "softmax"
        expected_scale = 8 ** -0.5
        np.testing.assert_allclose(layer.scale, expected_scale)
        # QKV sub-layer created in __init__ for linear mode.
        assert layer.qkv is not None

    def test_kan_key_mode(self):
        layer = SingleWindowAttention(
            dim=16, window_size=3, num_heads=2, attention_mode="kan_key"
        )
        assert layer.attention_mode == "kan_key"
        assert layer.query is not None
        assert layer.key is not None
        assert layer.value is not None

    def test_custom_scale(self):
        layer = SingleWindowAttention(
            dim=16, window_size=4, num_heads=2, qk_scale=0.25
        )
        assert layer.scale == 0.25
        assert layer.qk_scale == 0.25


# ==============================================================================
# 2. Input Validation
# ==============================================================================

class TestValidation:
    """Tests for __init__ validation."""

    def test_invalid_attention_mode(self):
        with pytest.raises(ValueError, match="Invalid attention_mode"):
            SingleWindowAttention(
                dim=16, window_size=4, num_heads=2, attention_mode="bogus"
            )

    @pytest.mark.parametrize(
        "bad_prob",
        ["routing", "deterministic_routing", "hierarchical", "hierarchical_routing"],
    )
    def test_invalid_probability_type(self, bad_prob):
        with pytest.raises(ValueError, match="Invalid probability_type"):
            SingleWindowAttention(
                dim=16, window_size=4, num_heads=2, probability_type=bad_prob
            )


# ==============================================================================
# 3. Forward Pass (output shape == input shape)
# ==============================================================================

class TestForward:
    """Forward-pass tests at minimal configs."""

    def test_output_shape_full_window(self):
        """N == window_size**2 -> no internal padding."""
        window_size = 4
        n = window_size * window_size  # 16
        x = keras.random.normal((2, n, 16))
        layer = SingleWindowAttention(dim=16, window_size=window_size, num_heads=2)
        out = layer(x)
        assert out.shape == (2, n, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_shape_partial_window(self):
        """N < window_size**2 -> layer pads internally then strips it back off."""
        x = keras.random.normal((2, 5, 16))  # 5 < 16
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        out = layer(x)
        # Output must equal the ACTUAL (unpadded) input shape.
        assert out.shape == (2, 5, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_output_shape_no_relative_bias(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(
            dim=16,
            window_size=3,
            num_heads=2,
            use_relative_position_bias=False,
        )
        out = layer(x)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_kan_key_forward(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(
            dim=16, window_size=3, num_heads=2, attention_mode="kan_key"
        )
        out = layer(x)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_deterministic_inference(self):
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        out1 = layer(x, training=False)
        out2 = layer(x, training=False)
        np.testing.assert_allclose(np.array(out1), np.array(out2), atol=1e-6)


# ==============================================================================
# 4. Masked Forward (attention_mask is accepted AND honored)
# ==============================================================================

class TestMaskedForward:
    """The layer accepts a [B, N] attention_mask combined with internal padding."""

    def test_masked_forward_shape(self):
        x = keras.random.normal((2, 9, 16))
        # Mask out the last 3 tokens of every sample.
        mask = np.ones((2, 9), dtype="float32")
        mask[:, -3:] = 0.0
        mask = keras.ops.convert_to_tensor(mask)
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        out = layer(x, attention_mask=mask)
        assert out.shape == (2, 9, 16)
        assert not np.any(np.isnan(np.array(out)))

    def test_mask_changes_output(self):
        """A non-trivial mask must actually change the attended output vs no mask
        (proves the mask is honored, unlike SpatialAttention's ignored mask)."""
        x = keras.random.normal((2, 9, 16))
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)

        out_unmasked = np.array(layer(x, training=False))

        mask = np.ones((2, 9), dtype="float32")
        mask[:, -4:] = 0.0
        mask = keras.ops.convert_to_tensor(mask)
        out_masked = np.array(layer(x, attention_mask=mask, training=False))

        assert not np.allclose(out_unmasked, out_masked, atol=1e-4), (
            "attention_mask had no effect on the output"
        )


# ==============================================================================
# 5. Serialization
# ==============================================================================

class TestSerialization:

    def test_get_config(self):
        layer = SingleWindowAttention(
            dim=32,
            window_size=4,
            num_heads=4,
            attention_mode="linear",
            dropout_rate=0.1,
            use_relative_position_bias=False,
        )
        config = layer.get_config()
        assert config["dim"] == 32
        assert config["window_size"] == 4
        assert config["num_heads"] == 4
        assert config["attention_mode"] == "linear"
        assert config["dropout_rate"] == 0.1
        assert config["use_relative_position_bias"] is False

    def test_from_config(self):
        layer = SingleWindowAttention(dim=16, window_size=3, num_heads=2)
        config = layer.get_config()
        rebuilt = SingleWindowAttention.from_config(config)
        assert rebuilt.dim == 16
        assert rebuilt.window_size == 3
        assert rebuilt.num_heads == 2
        assert rebuilt.attention_mode == "linear"

    def test_model_save_load_loop(self):
        """Full .keras save/load round-trip; deterministic -> assert exact output."""
        inputs = keras.Input(shape=(9, 16))
        x = SingleWindowAttention(dim=16, window_size=3, num_heads=2)(inputs)
        model = keras.Model(inputs, x)

        x_in = np.random.normal(size=(2, 9, 16)).astype("float32")
        pred_orig = model.predict(x_in, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "single_window_attention.keras")
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
        layer = SingleWindowAttention(dim=16, window_size=4, num_heads=2)
        # Output shape is identical to the input shape.
        assert layer.compute_output_shape((2, 9, 16)) == (2, 9, 16)

    def test_kwargs_passthrough(self):
        layer = SingleWindowAttention(
            dim=16, window_size=4, num_heads=2, name="swa_special"
        )
        assert layer.name == "swa_special"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
