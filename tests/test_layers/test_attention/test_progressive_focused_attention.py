"""Test suite for ProgressiveFocusedAttention layer.

First-ever regression coverage for ProgressiveFocusedAttention (PFA), the
windowed self-attention block from PFT-SR. Covers:
1. Initialization & Configuration
2. Forward pass (tuple output: (output, attn_weights), 4D [B, H, W, C])
3. Progressive focusing via prev_attn_map
4. NotImplementedError contract for sparsity_mode != 'none'
5. get_config / from_config round-trip
6. Full `.keras` model save/load round-trip

NOTE: PFA's ``call`` returns a tuple ``(output, attn_weights)``. Tuple-output
layers fail the Keras functional-API `.keras` round-trip in this repo, so the
serialization test wraps the layer in a ``keras.Model`` SUBCLASS whose ``call``
returns only ``output[0]`` (a single tensor) and round-trips that wrapper.
"""

import os
import tempfile

import pytest
import numpy as np
import tensorflow as tf
import keras

from dl_techniques.layers.attention.progressive_focused_attention import (
    ProgressiveFocusedAttention,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def minimal_config():
    """Minimal valid config: dim=16, 2 heads, window 4, no LePE-free path needed."""
    return {
        "dim": 16,
        "num_heads": 2,
        "window_size": 4,
        "shift_size": 0,
    }


@pytest.fixture
def sample_input():
    """4D input [B, H, W, C] with H, W divisible by window_size (4)."""
    return keras.random.normal((2, 8, 8, 16))


# ==============================================================================
# 1. Initialization & Configuration
# ==============================================================================

class TestInitialization:
    """Tests for layer initialization and parameter storage."""

    def test_defaults(self, minimal_config):
        layer = ProgressiveFocusedAttention(**minimal_config)
        assert layer._dim == 16
        assert layer._num_heads == 2
        assert layer._window_size == 4
        assert layer._shift_size == 0
        assert layer._head_dim == 8
        assert layer._window_area == 16
        assert layer._sparsity_mode == "none"

    def test_invalid_divisibility(self):
        with pytest.raises(ValueError, match="must be divisible"):
            ProgressiveFocusedAttention(dim=15, num_heads=2, window_size=4)

    def test_invalid_shift_size(self):
        with pytest.raises(ValueError, match="shift_size"):
            ProgressiveFocusedAttention(
                dim=16, num_heads=2, window_size=4, shift_size=4
            )


# ==============================================================================
# 2. Forward Pass (tuple output, 4D)
# ==============================================================================

class TestForwardPass:
    """Tests for the forward pass and output / attention-map shapes."""

    def test_forward_shapes(self, minimal_config, sample_input):
        layer = ProgressiveFocusedAttention(**minimal_config)
        result = layer(sample_input)

        # PFA returns a tuple (output, attn_weights)
        assert isinstance(result, tuple)
        assert len(result) == 2
        output, attn_weights = result

        # Output preserves the input spatial shape [B, H, W, C]
        assert tuple(output.shape) == (2, 8, 8, 16)

        # Attention weights: (B * num_windows, num_heads, window_area, window_area)
        # num_windows per image = (8/4) * (8/4) = 4; B=2 -> 8
        assert tuple(attn_weights.shape) == (8, 2, 16, 16)

    def test_forward_no_nans(self, minimal_config, sample_input):
        layer = ProgressiveFocusedAttention(**minimal_config)
        output, attn_weights = layer(sample_input)
        assert not np.any(np.isnan(np.asarray(output)))
        assert not np.any(np.isnan(np.asarray(attn_weights)))

    def test_progressive_focusing(self, minimal_config, sample_input):
        """Feeding back a prev_attn_map biases the output (focusing path runs)."""
        layer = ProgressiveFocusedAttention(**minimal_config)
        out1, attn1 = layer(sample_input, training=False)
        # Reuse the produced attention map as the previous-layer guidance.
        out2, attn2 = layer(sample_input, prev_attn_map=attn1, training=False)

        # The focused path multiplies scores by prev_attn_map, so the second
        # output must differ from the un-focused first pass.
        assert tuple(out2.shape) == (2, 8, 8, 16)
        assert not np.allclose(np.asarray(out1), np.asarray(out2), atol=1e-6)

    @pytest.mark.xfail(
        reason="SW-MSA mask broadcast in call() is dead-on-forward: self._attn_mask "
               "is 3D (num_windows, wa, wa); the [None, None, :, :] indexing yields a "
               "5D tensor, the tile uses a mismatched 4-tuple, and the final reshape "
               "collapses (B*heads, wa, wa) into (1, 1, wa, wa) -> InvalidArgumentError "
               "(4096 vs 256 values). Correct Swin masking needs a per-window reshape "
               "+ broadcast (>10-line restructure altering masked-attention semantics) "
               "-- needs dedicated plan. shift_size=0 (W-MSA) path is fully covered.",
        strict=False,
    )
    def test_shifted_window_forward(self):
        """SW-MSA path (shift_size > 0) runs and preserves shape.

        XFAIL: the shifted-window attention-mask branch is latent dead code
        (never exercised before this first-ever test). See xfail reason.
        """
        layer = ProgressiveFocusedAttention(
            dim=16, num_heads=2, window_size=4, shift_size=2
        )
        x = keras.random.normal((2, 8, 8, 16))
        output, attn_weights = layer(x)
        assert tuple(output.shape) == (2, 8, 8, 16)

    def test_no_lepe_forward(self):
        layer = ProgressiveFocusedAttention(
            dim=16, num_heads=2, window_size=4, use_lepe=False
        )
        x = keras.random.normal((2, 8, 8, 16))
        output, _ = layer(x)
        assert tuple(output.shape) == (2, 8, 8, 16)

    def test_determinism_inference(self, minimal_config, sample_input):
        layer = ProgressiveFocusedAttention(**minimal_config)
        out1, _ = layer(sample_input, training=False)
        out2, _ = layer(sample_input, training=False)
        np.testing.assert_allclose(
            np.asarray(out1), np.asarray(out2), atol=1e-6
        )


# ==============================================================================
# 3. Sparsity Contract (advertised NotImplementedError)
# ==============================================================================

class TestSparsityContract:
    """Locks the advertised contract that sparse modes are not implemented."""

    def test_top_k_raises(self):
        with pytest.raises(NotImplementedError, match="sparsity_mode"):
            ProgressiveFocusedAttention(
                dim=16, num_heads=2, window_size=4,
                sparsity_mode="top_k", top_k=4,
            )

    def test_threshold_raises(self):
        with pytest.raises(NotImplementedError, match="sparsity_mode"):
            ProgressiveFocusedAttention(
                dim=16, num_heads=2, window_size=4,
                sparsity_mode="threshold",
            )

    def test_invalid_sparsity_mode_raises_valueerror(self):
        """An unknown mode is rejected by validation before the NotImplemented gate."""
        with pytest.raises(ValueError, match="sparsity_mode must be one of"):
            ProgressiveFocusedAttention(
                dim=16, num_heads=2, window_size=4,
                sparsity_mode="bogus",
            )


# ==============================================================================
# 4. Serialization & Persistence
# ==============================================================================

class TestSerialization:
    """Config round-trip + `.keras` model save/load."""

    def test_get_config(self, minimal_config):
        layer = ProgressiveFocusedAttention(
            **minimal_config, use_lepe=False, attention_dropout=0.1
        )
        config = layer.get_config()
        assert config["dim"] == 16
        assert config["num_heads"] == 2
        assert config["window_size"] == 4
        assert config["use_lepe"] is False
        assert config["attention_dropout"] == 0.1

    def test_from_config(self, minimal_config):
        original = ProgressiveFocusedAttention(**minimal_config)
        config = original.get_config()
        rebuilt = ProgressiveFocusedAttention.from_config(config)
        assert rebuilt._dim == original._dim
        assert rebuilt._num_heads == original._num_heads
        assert rebuilt._window_size == original._window_size
        assert rebuilt._shift_size == original._shift_size
        assert rebuilt._use_lepe == original._use_lepe

    def test_config_equivalent_forward(self, minimal_config, sample_input):
        """A from_config clone produces a structurally-equivalent layer."""
        original = ProgressiveFocusedAttention(**minimal_config)
        original(sample_input)  # build
        rebuilt = ProgressiveFocusedAttention.from_config(original.get_config())
        out, attn = rebuilt(sample_input)
        assert tuple(out.shape) == (2, 8, 8, 16)
        assert tuple(attn.shape) == (8, 2, 16, 16)

    def test_model_save_load_loop(self, minimal_config):
        """Full `.keras` round-trip via a single-output keras.Model subclass.

        PFA returns a tuple; the functional Keras API cannot round-trip a
        tuple-output layer in this repo. We wrap it in a Model subclass whose
        `call` returns only output[0] (the spatial tensor) and round-trip THAT.
        """

        @keras.saving.register_keras_serializable()
        class PFAWrapper(keras.Model):
            def __init__(self, layer_config, **kwargs):
                super().__init__(**kwargs)
                self.layer_config = layer_config
                self.pfa = ProgressiveFocusedAttention(**layer_config)

            def call(self, inputs, training=None):
                output, _attn = self.pfa(inputs, training=training)
                return output

            def get_config(self):
                config = super().get_config()
                config.update({"layer_config": self.layer_config})
                return config

        model = PFAWrapper(layer_config=minimal_config)
        x_in = np.random.normal(size=(2, 8, 8, 16)).astype("float32")
        pred_orig = model(x_in, training=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "pfa_model.keras")
            model.save(path)
            loaded = keras.models.load_model(path)
            pred_load = loaded(x_in, training=False)

            np.testing.assert_allclose(
                np.asarray(pred_orig), np.asarray(pred_load), atol=1e-6
            )


# ==============================================================================
# 5. Misc
# ==============================================================================

class TestMisc:

    def test_compute_output_shape(self, minimal_config):
        layer = ProgressiveFocusedAttention(**minimal_config)
        out_shape, attn_shape = layer.compute_output_shape((2, 8, 8, 16))
        assert out_shape == (2, 8, 8, 16)
        # (B * num_windows, num_heads, window_area, window_area)
        assert attn_shape == (8, 2, 16, 16)

    def test_gradient_flow(self, minimal_config, sample_input):
        layer = ProgressiveFocusedAttention(**minimal_config)
        with tf.GradientTape() as tape:
            output, _ = layer(sample_input)
            loss = tf.reduce_mean(output)
        grads = tape.gradient(loss, layer.trainable_variables)
        assert len(grads) > 0
        assert all(g is not None for g in grads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
