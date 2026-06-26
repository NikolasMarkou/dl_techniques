"""
Test suite for the parameter-free `MatchChannels` layer.

Covers (SC1 of plan_2026-06-26_90d8cbe6):
- Zero-pad (in < target): output shape + padded channels exactly zero, leading
  channels equal the input.
- Slice (in > target): output equals ``x[..., :target]``.
- Passthrough (in == target): returns the input unchanged.
- Homogeneity (degree-1, both directions): ``f(alpha * x) == alpha * f(x)`` for
  positive and negative ``alpha`` (pad / slice are linear).
- Weightless: ``layer.weights == []`` and ``count_params() == 0`` after a forward
  pass.
- ``compute_output_shape`` resizes the last axis to ``target_channels``.
- ``get_config`` / ``from_config`` round-trip of the layer object.
- Tiny functional model ``.keras`` save / load round-trip on CPU (atol=1e-6).
"""

import os
import tempfile

import keras
import numpy as np
import pytest

from dl_techniques.layers.match_channels import MatchChannels


# Test fixtures
@pytest.fixture
def pad_input() -> np.ndarray:
    """Input with 4 channels, to be padded to 8."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((2, 8, 8, 4)).astype("float32")


@pytest.fixture
def slice_input() -> np.ndarray:
    """Input with 8 channels, to be sliced to 4."""
    rng = np.random.default_rng(43)
    return rng.standard_normal((2, 8, 8, 8)).astype("float32")


class TestMatchChannels:
    """Unit tests for the MatchChannels layer."""

    def test_pad_shape_and_zero_content(self, pad_input: np.ndarray) -> None:
        """in < target: pads with zeros; leading channels equal the input."""
        layer = MatchChannels(8)
        out = keras.ops.convert_to_numpy(layer(pad_input))

        assert out.shape == (2, 8, 8, 8)
        # Padded channels are exactly zero.
        assert np.all(out[..., 4:] == 0.0)
        # Original channels are preserved exactly.
        assert np.array_equal(out[..., :4], pad_input)

    def test_slice(self, slice_input: np.ndarray) -> None:
        """in > target: keeps the first `target` channels."""
        layer = MatchChannels(4)
        out = keras.ops.convert_to_numpy(layer(slice_input))

        assert out.shape == (2, 8, 8, 4)
        assert np.array_equal(out, slice_input[..., :4])

    def test_passthrough(self, pad_input: np.ndarray) -> None:
        """in == target: returns the input unchanged."""
        layer = MatchChannels(4)
        out = keras.ops.convert_to_numpy(layer(pad_input))

        assert out.shape == pad_input.shape
        assert np.allclose(out, pad_input, atol=1e-6)

    @pytest.mark.parametrize("alpha", [3.0, -2.0])
    def test_homogeneity_pad(self, pad_input: np.ndarray, alpha: float) -> None:
        """Pad is degree-1 homogeneous: f(alpha*x) == alpha*f(x)."""
        layer = MatchChannels(8)
        f_x = keras.ops.convert_to_numpy(layer(pad_input))
        f_alpha_x = keras.ops.convert_to_numpy(layer(alpha * pad_input))
        assert np.allclose(f_alpha_x, alpha * f_x, atol=1e-6)

    @pytest.mark.parametrize("alpha", [3.0, -2.0])
    def test_homogeneity_slice(self, slice_input: np.ndarray, alpha: float) -> None:
        """Slice is degree-1 homogeneous: f(alpha*x) == alpha*f(x)."""
        layer = MatchChannels(4)
        f_x = keras.ops.convert_to_numpy(layer(slice_input))
        f_alpha_x = keras.ops.convert_to_numpy(layer(alpha * slice_input))
        assert np.allclose(f_alpha_x, alpha * f_x, atol=1e-6)

    def test_weightless(self, pad_input: np.ndarray) -> None:
        """Layer holds no weights after a forward pass."""
        layer = MatchChannels(8)
        _ = layer(pad_input)
        assert layer.weights == []
        assert layer.count_params() == 0

    def test_compute_output_shape(self) -> None:
        """compute_output_shape resizes the last axis to target_channels."""
        layer = MatchChannels(8)
        out_shape = layer.compute_output_shape((None, 8, 8, 4))
        assert out_shape[-1] == 8

    def test_config_round_trip(self) -> None:
        """get_config / from_config round-trips target_channels."""
        layer = MatchChannels(8)
        restored = MatchChannels.from_config(layer.get_config())
        assert restored.target_channels == 8

    def test_keras_model_round_trip(self) -> None:
        """Tiny functional model with MatchChannels round-trips through .keras (CPU)."""
        inputs = keras.Input(shape=(8, 8, 4))
        x = MatchChannels(8)(inputs)          # pad 4 -> 8
        x = MatchChannels(2)(x)               # slice 8 -> 2
        outputs = keras.layers.Conv2D(3, 1)(x)
        model = keras.Model(inputs, outputs)

        rng = np.random.default_rng(7)
        sample = rng.standard_normal((2, 8, 8, 4)).astype("float32")
        pre = keras.ops.convert_to_numpy(model(sample, training=False))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "match_channels_model.keras")
            model.save(path)
            try:
                loaded = keras.models.load_model(path)
            except Exception:
                # MatchChannels is registered, but pass defensively if needed.
                loaded = keras.models.load_model(
                    path, custom_objects={"MatchChannels": MatchChannels}
                )
            post = keras.ops.convert_to_numpy(loaded(sample, training=False))

        assert np.allclose(pre, post, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
