"""Tests for the FFTLayer / IFFTLayer spectral transforms."""

import os
import keras
import numpy as np
import pytest

from dl_techniques.layers.fft_layers import FFTLayer, IFFTLayer

B, H, W, C = 2, 8, 8, 3


@pytest.fixture
def sample_image():
    return np.random.default_rng(0).standard_normal((B, H, W, C)).astype("float32")


class TestFFTLayer:

    def test_forward_pass(self, sample_image):
        out = FFTLayer()(sample_image)
        assert tuple(out.shape) == (B, H, W, 2 * C)

    def test_compute_output_shape(self):
        assert FFTLayer().compute_output_shape((B, H, W, C)) == (B, H, W, 2 * C)

    def test_serialization_round_trip(self, sample_image, tmp_path):
        inp = keras.Input(shape=(H, W, C))
        out = FFTLayer(name="fft")(inp)
        model = keras.Model(inp, out)
        y0 = model(sample_image)
        path = os.path.join(tmp_path, "fft.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"FFTLayer": FFTLayer})
        y1 = loaded(sample_image)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )


class TestIFFTLayer:

    def test_forward_pass(self, sample_image):
        spectral = FFTLayer()(sample_image)
        out = IFFTLayer()(spectral)
        assert tuple(out.shape) == (B, H, W, C)

    def test_round_trip_reconstruction(self, sample_image):
        """FFT followed by IFFT should approximately reconstruct the input."""
        spectral = FFTLayer()(sample_image)
        recon = IFFTLayer()(spectral)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(recon), sample_image, rtol=1e-4, atol=1e-4
        )

    def test_compute_output_shape(self):
        assert IFFTLayer().compute_output_shape((B, H, W, 2 * C)) == (B, H, W, C)

    def test_compute_output_shape_odd_raises(self):
        with pytest.raises(ValueError):
            IFFTLayer().compute_output_shape((B, H, W, 3))

    def test_serialization_round_trip(self, sample_image, tmp_path):
        spectral = keras.ops.convert_to_numpy(FFTLayer()(sample_image))
        inp = keras.Input(shape=(H, W, 2 * C))
        out = IFFTLayer(name="ifft")(inp)
        model = keras.Model(inp, out)
        y0 = model(spectral)
        path = os.path.join(tmp_path, "ifft.keras")
        model.save(path)
        loaded = keras.models.load_model(path, custom_objects={"IFFTLayer": IFFTLayer})
        y1 = loaded(spectral)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(y0), keras.ops.convert_to_numpy(y1),
            rtol=1e-5, atol=1e-5,
        )
