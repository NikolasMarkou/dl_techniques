"""
Test suite for FFTNet (adaptive spectral filtering vision foundation model).

Covers construction (from_variant + create_fftnet), a forward pass, and the M2
full .keras save -> load -> identical-output round-trip. The FFTMixer uses a
documented raw-tf FFT path (accepted §L2-5 exception); the model still serializes
and round-trips cleanly. call() takes (B, H, W, 3) and returns a dict.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.fftnet.model import FFTNet, create_fftnet


def _model():
    return create_fftnet(variant="tiny", image_size=32, patch_size=16)


def _images(batch=2):
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


class TestFFTNet:

    def test_forward_dict(self):
        out = _model()(_images(), training=False)
        assert {"last_hidden_state", "cls_token", "patch_features"} <= set(out)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out["last_hidden_state"])))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            FFTNet.from_variant("nonexistent")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False)["last_hidden_state"])

        path = os.path.join(str(tmp_path), "fftnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["last_hidden_state"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="FFTNet differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
