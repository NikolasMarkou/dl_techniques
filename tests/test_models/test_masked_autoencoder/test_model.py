"""
Test suite for the MaskedAutoencoder (MAE) model.

MAE wraps a user-provided encoder (keras.Model) and reconstructs masked patches
via a ConvDecoder, with a PatchMasking layer. Covers construction (incl. the
encoder TypeError path), a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.

The `reconstruction`/`mask` outputs depend on random masking; the round-trip
identity is asserted on the deterministic `encoded` output (pure encoder path).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.masked_autoencoder import create_mae_model
from dl_techniques.models.masked_autoencoder.mae import MaskedAutoencoder

INPUT_SHAPE = (32, 32, 3)


def _encoder(shape=INPUT_SHAPE):
    inp = keras.Input(shape=shape)
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    return keras.Model(inp, x, name="tiny_mae_encoder")


def _model():
    return create_mae_model(encoder=_encoder(), patch_size=16,
                            input_shape=INPUT_SHAPE)


def _images(batch=2):
    return np.random.default_rng(0).random((batch, *INPUT_SHAPE)).astype("float32")


class TestMAE:

    def test_forward_dict(self):
        out = _model()(_images(), training=False)
        assert {"reconstruction", "mask", "masked_input", "encoded"} <= set(out)
        recon = out["reconstruction"]
        # reconstruction is a 4D image batch with the input's channel count
        assert recon.shape.rank == 4
        assert int(recon.shape[0]) == 2 and int(recon.shape[-1]) == INPUT_SHAPE[-1]
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(recon)))

    def test_invalid_encoder_raises(self):
        with pytest.raises(TypeError, match="encoder"):
            MaskedAutoencoder(encoder="not_a_model")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        # `encoded` is the deterministic encoder output (mask path is random)
        before = keras.ops.convert_to_numpy(model(x, training=False)["encoded"])

        path = os.path.join(str(tmp_path), "mae.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False)["encoded"])

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="MAE encoded output differs after round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
