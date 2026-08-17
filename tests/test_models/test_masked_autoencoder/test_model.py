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
    """A /16 encoder, matching the default decoder_depth=4 (16x upsampling).

    This used to be a /4 encoder, which violated MAE's scale contract: the
    decoder emitted 128x128 for a 32x32 input and `compute_loss` could not
    broadcast. The forward test asserted only rank and channel count, so it
    passed on a model that could not train.
    """
    inp = keras.Input(shape=shape)
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
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
        # reconstruction is a 4D image batch at the INPUT's resolution. The
        # spatial assertion is the one that matters: rank+channels alone passed
        # against a decoder emitting 128x128 for a 32x32 input.
        assert recon.shape.rank == 4
        assert int(recon.shape[0]) == 2 and int(recon.shape[-1]) == INPUT_SHAPE[-1]
        assert tuple(int(d) for d in recon.shape[1:3]) == INPUT_SHAPE[:2]
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(recon)))

    def test_invalid_encoder_raises(self):
        with pytest.raises(TypeError, match="encoder"):
            MaskedAutoencoder(encoder="not_a_model")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        # The mask path is NOT random under `training=False`: `PatchMasking._create_mask`
        # returns `ops.zeros(...)` whenever `training` is falsy, so `reconstruction` is
        # deterministic here. Comparing ONLY `encoded` left the decoder's restored
        # weights asserted by nothing at all — the encoder is a plain functional
        # `keras.Model` and cannot exercise the container shapes the decoder uses.
        before = model(x, training=False)
        before_encoded = keras.ops.convert_to_numpy(before["encoded"])
        before_recon = keras.ops.convert_to_numpy(before["reconstruction"])
        # Anti-vacuity: an all-zero reconstruction would satisfy any allclose below.
        assert np.abs(before_recon).max() > 0.0

        path = os.path.join(str(tmp_path), "mae.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = loaded(x, training=False)

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(
            before_encoded,
            keras.ops.convert_to_numpy(after["encoded"]),
            atol=1e-4,
            err_msg="MAE encoded output differs after round-trip")
        np.testing.assert_allclose(
            before_recon,
            keras.ops.convert_to_numpy(after["reconstruction"]),
            atol=1e-4,
            err_msg="MAE reconstruction differs after round-trip: the DECODER's "
                    "weights were not restored")

    def test_the_round_trip_assertion_would_catch_a_lost_decoder(self, tmp_path):
        """RED proof, kept in the suite: transplant fresh decoder weights into the
        loaded model and require the `reconstruction` comparison above to fire.

        A shape-and-count assertion cannot do this — the transplanted model has an
        identical output shape and an identical `count_params()`.
        """
        model = _model()
        x = _images()
        before = model(x, training=False)
        before_recon = keras.ops.convert_to_numpy(before["reconstruction"])

        path = os.path.join(str(tmp_path), "mae.keras")
        model.save(path)
        loaded = keras.models.load_model(path)

        fresh = keras.models.load_model(path)
        rng = np.random.default_rng(99)
        for v in fresh.weights:
            v.assign(keras.ops.convert_to_tensor(
                rng.normal(scale=0.5, size=v.shape).astype("float32")))
        sabotaged = keras.ops.convert_to_numpy(
            fresh(x, training=False)["reconstruction"])

        # The two quantities the trap preserves are still equal ...
        assert sabotaged.shape == before_recon.shape
        assert fresh.count_params() == loaded.count_params()
        # ... while the VALUE assertion fires.
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(before_recon, sabotaged, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
