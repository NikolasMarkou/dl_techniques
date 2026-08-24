"""RED proof for MAE's encoder/decoder scale contract (review finding C-12).

`ConvDecoder` upsamples exactly 2x per entry in `decoder_dims`, so the encoder
must downsample by exactly `2 ** len(decoder_dims)`. The module docstring
documented that contract; nothing enforced it. A `/4` encoder with the default
`decoder_depth=4` builds, runs a forward pass, and returns a 128x128
reconstruction for a 32x32 input — the failure only appears as a broadcast
error inside `compute_loss` on the first `fit()` step.

Two assertions, two independent mutations:

1. `test_mismatched_scale_raises_at_construction` — the shipped-fixture
   configuration must raise a `ValueError` naming both the encoder feature-map
   size and the decoder's upsampling factor, at construction. RED against the
   unfixed `mae.py`: no exception at all (`DID NOT RAISE`).
2. `test_matched_configuration_trains` — anti-vacuity. The guard must not
   reject a valid configuration, and a real `fit()` step on the documented
   configuration must succeed. RED against a guard that raises unconditionally,
   and RED against the unfixed code's own test fixture (which is where the
   broadcast error fired).

CPU only: no GPU op is involved and no tolerance is asserted.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder import create_mae_model

INPUT_SHAPE = (32, 32, 3)


def _encoder(num_strided: int, shape=INPUT_SHAPE):
    """An encoder downsampling by `2 ** num_strided`."""
    inp = keras.Input(shape=shape)
    x = inp
    for i in range(num_strided):
        x = keras.layers.Conv2D(
            8 if i == 0 else 16, 3, strides=2, padding="same", activation="relu"
        )(x)
    return keras.Model(inp, x, name=f"mae_encoder_div{2 ** num_strided}")


def _images(batch=2):
    return np.random.default_rng(0).random((batch, *INPUT_SHAPE)).astype("float32")


class TestScaleContract:

    def test_mismatched_scale_raises_at_construction(self):
        """A /4 encoder with decoder_depth=4 (16x up) must be rejected."""
        with pytest.raises(ValueError) as excinfo:
            create_mae_model(
                encoder=_encoder(2),  # 32x32 -> 8x8
                patch_size=16,
                input_shape=INPUT_SHAPE,
            )
        message = str(excinfo.value)
        # The message must name both sides of the contract, not just say
        # "shape mismatch": 8x8 is what the encoder produced, 16x is what the
        # decoder does to it.
        assert "8x8" in message, message
        assert "16x" in message, message
        assert "128x128" in message, message

    def test_mismatch_is_not_deferred_to_the_loss(self):
        """The failure must be a construction-time ValueError, not a broadcast.

        Distinguishes the fix from 'it fails eventually anyway': at HEAD the
        exception that eventually fired came from `Sub` inside `compute_loss`
        during `fit()`, naming a TF node rather than the encoder or decoder.
        """
        with pytest.raises(ValueError) as excinfo:
            create_mae_model(
                encoder=_encoder(2), patch_size=16, input_shape=INPUT_SHAPE
            )
        message = str(excinfo.value)
        assert "Encoder/decoder scale mismatch" in message, message
        assert "Sub" not in message and "node" not in message, message

    def test_matched_configuration_trains(self):
        """ANTI-VACUITY: a /16 encoder builds AND survives a real fit() step."""
        model = create_mae_model(
            encoder=_encoder(4),  # 32x32 -> 2x2, decoder 16x -> 32x32
            patch_size=16,
            input_shape=INPUT_SHAPE,
        )
        recon = model(_images(), training=False)["reconstruction"]
        assert tuple(int(d) for d in recon.shape[1:3]) == INPUT_SHAPE[:2]

        model.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3))
        history = model.fit(_images(4), epochs=1, batch_size=2, verbose=0)
        loss = float(history.history["loss"][-1])
        assert np.isfinite(loss), f"non-finite training loss: {loss}"

    def test_explicit_decoder_dims_are_validated_too(self):
        """The contract is on len(decoder_dims), not on decoder_depth."""
        # A /4 encoder with two decoder entries (4x up) is valid...
        create_mae_model(
            encoder=_encoder(2), patch_size=16, input_shape=INPUT_SHAPE,
            decoder_dims=[32, 16],
        )
        # ...and three entries (8x up) is not.
        with pytest.raises(ValueError, match="Encoder/decoder scale mismatch"):
            create_mae_model(
                encoder=_encoder(2), patch_size=16, input_shape=INPUT_SHAPE,
                decoder_dims=[32, 16, 8],
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
