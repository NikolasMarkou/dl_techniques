"""
`history["loss"]` must be an epoch mean, like every other Keras model -- D-133.

`MAE.train_step` / `test_step` return two keys carrying the SAME quantity:

    self.reconstruction_loss_tracker.update_state(loss)
    return {"loss": loss,                                    # <- raw last batch
            "reconstruction_loss": tracker.result()}         # <- epoch mean

So `history["loss"]` was the final batch's value while `reconstruction_loss`,
fed from that same variable one line above, was the running mean. Nothing warned,
and the two keys silently disagreed by the width of the batch-to-batch spread.

The detector has to be a MULTI-BATCH fit whose per-batch losses genuinely differ,
or the mean and the last batch coincide and the test is vacuous either way. This
file therefore asserts BOTH halves:

  * the two reported keys agree exactly (the fix), and
  * the per-batch losses in the run actually vary (the control) -- without it a
    single-batch fit would pass against the defect.

RED proof: restoring `"loss": loss` at both sites fails
`test_the_two_keys_agree` (verified 2026-08-21).
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.masked_autoencoder import MaskedAutoencoder

IMAGE_SIZE, PATCH_SIZE, CHANNELS = 32, 16, 3
SEED = 20260821


def _tiny_encoder():
    inp = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
    x = inp
    for _ in range(4):
        x = keras.layers.Conv2D(16, 3, strides=2, padding="same")(x)
    return keras.Model(inp, x)


@pytest.fixture(scope="module")
def trained():
    keras.utils.set_random_seed(SEED)
    model = MaskedAutoencoder(
        encoder=_tiny_encoder(),
        patch_size=PATCH_SIZE,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3))

    rng = np.random.default_rng(SEED)
    # Deliberately heterogeneous batches: three constant-valued images at very
    # different scales, so per-batch losses cannot coincide by construction.
    x = np.concatenate(
        [np.full((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), v, "float32")
         for v in (0.0, 5.0, 50.0)]
    ) + rng.normal(0, 1e-3, (3, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)).astype("float32")

    per_batch = []
    cb = keras.callbacks.LambdaCallback(
        on_train_batch_end=lambda b, logs: per_batch.append(float(logs["loss"]))
    )
    history = model.fit(x, epochs=1, batch_size=1, shuffle=False, verbose=0, callbacks=[cb])
    return history, per_batch


def test_the_two_keys_agree(trained):
    history, _ = trained
    loss = history.history["loss"][-1]
    recon = history.history["reconstruction_loss"][-1]
    assert loss == pytest.approx(recon, rel=1e-6), (loss, recon)


def test_the_batches_actually_differed(trained):
    """Control: without this, a single-batch fit would pass against the defect."""
    _, per_batch = trained
    assert len(per_batch) >= 3, per_batch
    assert max(per_batch) > 2.0 * min(per_batch) + 1e-6, per_batch


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
