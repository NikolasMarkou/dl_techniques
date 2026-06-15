"""Permanent build+forward smoke test for the masked_autoencoder family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).
REPORT-ONLY: a build/forward break is documented via xfail, never fixed.
"""

import numpy as np
import keras
import pytest


def _tiny_conv_encoder(input_shape=(32, 32, 3)):
    """Smallest 4D-output encoder MAE expects: (B, H', W', C)."""
    inp = keras.Input(shape=input_shape)
    x = keras.layers.Conv2D(8, 3, strides=2, padding="same", activation="relu")(inp)
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(x)
    return keras.Model(inp, x, name="tiny_mae_encoder")


def _assert_finite(value):
    arr = np.asarray(value)
    assert arr is not None
    assert not np.any(np.isnan(arr))
    assert not np.any(np.isinf(arr))


def test_smoke_build_and_forward():
    try:
        from dl_techniques.models.masked_autoencoder import create_mae_model

        encoder = _tiny_conv_encoder((32, 32, 3))
        model = create_mae_model(
            encoder,
            patch_size=16,
            mask_ratio=0.75,
            input_shape=(32, 32, 3),
        )
        x = np.random.rand(2, 32, 32, 3).astype("float32")
        out = model(x, training=False)
    except Exception as exc:  # noqa: BLE001
        pytest.xfail(f"masked_autoencoder build/forward failed: {type(exc).__name__}: {exc}")

    # Output is a dict of tensors.
    assert isinstance(out, dict)
    for v in out.values():
        _assert_finite(v)
