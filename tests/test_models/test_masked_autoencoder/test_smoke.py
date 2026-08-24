"""Permanent build+forward smoke test for the masked_autoencoder family.

Part of the 2026-06-15 model build/forward sweep (plan_2026-06-15_b5cec9e4).

Originally REPORT-ONLY: the whole body sat inside a `try` whose `except` called
`pytest.xfail`, so any construction or forward break was reported as a pass. That
converted a real defect into a green test -- this file's own encoder violated MAE's
encoder/decoder scale contract and the xfail would have swallowed the resulting
`ValueError`. The xfail is now narrowed to the import alone, which is the only
part of this smoke test that legitimately depends on something outside the
package; a build or forward break is a failure.

What survived that narrowing was a contract of `isinstance(out, dict)` plus
per-value finiteness, which a forward returning `{"anything": 0.0}` satisfies. It
now asserts the exact key set and each value's shape.

Keys and shapes MEASURED at 32x32 input, patch_size 16, mask_ratio 0.75: a
``reconstruction`` and a ``masked_input`` at the input geometry, the boolean-ish
``mask`` over the 4 patches, and the ``encoded`` feature map at the encoder's 16x
downsample. ``mask``'s length is the assertion that ties this test to
``patch_size``: at 32x32 there are exactly ``(32/16)**2 = 4`` patches.
"""

import keras
import numpy as np
import pytest

from ..smoke_contract_oracle import assert_finite
from .conftest import tiny_encoder

BATCH, IMAGE_SIZE, PATCH_SIZE, CHANNELS = 2, 32, 16, 3
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4
ENCODER_STRIDE, ENCODER_WIDTH = 16, 16

EXPECTED_SHAPES = {
    "reconstruction": (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    "mask": (BATCH, NUM_PATCHES),
    "masked_input": (BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    "encoded": (
        BATCH,
        IMAGE_SIZE // ENCODER_STRIDE,
        IMAGE_SIZE // ENCODER_STRIDE,
        ENCODER_WIDTH,
    ),
}


def _tiny_conv_encoder(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)):
    """Smallest 4D-output encoder MAE expects: (B, H', W', C).

    Downsamples by 16x, matching the default `decoder_depth=4`. It was /4 until
    2026-08-15, which the constructor now rejects outright.
    """
    return tiny_encoder(
        image_size=input_shape[0],
        channels=input_shape[2],
        filters=(8, 16, 16, 16),
        activation="relu",
        name="tiny_mae_encoder",
    )


def _inputs():
    return np.random.rand(BATCH, IMAGE_SIZE, IMAGE_SIZE, CHANNELS).astype("float32")


def _assert_contract(out):
    """The smoke assertion. Shared with the meta-test so it is proven falsifiable."""
    assert isinstance(out, dict), f"expected a dict of tensors, got {type(out)}"
    assert set(out) == set(EXPECTED_SHAPES), sorted(out)
    for key, expected in EXPECTED_SHAPES.items():
        assert tuple(out[key].shape) == expected, f"{key}: {tuple(out[key].shape)}"
    assert_finite(out)


def test_smoke_build_and_forward():
    try:
        from dl_techniques.models.masked_autoencoder import create_mae_model
    except ImportError as exc:
        pytest.xfail(f"masked_autoencoder is not importable: {exc}")

    model = create_mae_model(
        _tiny_conv_encoder(),
        patch_size=PATCH_SIZE,
        mask_ratio=0.75,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
    )
    _assert_contract(model(_inputs(), training=False))
