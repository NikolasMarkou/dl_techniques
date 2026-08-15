"""
Test suite for the DarkIR low-light image restoration model.

DarkIR is a pure functional U-Net (create_darkir_model) using FreMLP (FFT path).
Covers a forward pass, the use_side_loss variant, and the M2 full .keras
save -> load -> identical-output round-trip. NHWC float32 input -> restored
image (B, H, W, 3).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.darkir.model import create_darkir_model

SPATIAL = 32


def _images(batch=2):
    return np.random.default_rng(0).random(
        (batch, SPATIAL, SPATIAL, 3)).astype("float32")


def _primary(out):
    return out[0] if isinstance(out, (list, tuple)) else out


class TestForward:

    def test_forward_shape(self):
        model = create_darkir_model(img_channels=3, width=16)
        out = model(_images(), training=False)
        y = _primary(out)
        assert tuple(y.shape) == (2, SPATIAL, SPATIAL, 3)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(y)))

    def test_side_loss_variant_builds(self):
        enc_blk_nums = [1, 2, 3]
        model = create_darkir_model(
            img_channels=3, width=16, use_side_loss=True,
            enc_blk_nums=enc_blk_nums, dec_blk_nums=[3, 1, 1],
        )
        out = model(_images(), training=False)
        # Two outputs, and the second one is at BOTTLENECK resolution. The
        # `_primary(out)`-only assertion this replaced passed identically for a
        # single-output model, so it could not see C-23 at all.
        assert isinstance(out, (list, tuple)) and len(out) == 2, type(out)
        assert _primary(out).shape[1:] == (SPATIAL, SPATIAL, 3)

        factor = 2 ** len(enc_blk_nums)
        assert tuple(out[1].shape[1:]) == (
            SPATIAL // factor, SPATIAL // factor, 3,
        ), out[1].shape


class TestKerasRoundTrip:

    def test_save_load_identical(self, tmp_path):
        model = create_darkir_model(img_channels=3, width=16)
        x = _images()
        before = keras.ops.convert_to_numpy(_primary(model(x, training=False)))

        path = os.path.join(str(tmp_path), "darkir.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(_primary(loaded(x, training=False)))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Outputs differ after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
