"""
Test suite for PFTSR (permuted self-attention super-resolution).

create_pft_sr(scale, variant) builds the model; window_size=8 so input H/W are
kept divisible by 8. NHWC float32 input (B, H, W, 3); at scale=2 the output is
the upsampled image (B, 2H, 2W, 3). Covers a forward pass and the M2 full
.keras save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.pft_sr.model import create_pft_sr, PFTSR

SIZE = 32
SCALE = 2


def _model():
    return create_pft_sr(scale=SCALE, variant="light")


def _images(batch=2):
    return np.random.default_rng(0).random((batch, SIZE, SIZE, 3)).astype("float32")


class TestPFTSR:

    def test_factory_construction(self):
        assert isinstance(_model(), PFTSR)

    def test_forward_upsamples(self):
        out = _model()(_images(), training=False)
        assert tuple(out.shape) == (2, SIZE * SCALE, SIZE * SCALE, 3)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            create_pft_sr(scale=2, variant="nonexistent")

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "pft_sr.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="PFTSR differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
