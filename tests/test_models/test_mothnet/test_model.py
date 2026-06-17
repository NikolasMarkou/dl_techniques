"""
Test suite for MothNet (bio-mimetic feature generator / classifier).

al_units is inferred from the input dimension at build time. Input is a 2D
tabular tensor (B, F); output is class logits (B, num_classes). Covers
construction (incl. a ValueError path), a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.mothnet.model import MothNet

NUM_FEATURES = 64
NUM_CLASSES = 10


def _features(batch=2):
    return np.random.default_rng(0).random((batch, NUM_FEATURES)).astype("float32")


class TestMothNet:

    def test_forward_logits(self):
        model = MothNet(num_classes=NUM_CLASSES)
        out = model(_features(), training=False)
        assert tuple(out.shape) == (2, NUM_CLASSES)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))

    def test_undefined_last_dim_raises(self):
        model = MothNet(num_classes=NUM_CLASSES)
        with pytest.raises(ValueError, match="Last dimension"):
            model.build((None, None))

    def test_keras_round_trip(self, tmp_path):
        model = MothNet(num_classes=NUM_CLASSES)
        x = _features()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "mothnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="MothNet differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
