"""
M2 .keras round-trip test for ResNet.

ResNet(num_classes, blocks_per_stage, filters_per_stage, ...) -> classifier
logits (B, num_classes). Pins numerically-identical outputs across a full
save -> load cycle (deterministic forward).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.resnet.model import ResNet


def _model():
    return ResNet(
        num_classes=10,
        blocks_per_stage=[2, 2, 2, 2],
        filters_per_stage=[64, 128, 256, 512],
        block_type="basic",
        normalization_type="batch_norm",
        input_shape=(32, 32, 3),
    )


def _images(batch=2):
    return np.random.default_rng(0).random((batch, 32, 32, 3)).astype("float32")


class TestResNetRoundTrip:

    def test_forward_logits(self):
        out = _model()(_images(), training=False)
        assert tuple(out.shape) == (2, 10)

    def test_keras_round_trip(self, tmp_path):
        model = _model()
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "resnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="ResNet differs after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
