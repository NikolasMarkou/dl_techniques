"""
Test suite for CoShNet (complex shearlet network).

Covers construction (including ValueError validation paths), the from_variant /
create_coshnet factory, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.

`create_coshnet(variant, num_classes, input_shape)` -> CoShNet.from_variant.
NHWC float32 image input; classifier head returns logits (B, num_classes).
"""

import os
import keras
import pytest
import numpy as np

from dl_techniques.models.coshnet.model import CoShNet, create_coshnet

INPUT_SHAPE = (32, 32, 3)
NUM_CLASSES = 10


def _images(batch=2):
    return np.random.default_rng(0).random((batch, *INPUT_SHAPE)).astype("float32")


class TestConstruction:

    def test_create_coshnet_factory(self):
        model = create_coshnet("base", NUM_CLASSES, INPUT_SHAPE)
        assert isinstance(model, CoShNet)
        assert model.num_classes == NUM_CLASSES

    def test_from_variant(self):
        model = CoShNet.from_variant("base", num_classes=NUM_CLASSES,
                                     input_shape=INPUT_SHAPE)
        assert isinstance(model, CoShNet)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError):
            CoShNet.from_variant("nonexistent", num_classes=NUM_CLASSES,
                                 input_shape=INPUT_SHAPE)

    def test_invalid_num_classes_raises(self):
        with pytest.raises(ValueError):
            CoShNet(num_classes=0, input_shape=INPUT_SHAPE)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError):
            CoShNet(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE,
                    dropout_rate=1.5)


class TestForward:

    def test_forward_logits_shape(self):
        model = create_coshnet("base", NUM_CLASSES, INPUT_SHAPE)
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, NUM_CLASSES)
        assert not np.any(np.isnan(keras.ops.convert_to_numpy(out)))


class TestKerasRoundTrip:

    def test_save_load_identical(self, tmp_path):
        model = create_coshnet("base", NUM_CLASSES, INPUT_SHAPE)
        x = _images()
        before = keras.ops.convert_to_numpy(model(x, training=False))

        path = os.path.join(str(tmp_path), "coshnet.keras")
        model.save(path)
        loaded = keras.models.load_model(path)
        after = keras.ops.convert_to_numpy(loaded(x, training=False))

        # GPU fp32 reduction noise -> atol 1e-4 (SYSTEM invariant)
        np.testing.assert_allclose(before, after, atol=1e-4,
                                   err_msg="Outputs differ after .keras round-trip")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
