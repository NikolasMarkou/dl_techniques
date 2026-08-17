"""
Test suite for CoShNet (complex shearlet network).

Covers construction (including ValueError validation paths), the from_variant /
create_coshnet factory, a forward pass, and the M2 full .keras
save -> load -> identical-output round-trip.

`create_coshnet(variant, num_classes, input_shape)` -> CoShNet.from_variant.
NHWC float32 image input; the classifier head is `Dense(activation="softmax")`,
so it returns PROBABILITIES, not logits — compile with `from_logits=False`. The
method below used to be named `test_forward_logits_shape` and this line used to say
"logits"; nothing asserted the rows summed to 1, so the contradiction with
`model.py:515-521` survived.
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

    def test_forward_returns_a_probability_distribution(self):
        model = create_coshnet("base", NUM_CLASSES, INPUT_SHAPE)
        out = model(_images(), training=False)
        assert tuple(out.shape) == (2, NUM_CLASSES)
        values = keras.ops.convert_to_numpy(out)
        assert not np.any(np.isnan(values))
        # The head is `Dense(num_classes, activation="softmax")`. Asserting the shape
        # alone cannot tell probabilities from logits, which is how the suite came to
        # call this output "logits" while the module docstring correctly told callers
        # to compile with `from_logits=False`.
        np.testing.assert_allclose(values.sum(axis=-1), 1.0, atol=1e-5)
        assert np.all(values >= 0.0)
        assert np.all(values <= 1.0)


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


class TestDocumentedParameterCounts:
    """Re-derive every count `model.py`'s module docstring prints.

    The factory docstring 620 lines below used to carry a SECOND, contradicting set
    ("tiny (~50k parameters)", "base (~800k parameters)") — neither variant's number.
    The labels had shifted one rung when `nano` was added: ~55k is nano's figure, not
    tiny's. That set was deleted rather than corrected, and this class is what keeps
    the surviving one honest.
    """

    # The exact figures printed in `models/coshnet/model.py`'s module docstring,
    # measured at (32, 32, 3) with 10 classes.
    DOCUMENTED_TOTAL = {
        "nano": 55_282,
        "tiny": 101_850,
        "base": 927_632,
        "large": 4_630_858,
        "cifar10": 592_282,
        "imagenet": 5_627_466,
    }

    @pytest.mark.parametrize("variant", sorted(DOCUMENTED_TOTAL))
    def test_the_documented_count_is_the_measured_count(self, variant):
        model = create_coshnet(variant, num_classes=10, input_shape=(32, 32, 3))
        assert model.count_params() == self.DOCUMENTED_TOTAL[variant]

    def test_nano_trainable_share_is_the_documented_one(self):
        """The docstring's one trainable-vs-total figure. The shearlet filter bank is a
        large FIXED contribution, so quoting only `count_params()` would overstate what
        the optimizer touches by more than 2x on `nano`."""
        model = create_coshnet("nano", num_classes=10, input_shape=(32, 32, 3))
        trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
        assert trainable == 22_514
        assert trainable < model.count_params()

    def test_the_factory_docstring_no_longer_restates_a_count(self):
        """One home for the number: the module docstring. A second copy drifted once
        already and would drift again."""
        import inspect

        doc = inspect.getdoc(create_coshnet) or ""
        assert "~50k" not in doc
        assert "~800k" not in doc
