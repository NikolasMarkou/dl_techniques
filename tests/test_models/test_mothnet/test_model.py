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


class TestTrainHebbianOnAnUnbuiltModel:
    """`train_hebbian` is MothNet's ONLY training entry point and had no test.

    Every documented path -- the class docstring's "Usage Paradigms" 1 and 2 and
    the four README snippets, including the README's own "CRITICAL TEST" -- goes
    `MothNet(...)` then straight to `train_hebbian(...)`. On an unbuilt instance
    `self.antennal_lobe` is still the `None` that `__init__` assigned, so the
    first mini-batch was `None(batch_x_tensor, training=True)`. Only the class
    docstring's `Example` block, which calls `model.build(...)` first, ever
    worked.
    """

    def test_train_hebbian_builds_itself_on_an_unbuilt_instance(self):
        model = MothNet(num_classes=NUM_CLASSES)
        assert not model.built, "fixture precondition: the defect needs an unbuilt model"

        x = _features(batch=8)
        y = keras.utils.to_categorical(
            np.arange(8) % NUM_CLASSES, num_classes=NUM_CLASSES
        )

        history = model.train_hebbian(x, y, epochs=1, batch_size=4, verbose=0)

        assert model.built
        assert model.antennal_lobe is not None
        assert len(history["loss"]) == 1
        assert np.isfinite(history["loss"][0])

    def test_train_hebbian_infers_al_units_from_the_training_data(self):
        """The self-build must use the TRAINING data's feature count -- `al_units`
        defaults to the input dimension, so building against a wrong shape would
        silently produce a differently-sized antennal lobe."""
        model = MothNet(num_classes=NUM_CLASSES, al_units=None)
        x = _features(batch=4)
        y = keras.utils.to_categorical(
            np.arange(4) % NUM_CLASSES, num_classes=NUM_CLASSES
        )

        model.train_hebbian(x, y, epochs=1, batch_size=4, verbose=0)

        assert model.antennal_lobe.units == NUM_FEATURES

    def test_train_hebbian_on_an_already_built_model_is_unchanged(self):
        """The self-build must be a no-op for the one path that already worked --
        rebuilding would discard the Hebbian weights learned so far."""
        model = MothNet(num_classes=NUM_CLASSES)
        model.build((None, NUM_FEATURES))
        readout_before = model.readout

        x = _features(batch=4)
        y = keras.utils.to_categorical(
            np.arange(4) % NUM_CLASSES, num_classes=NUM_CLASSES
        )
        model.train_hebbian(x, y, epochs=1, batch_size=4, verbose=0)

        assert model.readout is readout_before
