"""C-39: `PW_FNet`'s depth is FIXED at 2, and `call` constructs no layers.

Two defects, one finding:

1. The docstring said "Length [of ``enc_blk_nums``] determines number of
   scales" and the constructor validated only that the two lists AGREE in
   length, while the body read exactly indices ``[0]`` and ``[1]``. So
   ``enc_blk_nums=[2, 2, 2]`` silently built a 2-level U-Net and DROPPED the
   third entry, and ``enc_blk_nums=[2]`` raised a bare ``IndexError`` from the
   middle of ``__init__`` rather than from the validation block.
2. ``call`` constructed two ``keras.layers.AveragePooling2D`` OBJECTS on every
   trace -- untracked, absent from ``model.layers``, rebuilt per call -- where
   ``keras.ops.average_pool`` is a stateless one-liner.
"""

import numpy as np
import keras
import pytest

from dl_techniques.models.pw_fnet.model import PW_FNet, _NUM_SCALES


def _tiny(**kwargs) -> PW_FNet:
    defaults = dict(
        img_channels=3, width=8, middle_blk_num=1,
        enc_blk_nums=[1, 1], dec_blk_nums=[1, 1],
    )
    defaults.update(kwargs)
    return PW_FNet(**defaults)


class TestTheDepthIsFixedAndSaysSo:
    """A wrong-length block list must be REFUSED, by name, from validation."""

    @pytest.mark.parametrize("blocks", [[2], [2, 2, 2], [1, 1, 1, 1]])
    def test_a_wrong_length_raises_from_validation(self, blocks):
        with pytest.raises(ValueError) as excinfo:
            PW_FNet(enc_blk_nums=list(blocks), dec_blk_nums=list(blocks))

        message = str(excinfo.value)
        assert "enc_blk_nums" in message and "dec_blk_nums" in message, (
            f"ASSERT-NAMED-VALIDATION-ERROR: the error must name the offending "
            f"parameters; got {message!r}"
        )
        assert str(len(blocks)) in message, (
            "the error must report the length it was given"
        )

    def test_a_third_entry_is_not_silently_dropped(self):
        """The exact defect: `[2, 2, 2]` used to build a 2-level network."""
        two_level = _tiny(enc_blk_nums=[2, 2], dec_blk_nums=[2, 2])
        two_level(np.zeros((1, 32, 32, 3), dtype="float32"))

        with pytest.raises(ValueError):
            PW_FNet(
                img_channels=3, width=8, middle_blk_num=1,
                enc_blk_nums=[2, 2, 2], dec_blk_nums=[2, 2, 2],
            )

    def test_a_short_list_does_not_raise_indexerror(self):
        """`[2]` used to die on `enc_blk_nums[1]` deep inside `__init__`."""
        with pytest.raises(ValueError):
            PW_FNet(enc_blk_nums=[2], dec_blk_nums=[2])

        # An IndexError is NOT an acceptable pass for the test above: pin it.
        try:
            PW_FNet(enc_blk_nums=[2], dec_blk_nums=[2])
        except IndexError:  # pragma: no cover - this is the defect
            pytest.fail(
                "ASSERT-NOT-INDEXERROR: `[2]` still escapes validation and "
                "raises IndexError from inside the constructor body"
            )
        except ValueError:
            pass

    def test_the_return_arity_is_what_pins_the_depth(self):
        """Three outputs, always -- the reason N scales was refused."""
        model = _tiny()
        outputs = model(np.zeros((2, 32, 32, 3), dtype="float32"))
        assert isinstance(outputs, list) and len(outputs) == 3
        assert tuple(outputs[0].shape) == (2, 32, 32, 3)
        assert tuple(outputs[1].shape) == (2, 16, 16, 3)
        assert tuple(outputs[2].shape) == (2, 8, 8, 3)
        assert _NUM_SCALES == 2

    def test_the_valid_length_still_builds(self):
        """Anti-vacuity: the supported configuration is untouched."""
        model = _tiny(enc_blk_nums=[2, 3], dec_blk_nums=[1, 2])
        model(np.zeros((1, 32, 32, 3), dtype="float32"))
        assert len(model.encoder_level1) == 2
        assert len(model.encoder_level2) == 3
        assert len(model.decoder_level2) == 1
        assert len(model.decoder_level1) == 2


class TestCallCreatesNoLayers:
    """`call` must not construct layer objects per trace."""

    def test_layer_count_is_stable_across_two_forward_passes(self):
        model = _tiny()
        x = np.random.rand(1, 32, 32, 3).astype("float32")

        model(x)
        layers_after_first = len(model.layers)
        weights_after_first = len(model.weights)

        model(x)
        model(x)

        assert len(model.layers) == layers_after_first, (
            f"ASSERT-NO-LAYER-PER-CALL: model.layers grew from "
            f"{layers_after_first} to {len(model.layers)} across forward "
            f"passes -- `call` is constructing layer objects."
        )
        assert len(model.weights) == weights_after_first

    def test_no_pooling_layer_is_CONSTRUCTED_during_a_forward_pass(
            self, monkeypatch):
        """Count CONSTRUCTIONS -- the only detector that sees this defect.

        MEASURED: the two `AveragePooling2D` objects were never TRACKED even
        while they existed, so `len(model.layers)`, `len(model.weights)` and a
        walk of the layer tree are all BLIND to them. Re-injecting the defect
        leaves every one of those assertions green (10 passed). A construction
        counter on the class is what actually fires.
        """
        constructions = []
        original_init = keras.layers.AveragePooling2D.__init__

        def counting_init(self, *args, **kwargs):
            constructions.append(1)
            original_init(self, *args, **kwargs)

        model = _tiny()
        x = np.random.rand(1, 32, 32, 3).astype("float32")
        model(x)  # first trace, before the counter is installed

        monkeypatch.setattr(
            keras.layers.AveragePooling2D, "__init__", counting_init)
        model(x)
        model(x)

        assert not constructions, (
            f"ASSERT-NO-POOLING-CONSTRUCTED-IN-CALL: {len(constructions)} "
            f"AveragePooling2D object(s) were constructed during 2 forward "
            f"passes -- `call` is building untracked layers per trace."
        )

    def test_the_downsampled_targets_are_numerically_unchanged(self):
        """`ops.average_pool` must equal the layer it replaced, exactly."""
        from keras import ops

        x = np.random.rand(2, 32, 32, 3).astype("float32")
        by_layer = np.asarray(keras.layers.AveragePooling2D(pool_size=2)(x))
        by_op = np.asarray(
            ops.average_pool(x, pool_size=2, strides=2, padding="valid"))

        assert by_layer.shape == by_op.shape
        assert np.max(np.abs(by_layer - by_op)) == 0.0, (
            "ASSERT-POOL-BIT-IDENTICAL: the op must reproduce the layer exactly"
        )
