"""SqueezeNet's bypass variant must really wire bypasses, and its head must
emit a class distribution.

Guard for C-9 (plan-2026-08-14T233721-d4f9beb2, step 32).

Two defects, both invisible to the pre-existing suite:

1. ``squeezenet_v1.py`` selected ``bypass_indices = [2, 4, 6, 8]`` for
   ``use_bypass="simple"``. Fire modules are named ``fire{idx+2}``, so fire3/5/7/9
   -- the channel-matched positions -- are ``idx = 1, 3, 5, 7``. The listed
   indices are all channel-MISMATCHED (plus an out-of-range 8), so the
   ``identity.shape[-1] == x.shape[-1]`` guard was false everywhere and
   ``create_squeezenet_v1("1.0_bypass")`` built a graph topologically identical
   to plain ``"1.0"`` while still reporting ``use_bypass == "simple"``.
   ``test_bypass_configurations`` asserts only that the model builds and the
   output shape is right, and ``test_model_variants`` asserts the stored flag --
   which is exactly how this survived.

2. ``squeezenet_v2.py``'s head selected ``'sigmoid'`` at ``num_classes == 2``
   under a comment reading "Softmax activation for binary classification". With
   ``Conv2D(2) -> GAP -> sigmoid`` the two outputs are independent and do not sum
   to 1, so the package's own documented ``num_classes=2`` nodule examples,
   compiled with ``categorical_crossentropy``, optimize something else.
   ``SqueezeNetV1`` softmaxes on the same argument.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.squeezenet.squeezenet_v1 import (
    SqueezeNetV1,
    create_squeezenet_v1,
)
from dl_techniques.models.squeezenet.squeezenet_v2 import SqueezeNoduleNetV2


def _add_layers(model):
    return [l for l in model.layers if isinstance(l, keras.layers.Add)]


def _channel_matched_positions(fire_configs):
    """Positions where a residual Add is dimensionally possible, derived from the
    variant config alone (input channels of fire idx == output of fire idx-1)."""
    widths = [c["e1x1"] + c["e3x3"] for c in fire_configs]
    return [idx for idx in range(1, len(widths)) if widths[idx - 1] == widths[idx]]


class TestSimpleBypassIsRealWiring:
    def test_bypass_variant_builds_add_layers(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        adds = _add_layers(model)
        assert len(adds) > 0, "1.0_bypass built no residual Add layers at all"

    def test_add_count_equals_the_channel_matched_positions(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        expected = _channel_matched_positions(
            SqueezeNetV1.MODEL_VARIANTS["1.0_bypass"]["fire_configs"]
        )
        adds = _add_layers(model)
        assert len(adds) == len(expected)
        # fire modules are named fire{idx+2}; the Add is add_fire{idx+2}
        assert sorted(l.name for l in adds) == sorted(
            f"add_fire{idx + 2}" for idx in expected
        )

    def test_bypass_variant_differs_topologically_from_plain(self):
        """Anti-vacuity: the flag alone is not the property. At HEAD the two
        graphs had an identical Add count (zero)."""
        plain = create_squeezenet_v1("1.0", num_classes=10)
        bypass = create_squeezenet_v1("1.0_bypass", num_classes=10)
        assert bypass.use_bypass == "simple"
        assert plain.use_bypass is False or plain.use_bypass == False  # noqa: E712
        assert len(_add_layers(bypass)) > len(_add_layers(plain))

    def test_bypass_model_still_forwards(self):
        model = create_squeezenet_v1("1.0_bypass", num_classes=10)
        out = model(np.random.rand(2, 224, 224, 3).astype("float32"))
        arr = keras.ops.convert_to_numpy(out)
        assert arr.shape == (2, 10)
        assert np.all(np.isfinite(arr))


class TestV2HeadEmitsAClassDistribution:
    @pytest.mark.parametrize("num_classes", [2, 3])
    def test_outputs_sum_to_one(self, num_classes):
        model = SqueezeNoduleNetV2.from_variant(
            "v2", num_classes=num_classes, input_shape=(64, 64, 1)
        )
        out = keras.ops.convert_to_numpy(
            model(np.random.rand(4, 64, 64, 1).astype("float32"))
        )
        assert out.shape == (4, num_classes)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(4), rtol=1e-5, atol=1e-5)

    def test_v1_and_v2_agree_on_the_binary_head(self):
        v1 = SqueezeNetV1.from_variant("1.1", num_classes=2, input_shape=(64, 64, 3))
        v2 = SqueezeNoduleNetV2.from_variant("v2", num_classes=2, input_shape=(64, 64, 1))
        v1_out = keras.ops.convert_to_numpy(
            v1(np.random.rand(2, 64, 64, 3).astype("float32"))
        )
        v2_out = keras.ops.convert_to_numpy(
            v2(np.random.rand(2, 64, 64, 1).astype("float32"))
        )
        np.testing.assert_allclose(v1_out.sum(axis=-1), np.ones(2), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(v2_out.sum(axis=-1), np.ones(2), rtol=1e-5, atol=1e-5)


class TestDocstringExampleIsConstructible:
    """The class docstring advertises `1.0_bypass` at (32, 32, 3)."""

    def test_thirty_two_pixel_example_builds_and_forwards(self):
        model = SqueezeNetV1.from_variant(
            "1.0_bypass", num_classes=10, input_shape=(32, 32, 3)
        )
        out = keras.ops.convert_to_numpy(
            model(np.random.rand(2, 32, 32, 3).astype("float32"))
        )
        assert out.shape == (2, 10)
