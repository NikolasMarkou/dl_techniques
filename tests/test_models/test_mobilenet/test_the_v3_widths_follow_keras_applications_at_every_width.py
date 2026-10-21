"""MobileNetV3 follows one reference, Keras Applications, at every width.

The head already followed ``keras.applications.MobileNetV3{Small,Large}``
(``last_point`` widened only above 1.0); since plan-2026-10-14T062616-81938e81
step 10.3 the backbone does too:

* the stem is 16 channels at every width (``Conv2D(16, ...)``, no ``alpha``);
* a block's expanded width is ``_depth(infilters * expansion)``: the block's
  actual input width times the table's expansion ratio, not torchvision's
  ``adjust_channels(exp) = _make_divisible(exp * alpha)``;
* the first row (expansion 1) has no expansion conv (``if block_id:``);
* the last conv is ``_depth(last block width * 6)``.

Before the change the port matched Keras Applications only at
``width_multiplier=1.0``, and there with 320 extra parameters (the first row's
16 x 16 expansion conv and its BatchNorm); at 0.75 it was 65,584 (Large) and
21,264 (Small) parameters short. The reference is a live model built with
``weights=None`` and ``minimalistic=False``; both sides run at 224 x 224 x 3
with 1000 classes, with and without the top.
"""

import functools

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mobilenet.mobilenet_v3 import MobileNetV3

CTOR = {"small": "MobileNetV3Small", "large": "MobileNetV3Large"}
#: 0.75 and 1.0 are the widths Keras Applications ships weights for; the other
#: three cover a narrower stem than 16, widths where ``_depth`` rounds up, and
#: the widened head above 1.0.
WIDTHS = (0.25, 0.5, 0.75, 1.0, 1.25)


@functools.lru_cache(maxsize=None)
def _reference(variant: str, alpha: float, include_top: bool) -> keras.Model:
    return getattr(keras.applications, CTOR[variant])(
        input_shape=(224, 224, 3), alpha=alpha, minimalistic=False,
        include_top=include_top, weights=None, classes=1000,
        include_preprocessing=False)


def _port(variant: str, alpha: float, include_top: bool) -> MobileNetV3:
    model = MobileNetV3(num_classes=1000, variant=variant,
                        width_multiplier=alpha, include_top=include_top)
    model(np.zeros((1, 224, 224, 3), "float32"))
    return model


def _shapes(model: keras.Model) -> list:
    """Every weight's shape with leading 1s dropped, sorted.

    A 1x1 ``Conv2D`` head kernel ``(1, 1, C, K)`` and a ``Dense`` kernel
    ``(C, K)`` compare equal; nothing else changes."""
    out = []
    for w in model.weights:
        shape = tuple(int(d) for d in w.shape)
        while len(shape) > 1 and shape[0] == 1:
            shape = shape[1:]
        out.append(shape)
    return sorted(out)


@pytest.mark.parametrize("include_top", [True, False])
@pytest.mark.parametrize("alpha", WIDTHS)
@pytest.mark.parametrize("variant", ["small", "large"])
def test_the_count_equals_keras_applications(variant, alpha, include_top):
    port = _port(variant, alpha, include_top)
    reference = _reference(variant, alpha, include_top)
    assert port.count_params() == reference.count_params(), (
        f"{variant} x{alpha} include_top={include_top}: "
        f"{port.count_params():,} != keras.applications "
        f"{reference.count_params():,}")


@pytest.mark.parametrize("alpha", WIDTHS)
@pytest.mark.parametrize("variant", ["small", "large"])
def test_every_weight_shape_equals_keras_applications(variant, alpha):
    """The count can balance by accident; the multiset of shapes cannot."""
    port = _shapes(_port(variant, alpha, True))
    reference = _shapes(_reference(variant, alpha, True))
    missing = sorted(set(reference) - set(port))
    extra = sorted(set(port) - set(reference))
    assert port == reference, (
        f"{variant} x{alpha}: shapes only in keras.applications {missing}, "
        f"only in the port {extra}")


@pytest.mark.parametrize("alpha", WIDTHS)
@pytest.mark.parametrize("variant", ["small", "large"])
def test_each_block_expands_to_the_keras_applications_width(variant, alpha):
    reference = [
        layer.input.shape[-1] for layer in _reference(variant, alpha, False).layers
        if isinstance(layer, keras.layers.DepthwiseConv2D)]
    port = [int(b.dw1.kernel.shape[-2]) for b in _port(variant, alpha, False).blocks]
    assert port == reference


@pytest.mark.parametrize("variant", ["small", "large"])
def test_the_first_row_has_no_expansion_conv(variant):
    port = _port(variant, 1.0, False)
    assert port.blocks[0].expand_conv is None
    assert all(b.expand_conv is not None for b in port.blocks[1:])
