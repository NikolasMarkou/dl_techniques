"""MobileNetV2's first stage (``t=1``) has no expansion conv, as the references.

Keras Applications ``mobilenet_v2._inverted_res_block`` expands only
``if block_id:`` and torchvision ``InvertedResidual`` only
``if expand_ratio != 1``; the paper's Table 2 first bottleneck has ``t=1``.
Until plan-2026-10-14T062616-81938e81 step 10.3 the universal block always
built the expansion, so every variant carried ``C*C + 4*C`` extra parameters
(a ``C x C`` 1x1 conv and its BatchNorm, ``C = make_divisible(32 * alpha)``):
2,496 (large), 1,152 (medium), 672 (small), 320 (nano and pico). The
reference is a live ``keras.applications.MobileNetV2`` built with
``weights=None`` at the variant's ``alpha``, 224 x 224 x 3, 1000 classes.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.mobilenet.mobilenet_v2 import MobileNetV2

VARIANTS = sorted(MobileNetV2.MODEL_VARIANTS)


def _port(variant: str, include_top: bool = True) -> MobileNetV2:
    model = MobileNetV2.from_variant(variant, num_classes=1000,
                                     include_top=include_top)
    model(np.zeros((1, 224, 224, 3), "float32"))
    return model


def _reference(variant: str, include_top: bool = True) -> keras.Model:
    alpha = MobileNetV2.MODEL_VARIANTS[variant]["width_multiplier"]
    return keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), alpha=alpha, include_top=include_top,
        weights=None, classes=1000)


def _shapes(model: keras.Model) -> list:
    return sorted(tuple(int(d) for d in w.shape) for w in model.weights)


@pytest.mark.parametrize("include_top", [True, False])
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_count_equals_keras_applications(variant, include_top):
    port, reference = _port(variant, include_top), _reference(variant, include_top)
    assert port.count_params() == reference.count_params(), (
        f"{variant} include_top={include_top}: {port.count_params():,} != "
        f"keras.applications {reference.count_params():,}")


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_backbone_weight_shape_equals_keras_applications(variant):
    assert _shapes(_port(variant, False)) == _shapes(_reference(variant, False))


def test_only_the_first_block_skips_the_expansion():
    blocks = _port("medium", False).blocks
    assert blocks[0].expand_conv is None
    assert all(b.expand_conv is not None for b in blocks[1:])
