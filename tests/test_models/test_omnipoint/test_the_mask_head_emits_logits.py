"""The mask head emits unbounded linear logits, not probabilities.

D-010 (plan-2026-09-11T050223-1b47bcf6): ``MaskLoss`` wraps
``BinaryCrossentropy(from_logits=True)``, so a sigmoid on the head would be applied twice.
The mask head is a bare ``DPTDecoder`` built in ``OmniPoint.__init__`` with
``output_activation="linear"``; nothing but this file fails if that argument is dropped or
changed, because the default of ``DPTDecoder`` is also ``"linear"`` and a sigmoid swapped in
keeps every shape the other tests assert.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.depth_anything.components import DPTDecoder
from dl_techniques.models.vision.omnipoint import OmniPoint

_IMAGE_SHAPE = (28, 28, 3)


def _model() -> OmniPoint:
    return OmniPoint(
        image_shape=_IMAGE_SHAPE, vit_scale="pico", decoder_dims=[16, 8, 8, 4],
        metric_hidden_dim=8,
    )


def _images() -> np.ndarray:
    return np.random.default_rng(0).uniform(0.0, 1.0, (2,) + _IMAGE_SHAPE).astype("float32")


def test_the_mask_head_is_a_bare_one_channel_dpt_decoder():
    model = _model()
    head = model.mask_head
    assert type(head) is DPTDecoder
    assert head.name == "mask_head"
    assert head.output_conv.filters == 1
    assert head.dims == [16, 8, 8, 4]


def test_the_mask_head_applies_no_output_activation():
    head = _model().mask_head
    assert head.output_conv.activation is keras.activations.linear


@pytest.mark.parametrize("bias", [1e3, -1e3])
def test_the_mask_logit_leaves_the_unit_interval(bias):
    """A sigmoid or any bounded activation on the head would keep ``mask_logit`` in (0, 1)."""
    model = _model()
    x = _images()
    model(x)
    conv = model.mask_head.output_conv
    conv.bias.assign(np.full(tuple(conv.bias.shape), bias, dtype="float32"))
    mask_logit = keras.ops.convert_to_numpy(model(x, training=False)[3])
    if bias > 0:
        assert mask_logit.min() > 1.0
    else:
        assert mask_logit.max() < 0.0
