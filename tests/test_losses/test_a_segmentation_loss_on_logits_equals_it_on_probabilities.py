"""A segmentation loss on logits equals the same loss on their probabilities.

Guard for plan-2026-10-14T062616-81938e81 iteration 2 step 2.1 (D-143).
AccUNet now emits logits, so ``LossConfig`` carries ``from_logits``. For each of
the nine ``SegmentationWrapperLoss`` names and for one, two and three channels:
the loss with ``from_logits=True`` on a logit map ``z`` equals the loss with the
default ``from_logits=False`` on ``sigmoid(z)`` (one channel) or
``softmax(z)`` (more), within a bound derived from float32.

Two arms see what the value comparison alone cannot:
- one channel is a sigmoid, never a softmax over a single channel (the constant
  1, under which every loss would ignore the model);
- cross-entropy reads logits in log space, so a pixel whose sigmoid saturates
  still gets a gradient (the probability path clips it to zero, the dead zone
  ``sam2_video_loss.py`` documents).

The default path stays bit-identical for every other consumer; that was
measured once (60 cases, values and gradients) and is held by those consumers'
own tests, not re-pinned here.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf

from dl_techniques.losses.segmentation_loss import LossConfig
from dl_techniques.losses.segmentation_wrapper_loss import (
    SegmentationWrapperLoss,
    _LOSS_METHOD_MAP,
)
from tests.numerics import reassociation_atol

SHAPE = (2, 16, 16)


def _case(channels: int, seed: int = 0):
    rng = np.random.default_rng(seed + channels)
    z = rng.normal(0.0, 3.0, (*SHAPE, channels)).astype("float32")
    if channels == 1:
        p = keras.ops.convert_to_numpy(keras.ops.sigmoid(z))
        y = (rng.random((*SHAPE, 1)) > 0.5).astype("float32")
    else:
        p = keras.ops.convert_to_numpy(keras.ops.softmax(z, axis=-1))
        y = np.eye(channels, dtype="float32")[rng.integers(0, channels, SHAPE)]
    return z, p, y


@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("name", sorted(_LOSS_METHOD_MAP))
def test_the_loss_on_logits_equals_the_loss_on_their_probabilities(name, channels):
    z, p, y = _case(channels)
    on_probabilities = float(SegmentationWrapperLoss(name, LossConfig(num_classes=channels))(y, p))
    on_logits = float(SegmentationWrapperLoss(
        name, LossConfig(num_classes=channels, from_logits=True))(y, z))
    # The two paths differ by a sigmoid/softmax-then-log versus a log_sigmoid /
    # log_softmax per element (two rounded steps) and then one mean over every
    # element; reassociation_atol bounds that random-walk accumulation.
    bound = reassociation_atol([z.size], num_steps=2, scale=abs(on_probabilities))
    assert abs(on_logits - on_probabilities) <= bound, (name, channels, on_logits, on_probabilities, bound)


@pytest.mark.parametrize("name", ["dice", "tversky", "lovasz", "boundary"])
def test_one_channel_logits_are_a_sigmoid_not_a_softmax(name):
    z, p, y = _case(1)
    on_logits = float(SegmentationWrapperLoss(name, LossConfig(num_classes=1, from_logits=True))(y, z))
    on_ones = float(SegmentationWrapperLoss(name, LossConfig(num_classes=1))(y, np.ones_like(p)))
    on_probabilities = float(SegmentationWrapperLoss(name, LossConfig(num_classes=1))(y, p))
    assert on_logits != pytest.approx(on_ones, rel=1e-3)
    assert on_logits == pytest.approx(on_probabilities, rel=1e-5)


@pytest.mark.parametrize("channels", [1, 3])
def test_cross_entropy_on_logits_has_a_gradient_where_the_sigmoid_saturates(channels):
    # Every pixel confidently wrong: |z| = 30 puts each probability of the true
    # class below 1e-7, inside the probability path's clip.
    z = np.full((*SHAPE, channels), -30.0, dtype="float32")
    if channels == 1:
        y = np.ones((*SHAPE, 1), dtype="float32")
    else:
        z[..., 0] = 30.0
        y = np.eye(channels, dtype="float32")[np.ones(SHAPE, dtype=int)]
    logits = tf.Variable(z)
    with tf.GradientTape() as tape:
        loss = SegmentationWrapperLoss(
            "cross_entropy", LossConfig(num_classes=channels, from_logits=True))(y, logits)
    grad = keras.ops.convert_to_numpy(tape.gradient(loss, logits))
    assert np.isfinite(float(loss)) and float(loss) > 25.0
    # d(-log p_true)/dz_true = p_true - 1, about -1 per pixel before the mean.
    assert np.abs(grad).max() > 0.5 / np.prod(SHAPE)


def test_from_logits_survives_the_config_round_trip():
    loss = SegmentationWrapperLoss("combo", LossConfig(num_classes=2, from_logits=True))
    restored = SegmentationWrapperLoss.from_config(loss.get_config())
    assert restored.config.from_logits is True


def test_a_non_bool_from_logits_is_refused():
    with pytest.raises(ValueError, match="from_logits"):
        SegmentationWrapperLoss("dice", LossConfig(num_classes=1, from_logits="yes"))
