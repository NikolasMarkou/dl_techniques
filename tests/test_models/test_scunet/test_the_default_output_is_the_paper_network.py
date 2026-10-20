"""SCUNet's default output is the paper's network, tail included.

cszn/SCUNet ``models/network_scunet.py`` ends ``SCUNet.forward`` with
``x = self.m_tail(x + x1)``, ``m_tail = nn.Conv2d(dim, in_nc, 3, 1, 1,
bias=False)``: the long skip from the stem conv, then one bias-free 3x3 conv.
From 2026-10-10 (3b281907a) to 2026-10-15 the default ``include_top`` head
replaced BOTH with a 3-ConvBlock ``EnhancementHead`` on the ``up1`` output, so
the default model had no long skip and was not SCUNet (only
``include_top=False`` was). Since plan-2026-10-14T062616-81938e81 step 6.1
(D-022) the head IS ``m_tail`` (``block_dims=()``, ``output_use_bias=False``),
the model applies the long skip, and ``include_top=False`` returns the
``dim``-channel features the tail reads.

``test_golden_values.py`` pins the external evidence: the default model
reproduces, at ``atol=1e-8``, the values that file pinned before 3b281907a.
This file pins the mechanism, at 64 px (no pad, so no crop):

* the headed output is the head's single bias-free conv applied to the
  headless features, exactly;
* the headless features are ``up1(...) + x1``, recomputed from the model's own
  stages, exactly.
"""

import keras
import numpy as np
import pytest

from dl_techniques.models.vision.image_restoration.scunet.model import SCUNet

CONFIG = dict(in_nc=3, config=[1] * 7, dim=8, head_dim=4, window_size=4,
              input_resolution=64)
SIZE = 64


def _x():
    return np.random.default_rng(5).standard_normal((1, SIZE, SIZE, 3)).astype("float32")


def _pair():
    """A headed and a headless SCUNet with the SAME backbone weights."""
    keras.utils.set_random_seed(0)
    headed = SCUNet(**CONFIG)
    headless = SCUNet(include_top=False, **CONFIG)
    x = _x()
    headed(x)
    headless(x)
    by_path = {w.path.split("/", 1)[1]: w for w in headed.weights}
    for w in headless.weights:
        w.assign(by_path[w.path.split("/", 1)[1]])
    return headed, headless, x


def test_the_head_is_one_bias_free_conv_on_the_headless_features(golden_reference_device):
    with keras.device(golden_reference_device):
        headed, headless, x = _pair()
        y = keras.ops.convert_to_numpy(headed(x, training=False))
        f = headless(x, training=False)
        head = headed.enhancement_head
        assert [w.path.split("/", 1)[1] for w in head.weights] == [
            "enhancement_head/output_conv/kernel"]
        ref = keras.ops.convert_to_numpy(
            keras.ops.conv(f, head.output_conv.kernel, padding="same"))
    assert tuple(f.shape) == (1, SIZE, SIZE, CONFIG["dim"])
    assert y.shape == (1, SIZE, SIZE, CONFIG["in_nc"])
    assert float(np.abs(y).max()) > 1e-3
    np.testing.assert_array_equal(y, ref)


def test_the_headless_features_carry_the_long_skip(golden_reference_device):
    with keras.device(golden_reference_device):
        _, headless, x = _pair()
        f = keras.ops.convert_to_numpy(headless(x, training=False))
        m = headless
        x1 = m.m_head(x)
        x2 = m.m_down1(x1, training=False)
        x3 = m.m_down2(x2, training=False)
        x4 = m.m_down3(x3, training=False)
        h = m.m_body(x4, training=False)
        h = m.m_up3(h + x4, training=False)
        h = m.m_up2(h + x3, training=False)
        up1 = m.m_up1(h + x2, training=False)
        with_skip = keras.ops.convert_to_numpy(up1 + x1)
        without_skip = keras.ops.convert_to_numpy(up1)
    np.testing.assert_array_equal(f, with_skip)
    # Anti-vacuity: the skip moves the features, so the equality above can
    # tell a model that drops it.
    assert float(np.abs(with_skip - without_skip).max()) > 1e-3


@pytest.mark.parametrize("include_top, channels", [(True, 3), (False, 8)])
def test_the_output_channels_follow_the_flag(include_top, channels):
    model = SCUNet(include_top=include_top, **CONFIG)
    assert tuple(model(_x()).shape) == (1, SIZE, SIZE, channels)
